import json
from decimal import Decimal

import pytest

from banking import expected_database, make_case
from walkthrough import check_leakage, installed_versions, money_patterns, register_attempt


def test_leakage_catches_currency_formats_and_new_report_fields():
    case = make_case(54)
    report = expected_database(case).reports[0]
    for text in (
        case.case_id,
        f"${case.savings[0].statement_balance:,.2f}",
        f"{report.expected_interest:.2f}",
        f"{report.net_interest_before_run:.2f}",
        f"The expected interest is {report.expected_interest:.2f}.",
    ):
        result = check_leakage([{"markdown": text}], [54])
        assert result["exact_heldout_identifiers_balances_or_corrections"]
    assert not check_leakage([{"markdown": "Apply the two caps before choosing a package."}], [54])[
        "exact_heldout_identifiers_balances_or_corrections"
    ]
    import re

    assert any(re.search(pattern, "$11,631.60") for pattern in money_patterns(Decimal("11631.6")))
    assert any(re.search(pattern, "$11,631.60.") for pattern in money_patterns(Decimal("11631.6")))


def test_verification_attempts_are_append_only_and_reject_overlap(tmp_path):
    register_attempt(tmp_path, tmp_path / "first", [10, 11], "first-hash")
    with pytest.raises(ValueError, match="overlap"):
        register_attempt(tmp_path, tmp_path / "second", [11, 12], "second-hash")
    register_attempt(tmp_path, tmp_path / "third", [12, 13], "third-hash")
    rows = [
        json.loads(line)
        for line in (tmp_path / "verification-attempts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["seeds"] == [10, 11]


def test_installed_dependency_versions_are_recorded():
    assert set(installed_versions()) == {
        "pydantic",
        "pydantic-ai-slim",
        "anthropic",
        "openai",
        "httpx",
    }
    assert all(installed_versions().values())


def test_repeated_indices_are_reserved_before_calls_and_cannot_be_reused(tmp_path):
    register_attempt(tmp_path, tmp_path / "first", [3, 3], "first", [0, 1])
    with pytest.raises(ValueError, match="indices overlap"):
        register_attempt(tmp_path, tmp_path / "second", [3, 3], "second", [1, 2])
    register_attempt(tmp_path, tmp_path / "third", [3, 3], "third", [2, 3])
    rows = [json.loads(s) for s in (tmp_path / "repeated-attempts.jsonl").read_text().splitlines()]
    assert [r["run_indices"] for r in rows] == [[0, 1], [2, 3]]


def test_repeat_mode_makes_six_calls_for_three_pairs_instead_of_reusing_results(
    tmp_path, monkeypatch
):
    import argparse
    import asyncio

    import walkthrough
    from runner import source_hashes, write_json

    learned = tmp_path / "learned"
    learned.mkdir()
    write_json(
        learned / "config.json",
        {
            "source": source_hashes(),
            "installed_versions": installed_versions(),
            "student": "test-model",
            "effort": "none",
            "seed": 3,
            "teacher_seed": 54,
            "screening_seeds": [0, 1, 2, 3],
        },
    )
    write_json(learned / "agent.json", {"uuid": "example"})
    write_json(learned / "skill-pin.json", [])
    skills = [{"markdown": "An invented package comparison procedure."}]
    calls = []

    async def fake_episode(path, case, model, effort, skills=None, teaching=False):
        path.mkdir()
        calls.append(path.name)
        row = {"passed": bool(skills)}
        write_json(path / "tool-calls.json", [{"invocation": len(calls)}])
        write_json(path / "database.json", {"passed": bool(skills)})
        write_json(path / "result.json", row)
        return row

    monkeypatch.setattr(walkthrough, "episode", fake_episode)
    monkeypatch.setattr(walkthrough, "Hosted", lambda path: None)
    monkeypatch.setattr(walkthrough, "live_skills", lambda *args: skills)
    args = argparse.Namespace(
        learned_run=str(learned),
        output=str(tmp_path / "check"),
        count=3,
        repeat_case=True,
        first_run=0,
        first_seed=10000,
    )
    asyncio.run(walkthrough.verify(args))
    assert len(calls) == len(set(calls)) == 6
    assert calls[0:2] == ["without-skill-seed3-run0", "with-skill-seed3-run0"]
    assert calls[2:4] == ["with-skill-seed3-run1", "without-skill-seed3-run1"]
    summary = json.loads((tmp_path / "check" / "summary.json").read_text())
    assert summary["accepted"] and summary["distinct_cases"] == 1
    assert summary["distinct_tool_sequences"] == {"with-skill": 3, "without-skill": 3}
    assert summary["distinct_final_databases"] == {"with-skill": 1, "without-skill": 1}
    args.output = str(tmp_path / "duplicate")
    with pytest.raises(ValueError, match="indices overlap"):
        asyncio.run(walkthrough.verify(args))
    assert len(calls) == 6

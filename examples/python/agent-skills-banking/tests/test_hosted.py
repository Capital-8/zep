import json

import httpx
import pytest

from hosted import FAMILY, Hosted, readable_skills


def test_public_api_upload_and_call_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEP_API_KEY", "local-test-value")
    requests = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request.url.path, body))
        if request.method != "GET":
            assert request.headers["Idempotency-Key"]
        if request.url.path.endswith("/close"):
            return httpx.Response(
                202, json={"trace": {"verification": "none"}, "task": {"uuid": "close-task"}}
            )
        if "/tasks/" in request.url.path:
            return httpx.Response(200, json={"status": "succeeded"})
        if request.method == "GET":
            return httpx.Response(200, json={"verification": "external", "status": "closed"})
        return httpx.Response(200, json={"uuid": "example-trace"})

    hosted = Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler))
    hosted.create_agent()
    checkpoint = tmp_path / "upload.json"
    result = hosted.upload(
        "example-agent", [{"event_type": "decision", "content": "Example"}], checkpoint
    )
    assert result["events"] == 1
    assert requests[0][1]["memory_settings"]["auto_approval_task_families"] == [FAMILY]
    records = [json.loads(line) for line in (tmp_path / "zep.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert [record["status"] for record in records] == [200, 200, 200, 202, 200, 200]
    assert json.loads(checkpoint.read_text())["stage"] == "close_verified"
    hosted.complete_upload("example-agent", checkpoint)
    assert len([item for item in requests if item[0].endswith("/events")]) == 1
    assert "local-test-value" not in (tmp_path / "zep.jsonl").read_text()


def test_failed_service_call_is_preserved_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEP_API_KEY", "local-test-value")
    hosted = Hosted(
        tmp_path / "zep.jsonl", httpx.MockTransport(lambda _: httpx.Response(503, json={}))
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        hosted.create_agent()
    assert len((tmp_path / "zep.jsonl").read_text().splitlines()) == 1


def test_skill_requires_uuid_version_and_markdown():
    assert readable_skills({"items": [{"uuid": "example", "version": 1}]}) == []
    skill = {"uuid": "example", "version": 1, "markdown": "An example."}
    assert readable_skills({"items": [skill]}) == [skill]


def test_trace_uuid_survives_an_event_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEP_API_KEY", "local-test-value")

    def handler(request):
        if request.url.path.endswith("/events"):
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"uuid": "saved-trace"})

    hosted = Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler))
    checkpoint = tmp_path / "upload.json"
    with pytest.raises(RuntimeError, match="HTTP 503"):
        hosted.upload("agent", [{"content": "Example"}], checkpoint)
    assert json.loads(checkpoint.read_text())["trace_uuid"] == "saved-trace"
    with pytest.raises(RuntimeError, match="interrupted"):
        hosted.complete_upload("agent", checkpoint)
    assert len((tmp_path / "zep.jsonl").read_text().splitlines()) == 2


def test_pin_waits_for_a_terminal_compiler_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEP_API_KEY", "local-test-value")
    monkeypatch.setattr("hosted.time.sleep", lambda _: None)
    reads = 0

    def handler(request):
        nonlocal reads
        if request.url.path.endswith("/learning"):
            reads += 1
            return httpx.Response(200, json={"compilation_pending": reads == 1})
        if request.url.path.endswith("/learning-runs"):
            return httpx.Response(200, json={"items": [{"decision": "produced"}]})
        assert reads > 1
        return httpx.Response(
            200,
            json={"items": [{"uuid": "skill", "version": 1, "markdown": "An example procedure."}]},
        )

    hosted = Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler))
    assert hosted.wait("agent")["items"][0]["uuid"] == "skill"
    assert reads == 4

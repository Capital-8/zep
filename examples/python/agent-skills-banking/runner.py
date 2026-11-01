"""One PydanticAI loop, the same local tools, and a retrieved live Skill."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelRetry, capture_run_messages
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from audit import AuditedTransport
from banking import (
    DOCUMENTS,
    Case,
    Credit,
    Database,
    PositiveMoney,
    Report,
    calculate,
    make_case,
    quote_package,
    score,
)

SYSTEM = (
    "You are a banking service agent. Resolve the customer's request using the local "
    "banking tools and the bank's policies. The customer is already verified and has "
    "authorized the necessary correction credits and reports. Do not ask for more input. "
    "Use the records and policy evidence instead of inventing account facts or rates. "
    "Give a short final summary of the actions and calculations. "
    "If a relevant Skill is in the catalog, read the Skill before acting. "
    "The Skill catalog is: {catalog}"
)
USER = (
    "Please investigate the savings interest for statement month {month} on my two "
    "savings accounts. Correct each underpayment and save the discrepancy reports. "
    "You have my permission to complete these actions. My case ID is {case_id}."
)
TEACHING_REQUEST = (
    "For this teaching run, keep the action summary short. Then show a comparison table "
    "of every eligible, permitted checking/card package for each savings account, "
    "including its capped bonus contribution and the selected package. "
    "Use quote_interest to evaluate each package instead of reimplementing its arithmetic. "
    "After the tables, document the reusable procedure that another banking agent "
    "can follow on different account records. State the general selection rule with "
    "variables and policy caps. Do not use customer identifiers or case-specific "
    "amounts in the reusable procedure."
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir())
        if p.suffix == ".py" or p.name in {"uv.lock", "pyproject.toml"}
    }


def model_identifier(model_name: str) -> tuple[str, str]:
    provider, separator, name = model_name.partition(":")
    if not separator:
        provider = ""
        name = model_name
        if name.startswith("claude-"):
            provider = "anthropic"
        elif name.startswith(("gpt-", "o1", "o3", "o4")):
            provider = "openai"
    if provider not in {"anthropic", "openai"} or not name:
        raise ValueError("Use an anthropic:MODEL or openai:MODEL identifier.")
    return provider, name


def build_agent(
    case: Case,
    model: Model,
    effort: str,
    skills: list[dict[str, Any]],
    db: Database,
    calls: list[dict[str, Any]],
) -> Agent:
    catalog = [
        {
            "uuid": s["uuid"],
            "name": s.get("name", ""),
            "description": s.get("description", ""),
            "version": s["version"],
        }
        for s in skills
    ]
    anthropic_settings = AnthropicModelSettings(
        anthropic_thinking={"type": "disabled"} if effort == "none" else {"type": "adaptive"},
        anthropic_effort=None if effort == "none" else cast(Any, effort),
        max_tokens=16384,
        timeout=120,
        parallel_tool_calls=False,
    )
    if effort == "none":
        anthropic_settings["temperature"] = 0
    agent = Agent(
        model,
        instructions=SYSTEM.format(catalog=json.dumps(catalog)),
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort=cast(Any, effort),
            max_tokens=16384,
            timeout=120,
            parallel_tool_calls=False,
        )
        if isinstance(model, OpenAIResponsesModel)
        else anthropic_settings,
        retries=2,
    )

    def record(name: str, arguments: dict, result: Any) -> Any:
        calls.append({"name": name, "arguments": arguments, "result": result})
        return result

    @agent.tool_plain
    def get_case() -> dict:
        """Read the case's savings, checking, cards, and interest ledger for all months."""
        return record("get_case", {}, case.model_dump(mode="json"))

    @agent.tool_plain
    def list_documents() -> list[dict[str, str]]:
        """List the titles and IDs of all available bank policy documents."""
        return record(
            "list_documents",
            {},
            [{"id": key, "title": value[0]} for key, value in DOCUMENTS.items()],
        )

    @agent.tool_plain
    def read_document(document_id: str) -> dict:
        """Read the complete bank policy document with this ID."""
        if document_id not in DOCUMENTS:
            raise ModelRetry("The document ID is not in list_documents.")
        title, content = DOCUMENTS[document_id]
        return record(
            "read_document", {"document_id": document_id}, {"title": title, "content": content}
        )

    @agent.tool_plain
    def calculator(expression: str) -> str:
        """Evaluate +, -, *, / and parentheses. Results round half-up to cents: use the full formula in one call."""
        try:
            value = calculate(expression)
        except (ValueError, ArithmeticError, SyntaxError) as error:
            raise ModelRetry(str(error)) from error
        return record("calculator", {"expression": expression}, value)

    @agent.tool_plain
    def quote_interest(
        account_id: str, checking_account_id: str | None = None, card_id: str | None = None
    ) -> dict:
        """Quote statement interest for one proposed checking/card package. This does not post a credit."""
        try:
            quote = quote_package(case, account_id, checking_account_id, card_id)
        except ValueError as error:
            raise ModelRetry(str(error)) from error
        return record(
            "quote_interest",
            {
                "account_id": account_id,
                "checking_account_id": checking_account_id,
                "card_id": card_id,
            },
            quote.model_dump(mode="json"),
        )

    @agent.tool_plain
    def credit_interest(
        account_id: str,
        statement_month: str,
        amount: PositiveMoney,
        reason_code: str,
    ) -> dict:
        """Post one interest correction credit. Use the policy to determine its amount."""
        if account_id not in {a.account_id for a in case.savings}:
            raise ModelRetry("The savings account is not in this case.")
        credit = Credit(
            credit_id=f"credit-{account_id}-{statement_month}",
            account_id=account_id,
            statement_month=statement_month,
            amount=amount,
            reason_code=reason_code,
        )
        if any(c.credit_id == credit.credit_id for c in db.credits):
            raise ModelRetry("A correction for this account and month already exists.")
        db.credits.append(credit)
        return record(
            "credit_interest",
            credit.model_dump(mode="json"),
            {"saved": True, "credit_id": credit.credit_id},
        )

    @agent.tool_plain
    def file_discrepancy_report(report: Report) -> dict:
        """Save a discrepancy report in the local database after the correction credit."""
        credit = next((c for c in db.credits if c.credit_id == report.correction_credit_id), None)
        if not credit or credit.account_id != report.account_id:
            raise ModelRetry("The report must refer to a saved credit on the same account.")
        if any(
            r.account_id == report.account_id and r.statement_month == report.statement_month
            for r in db.reports
        ):
            raise ModelRetry("A report already exists for this account and month.")
        db.reports.append(report)
        return record("file_discrepancy_report", report.model_dump(mode="json"), {"saved": True})

    @agent.tool_plain
    def read_skill(skill_uuid: str) -> dict:
        """Read the complete Markdown of a Skill from the supplied live search catalog."""
        skill = next((s for s in skills if s["uuid"] == skill_uuid), None)
        if skill is None:
            raise ModelRetry("The Skill UUID is not in the current catalog.")
        return record(
            "read_skill",
            {"skill_uuid": skill_uuid},
            {"uuid": skill_uuid, "version": skill["version"], "markdown": skill["markdown"]},
        )

    return agent


async def episode(
    path: Path,
    case: Case,
    model_name: str = "claude-haiku-5-5",
    effort: str = "low",
    skills: list[dict[str, Any]] | None = None,
    test_model: Model | None = None,
    teaching: bool = False,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    skills = skills or []
    db, provider_calls, tool_calls = Database(), [], []
    write_json(
        path / "start.json",
        {
            "case": case.model_dump(mode="json"),
            "model": model_name,
            "effort": effort,
            "teaching": teaching,
            "source_sha256": source_hashes(),
            "skill_identity": [
                {
                    "uuid": s["uuid"],
                    "version": s["version"],
                    "sha256": hashlib.sha256(s["markdown"].encode()).hexdigest(),
                }
                for s in skills
            ],
        },
    )
    client = httpx.AsyncClient(transport=AuditedTransport(provider_calls, "agent"), timeout=120)
    if test_model is not None:
        model = test_model
    elif model_identifier(model_name)[0] == "openai":
        model = OpenAIResponsesModel(
            model_identifier(model_name)[1],
            provider=OpenAIProvider(
                openai_client=AsyncOpenAI(max_retries=0, timeout=120, http_client=cast(Any, client))
            ),
        )
    else:
        model = AnthropicModel(
            model_identifier(model_name)[1],
            provider=AnthropicProvider(
                anthropic_client=AsyncAnthropic(max_retries=0, timeout=120, http_client=client)
            ),
        )
    agent = build_agent(case, model, effort, skills, db, tool_calls)
    write_json(path / "settings.json", agent.model_settings)
    started = time.monotonic()
    transcript = []
    try:
        async with asyncio.timeout(900):
            with capture_run_messages() as transcript:
                result = await agent.run(
                    USER.format(month=case.statement_month, case_id=case.case_id)
                    + ("\n\n" + TEACHING_REQUEST if teaching else ""),
                    usage_limits=UsageLimits(request_limit=100, total_tokens_limit=1000000),
                )
        elapsed = time.monotonic() - started
        transcript = result.all_messages()
        (path / "messages.json").write_bytes(
            ModelMessagesTypeAdapter.dump_json(transcript, indent=2)
        )
        usage = result.usage
        if not result.output.strip():
            raise RuntimeError("The agent returned an empty final answer.")
        if any(c["status"] != 200 for c in provider_calls):
            raise RuntimeError("A provider call failed.")
        if test_model is None:
            input_count = sum(
                c["usage"].get("input_tokens", 0)
                + c["usage"].get("cache_read_input_tokens", 0)
                + c["usage"].get("cache_creation_input_tokens", 0)
                for c in provider_calls
            )
            output_count = sum(c["usage"].get("output_tokens", 0) for c in provider_calls)
            if (len(provider_calls), input_count, output_count) != (
                usage.requests,
                usage.input_tokens,
                usage.output_tokens,
            ):
                raise RuntimeError("Provider request or token accounting did not reconcile.")
        returns = [p for m in transcript for p in m.parts if isinstance(p, ToolReturnPart)]
        if len(returns) != len(tool_calls):
            raise RuntimeError("Tool return accounting did not reconcile.")
        validation = [p for m in transcript for p in m.parts if isinstance(p, RetryPromptPart)]
        reads = [c["result"] for c in tool_calls if c["name"] == "read_skill"]
        if skills and not reads:
            raise RuntimeError("The agent did not read a live Skill.")
        evaluation = score(case, db)
        row = {
            "case_id": case.case_id,
            "model": model_name,
            "effort": effort,
            "passed": evaluation["passed"],
            "tokens": usage.total_tokens,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "seconds": elapsed,
            "requests": usage.requests,
            "provider_calls": len(provider_calls),
            "failed_provider_calls": 0,
            "validation_corrections": len(validation),
            "skill_reads": [{"uuid": s["uuid"], "version": s["version"]} for s in reads],
            "termination": "final_answer",
            "final_answer": result.output,
        }
        write_json(path / "evaluation.json", evaluation)
        write_json(path / "result.json", row)
        print(
            f"{path.name}: {'PASS' if row['passed'] else 'FAIL'}; "
            f"{row['tokens']:,} tokens; {elapsed:.1f}s",
            flush=True,
        )
        return row
    except BaseException as error:
        write_json(path / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        (path / "messages.json").write_bytes(
            ModelMessagesTypeAdapter.dump_json(transcript, indent=2)
        )
        write_json(path / "provider-calls.json", provider_calls)
        write_json(path / "tool-calls.json", tool_calls)
        write_json(path / "database.json", db.model_dump(mode="json"))
        await client.aclose()


def trace_events(path: Path) -> list[dict[str, Any]]:
    messages = ModelMessagesTypeAdapter.validate_json((path / "messages.json").read_text())
    events = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                events.append(
                    {
                        "event_type": "tool_call",
                        "content": part.args_as_json_str(),
                        "tool_call_id": part.tool_call_id,
                        "context": {"tools": [{"name": part.tool_name}]},
                    }
                )
            elif isinstance(part, ToolReturnPart):
                events.append(
                    {
                        "event_type": "tool_result",
                        "content": json.dumps(part.content),
                        "tool_call_id": part.tool_call_id,
                        "context": {"tools": [{"name": part.tool_name}]},
                    }
                )
            elif isinstance(part, RetryPromptPart):
                events.append(
                    {
                        "event_type": "tool_result",
                        "tool_call_id": part.tool_call_id,
                        "content": part.model_response(),
                        "context": {"tools": [{"name": part.tool_name or "validation"}]},
                    }
                )
            elif isinstance(part, (TextPart, UserPromptPart)) and str(part.content).strip():
                events.append(
                    {
                        "event_type": "decision" if part.part_kind == "text" else "input_reference",
                        "content": str(part.content),
                    }
                )
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--skills-json")
    args = parser.parse_args()
    skills = json.loads(Path(args.skills_json).read_text()) if args.skills_json else []
    asyncio.run(episode(Path(args.output), make_case(args.seed), args.model, args.effort, skills))

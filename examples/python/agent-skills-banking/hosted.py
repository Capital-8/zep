"""The public Zep Python SDK, with an append-only call ledger."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from zep_cloud import AgentMemorySettings, AgentTraceCloseVerifier, AgentTraceEventContext, Zep

FAMILY = "banking.synthetic-monthly-correction"
QUERY = "Investigate savings interest, apply qualifying product boosts, credit shortfalls and save reports"


class LedgerTransport(httpx.BaseTransport):
    def __init__(self, ledger: Path, transport: httpx.BaseTransport | None = None):
        self.ledger = ledger
        self.transport = transport if transport is not None else httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        record: dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "status": None,
        }
        start = time.monotonic()
        try:
            response = self.transport.handle_request(request)
            record.update(
                status=response.status_code, request_id=response.headers.get("x-request-id")
            )
            return response
        finally:
            record["seconds"] = time.monotonic() - start
            with self.ledger.open("a") as file:
                file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.transport.close()


class Hosted:
    def __init__(self, ledger: Path, transport: httpx.BaseTransport | None = None):
        api_key = os.environ["ZEP_API_KEY"]
        base = os.getenv("ZEP_BASE_URL")
        if base is not None:
            if urlsplit(base).scheme != "https":
                raise ValueError("ZEP_BASE_URL must use HTTPS.")
            base = base.rstrip("/").removesuffix("/api/v4") + "/api/v4"
        self.ledger = ledger
        self.http_client = httpx.Client(
            transport=LedgerTransport(ledger, transport), timeout=120, follow_redirects=False
        )
        self.client = Zep(
            api_key=api_key,
            base_url=base,
            httpx_client=self.http_client,
            timeout=120,
            max_retries=0,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.http_client.close()

    def create_agent(self) -> dict[str, Any]:
        return self.client.agent.create(
            agent_id="simple-banking-" + uuid.uuid4().hex,
            name="Synthetic monthly interest correction demo",
            security_domain="simple-banking-demo",
            memory_settings=AgentMemorySettings(
                compilation_mode="on_close",
                approval="auto",
                auto_approval_task_families=[FAMILY],
            ),
            idempotency_key=str(uuid.uuid4()),
        ).model_dump(mode="json", by_alias=True, exclude_none=True)

    def search(self, agent: str) -> dict[str, Any]:
        results = self.client.agent.skill.search(
            agent,
            page_size=20,
            query=QUERY,
            task_family=FAMILY,
            include_markdown=True,
            markdown_format="agent",
        )
        return {
            "items": [
                skill.model_dump(mode="json", by_alias=True, exclude_none=True) for skill in results
            ]
        }

    def learning_state(self, agent: str) -> dict[str, Any]:
        return self.client.agent.learning.get(agent, task_family=FAMILY).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )

    def learning_runs(self, agent: str) -> dict[str, Any]:
        results = self.client.agent.learning.list_runs(agent, task_family=FAMILY)
        return results.response.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _save_checkpoint(self, checkpoint: Path | None, state: dict[str, Any]) -> None:
        if checkpoint is None:
            return
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, default=str) + "\n")
        temporary.replace(checkpoint)

    def complete_upload(self, agent: str, checkpoint: Path, seconds: int = 900) -> dict[str, Any]:
        saved = json.loads(checkpoint.read_text())
        if saved["stage"] not in {"close_accepted", "close_verified"}:
            raise RuntimeError(
                "The Trace upload was interrupted before close confirmation. "
                "Inspect the saved Trace before any manual recovery; do not duplicate its events."
            )
        trace_uuid = saved["trace_uuid"]
        close = saved["close"]
        task_uuid = close.get("task", {}).get("uuid")
        if not task_uuid:
            raise RuntimeError("Zep did not return a close Task UUID.")
        deadline = time.monotonic() + seconds
        while True:
            task = self.client.task.get(task_uuid).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            if task["status"] == "succeeded":
                trace = self.client.agent.trace.get(agent, trace_uuid).model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                if trace.get("status") != "closed" or trace.get("verification") != "external":
                    raise RuntimeError("Zep did not close the externally verified Trace.")
                self._save_checkpoint(
                    checkpoint,
                    {**saved, "stage": "close_verified", "task": task, "final_trace": trace},
                )
                return {
                    "trace_uuid": trace_uuid,
                    "close": close,
                    "task": task,
                    "final_trace": trace,
                    "events": saved["events"],
                }
            if task["status"] not in {"pending", "processing"}:
                raise RuntimeError(f"Trace close Task ended in {task['status']}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Trace close Task is still running. Resume without re-uploading."
                )
            time.sleep(5)

    def wait(self, agent: str, seconds: int = 900) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        last_pin = None
        stable = 0
        while True:
            learning = self.learning_state(agent)
            runs = self.learning_runs(agent)
            with (self.ledger.parent / "learning-progress.jsonl").open("a") as file:
                file.write(json.dumps({"learning": learning, "runs": runs}) + "\n")
            outcomes = runs.get("items", [])
            if outcomes and not learning.get("compilation_pending", True):
                latest = outcomes[0]
                if latest["decision"] != "produced":
                    raise RuntimeError(
                        f"The compiler did not produce a Skill: {latest['decision']}; "
                        f"reason: {latest.get('rejection_class', latest.get('reason_text', 'unknown'))}."
                    )
                response = self.search(agent)
                skills = readable_skills(response)
                pin = sorted((item["uuid"], item["version"], item["markdown"]) for item in skills)
                if pin and pin == last_pin:
                    stable += 1
                else:
                    last_pin, stable = pin, 1
                if stable >= 3:
                    return response
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "No complete admitted Skill is searchable. Resume after checking Zep."
                )
            time.sleep(5)

    def upload(
        self,
        agent: str,
        events: list[dict[str, Any]],
        checkpoint: Path | None = None,
        seconds: int = 900,
    ) -> dict[str, Any]:
        trace = self.client.agent.trace.create(
            agent,
            objective=QUERY,
            task_family=FAMILY,
            metadata={"source": "simple-banking-v3", "data": "invented"},
            idempotency_key=str(uuid.uuid4()),
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        self._save_checkpoint(
            checkpoint,
            {"stage": "created", "trace_uuid": trace["uuid"], "events": len(events), "sent": 0},
        )
        for index, event in enumerate(events, start=1):
            self.client.agent.trace.append_event(
                agent,
                trace["uuid"],
                event_type=event["event_type"],
                content=event.get("content"),
                tool_call_id=event.get("tool_call_id"),
                context=(
                    AgentTraceEventContext.model_validate(event["context"])
                    if event.get("context") is not None
                    else None
                ),
                idempotency_key=str(uuid.uuid4()),
            )
            self._save_checkpoint(
                checkpoint,
                {
                    "stage": "uploading",
                    "trace_uuid": trace["uuid"],
                    "events": len(events),
                    "sent": index,
                },
            )
        self._save_checkpoint(
            checkpoint,
            {
                "stage": "events_sent",
                "trace_uuid": trace["uuid"],
                "events": len(events),
                "sent": len(events),
            },
        )
        close = self.client.agent.trace.close(
            agent,
            trace["uuid"],
            outcome="succeeded",
            verification="external",
            verifier=AgentTraceCloseVerifier(
                verifier_id="simple-banking-state-v3",
                class_="harness",
                outcome="succeeded",
                evidence_reference=trace["uuid"],
            ),
            idempotency_key=str(uuid.uuid4()),
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        self._save_checkpoint(
            checkpoint,
            {
                "stage": "close_accepted",
                "trace_uuid": trace["uuid"],
                "events": len(events),
                "close": close,
            },
        )
        if checkpoint is not None:
            return self.complete_upload(agent, checkpoint, seconds)
        if not close.get("task", {}).get("uuid"):
            raise RuntimeError("Zep did not return a close Task UUID.")
        return {"trace_uuid": trace["uuid"], "close": close, "events": len(events)}


def readable_skills(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("items", [])
        if item.get("uuid")
        and isinstance(item.get("version"), int)
        and str(item.get("markdown") or "").strip()
    ]

"""The documented public Zep Agent Skills API, with an append-only call ledger."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

FAMILY = "banking.synthetic-monthly-correction"
QUERY = "Investigate savings interest, apply qualifying product boosts, credit shortfalls and save reports"


class Hosted:
    def __init__(self, ledger: Path, transport: httpx.BaseTransport | None = None):
        base = os.getenv("ZEP_BASE_URL", "https://api.getzep.com").rstrip("/")
        if urlsplit(base).scheme != "https":
            raise ValueError("ZEP_BASE_URL must use HTTPS.")
        self.ledger = ledger
        self.client = httpx.Client(
            base_url=base.removesuffix("/api/v4") + "/api/v4/",
            headers={"Authorization": "Api-Key " + os.environ["ZEP_API_KEY"]},
            timeout=120,
            follow_redirects=False,
            transport=transport,
        )

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = {"Idempotency-Key": str(uuid.uuid4())} if method != "GET" else {}
        record: dict[str, Any] = {"method": method, "path": path, "status": None}
        start = time.monotonic()
        try:
            response = self.client.request(method, path, headers=headers, **kwargs)
            record.update(
                status=response.status_code, request_id=response.headers.get("x-request-id")
            )
            if not response.is_success:
                raise RuntimeError(f"Zep returned HTTP {response.status_code} for {method} {path}.")
            body = response.json()
            if not isinstance(body, dict) or body.get("error"):
                raise RuntimeError("Zep returned an invalid response.")
            return body
        finally:
            record["seconds"] = time.monotonic() - start
            with self.ledger.open("a") as file:
                file.write(json.dumps(record) + "\n")

    def create_agent(self) -> dict[str, Any]:
        return self.request(
            "POST",
            "agents",
            json={
                "agent_id": "simple-banking-" + uuid.uuid4().hex,
                "name": "Synthetic monthly interest correction demo",
                "security_domain": "simple-banking-demo",
                "memory_settings": {
                    "compilation_mode": "on_close",
                    "approval": "auto",
                    "auto_approval_task_families": [FAMILY],
                },
            },
        )

    def search(self, agent: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"agents/{agent}/skills/search",
            params={"page_size": 20},
            json={
                "query": QUERY,
                "task_family": FAMILY,
                "include_markdown": True,
                "markdown_format": "agent",
            },
        )

    def learning_state(self, agent: str) -> dict[str, Any]:
        return self.request("GET", f"agents/{agent}/learning", params={"task_family": FAMILY})

    def learning_runs(self, agent: str) -> dict[str, Any]:
        return self.request("GET", f"agents/{agent}/learning-runs", params={"task_family": FAMILY})

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
            task = self.request("GET", f"tasks/{task_uuid}")
            if task["status"] == "succeeded":
                trace = self.request("GET", f"agents/{agent}/traces/{trace_uuid}")
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
        trace = self.request(
            "POST",
            f"agents/{agent}/traces",
            json={
                "objective": QUERY,
                "task_family": FAMILY,
                "metadata": {"source": "simple-banking-v3", "data": "invented"},
            },
        )
        self._save_checkpoint(
            checkpoint,
            {"stage": "created", "trace_uuid": trace["uuid"], "events": len(events), "sent": 0},
        )
        path = f"agents/{agent}/traces/{trace['uuid']}"
        for index, event in enumerate(events, start=1):
            self.request("POST", path + "/events", json=event)
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
        close = self.request(
            "POST",
            path + "/close",
            json={
                "outcome": "succeeded",
                "verification": "external",
                "verifier": {
                    "verifier_id": "simple-banking-state-v3",
                    "class": "harness",
                    "outcome": "succeeded",
                    "evidence_reference": trace["uuid"],
                },
            },
        )
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

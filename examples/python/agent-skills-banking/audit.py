"""Content-free records of provider calls, including unsuccessful requests."""

import json
import time
from typing import Any

import httpx


class AuditedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, records: list[dict[str, Any]], role: str):
        super().__init__(retries=0)
        self.records, self.role = records, role

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        record: dict[str, Any] = {"role": self.role, "status": None}
        try:
            response = await super().handle_async_request(request)
            record["status"] = response.status_code
            content = await response.aread()
            try:
                body = json.loads(content)
                record["usage"] = body.get("usage", {})
                record["model"] = body.get("model")
                record["request_id"] = response.headers.get("request-id")
            except (ValueError, AttributeError):
                pass
            return response
        finally:
            record["seconds"] = time.monotonic() - started
            self.records.append(record)

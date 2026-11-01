import json

import httpx
import pytest
from zep_cloud import Zep
from zep_cloud.core.api_error import ApiError

from hosted import FAMILY, Hosted, readable_skills


@pytest.fixture(autouse=True)
def local_credentials(monkeypatch):
    monkeypatch.setenv("ZEP_API_KEY", "local-test-value")
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)


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
    with pytest.raises(ApiError) as error:
        hosted.create_agent()
    assert error.value.status_code == 503
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
    with pytest.raises(ApiError) as error:
        hosted.upload("agent", [{"event_type": "decision", "content": "Example"}], checkpoint)
    assert error.value.status_code == 503
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


@pytest.mark.parametrize(
    ("base", "host"),
    [
        (None, "api.getzep.com"),
        ("https://api.eu.getzep.com", "api.eu.getzep.com"),
        ("https://api.eu.getzep.com/api/v4/", "api.eu.getzep.com"),
    ],
)
def test_sdk_owns_authentication_and_endpoint_routing(tmp_path, monkeypatch, base, host):
    if base is not None:
        monkeypatch.setenv("ZEP_BASE_URL", base)

    def handler(request):
        assert request.method == "POST"
        assert request.url.host == host
        assert request.url.path == "/api/v4/agents"
        assert request.headers["Authorization"] == "Api-Key local-test-value"
        assert "zep-cloud" in request.headers["User-Agent"].lower()
        assert request.headers["Idempotency-Key"]
        return httpx.Response(201, json={"uuid": "agent-uuid"})

    with Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler)) as hosted:
        assert isinstance(hosted.client, Zep)
        agent = hosted.create_agent()
        assert agent["uuid"] == "agent-uuid"
        assert "uuid_" not in agent
    assert hosted.http_client.is_closed


@pytest.mark.parametrize("base", ["http://api.getzep.com", "api.getzep.com", ""])
def test_non_https_endpoint_is_rejected_before_requests(tmp_path, monkeypatch, base):
    monkeypatch.setenv("ZEP_BASE_URL", base)
    with pytest.raises(ValueError, match="HTTPS"):
        Hosted(tmp_path / "zep.jsonl")
    assert not (tmp_path / "zep.jsonl").exists()


def test_sdk_preserves_tool_context_and_verifier_aliases(tmp_path):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if request.url.path.endswith("/close"):
            return httpx.Response(202, json={"task": {"uuid": "close-task"}})
        return httpx.Response(201, json={"uuid": "trace-uuid"})

    event = {
        "event_type": "tool_call",
        "content": '{"account_id": "invented-account"}',
        "tool_call_id": "call-1",
        "context": {"tools": [{"name": "quote_interest"}]},
    }
    with Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler)) as hosted:
        result = hosted.upload("agent-uuid", [event])
    assert result["trace_uuid"] == "trace-uuid"
    assert result["close"]["task"]["uuid"] == "close-task"
    assert bodies[1] == event
    assert bodies[2]["verification"] == "external"
    assert bodies[2]["verifier"] == {
        "class": "harness",
        "outcome": "succeeded",
        "verifier_id": "simple-banking-state-v3",
        "evidence_reference": "trace-uuid",
    }


def test_sdk_search_reads_all_pages_and_preserves_skill_identity(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/v4/agents/agent-uuid/skills/search"
        assert request.url.params["page_size"] == "20"
        body = json.loads(request.content)
        assert body["include_markdown"] is True
        assert body["markdown_format"] == "agent"
        assert body["task_family"] == FAMILY
        page = "2" if request.url.params.get("cursor") else "1"
        return httpx.Response(
            200,
            json={
                "items": [{"uuid": f"skill-{page}", "version": 1, "markdown": "Procedure."}],
                "next_cursor": "page-2" if page == "1" else None,
            },
        )

    with Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler)) as hosted:
        skills = readable_skills(hosted.search("agent-uuid"))
    assert [skill["uuid"] for skill in skills] == ["skill-1", "skill-2"]
    assert len(requests) == 2
    assert requests[1].url.params["cursor"] == "page-2"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_sdk_errors_keep_status_and_are_not_retried(tmp_path, status):
    with (
        Hosted(
            tmp_path / "zep.jsonl",
            httpx.MockTransport(
                lambda _: httpx.Response(status, json={"error": {"message": "Service error."}})
            ),
        ) as hosted,
        pytest.raises(ApiError) as error,
    ):
        hosted.create_agent()
    assert error.value.status_code == status
    records = [json.loads(line) for line in (tmp_path / "zep.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == status
    assert "local-test-value" not in (tmp_path / "zep.jsonl").read_text()


def test_sdk_transport_failure_is_recorded_and_client_is_closed(tmp_path):
    def handler(request):
        raise httpx.ConnectError("The connection failed.", request=request)

    with (
        Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler)) as hosted,
        pytest.raises(httpx.ConnectError),
    ):
        hosted.create_agent()
    records = [json.loads(line) for line in (tmp_path / "zep.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["status"] is None
    assert hosted.http_client.is_closed


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_sdk_close_timeout_can_resume_without_uploading_again(tmp_path, status):
    checkpoint = tmp_path / "upload.json"
    checkpoint.write_text(
        json.dumps(
            {
                "stage": "close_accepted",
                "trace_uuid": "trace-uuid",
                "close": {"task": {"uuid": "close-task"}},
                "events": 3,
            }
        )
    )
    requests = []
    task_status = status

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        if request.url.path == "/api/v4/tasks/close-task":
            return httpx.Response(200, json={"status": task_status})
        assert request.url.path == "/api/v4/agents/agent-uuid/traces/trace-uuid"
        return httpx.Response(200, json={"verification": "external", "status": "closed"})

    with Hosted(tmp_path / "zep.jsonl", httpx.MockTransport(handler)) as hosted:
        with pytest.raises(TimeoutError, match="Resume"):
            hosted.complete_upload("agent-uuid", checkpoint, seconds=0)
        assert json.loads(checkpoint.read_text())["stage"] == "close_accepted"
        task_status = "succeeded"
        result = hosted.complete_upload("agent-uuid", checkpoint)
    assert result["events"] == 3
    assert json.loads(checkpoint.read_text())["stage"] == "close_verified"
    assert len(requests) == 3

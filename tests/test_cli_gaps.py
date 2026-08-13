"""Tests for the paths a coverage audit found unexercised, ranked by consequence.

WHY THIS FILE EXISTS. The 82 tests in `test_cli.py` drove `nerlo check` to 100%
statement coverage and left the rest of the CLI at 85%, with the misses
concentrated in exactly the places a wrong answer is expensive:

  * `rescan` — an authenticated write command at 4.3% (only its decorator ran).
  * every non-zero EXIT PATH of `search` / `info` / `submit` / `_resolve_skill`.
  * the config WRITER's refusals (`_write_mcp_entry`): an existing entry, a
    corrupt config, a config that is not a JSON object.
  * `_build_mcp_entry`'s exact-host guard — the check that stops
    `evilnpmjs.com` from producing a runnable `npx` entry.
  * `_git_shallow_clone`'s scheme guard and every one of its failure modes.
  * the plain-HTTP token warning in `_client`.
  * `nerlo_cli/main.py`, at 0% — nothing asserted that the commands are wired up.

The organising rule is this repo's own: an ABSENT ANSWER MUST NEVER RENDER AS A
GOOD ONE. Several tests below therefore assert a negative as well as a positive
— that a registry 500 does NOT print "No results found", that a refused install
did NOT touch the config file, that an unresolvable identifier did NOT issue the
write request.

Kept separate from `test_cli.py` deliberately: that file is 1744 lines and is
being edited concurrently. The shared helpers are duplicated here rather than
hoisted into a `conftest.py`, which would be a refactor of a file this change
does not otherwise touch.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from nerlo_cli import _logging, commands, main

# Bound at import time, BEFORE the autouse fixture below replaces the module
# attribute — the only way to reach the real factory from a test.
_REAL_TELEMETRY_CLIENT = commands._telemetry_client

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# harness (mirrors test_cli.py)                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep telemetry off the real network and out of the developer's ~/.nerlo."""
    monkeypatch.setenv("NERLO_HOME", str(tmp_path / ".nerlo"))
    monkeypatch.delenv("NERLO_TELEMETRY", raising=False)

    def _no_network(api_url: str) -> httpx.Client:
        raise RuntimeError("telemetry client not stubbed for this test")

    monkeypatch.setattr(commands, "_telemetry_client", _no_network)


def _use_handler(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    def _fake_client(api_url: str, token: str | None = None) -> httpx.Client:
        return httpx.Client(base_url=api_url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(commands, "_client", _fake_client)


def _json_response(request: httpx.Request, status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body, request=request)


def _json_payload(result: Result) -> Any:
    lines = result.output.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() in ("{", "["):
            return json.loads("".join(lines[i:]))
    raise AssertionError(f"no JSON payload found in output: {result.output!r}")


def _combined(result: Result) -> str:
    err = ""
    try:
        err = result.stderr
    except ValueError:
        err = ""
    return result.output + err


def _fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


# --------------------------------------------------------------------------- #
# TIER 1 — `nerlo rescan`, an authenticated write command with no tests at all #
# --------------------------------------------------------------------------- #
#
# Coverage before this section: 1 of 23 lines (the @click.command decorator).
# Every branch below — auth refusal, UUID passthrough, slug resolution,
# unresolvable slug, non-2xx, --json — was unexercised.

_UUID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def _rescan_handler(
    requests: list[httpx.Request],
    *,
    rescan_status: int = 202,
    skill: dict[str, Any] | None = None,
    search_rows: list[dict[str, Any]] | None = None,
) -> Handler:
    """Registry stub that RECORDS every request it is asked to serve.

    Recording is the point: several assertions below are about a request that
    must NOT have been issued (no rescan POST after a failed resolution, no
    network at all without a token). Asserting only on the exit code would let a
    command that fires the write and then errors out look identical to one that
    refused before acting — and "no action taken" is the stated contract.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.startswith("/api/v1/skills/"):
            if skill is None:
                return _json_response(request, 404, {})
            return _json_response(request, 200, skill)
        if path == "/api/v1/servers":
            return _json_response(request, 200, {"results": search_rows or []})
        if path.endswith("/rescan"):
            if rescan_status not in (200, 202):
                return _json_response(request, rescan_status, {"detail": "nope"})
            return _json_response(
                request, rescan_status, {"scan_job_id": "job-1", "dispatch": "queued"}
            )
        return _json_response(request, 404, {})

    return handler


def test_rescan_requires_a_token_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    _use_handler(monkeypatch, _rescan_handler(requests))
    result = CliRunner().invoke(commands.rescan, [_UUID], env={"NERLO_API_TOKEN": ""})
    assert result.exit_code == 1
    assert "authentication required" in _combined(result)
    # Req 11.10/11.11: refused BEFORE acting, not after.
    assert requests == []


def test_rescan_accepts_a_uuid_without_resolving_a_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    _use_handler(monkeypatch, _rescan_handler(requests))
    result = CliRunner().invoke(commands.rescan, [_UUID, "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert "Re-scan queued" in result.output
    assert "job-1" in result.output and "queued" in result.output
    # Exactly one request, straight at the rescan endpoint — a UUID is already a
    # server id, so no /skills/ lookup should be issued.
    assert [r.url.path for r in requests] == [f"/api/v1/servers/{_UUID}/rescan"]
    assert requests[0].method == "POST"


def test_rescan_resolves_a_slug_to_a_server_id(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    _use_handler(
        monkeypatch,
        _rescan_handler(
            requests,
            skill={"skill_id": "demo-skill", "name": "demo"},
            search_rows=[{"id": "srv-9", "name": "demo"}],
        ),
    )
    result = CliRunner().invoke(commands.rescan, ["demo", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert [r.url.path for r in requests] == [
        "/api/v1/skills/demo",
        "/api/v1/servers",
        "/api/v1/servers/srv-9/rescan",
    ]


def test_rescan_uses_the_skills_own_server_id_when_it_carries_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_server_id`'s shortcut arm: no search when the skill knows its id."""
    requests: list[httpx.Request] = []
    _use_handler(
        monkeypatch,
        _rescan_handler(requests, skill={"name": "demo", "mcp_server_id": "srv-direct"}),
    )
    result = CliRunner().invoke(commands.rescan, ["demo", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert [r.url.path for r in requests] == [
        "/api/v1/skills/demo",
        "/api/v1/servers/srv-direct/rescan",
    ]


def test_rescan_unresolvable_slug_exits_1_without_posting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    _use_handler(
        monkeypatch,
        # The skill exists but no search row matches its name, so there is no
        # server id — the command must refuse rather than guess one.
        _rescan_handler(
            requests,
            skill={"skill_id": "demo-skill", "name": "demo"},
            search_rows=[{"id": "srv-9", "name": "something-else"}],
        ),
    )
    result = CliRunner().invoke(commands.rescan, ["demo", "--token", "t"])
    assert result.exit_code == 1
    assert "cannot resolve" in _combined(result)
    assert not any(r.url.path.endswith("/rescan") for r in requests)


def test_rescan_unknown_slug_is_not_found_not_a_silent_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    _use_handler(monkeypatch, _rescan_handler(requests, skill=None))  # /skills/ -> 404
    result = CliRunner().invoke(commands.rescan, ["ghost", "--token", "t"])
    assert result.exit_code == 1
    assert "skill not found" in _combined(result)
    assert not any(r.url.path.endswith("/rescan") for r in requests)


def test_rescan_non_2xx_response_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry that refused the re-scan must not read as "Re-scan queued"."""
    requests: list[httpx.Request] = []
    _use_handler(monkeypatch, _rescan_handler(requests, rescan_status=500))
    result = CliRunner().invoke(commands.rescan, [_UUID, "--token", "t"])
    assert result.exit_code == 1
    assert "rescan failed (HTTP 500)" in _combined(result)
    assert "Re-scan queued" not in result.output


def test_rescan_json_payload_is_the_registry_body(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    _use_handler(monkeypatch, _rescan_handler(requests))
    result = CliRunner().invoke(commands.rescan, [_UUID, "--token", "t", "--json"])
    assert result.exit_code == 0, _combined(result)
    assert _json_payload(result) == {"scan_job_id": "job-1", "dispatch": "queued"}


def test_rescan_401_aborts_with_no_action_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _json_response(request, 401, {})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.rescan, [_UUID, "--token", "bad"])
    assert result.exit_code == 1
    assert "authentication failed (HTTP 401)" in _combined(result)
    assert "no action taken" in _combined(result)


# --------------------------------------------------------------------------- #
# TIER 1 — non-zero exit paths of search / info / submit                       #
# --------------------------------------------------------------------------- #


def test_search_http_error_is_not_rendered_as_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The house rule, applied to `search`: a registry that could not answer must
    not produce the same output as a registry that answered "nothing matches"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 503, {"detail": "down"})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["demo"])
    assert result.exit_code == 1
    assert "search failed (HTTP 503)" in _combined(result)
    assert "No results found" not in result.output


def test_search_empty_result_set_is_an_explicit_exit_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control for the test above: a genuine empty answer exits 0."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"results": []})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["demo"])
    assert result.exit_code == 0, _combined(result)
    assert "No results found for 'demo'." in result.output


def test_search_rejects_an_over_long_query() -> None:
    """The upper half of the 2-100 window; only the lower half had a test."""
    result = CliRunner().invoke(commands.search, ["x" * 101])
    assert result.exit_code == 1
    assert "2-100 characters" in _combined(result)


def test_info_unknown_skill_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 404, {})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.info, ["ghost"])
    assert result.exit_code == 1
    assert "skill not found" in _combined(result)


def test_info_lookup_failure_is_distinct_from_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500 is "we could not ask", 404 is "we asked and it is absent". Different
    messages, because collapsing them is how an outage reads as an absence."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 500, {})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.info, ["demo"])
    assert result.exit_code == 1
    combined = _combined(result)
    assert "lookup failed (HTTP 500)" in combined
    assert "skill not found" not in combined


def test_submit_non_2xx_exits_1_without_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 409, {"detail": "already exists"})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(
        commands.submit, ["https://github.com/o/r", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "submit failed (HTTP 409)" in _combined(result)
    assert "Submitted." not in result.output


def test_submit_json_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` is a machine contract: the registry body passes through intact."""
    body = {"mcp_server_id": "srv-1", "scan_job_id": "job-1", "status": "queued"}

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 201, body)

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(
        commands.submit, ["https://github.com/o/r", "--token", "t", "--json"]
    )
    assert result.exit_code == 0, _combined(result)
    assert _json_payload(result) == body


def test_submit_human_output_names_the_server_and_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 202, {"mcp_server_id": "srv-1", "scan_job_id": "job-7"})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.submit, ["https://github.com/o/r", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert "Submitted." in result.output
    assert "srv-1" in result.output and "job-7" in result.output


def test_request_transport_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection that never completed is an error, not an empty answer.

    `_request(fatal=True)` is what every command except `check` uses, and its
    transport-error arm had no test — a regression there would surface as a
    traceback, or worse as a swallowed failure.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["demo"])
    assert result.exit_code == 1
    assert "cannot reach registry API: ConnectError" in _combined(result)


def test_request_returns_the_response_when_the_transport_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the test above — the same call path, no exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"results": [{"name": "demo"}]})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["demo", "--json"])
    assert result.exit_code == 0, _combined(result)
    assert _json_payload(result) == [{"name": "demo"}]


# --------------------------------------------------------------------------- #
# TIER 1 — `info`'s full render: detail, install stats, per-scanner scoresheets #
# --------------------------------------------------------------------------- #


def _full_info_handler() -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/api/v1/skills/"):
            return _json_response(
                request,
                200,
                {
                    "skill_id": "demo-skill",
                    "name": "demo",
                    "artifact_type": "mcp_server",
                    "current_badge": "Caution",
                    "current_security_score": 62.5,
                    "repository_url": "https://github.com/o/r",
                },
            )
        if path == "/api/v1/servers":
            return _json_response(request, 200, {"results": [{"id": "srv-1", "name": "demo"}]})
        if path.endswith("/installation-stats"):
            return _json_response(request, 200, {"total": 42, "last_30d": 7})
        if path == "/api/v1/servers/srv-1":
            return _json_response(
                request,
                200,
                {
                    "id": "srv-1",
                    "scanner_reports": [
                        {"scanner_name": "cisco", "score": 60, "badge": "Caution",
                         "findings": [{"id": 1}, {"id": 2}]},
                        {"tool_name": "semgrep", "score": 88, "badge": "Verified",
                         "findings": []},
                    ],
                },
            )
        return _json_response(request, 404, {})

    return handler


def test_info_renders_per_scanner_scoresheets_and_install_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregator stance (Req 11.9): the per-scanner table is the primary view.

    Untested before this: the entire detail-fetch, install-stats and
    scoresheet-table block — 12 of `info`'s 30 statements.
    """
    _use_handler(monkeypatch, _full_info_handler())
    result = CliRunner().invoke(commands.info, ["demo"])
    assert result.exit_code == 0, _combined(result)
    out = result.output
    assert "per-scanner scoresheets" in out
    # Both naming conventions the API uses for a scanner are rendered.
    assert "cisco" in out and "semgrep" in out
    # Findings are counted, not listed.
    assert "2" in out
    # Req 29.5: labelled as an install count, never as "popular" or "trusted".
    assert "installed via Nerlo: 42 total (7 in last 30d, CLI installs only)" in out
    assert "popular" not in out.lower() and "trusted" not in out.lower()


def test_info_json_shape_carries_skill_detail_and_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three-key `--json` envelope machines consume."""
    _use_handler(monkeypatch, _full_info_handler())
    result = CliRunner().invoke(commands.info, ["demo", "--json"])
    assert result.exit_code == 0, _combined(result)
    payload = _json_payload(result)
    assert set(payload) == {"skill", "detail", "install_stats"}
    assert payload["skill"]["current_badge"] == "Caution"
    assert payload["install_stats"] == {"total": 42, "last_30d": 7}
    assert len(payload["detail"]["scanner_reports"]) == 2


def test_info_json_keeps_null_detail_when_the_server_is_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing detail is reported as null, not omitted — an absent key would
    make a consumer's `payload["detail"]` raise instead of reading as unknown."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/skills/"):
            return _json_response(request, 200, {"skill_id": "s", "name": "demo"})
        return _json_response(request, 500, {})  # search fails -> no server id

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.info, ["demo", "--json"])
    assert result.exit_code == 0, _combined(result)
    payload = _json_payload(result)
    assert payload["detail"] is None
    assert payload["install_stats"] is None


def test_info_does_not_search_for_a_name_under_the_query_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_server_id` refuses to issue a query the API would reject.

    A 1-character name is under the registry's 2-character `q=` floor. The
    branch returns None WITHOUT searching; this pins that no query is sent, so a
    future edit cannot turn it into a search that 422s and looks like a miss.
    """
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.startswith("/api/v1/skills/"):
            return _json_response(request, 200, {"skill_id": "s", "name": "a"})
        return _json_response(request, 200, {"results": []})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.info, ["a", "--json"])
    assert result.exit_code == 0, _combined(result)
    assert paths == ["/api/v1/skills/a"]  # QUERIES ISSUED: none
    assert _json_payload(result)["detail"] is None


# --------------------------------------------------------------------------- #
# TIER 1 — the config WRITER: a mutated config file is the expensive failure   #
# --------------------------------------------------------------------------- #


def _verified_skill_handler(repo: str = "https://www.npmjs.com/package/demo") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/skills/"):
            return _json_response(
                request,
                200,
                {
                    "skill_id": "demo-skill",
                    "name": "demo",
                    "current_badge": "Verified",
                    "repository_url": repo,
                },
            )
        return _json_response(request, 404, {})

    return handler


def _silent_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry that succeeds silently, so install tests exercise the real path."""

    def _client(api_url: str) -> httpx.Client:
        return httpx.Client(
            base_url=api_url,
            transport=httpx.MockTransport(lambda r: httpx.Response(202, json={}, request=r)),
        )

    monkeypatch.setattr(commands, "_telemetry_client", _client)


def test_install_refuses_to_replace_an_existing_entry_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.json"
    original = json.dumps({"mcpServers": {"demo-skill": {"command": "mine"}}}, indent=2) + "\n"
    config.write_text(original, encoding="utf-8")
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "--force" in _combined(result)
    # The user's config is byte-for-byte untouched. Not "still parses" — the
    # atomic-replace writer rewrites the whole file, so anything short of an
    # exact match would mean it ran.
    assert config.read_text(encoding="utf-8") == original


def test_install_force_replaces_the_existing_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control for the refusal above."""
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"demo-skill": {"command": "mine"}}}), encoding="utf-8"
    )
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t", "--force"]
    )
    assert result.exit_code == 0, _combined(result)
    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["mcpServers"]["demo-skill"] == {"command": "npx", "args": ["-y", "demo"]}


def test_install_preserves_unrelated_config_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """~/.claude.json is the user's live editor state, not a Nerlo-owned file.

    This is the highest-consequence write in the CLI: everything already in the
    config — sibling servers, unrelated top-level keys, the nested `projects`
    map `check` reads — must survive an install untouched.
    """
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps(
            {
                "numStartups": 17,
                "projects": {"/work/repo": {"mcpServers": {"other": {"command": "x"}}}},
                "mcpServers": {"already-there": {"command": "keep-me"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["numStartups"] == 17
    assert written["projects"] == {"/work/repo": {"mcpServers": {"other": {"command": "x"}}}}
    assert written["mcpServers"]["already-there"] == {"command": "keep-me"}
    assert "demo-skill" in written["mcpServers"]


def test_install_refuses_a_config_that_is_not_a_json_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.json"
    config.write_text('["not", "an", "object"]', encoding="utf-8")
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "refusing to overwrite" in _combined(result)
    assert config.read_text(encoding="utf-8") == '["not", "an", "object"]'


def test_install_refuses_an_unparseable_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt config must abort, never be silently replaced with a fresh one."""
    config = tmp_path / "mcp.json"
    config.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "cannot read" in _combined(result)
    assert "JSONDecodeError" in _combined(result)
    assert config.read_text(encoding="utf-8") == "{ this is not json"


def test_install_creates_missing_parent_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `.cursor/mcp.json` / `.gemini/settings.json` layouts nest one deep."""
    config = tmp_path / "nested" / "deeper" / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["demo-skill"]


def test_install_leaves_no_temp_file_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_write_mcp_entry` stages through `mkstemp` next to the target; a stray
    `.nerlo-tmp` in the user's home would be litter, and one left holding a
    partial config would be worse."""
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".nerlo", "mcp.json"]


# --------------------------------------------------------------------------- #
# TIER 1 — `_build_mcp_entry`: which repository URLs become RUNNABLE commands  #
# --------------------------------------------------------------------------- #
#
# The entry this writes is executed by the AI platform at startup. A host match
# that is too loose is arbitrary code execution, so the exact-host rule is a
# security control and was entirely untested: only the npmjs arm ran.


@pytest.mark.parametrize(
    "repo",
    [
        "https://evilnpmjs.com/package/pwn",
        "https://npmjs.com.evil.test/package/pwn",
        "https://www.npmjs.com.attacker.example/package/pwn",
        "https://notpypi.org/project/pwn",
        "https://pypi.org.evil.test/project/pwn",
    ],
)
def test_build_mcp_entry_does_not_make_a_lookalike_host_runnable(repo: str) -> None:
    """Suffix matching here would let `evilnpmjs.com` mint `npx -y pwn`."""
    entry = commands._build_mcp_entry({"repository_url": repo, "current_badge": "Verified"})
    assert "command" not in entry
    assert entry == {"repository": repo, "nerlo_badge": "Verified"}


@pytest.mark.parametrize(
    ("repo", "expected"),
    [
        ("https://www.npmjs.com/package/demo", {"command": "npx", "args": ["-y", "demo"]}),
        ("https://npmjs.com/package/demo", {"command": "npx", "args": ["-y", "demo"]}),
        (
            "https://www.npmjs.com/package/@scope/pkg",
            {"command": "npx", "args": ["-y", "@scope/pkg"]},
        ),
        ("https://pypi.org/project/demo/", {"command": "uvx", "args": ["demo"]}),
    ],
)
def test_build_mcp_entry_makes_real_registry_hosts_runnable(
    repo: str, expected: dict[str, Any]
) -> None:
    """Positive control for the lookalike test. The pypi/uvx arm had no test at
    all — `install` from a PyPI-hosted source was never exercised."""
    assert commands._build_mcp_entry({"repository_url": repo}) == expected


@pytest.mark.parametrize(
    "repo",
    [
        "https://www.npmjs.com/package/",  # right host, no package name
        "https://pypi.org/project/",
        "https://github.com/o/r",
        "",
    ],
)
def test_build_mcp_entry_falls_back_to_a_repository_reference(repo: str) -> None:
    entry = commands._build_mcp_entry({"repository_url": repo, "current_badge": "Verified"})
    assert entry == {"repository": repo, "nerlo_badge": "Verified"}


def test_install_from_a_non_package_source_warns_that_wiring_is_unfinished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The written entry is inert, and the CLI says so rather than implying the
    platform can launch it."""
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler(repo="https://github.com/o/r"))

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert "no runnable package source detected" in _combined(result)
    entry = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["demo-skill"]
    assert entry == {"repository": "https://github.com/o/r", "nerlo_badge": "Verified"}


def test_install_json_shape_for_an_mcp_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t", "--json"]
    )
    assert result.exit_code == 0, _combined(result)
    payload = _json_payload(result)
    assert set(payload) == {"installed", "target", "config_path", "entry"}
    assert payload["installed"] == "demo-skill"
    assert payload["target"] == "mcp"
    assert payload["config_path"] == str(config)
    assert payload["entry"] == {"command": "npx", "args": ["-y", "demo"]}


def test_install_gemini_placeholder_json_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`installed: null` — a machine reading this must not see a success shape."""
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "gemini", config)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            200,
            {
                "skill_id": "demo-skill",
                "name": "demo",
                "artifact_type": "gemini_extension",
                "current_badge": "Verified",
                "repository_url": "https://github.com/o/r",
            },
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "gemini", "--token", "t", "--json"]
    )
    assert result.exit_code == 0, _combined(result)
    payload = _json_payload(result)
    assert payload["installed"] is None
    assert payload["artifact_type"] == "gemini_extension"
    assert not config.exists()


# --------------------------------------------------------------------------- #
# TIER 1 — `_git_shallow_clone`: a registry-supplied URL reaches a subprocess  #
# --------------------------------------------------------------------------- #

_SKILL_REPO_URL = "https://github.com/o/skill-repo"


def _claude_skill_handler(repo: str, *, skill_id: str = "demo-skill") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            200,
            {
                "skill_id": skill_id,
                "name": "demo",
                "artifact_type": "claude_skill",
                "current_badge": "Verified",
                "repository_url": repo,
            },
        )

    return handler


@pytest.mark.parametrize(
    "repo",
    [
        "file:///etc/passwd",
        "ext::sh -c whoami",
        "git@github.com:o/r.git",
        "ssh://git@github.com/o/r.git",
        "/tmp/local-repo",
    ],
)
def test_claude_skill_clone_refuses_non_http_transports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: str
) -> None:
    """git's non-http transports can run local commands (`ext::`) or read local
    paths. The URL comes from the registry, so the scheme allowlist is the
    control — and `subprocess.run` must not be reached at all."""
    _fake_home(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        commands.subprocess,
        "run",
        lambda args, **kw: calls.append(list(args)),  # type: ignore[arg-type,return-value]
    )
    _use_handler(monkeypatch, _claude_skill_handler(repo))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "only http(s) repository URLs are supported" in _combined(result)
    assert calls == []  # never handed to git


def test_claude_skill_clone_accepts_https(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: the same code path DOES invoke git for an https URL."""
    _fake_home(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        dest = Path(args[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text("# demo\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    assert len(calls) == 1 and _SKILL_REPO_URL in calls[0]


def test_claude_skill_clone_reports_a_missing_git_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A machine without git gets an instruction, not a FileNotFoundError."""
    home = _fake_home(monkeypatch, tmp_path)

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    combined = _combined(result)
    assert "`git` was not found on PATH" in combined
    assert "install git and retry" in combined
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not (home / ".claude" / "skills" / "demo-skill").exists()


def test_claude_skill_clone_reports_a_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=commands.GIT_CLONE_TIMEOUT_SECONDS)

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "timed out after 120s" in _combined(result)
    assert not (home / ".claude" / "skills" / "demo-skill").exists()


def test_claude_skill_clone_surfaces_gits_own_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A private/deleted repo is the common case; git's last stderr line is the
    only thing that tells the user which."""
    home = _fake_home(monkeypatch, tmp_path)

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 128, stdout="", stderr="Cloning into 'x'...\nfatal: repository not found\n"
        )

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    combined = _combined(result)
    assert "exit 128" in combined
    assert "fatal: repository not found" in combined
    assert not (home / ".claude" / "skills" / "demo-skill").exists()


def test_claude_skill_clone_error_without_stderr_still_reports_the_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_home(monkeypatch, tmp_path)

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "exit 1" in _combined(result)


# --------------------------------------------------------------------------- #
# TIER 1 — `_install_claude_skill` guards: what lands on disk, and where       #
# --------------------------------------------------------------------------- #


def _stub_clone(monkeypatch: pytest.MonkeyPatch, populate: Callable[[Path], None]) -> None:
    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        dest = Path(args[-1])
        dest.mkdir(parents=True, exist_ok=True)
        populate(dest)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)


def _flat_skill(dest: Path) -> None:
    (dest / "SKILL.md").write_text("# demo\n", encoding="utf-8")


@pytest.mark.parametrize(
    "slug",
    # NOT `""`: `install` resolves the slug as `skill_id or skill_name`, so an
    # empty skill_id falls back to the CLI argument and never reaches the guard.
    # That fallback is pinned by its own test below.
    ["../escape", "a/b", "..", ".hidden", "with space", "sub\\dir", "-leading-dash"],
)
def test_claude_skill_refuses_an_unsafe_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slug: str
) -> None:
    """The install directory name comes from the API. `~/.claude/skills/../x`
    would write outside the skills root; the slug allowlist is what stops it.

    Note `sub\\dir` is rejected on every OS, not only on Windows where the
    backslash is a path separator — the check is a character allowlist, not a
    per-platform path parse, so the matrix cannot disagree about it.
    """
    home = _fake_home(monkeypatch, tmp_path)
    _stub_clone(monkeypatch, _flat_skill)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL, skill_id=slug))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "not a safe directory name" in _combined(result)
    # Nothing anywhere under the fake home, and nothing beside it either — a
    # traversing slug would land next to `home/`, not inside it, so checking
    # only under the skills root would miss exactly the case this guards.
    assert not (home / ".claude").exists()
    assert {p.name for p in tmp_path.iterdir()} <= {".nerlo", "home"}


def test_claude_skill_slug_falls_back_to_the_requested_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A record with no `skill_id` installs under the name the user asked for —
    and that fallback is itself slug-checked, because Click will happily pass
    `../escape` as the argument."""
    home = _fake_home(monkeypatch, tmp_path)
    _stub_clone(monkeypatch, _flat_skill)
    _silent_telemetry(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            200,
            {
                "name": "demo",
                "artifact_type": "claude_skill",
                "current_badge": "Verified",
                "repository_url": _SKILL_REPO_URL,
            },
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    assert (home / ".claude" / "skills" / "demo" / "SKILL.md").is_file()


def test_claude_skill_refuses_a_traversing_name_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback path's own guard: no `skill_id`, and the argument traverses."""
    home = _fake_home(monkeypatch, tmp_path)
    _stub_clone(monkeypatch, _flat_skill)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            200,
            {
                "name": "demo",
                "artifact_type": "claude_skill",
                "current_badge": "Verified",
                "repository_url": _SKILL_REPO_URL,
            },
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(
        commands.install, ["../../escape", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "not a safe directory name" in _combined(result)
    assert not (home / ".claude").exists()
    assert {p.name for p in tmp_path.iterdir()} <= {".nerlo", "home"}


def test_claude_skill_refuses_a_record_without_a_repository_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    _use_handler(monkeypatch, _claude_skill_handler(""))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "no repository_url" in _combined(result)
    assert not (home / ".claude").exists()


def test_claude_skill_refuses_to_clobber_an_existing_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    existing = home / ".claude" / "skills" / "demo-skill"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("# MINE, hand-edited\n", encoding="utf-8")
    _stub_clone(monkeypatch, _flat_skill)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "already exists" in _combined(result)
    assert (existing / "SKILL.md").read_text(encoding="utf-8") == "# MINE, hand-edited\n"


def test_claude_skill_force_replaces_an_existing_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control for the refusal above, and the only test of the
    rmtree-then-replace arm. The stale file must be GONE, not merged over."""
    home = _fake_home(monkeypatch, tmp_path)
    existing = home / ".claude" / "skills" / "demo-skill"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("# stale\n", encoding="utf-8")
    (existing / "leftover.txt").write_text("old\n", encoding="utf-8")
    _stub_clone(monkeypatch, _flat_skill)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t", "--force"]
    )
    assert result.exit_code == 0, _combined(result)
    assert (existing / "SKILL.md").read_text(encoding="utf-8") == "# demo\n"
    assert not (existing / "leftover.txt").exists()
    # No staging directory survived the swap.
    skills_root = home / ".claude" / "skills"
    assert sorted(p.name for p in skills_root.iterdir()) == ["demo-skill"]


def test_claude_skill_picks_the_directory_matching_the_slug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A monorepo of skills: several SKILL.md files, one named for this slug."""
    home = _fake_home(monkeypatch, tmp_path)

    def populate(dest: Path) -> None:
        for name in ("alpha", "demo-skill", "zeta"):
            d = dest / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")

    _stub_clone(monkeypatch, populate)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    installed = home / ".claude" / "skills" / "demo-skill" / "SKILL.md"
    assert installed.read_text(encoding="utf-8") == "# demo-skill\n"


def test_claude_skill_refuses_to_guess_between_ambiguous_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Several SKILL.md files and none named for the slug — installing the wrong
    one would put unreviewed code in ~/.claude/skills under a trusted name."""
    home = _fake_home(monkeypatch, tmp_path)

    def populate(dest: Path) -> None:
        for name in ("alpha", "zeta"):
            d = dest / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")

    _stub_clone(monkeypatch, populate)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    combined = _combined(result)
    assert "multiple SKILL.md" in combined and "refusing to guess" in combined
    assert not (home / ".claude" / "skills" / "demo-skill").exists()


def test_claude_skill_installs_from_the_repository_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The single-skill repo layout (SKILL.md at the top level) — the early
    return in `_find_skill_dir`, which no test reached."""
    home = _fake_home(monkeypatch, tmp_path)

    def populate(dest: Path) -> None:
        (dest / "SKILL.md").write_text("# root skill\n", encoding="utf-8")
        (dest / "helper.py").write_text("x = 1\n", encoding="utf-8")

    _stub_clone(monkeypatch, populate)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _claude_skill_handler(_SKILL_REPO_URL))

    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t", "--json"]
    )
    assert result.exit_code == 0, _combined(result)
    installed = home / ".claude" / "skills" / "demo-skill"
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == "# root skill\n"
    assert (installed / "helper.py").exists()
    payload = _json_payload(result)
    assert set(payload) == {"installed", "artifact_type", "target", "path"}
    assert payload["installed"] == "demo-skill"
    assert payload["artifact_type"] == "claude_skill"
    assert payload["target"] == "claude-code"
    assert Path(payload["path"]) == installed


# --------------------------------------------------------------------------- #
# TIER 2 — the plain-HTTP credential warning                                   #
# --------------------------------------------------------------------------- #
#
# `_client` is monkeypatched away by every other test in the suite, so its
# warning arm — the only thing standing between a bearer token and a cleartext
# hop — had never executed.


def test_client_warns_when_a_token_would_cross_plain_http(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with commands._client("http://registry.example.com", "secret-token") as client:
        assert client.headers["Authorization"] == "Bearer secret-token"
    err = capsys.readouterr().err
    assert "plain HTTP" in err
    assert "use https" in err
    # The warning must not itself leak the credential it is warning about.
    assert "secret-token" not in err


@pytest.mark.parametrize(
    "api_url",
    [
        "https://api.nerlo.ai",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
def test_client_stays_quiet_for_https_and_loopback(
    capsys: pytest.CaptureFixture[str], api_url: str
) -> None:
    """Positive control: the warning fires on the insecure case ONLY. Without
    this, a warning that fired unconditionally would pass the test above."""
    with commands._client(api_url, "secret-token") as client:
        assert client.headers["Authorization"] == "Bearer secret-token"
    assert "plain HTTP" not in capsys.readouterr().err


def test_client_sends_no_authorization_header_without_a_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with commands._client("http://registry.example.com") as client:
        assert "Authorization" not in client.headers
        assert client.headers["User-Agent"] == "nerlo-cli"
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# TIER 2 — `_package_from_command`, the inverse of the writer (check's reader) #
# --------------------------------------------------------------------------- #
#
# Getting this wrong in EITHER direction is a wrong verdict: too eager invents
# an identity that could inherit some other project's Verified badge; too shy
# leaves a real, resolvable artifact reported as `unknown`.


@pytest.mark.parametrize(
    ("command", "args", "expected"),
    [
        ("npx", ["-y", "demo"], "demo"),
        ("bunx", ["demo"], "demo"),
        ("uvx", ["demo"], "demo"),
        # Wrapper subcommands are skipped, not mistaken for the package.
        ("pnpm", ["dlx", "demo"], "demo"),
        ("pnpm", ["exec", "demo"], "demo"),
        ("uv", ["tool", "run", "demo"], "demo"),
        # A version pin is stripped; an npm scope's leading @ is not.
        ("npx", ["-y", "demo@1.2.3"], "demo"),
        ("npx", ["-y", "@scope/pkg"], "@scope/pkg"),
        ("npx", ["-y", "@scope/pkg@1.2.3"], "@scope/pkg"),
        # A POSIX absolute path to the runner still resolves (Path().name works
        # on "/" separators under both pathlib flavours, so this holds on the
        # Windows matrix leg too).
        ("/usr/local/bin/npx", ["-y", "demo"], "demo"),
        # `.exe` is stripped, so a Windows-recorded runner still resolves.
        ("npx.exe", ["-y", "demo"], "demo"),
        ("NPX.EXE", ["-y", "demo"], "demo"),
    ],
)
def test_package_from_command_reads_a_runnable_entry(
    command: str, args: list[str], expected: str
) -> None:
    assert commands._package_from_command(command, args) == expected


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("node", ["server.js"]),  # not a package runner
        ("docker", ["run", "some/image"]),
        ("npx", ["-y"]),  # flags only — nothing to name
        ("npx", "not-a-list"),  # malformed config value
        ("npx", None),
        ("", ["demo"]),
        ("pnpm", ["dlx"]),  # subcommand consumed, no package left
    ],
)
def test_package_from_command_invents_nothing_it_cannot_read(
    command: str, args: Any
) -> None:
    """Returning a guess here would attach another project's badge to this
    entry. `None` sends it down the config-key path instead."""
    assert commands._package_from_command(command, args) is None


def test_package_from_command_reads_a_windows_batch_shim() -> None:
    """REGRESSION TEST for the defect this coverage audit turned up.

    npm installs its runners on Windows as batch shims — `npx.cmd`, `pnpm.cmd`
    — and only `.exe` was stripped. A Windows config saying
    `{"command": "npx.cmd", "args": ["-y", "pkg"]}` therefore yielded no package
    name, so `check` searched only the user-chosen config key, matched nothing,
    and reported a registry-listed artifact as `unknown`. `unknown` does not
    fail `--fail-on unsafe`: the gate degraded on one platform and said nothing.

    Matrix-safe by construction: a bare `npx.cmd` contains no separator, so
    `Path(...).name` returns it unchanged under both pathlib flavours and this
    assertion means the same thing on all three OSes in CI.
    """
    assert commands._package_from_command("npx.cmd", ["-y", "demo"]) == "demo"
    assert commands._package_from_command("NPX.CMD", ["-y", "demo"]) == "demo"
    assert commands._package_from_command("pnpm.cmd", ["dlx", "demo"]) == "demo"
    assert commands._package_from_command("uv.bat", ["tool", "run", "demo"]) == "demo"


def test_stripping_an_executable_suffix_does_not_invent_a_runner() -> None:
    """The negative control for the widening above: the stripped name still has
    to BE a known runner, so an arbitrary `.cmd` cannot become one."""
    assert commands._package_from_command("evil.cmd", ["demo"]) is None
    assert commands._package_from_command("node.exe", ["server.js"]) is None
    assert commands._package_from_command(".cmd", ["demo"]) is None


@pytest.mark.skipif(os.name != "nt", reason="backslash is only a path separator on Windows")
def test_package_from_command_strips_a_windows_directory_prefix() -> None:
    """PLATFORM-SPECIFIC BY CONSTRUCTION, and skipped rather than asserted
    per-OS so the difference stays visible.

    `Path(r"C:\\...\\npx.exe").name` is `npx.exe` under `WindowsPath` and the
    whole string under `PosixPath`, so a real Windows config entry — which
    records an absolute shim path — only resolves on the Windows matrix leg.
    The CI matrix runs this; a POSIX-only run reports it as skipped.
    """
    assert (
        commands._package_from_command(r"C:\Program Files\nodejs\npx.exe", ["-y", "demo"])
        == "demo"
    )


# --------------------------------------------------------------------------- #
# TIER 2 — malformed registry responses inside `check`                         #
# --------------------------------------------------------------------------- #


def _project_with(tmp_path: Path, servers: dict[str, Any]) -> Path:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return root


def test_check_search_body_that_is_not_an_object_is_error_not_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A top-level JSON ARRAY (or string) where the schema promises an object.

    The already-tested case is a response object missing its `results` list.
    This is the sibling: a body that is not an object at all. Both must land on
    `error`/exit 3 — a response we could not parse is not an absence.
    """
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, ["not", "an", "object"])

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project), "--json"])
    assert result.exit_code == 3
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "error"
    assert "malformed search response" in payload["artifacts"][0]["note"]
    assert payload["summary"]["unknown"] == 0


def test_check_detail_body_that_is_not_an_object_is_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The row matched, so the artifact IS in the registry — but its verdict
    could not be read. Falling back to the search row here is the silent
    fallback the module refuses; the result must be `error`, not `verified`."""
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request,
                200,
                {
                    "results": [{"id": "srv-1", "name": "demo", "current_badge": "Verified"}],
                    "total_count": 1,
                    "total_pages": 1,
                },
            )
        return _json_response(request, 200, "just a string")

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project), "--json"])
    assert result.exit_code == 3
    artifact = _json_payload(result)["artifacts"][0]
    assert artifact["status"] == "error"
    assert "malformed detail response" in artifact["note"]
    assert artifact["badge"] is None


def test_check_skips_non_object_rows_without_losing_the_real_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A junk element in `results` must not crash the matcher, and must not
    stop the genuine Unsafe row on the same page from being found."""
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request,
                200,
                {
                    "results": ["garbage", None, 17, {"id": "srv-1", "name": "demo"}],
                    "total_count": 4,
                    "total_pages": 1,
                },
            )
        return _json_response(
            request,
            200,
            {"id": "srv-1", "name": "demo", "composite_badge": "Unsafe", "composite_score": 3.0},
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project), "--json"])
    assert result.exit_code == 1
    assert _json_payload(result)["artifacts"][0]["status"] == "unsafe"


def test_check_renders_an_unscannable_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`unscannable_reason` is the registry explaining why it has no verdict;
    dropping it leaves the user with an unscored row and no reason."""
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request,
                200,
                {"results": [{"id": "srv-1", "name": "demo"}], "total_count": 1, "total_pages": 1},
            )
        return _json_response(
            request,
            200,
            {"id": "srv-1", "name": "demo", "unscannable_reason": "repository is archived"},
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project), "--json"])
    assert result.exit_code == 0  # unscored does not fail at --fail-on unsafe
    artifact = _json_payload(result)["artifacts"][0]
    assert artifact["status"] == "unscored"
    assert artifact["note"] == "repository is archived"


@pytest.mark.parametrize("score", [["not", "a", "number"], {"v": 1}, "N/A", None])
def test_check_unreadable_score_renders_as_a_dash_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, score: Any
) -> None:
    """`_coerce_score`'s non-scalar arm. The BADGE decides the gate; a score
    field we could not parse must neither crash the renderer (which the
    contract would read as exit 1 = policy violated) nor invent a number."""
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request,
                200,
                {"results": [{"id": "srv-1", "name": "demo"}], "total_count": 1, "total_pages": 1},
            )
        return _json_response(
            request,
            200,
            {
                "id": "srv-1",
                "name": "demo",
                "composite_badge": "Verified",
                "composite_score": score,
            },
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project)])
    assert result.exit_code == 0, _combined(result)
    assert "PASS" in result.output
    assert "VERIFIED" in result.output


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ('["a", "list"]', "config root is not a JSON object"),
        ('{"projects": ["a", "list"]}', "projects is not a JSON object"),
    ],
)
def test_check_config_of_the_wrong_shape_is_incomplete_not_empty(
    tmp_path: Path, body: str, why: str
) -> None:
    """A config whose ROOT or whose `projects` map is the wrong type could be
    hiding entries. Reading it as "nothing configured" is the collapse this
    command exists to prevent — no network stub needed, it never gets that far.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text(body, encoding="utf-8")

    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3, why
    combined = _combined(result)
    assert "could not read config" in combined
    assert "ValueError" in combined
    assert "Nothing was verified" in combined
    assert "nothing to check" not in combined


def test_check_json_lists_the_unreadable_config(tmp_path: Path) -> None:
    """The `--json` envelope names the file, so CI can print what to fix."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    assert result.exit_code == 3
    payload = _json_payload(result)
    assert payload["exit_code"] == 3
    assert len(payload["unreadable_configs"]) == 1
    assert "mcp.json" in payload["unreadable_configs"][0]
    assert payload["artifacts"] == []


def test_check_zero_artifacts_from_unreadable_configs_says_READ_not_CONFIGURED(
    tmp_path: Path,
) -> None:
    """The all-unreadable headline must say "could be READ", never "configured".

    Found by mutation, 2026-08-13 PT. The existing coverage of this branch pins
    the tail of the message ("Nothing was verified", `grep -n 'Nothing was
    verified' tests/test_cli_gaps.py`) and the exit code, but nothing pinned the
    HEADLINE. Substituting

        "No AI artifacts could be read in {scope}: ..."
     -> "No AI artifacts configured in {scope}: ..."

    left the whole suite green while the first line a human reads flipped from
    "we could not read your configs" to "you have no configs" — the empty-vs-
    unreadable collapse, reached through the wording instead of the exit code.
    Exit 3 was still correct, which is exactly why no test noticed: the machine
    contract held and the human one did not, and on this command the human
    sentence is half the product.

    The two readings must stay distinguishable in both directions, so this pins
    the negative too: the "nothing to check" wording belongs to the genuinely
    empty tree and must never appear here.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text("{ this is not json", encoding="utf-8")

    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3
    combined = _combined(result)
    assert "could not be read" in combined or "could be read" in combined
    assert "No AI artifacts could be read" in combined
    assert "config(s) could not be parsed" in combined
    # The empty-tree wording is reserved for the empty tree.
    assert "configured in" not in combined
    assert "nothing to check" not in combined


def test_check_rejects_a_server_id_that_could_steer_the_detail_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An `id` carrying path separators is an error, not a redirected fetch.

    Found by mutation, 2026-08-13 PT: deleting the `"/" in server_id` and
    `"?" in server_id` guards (`grep -n 'unusable server id' nerlo_cli/
    commands.py`) left the suite green. The guard is the only thing stopping a
    malformed or hostile search row from choosing which endpoint the detail
    fetch hits — `id` is interpolated straight into
    `/api/v1/servers/{server_id}`, so an id of `../../healthz` or
    `x?admin=1` walks the request off the endpoint whose answer the gate then
    reports as a verdict.

    Both halves are asserted: the request must NOT be issued, and the artifact
    must land on `error` (in INCOMPLETE_STATUSES, in no FAIL_ON set) rather
    than inheriting the search row's badge. The second half matters most — the
    search row here is Verified, so a fallback would render a hostile id as a
    green pass.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request,
                200,
                {
                    "results": [
                        {
                            "id": "../../healthz",
                            "name": "demo",
                            "repository_url": "",
                            "composite_badge": "Verified",
                            "composite_score": 99.0,
                        }
                    ],
                    "total_count": 1,
                    "total_pages": 1,
                },
            )
        return _json_response(request, 200, {})

    _use_handler(monkeypatch, handler)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"demo": {"command": "npx", "args": ["-y", "demo"]}}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "error", artifact
    assert "unusable server id" in artifact["note"]
    # The Verified badge on the search row must NOT have been inherited.
    assert artifact["badge"] is None
    assert artifact["score"] is None
    # And no detail request was issued at all.
    assert all(not p.startswith("/api/v1/servers/") for p in requested), requested
    assert result.exit_code == 3


# --------------------------------------------------------------------------- #
# TIER 3 — `nerlo_cli/main.py`, which was at 0%                                #
# --------------------------------------------------------------------------- #


def test_version_command_prints_the_installed_version() -> None:
    result = CliRunner().invoke(main.cli, ["version"])
    assert result.exit_code == 0, _combined(result)
    assert result.output.startswith("nerlo ")
    assert result.output.split()[1]  # a non-empty version string


def test_version_command_says_so_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(main.metadata, "version", _boom)
    result = CliRunner().invoke(main.cli, ["version"])
    assert result.exit_code == 0, _combined(result)
    assert "unknown (not installed)" in result.output


def test_every_public_command_is_wired_into_the_group() -> None:
    """The shipped command surface. A command dropped from `ALL_COMMANDS` would
    otherwise vanish from the installed CLI with nothing to catch it, since
    every other test invokes the command objects directly.
    """
    assert set(main.cli.commands) == {
        "search",
        "info",
        "install",
        "rescan",
        "submit",
        "check",
        "version",
    }


def test_group_help_lists_the_commands() -> None:
    result = CliRunner().invoke(main.cli, ["--help"])
    assert result.exit_code == 0, _combined(result)
    for name in ("search", "info", "install", "rescan", "submit", "check", "version"):
        assert name in result.output


def test_unknown_command_is_a_usage_error_exit_2() -> None:
    result = CliRunner().invoke(main.cli, ["frobnicate"])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# TIER 3 — small helpers whose failure would be silent                         #
# --------------------------------------------------------------------------- #


def test_cli_version_falls_back_when_the_package_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This string goes into every telemetry POST; an exception here would be
    swallowed by `_emit_install_event` and the event would vanish silently."""

    def _boom(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(commands.metadata, "version", _boom)
    assert commands._cli_version() == "0.0.0+unknown"


def test_cli_version_is_clamped_to_the_contracts_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands.metadata, "version", lambda name: "9" * 200)
    assert commands._cli_version() == "9" * 50


def test_read_config_ignores_comments_and_junk_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The opt-out store. A parser that choked on a comment would silently
    re-enable telemetry for a user who had opted out."""
    home = tmp_path / "nerlo-home"
    home.mkdir()
    monkeypatch.setenv("NERLO_HOME", str(home))
    (home / "config").write_text(
        # The commented line is a KEY=VALUE pair and comes LAST, so a parser
        # that skipped only blank/pairless lines would let it overwrite the real
        # setting and silently switch telemetry back on. A comment containing no
        # `=` cannot detect that, which is how the first draft of this test
        # passed against a broken parser.
        "# a comment\n"
        "\n"
        "   \n"
        "not-a-pair\n"
        "telemetry = false \n"
        "  spaced = value  \n"
        "# telemetry=true\n"
        "   # spaced=clobbered\n",
        encoding="utf-8",
    )
    assert commands._read_config() == {"telemetry": "false", "spaced": "value"}
    assert commands._telemetry_enabled() is False


def test_read_config_of_a_missing_file_is_empty_and_leaves_telemetry_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NERLO_HOME", str(tmp_path / "absent"))
    assert commands._read_config() == {}
    assert commands._telemetry_enabled() is True


def test_table_of_no_rows_prints_nothing_not_a_bare_header(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lone header row reads as "checked, found none"; the callers print their
    own words for the empty case instead."""
    commands._table([], ["status", "artifact"])
    assert capsys.readouterr().out == ""


def test_check_treats_a_401_from_the_public_registry_as_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_request(fatal=False)`'s auth arm — `check`'s only caller of it.

    `check` is anonymous, so a 401/403 means the registry would not answer, not
    that the caller has bad credentials. `fatal=True` would exit 1 here, which
    the contract reads as "policy violated" — a registry outage manufacturing a
    verdict. It must raise instead and land on `error` / exit 3.
    """
    project = _project_with(tmp_path, {"demo": {"command": "npx", "args": ["-y", "demo"]}})

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 403, {"detail": "forbidden"})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(project), "--json"])
    assert result.exit_code == 3
    artifact = _json_payload(result)["artifacts"][0]
    assert artifact["status"] == "error"
    assert "HTTP 403" in artifact["note"]


def test_install_cleans_up_its_temp_file_when_the_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_write_mcp_entry` stages to a temp file then `os.replace`s it into place.

    If the replace fails (a locked file on Windows, a full disk, a permissions
    change mid-run) the staged file must be removed and the original config left
    exactly as it was. A surviving `.nerlo-tmp` holding a full config next to the
    real one is both litter and a confusing thing to find in `~/.claude/`.
    """
    config = tmp_path / "mcp.json"
    original = json.dumps({"mcpServers": {}})
    config.write_text(original, encoding="utf-8")
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _silent_telemetry(monkeypatch)
    _use_handler(monkeypatch, _verified_skill_handler())

    # Scoped to the staged file by name: `commands.os` IS the stdlib module, so
    # an unconditional stub would break `os.replace` for everything running
    # inside the invoke — including anything the harness or httpx does.
    real_replace = os.replace

    def boom(src: Any, dst: Any) -> None:
        if str(src).endswith(".nerlo-tmp"):
            raise OSError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(commands.os, "replace", boom)

    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)
    assert config.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.glob("*.nerlo-tmp")] == []


def test_telemetry_client_is_unauthenticated() -> None:
    """Every other test replaces this factory, so its body never ran.

    The install telemetry POST must NOT carry the bearer token: the body already
    contains a one-way hash of the installer identity, and sending the raw
    credential to an unauthenticated endpoint would be a needless exposure.
    Constructing a client opens no connection, so this touches no network.
    """
    with _REAL_TELEMETRY_CLIENT("https://api.nerlo.ai") as client:
        assert "Authorization" not in client.headers
        assert client.headers["User-Agent"] == "nerlo-cli"
        assert client.timeout.connect == 2.0
        # Telemetry must never delay an install: a short, bounded timeout.
        assert client.timeout.read == 3.0


# --------------------------------------------------------------------------- #
# TIER 3 — `_logging`, the observability contract                             #
# --------------------------------------------------------------------------- #


def _fresh_logger() -> Any:
    """A logger built NOW — `_CliLogger` samples NERLO_DEBUG in `__init__`, so
    the module-level `commands.logger` cannot see an env change made by a test."""
    return _logging.get_logger("test")


def test_logger_is_silent_by_default_and_verbose_under_nerlo_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """debug/info are gated on NERLO_DEBUG=1; the gate had no test in either
    direction, so a logger that printed nothing at all would have passed."""
    monkeypatch.delenv("NERLO_DEBUG", raising=False)
    quiet = _fresh_logger()
    quiet.debug("cli.check", scope="project")
    quiet.info("cli.install", target="mcp")
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("NERLO_DEBUG", "1")
    loud = _fresh_logger()
    loud.info("cli.install", target="mcp", badge="Verified")
    err = capsys.readouterr().err
    assert "[info] cli.install" in err
    assert "target=mcp" in err and "badge=Verified" in err


def test_logger_always_emits_warning_and_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """These two bypass the debug gate — a warning nobody sees is not a warning."""
    monkeypatch.delenv("NERLO_DEBUG", raising=False)
    log = _fresh_logger()
    log.warning("cli.check", unresolved=2)
    log.error("cli.install", reason="refused")
    err = capsys.readouterr().err
    assert "[warning] cli.check unresolved=2" in err
    assert "[error] cli.install reason=refused" in err


def test_logger_event_without_fields_has_no_trailing_space(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NERLO_DEBUG", raising=False)
    _fresh_logger().error("cli.check")
    # Normalised because this asserts on the WHOLE stream (to catch a trailing
    # space before the newline), and a bare "\n" is the one thing that could
    # legitimately differ per OS in a captured text stream.
    assert capsys.readouterr().err.replace("\r\n", "\n") == "[error] cli.check\n"

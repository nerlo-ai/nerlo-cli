"""CLI unit tests — Task 20.2 (Req 11.2, 11.10-11.13).

Exercises the Click commands in `nerlo_cli.commands` through Click's `CliRunner`,
with the registry HTTP layer stubbed via `httpx.MockTransport` (built into
httpx — no new dependency). `commands._client` is monkeypatched to return a
client wired to a per-test request handler, so the real command logic runs end
to end (argument validation, badge gating, `--json` rendering, auth handling)
without a live API.

Covers:
  * argument validation — malformed submit URL, out-of-range search query,
    unknown `--target` platform
  * badge-based install gating (Req 11.2) — Unsafe refused, no-badge refused,
    Caution prompts, Verified proceeds and writes the mcpServers entry
  * `--json` machine output
  * authentication handling (Req 11.10) — missing token refused before any
    network call; a 401/403 from the API aborts with no action taken
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from nerlo_cli import commands

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep telemetry (Ticket 30.5) off the real network and out of ~/.nerlo.

    Every test gets an isolated `NERLO_HOME` so installer-id / notice markers
    never touch the developer's home, and the telemetry client defaults to
    raising so no test can accidentally make a real POST. Tests that assert on
    telemetry override `_telemetry_client` after this runs.
    """
    monkeypatch.setenv("NERLO_HOME", str(tmp_path / ".nerlo"))
    monkeypatch.delenv("NERLO_TELEMETRY", raising=False)

    def _no_network(api_url: str) -> httpx.Client:
        raise RuntimeError("telemetry client not stubbed for this test")

    monkeypatch.setattr(commands, "_telemetry_client", _no_network)


def _use_handler(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    """Point `commands._client` at an httpx client backed by `handler`."""

    def _fake_client(api_url: str, token: str | None = None) -> httpx.Client:
        return httpx.Client(base_url=api_url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(commands, "_client", _fake_client)


def _json_response(request: httpx.Request, status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body, request=request)


def _json_payload(result: Result) -> Any:
    """Parse the JSON emitted by a `--json` command.

    Under the test harness structlog renders to stdout (no logging sink is
    configured), so a `cli.*` log line can precede the payload. The machine
    output is the trailing JSON value; slice from its opening bracket.
    """
    # `_echo_json` uses json.dumps(indent=2), so the payload's opening bracket
    # sits alone on its own line; find that line (a structlog line like
    # "[info ] cli.install ..." also contains "[", so a raw char search won't do).
    lines = result.output.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() in ("{", "["):
            return json.loads("".join(lines[i:]))
    raise AssertionError(f"no JSON payload found in output: {result.output!r}")


def _combined(result: Result) -> str:
    """stdout + stderr, so message assertions don't depend on which stream a
    given Click version routes an error to."""
    err = ""
    try:
        err = result.stderr
    except ValueError:  # no separate stderr captured
        err = ""
    return result.output + err


# --------------------------------------------------------------------------- #
# argument validation                                                         #
# --------------------------------------------------------------------------- #


def test_search_rejects_too_short_query() -> None:
    # Validated before any network call — no handler needed.
    result = CliRunner().invoke(commands.search, ["x"])
    assert result.exit_code == 1
    assert "2-100 characters" in _combined(result)


def test_submit_rejects_malformed_url() -> None:
    # URL is validated before the token check, so no token/handler is required.
    result = CliRunner().invoke(commands.submit, ["not-a-url"])
    assert result.exit_code == 1
    assert "malformed repository URL" in _combined(result)


def test_install_rejects_unknown_platform() -> None:
    # click.Choice rejects an unknown --target with a usage error (exit 2).
    result = CliRunner().invoke(
        commands.install, ["some-skill", "--target", "bogus", "--token", "t"]
    )
    assert result.exit_code == 2
    assert "bogus" in _combined(result)


# --------------------------------------------------------------------------- #
# badge-based install gating (Req 11.2)                                        #
# --------------------------------------------------------------------------- #


def _skill_handler(
    badge: str | None, *, repo: str = "https://www.npmjs.com/package/demo"
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/skills/"):
            return _json_response(
                request,
                200,
                {
                    "skill_id": "demo-skill",
                    "name": "demo",
                    "current_badge": badge,
                    "repository_url": repo,
                },
            )
        return _json_response(request, 404, {})

    return handler


def test_install_refuses_unsafe_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, _skill_handler("Unsafe"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "Unsafe badge" in _combined(result)


def test_install_refuses_unbadged_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, _skill_handler(None))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "no badge yet" in _combined(result)


def test_install_caution_aborts_on_decline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", tmp_path / "mcp.json")
    _use_handler(monkeypatch, _skill_handler("Caution"))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t"], input="n\n"
    )
    assert result.exit_code == 1
    assert "Aborted" in result.output
    assert not (tmp_path / "mcp.json").exists()  # no config written on abort


def test_install_verified_writes_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    written = json.loads(config.read_text(encoding="utf-8"))
    assert "demo-skill" in written["mcpServers"]
    # npmjs repo -> a runnable npx entry.
    assert written["mcpServers"]["demo-skill"]["command"] == "npx"


def test_install_caution_proceeds_on_confirm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Caution"))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t"], input="y\n"
    )
    assert result.exit_code == 0, _combined(result)
    assert config.exists()


# --------------------------------------------------------------------------- #
# --json output (Req 11.13)                                                    #
# --------------------------------------------------------------------------- #


def test_search_json_output_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        {"name": "alpha", "current_security_score": 88.0, "current_badge": "Verified", "id": "1"}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"results": results})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["alpha", "--json"])
    assert result.exit_code == 0
    parsed = _json_payload(result)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "alpha"


def test_install_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t", "--json"]
    )
    assert result.exit_code == 0, _combined(result)
    parsed = _json_payload(result)
    assert parsed["installed"] == "demo-skill"
    assert parsed["target"] == "mcp"


# --------------------------------------------------------------------------- #
# authentication handling (Req 11.10)                                         #
# --------------------------------------------------------------------------- #


def test_install_requires_token() -> None:
    # No token -> refused before any network call (Req 11.10).
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp"])
    assert result.exit_code == 1
    assert "authentication required" in _combined(result)


def test_submit_requires_token() -> None:
    result = CliRunner().invoke(commands.submit, ["https://github.com/o/r"])
    assert result.exit_code == 1
    assert "authentication required" in _combined(result)


def test_api_401_aborts_with_no_action(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 401, {"detail": "nope"})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.search, ["alpha"])
    assert result.exit_code == 1
    assert "authentication failed" in _combined(result)


# --------------------------------------------------------------------------- #
# install telemetry (Ticket 30.5)                                             #
# --------------------------------------------------------------------------- #

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _recording_telemetry_client(posts: list[httpx.Request]) -> Callable[[str], httpx.Client]:
    """A `_telemetry_client` replacement that records every request it sends."""

    def factory(api_url: str) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(request)
            return httpx.Response(202, json={}, request=request)

        return httpx.Client(base_url=api_url, transport=httpx.MockTransport(handler))

    return factory


def test_installer_token_hash_is_stable_and_hex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NERLO_HOME", str(tmp_path / ".nerlo"))
    # Anonymous path: stable across calls (installer-id persisted) and 64 hex.
    first = commands._installer_token_hash(None)
    second = commands._installer_token_hash(None)
    assert first == second
    assert _HEX64.match(first)
    # installer-id file is created 0600.
    id_path = tmp_path / ".nerlo" / "installer-id"
    assert id_path.exists()
    assert (id_path.stat().st_mode & 0o777) == 0o600
    # Token path: deterministic SHA-256 hex of the credential, 64 hex chars.
    token_hash = commands._installer_token_hash("tok")
    assert token_hash == hashlib.sha256(b"tok").hexdigest()
    assert _HEX64.match(token_hash)


def test_install_emits_telemetry_with_expected_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert len(posts) == 1
    request = posts[0]
    assert request.url.path == "/api/v1/installations"
    # Unauthenticated endpoint: the bearer token must not be sent.
    assert "authorization" not in {k.lower() for k in request.headers}
    body = json.loads(request.content)
    assert body["target_platform"] == "mcp"
    assert _HEX64.match(body["installer_token_hash"])
    assert 1 <= len(body["cli_version"]) <= 50


def test_install_prints_telemetry_notice_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    first = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert first.exit_code == 0, _combined(first)
    assert "anonymous install telemetry" in _combined(first)
    second = CliRunner().invoke(
        commands.install, ["demo", "--target", "mcp", "--token", "t", "--force"]
    )
    assert second.exit_code == 0, _combined(second)
    # Notice is one-time: it should not repeat on the second install.
    assert "anonymous install telemetry" not in _combined(second)


def test_telemetry_env_opt_out_suppresses_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NERLO_TELEMETRY", "0")
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert config.exists()  # install still happened
    assert posts == []  # opt-out -> no telemetry POST, no notice
    assert "anonymous install telemetry" not in _combined(result)


def test_telemetry_config_opt_out_suppresses_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nerlo_home = tmp_path / ".nerlo"
    nerlo_home.mkdir(parents=True)
    (nerlo_home / "config").write_text("telemetry=false\n", encoding="utf-8")
    monkeypatch.setenv("NERLO_HOME", str(nerlo_home))
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert posts == []


def test_telemetry_failure_does_not_break_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(api_url: str) -> httpx.Client:
        raise RuntimeError("network is down")

    monkeypatch.setattr(commands, "_telemetry_client", boom)
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    # Telemetry blew up but the install itself succeeded.
    assert result.exit_code == 0, _combined(result)
    assert config.exists()


# --------------------------------------------------------------------------- #
# type-aware submit + install (Ticket 33.9)                                   #
# --------------------------------------------------------------------------- #


def _capture_submit_body(captured: dict[str, Any]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/servers":
            captured["body"] = json.loads(request.content)
            return _json_response(request, 201, {"mcp_server_id": "s1", "scan_job_id": "j1"})
        return _json_response(request, 404, {})

    return handler


def test_submit_passes_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _use_handler(monkeypatch, _capture_submit_body(captured))
    result = CliRunner().invoke(
        commands.submit,
        ["https://github.com/o/r", "--type", "claude_skill", "--token", "t"],
    )
    assert result.exit_code == 0, _combined(result)
    assert captured["body"]["artifact_type"] == "claude_skill"
    assert captured["body"]["repository_url"] == "https://github.com/o/r"


def test_submit_without_type_omits_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _use_handler(monkeypatch, _capture_submit_body(captured))
    result = CliRunner().invoke(commands.submit, ["https://github.com/o/r", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    assert "artifact_type" not in captured["body"]


def test_submit_rejects_unknown_type() -> None:
    # click.Choice rejects an unknown --type with a usage error (exit 2).
    result = CliRunner().invoke(
        commands.submit, ["https://github.com/o/r", "--type", "bogus", "--token", "t"]
    )
    assert result.exit_code == 2
    assert "bogus" in _combined(result)


def _typed_skill_handler(
    artifact_type: str | None,
    *,
    badge: str | None = "Verified",
    repo: str = "https://www.npmjs.com/package/demo",
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/skills/"):
            body: dict[str, Any] = {
                "skill_id": "demo-skill",
                "name": "demo",
                "current_badge": badge,
                "repository_url": repo,
            }
            if artifact_type is not None:
                body["artifact_type"] = artifact_type
            return _json_response(request, 200, body)
        return _json_response(request, 404, {})

    return handler


def test_install_cursor_rule_still_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # cursor_rule has no defined install action in the ticket — the existing
    # clear refusal (no path guessing, nothing written) must be preserved.
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _typed_skill_handler("cursor_rule"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    combined = _combined(result)
    assert "cursor_rule" in combined
    assert "can only write MCP server config entries" in combined
    assert not config.exists()  # nothing written for an unsupported type


def test_install_allows_mcp_server_artifact_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _typed_skill_handler("mcp_server"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 0, _combined(result)
    written = json.loads(config.read_text(encoding="utf-8"))
    assert "demo-skill" in written["mcpServers"]


# --------------------------------------------------------------------------- #
# artifact_type display in search/info (Ticket 33.9)                          #
# --------------------------------------------------------------------------- #


def _search_results_handler(results: list[dict[str, Any]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"results": results})

    return handler


def test_search_table_shows_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        {
            "name": "alpha",
            "artifact_type": "claude_skill",
            "current_security_score": 88.0,
            "current_badge": "Verified",
            "author": "a",
            "id": "1",
        }
    ]
    _use_handler(monkeypatch, _search_results_handler(results))
    result = CliRunner().invoke(commands.search, ["alpha"])
    assert result.exit_code == 0, _combined(result)
    assert "TYPE" in result.output  # table header column
    assert "claude_skill" in result.output


def test_search_json_includes_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        {"name": "alpha", "artifact_type": "gemini_extension", "current_badge": "Verified"}
    ]
    _use_handler(monkeypatch, _search_results_handler(results))
    result = CliRunner().invoke(commands.search, ["alpha", "--json"])
    assert result.exit_code == 0, _combined(result)
    parsed = _json_payload(result)
    assert parsed[0]["artifact_type"] == "gemini_extension"


def _info_handler(artifact_type: str) -> Handler:
    """Skill detail with an artifact_type; server-id resolution finds nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/skills/"):
            return _json_response(
                request,
                200,
                {
                    "skill_id": "demo-skill",
                    "name": "demo",
                    "artifact_type": artifact_type,
                    "current_badge": "Verified",
                    "current_security_score": 91.0,
                    "repository_url": "https://github.com/o/r",
                },
            )
        return _json_response(request, 200, {"results": []})

    return handler


def test_info_table_shows_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, _info_handler("claude_skill"))
    result = CliRunner().invoke(commands.info, ["demo"])
    assert result.exit_code == 0, _combined(result)
    assert "type:" in result.output
    assert "claude_skill" in result.output


def test_info_json_includes_artifact_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, _info_handler("cursor_rule"))
    result = CliRunner().invoke(commands.info, ["demo", "--json"])
    assert result.exit_code == 0, _combined(result)
    parsed = _json_payload(result)
    assert parsed["skill"]["artifact_type"] == "cursor_rule"


# --------------------------------------------------------------------------- #
# claude_skill copy-install + gemini placeholder (Ticket 33.9)                #
# --------------------------------------------------------------------------- #

_SKILL_REPO_URL = "https://github.com/o/skill-repo"


def _fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point `Path.home()` (hence ~/.claude/skills) at an isolated tmp home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


def _stub_git_clone(
    monkeypatch: pytest.MonkeyPatch, populate: Callable[[Path], None]
) -> list[list[str]]:
    """Stub `subprocess.run` for the shallow-clone path — no real network/git.

    Records each argv; `populate` builds the cloned tree at the clone dest.
    """
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        assert args[:4] == ["git", "clone", "--depth", "1"]
        dest = Path(args[-1])
        dest.mkdir(parents=True, exist_ok=True)
        populate(dest)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    return calls


def _populate_skill_repo(dest: Path) -> None:
    """A repo whose skill lives in a nested directory (not the repo root)."""
    skill_dir = dest / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# demo skill\n", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (dest / "README.md").write_text("repo readme\n", encoding="utf-8")


def test_install_claude_skill_copies_skill_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    calls = _stub_git_clone(monkeypatch, _populate_skill_repo)
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    _use_handler(monkeypatch, _typed_skill_handler("claude_skill", repo=_SKILL_REPO_URL))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    installed = home / ".claude" / "skills" / "demo-skill"
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == "# demo skill\n"
    assert (installed / "scripts" / "run.py").exists()
    assert not (installed / "README.md").exists()  # only the skill dir, not the repo
    # The clone was shallow and pointed at the skill's repository URL.
    assert len(calls) == 1
    assert _SKILL_REPO_URL in calls[0]
    # Telemetry reports target_platform "claude-code" for claude_skill installs.
    assert len(posts) == 1
    assert json.loads(posts[0].content)["target_platform"] == "claude-code"


def test_install_claude_skill_requires_claude_code_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    _use_handler(monkeypatch, _typed_skill_handler("claude_skill", repo=_SKILL_REPO_URL))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "claude-code" in _combined(result)
    assert not (home / ".claude").exists()  # nothing written


def test_install_claude_skill_refuses_without_skill_md(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)

    def populate_no_skill(dest: Path) -> None:
        (dest / "README.md").write_text("no skill here\n", encoding="utf-8")

    _stub_git_clone(monkeypatch, populate_no_skill)
    _use_handler(monkeypatch, _typed_skill_handler("claude_skill", repo=_SKILL_REPO_URL))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "SKILL.md" in _combined(result)
    assert not (home / ".claude" / "skills" / "demo-skill").exists()


def test_install_claude_skill_honours_badge_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    _use_handler(
        monkeypatch,
        _typed_skill_handler("claude_skill", badge="Unsafe", repo=_SKILL_REPO_URL),
    )
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 1
    assert "Unsafe badge" in _combined(result)
    assert not (home / ".claude").exists()


def test_install_gemini_extension_placeholder_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    config = tmp_path / "mcp.json"
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "gemini", config)
    _use_handler(monkeypatch, _typed_skill_handler("gemini_extension"))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "gemini", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    assert "install path pending Google runtime API" in _combined(result)
    assert not config.exists()  # nothing written anywhere
    assert not (home / ".claude").exists()

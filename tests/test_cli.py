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
        # A shallow clone that disables symlink checkout (untrusted-repo
        # hardening): committed symlinks must never materialize as real links.
        assert args[0] == "git"
        assert "core.symlinks=false" in args
        assert "clone" in args and "--depth" in args and "1" in args
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


def _populate_skill_repo_with_symlink(secret: Path) -> Callable[[Path], None]:
    """A skill repo that commits a symlink pointing at an out-of-tree secret —
    the exfil-on-install trick the copytree(symlinks=True) fix defends against."""

    def _populate(dest: Path) -> None:
        skill_dir = dest / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# demo skill\n", encoding="utf-8")
        (skill_dir / "stolen").symlink_to(secret)

    return _populate


def test_install_claude_skill_does_not_dereference_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A malicious skill's `stolen -> <secret>` must be copied AS A LINK, never
    # dereferenced into a regular file holding the secret's content at install
    # time. (In real git the clone's core.symlinks=false makes it a placeholder
    # file; the stub creates a real symlink, exercising the copytree layer.)
    home = _fake_home(monkeypatch, tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n", encoding="utf-8")
    _stub_git_clone(monkeypatch, _populate_skill_repo_with_symlink(secret))
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    _use_handler(monkeypatch, _typed_skill_handler("claude_skill", repo=_SKILL_REPO_URL))
    result = CliRunner().invoke(
        commands.install, ["demo", "--target", "claude-code", "--token", "t"]
    )
    assert result.exit_code == 0, _combined(result)
    stolen = home / ".claude" / "skills" / "demo-skill" / "stolen"
    # The install preserved the symlink instead of materializing a regular file
    # with the secret's bytes — no install-time out-of-tree read amplification.
    assert stolen.is_symlink()


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


# --------------------------------------------------------------------------- #
# nerlo check — the CI gate (exit codes are the contract)                     #
# --------------------------------------------------------------------------- #


def _row(
    name: str,
    badge: str | None,
    *,
    row_id: str = "",
    repo: str = "",
    withheld: bool = False,
    score: float | None = 90.0,
) -> dict[str, Any]:
    """One registry search row, shaped like the live API's response.

    Real ids are opaque UUIDs, so the derived stub id is slug-safe — a package
    name like `@scope/pkg` must not leak a `/` into the detail URL.
    """
    return {
        "id": row_id or "id-" + re.sub(r"[^a-z0-9._-]", "-", name.lower()),
        "name": name,
        "repository_url": repo,
        "current_badge": badge,
        "current_security_score": None if badge is None else score,
        "aggregate_verdict_withheld": withheld,
        "artifact_type": "mcp_server",
    }


def _registry_handler(
    rows: list[dict[str, Any]],
    *,
    search_status: int = 200,
    detail_status: int = 200,
    detail_overrides: dict[str, dict[str, Any]] | None = None,
) -> Handler:
    """Stub registry.

    Search returns EVERY row regardless of `q` — the live `q=` is fuzzy (a
    query for "server" returns "@4everland/hosting-mcp"), so returning
    everything is both faithful and the strongest possible test that `check`
    does its own exact-identity matching rather than trusting the search hit.
    """
    by_id = {str(r["id"]): r for r in rows}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/servers":
            if search_status != 200:
                return _json_response(request, search_status, {})
            return _json_response(request, 200, {"results": rows, "total_count": len(rows)})
        if path.startswith("/api/v1/servers/"):
            if detail_status != 200:
                return _json_response(request, detail_status, {})
            row = by_id.get(path.rsplit("/", 1)[-1])
            if row is None:
                return _json_response(request, 404, {})
            detail = dict(row)
            detail["composite_badge"] = row["current_badge"]
            detail["composite_score"] = row["current_security_score"]
            detail["scanner_reports"] = [{"scanner_name": "s1"}, {"scanner_name": "s2"}]
            detail.update((detail_overrides or {}).get(str(row["id"]), {}))
            return _json_response(request, 200, detail)
        return _json_response(request, 404, {})

    return handler


def _project(tmp_path: Path, servers: dict[str, Any], filename: str = "mcp.json") -> Path:
    """A project checkout carrying a platform MCP config."""
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return root


def _npx(package: str) -> dict[str, Any]:
    """The runnable entry shape `_build_mcp_entry` writes for npm packages."""
    return {"command": "npx", "args": ["-y", package]}


# --- the one design rule: three outcomes, never collapsed ------------------- #


def test_check_reports_found_bad_found_good_and_not_found_distinctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # (a) found+Unsafe, (b) found+Verified, (c) not in the registry at all.
    # All three must be visibly distinct; (c) must NOT render as a pass.
    root = _project(
        tmp_path,
        {"bad": _npx("bad"), "good": _npx("good"), "ghost": _npx("ghost")},
    )
    _use_handler(
        monkeypatch,
        _registry_handler([_row("bad", "Unsafe"), _row("good", "Verified")]),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    statuses = {a["name"]: a["status"] for a in payload["artifacts"]}
    assert statuses == {"bad": "unsafe", "good": "verified", "ghost": "unknown"}
    # The three are distinct values — none of them is spelled as another.
    assert len(set(statuses.values())) == 3


def test_check_unknown_is_not_rendered_as_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An artifact nobody has scanned must be called out as unknown and pointed
    # at the submit funnel — never shown as verified/green.
    root = _project(tmp_path, {"ghost": _npx("ghost")})
    _use_handler(monkeypatch, _registry_handler([_row("other", "Verified")]))
    result = CliRunner().invoke(commands.check, [str(root)])
    combined = _combined(result)
    assert "UNKNOWN" in combined
    assert "VERIFIED" not in combined
    assert "NOT in the Nerlo registry" in combined
    assert "Unknown is not safe" in combined
    assert "nerlo submit" in combined  # the submission funnel


def test_check_withheld_verdict_is_not_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Live data has rows with aggregate_verdict_withheld=true and a null badge:
    # the registry declining to vouch. That is its own outcome, not a pass.
    root = _project(tmp_path, {"held": _npx("held")})
    _use_handler(monkeypatch, _registry_handler([_row("held", None, withheld=True)]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "withheld"
    # And it fails the strict level, because it is not a verification.
    strict = CliRunner().invoke(commands.check, [str(root), "--fail-on", "any"])
    assert strict.exit_code == 1


# --- exit codes: the contract ---------------------------------------------- #


def test_check_exits_1_on_unsafe_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _project(tmp_path, {"bad": _npx("bad")})
    _use_handler(monkeypatch, _registry_handler([_row("bad", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 1
    assert "FAIL" in _combined(result)


def test_check_exits_0_when_all_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(monkeypatch, _registry_handler([_row("good", "Verified")]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 0
    assert "PASS" in _combined(result)


def test_check_caution_passes_by_default_and_fails_at_caution_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _project(tmp_path, {"meh": _npx("meh")})
    _use_handler(monkeypatch, _registry_handler([_row("meh", "Caution")]))
    default = CliRunner().invoke(commands.check, [str(root)])
    assert default.exit_code == 0
    strict = CliRunner().invoke(commands.check, [str(root), "--fail-on", "caution"])
    assert strict.exit_code == 1


def test_check_unknown_passes_by_default_but_fails_on_any(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The deliberate policy call: unknown is loud but non-fatal at the verdict
    # levels (a gate that red-builds every repo on day one gets deleted), and
    # fatal at --fail-on any (absence of evidence treated as failure).
    root = _project(tmp_path, {"ghost": _npx("ghost")})
    _use_handler(monkeypatch, _registry_handler([]))
    assert CliRunner().invoke(commands.check, [str(root)]).exit_code == 0
    assert CliRunner().invoke(commands.check, [str(root), "--fail-on", "caution"]).exit_code == 0
    assert CliRunner().invoke(commands.check, [str(root), "--fail-on", "any"]).exit_code == 1


def test_check_network_failure_is_exit_3_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A check that cannot reach the registry has verified nothing. It must not
    # exit 0, and it must say so.
    root = _project(tmp_path, {"good": _npx("good")})

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("registry down", request=request)

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3
    combined = _combined(result)
    assert "ERROR" in combined
    assert "could not be resolved" in combined
    assert "PASS" not in combined


def test_check_search_http_error_is_exit_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A 500 from the registry is also "no answer", not "no problem".
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(monkeypatch, _registry_handler([_row("good", "Verified")], search_status=500))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3


def test_check_detail_failure_does_not_fall_back_to_the_search_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Search says Verified but the authoritative detail fetch fails. Reporting
    # the list row's badge here would turn "could not verify" into "verified".
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(monkeypatch, _registry_handler([_row("good", "Verified")], detail_status=503))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "error"
    assert payload["exit_code"] == 3
    assert result.exit_code == 3


def test_check_violation_outranks_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One Unsafe (known) plus one unresolvable: exit 1, the actionable signal,
    # and the unresolved row is still reported.
    root = _project(tmp_path, {"bad": _npx("bad"), "boom": _npx("boom")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("bad", "Unsafe"), _row("boom", "Verified")],
            detail_overrides={"id-boom": {}},
            detail_status=200,
        ),
    )
    # Make only the second artifact unresolvable by failing its detail call.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/servers":
            return _json_response(
                request, 200, {"results": [_row("bad", "Unsafe"), _row("boom", "Verified")]}
            )
        if request.url.path.endswith("id-boom"):
            return _json_response(request, 500, {})
        return _json_response(
            request, 200, {**_row("bad", "Unsafe"), "composite_badge": "Unsafe"}
        )

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    statuses = {a["name"]: a["status"] for a in payload["artifacts"]}
    assert statuses == {"bad": "unsafe", "boom": "error"}
    assert result.exit_code == 1  # violation wins over incomplete


def test_check_nonexistent_path_is_usage_error_exit_2() -> None:
    result = CliRunner().invoke(commands.check, ["/no/such/directory/anywhere"])
    assert result.exit_code == 2


def test_check_empty_is_an_explicit_pass_not_an_empty_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    _use_handler(monkeypatch, _registry_handler([]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 0
    assert "nothing to check" in _combined(result)
    assert "STATUS" not in result.output  # no bare table header


def test_check_unreadable_config_is_incomplete_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A config we could not parse is an UNCHECKED config. Treating it as "no
    # artifacts configured" would report a clean pass over a blind spot.
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text("{ this is not json", encoding="utf-8")
    _use_handler(monkeypatch, _registry_handler([]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3
    assert "could not read config" in _combined(result)


# --- discovery: the reader must parse what the writer produces -------------- #


def test_check_reads_back_what_install_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # End-to-end round trip: `install` writes an mcpServers entry, `check`
    # discovers and resolves that exact entry. This is what stops the reader
    # from drifting away from the writer's on-disk shape.
    posts: list[httpx.Request] = []
    monkeypatch.setattr(commands, "_telemetry_client", _recording_telemetry_client(posts))
    config = tmp_path / "proj" / "mcp.json"
    config.parent.mkdir()
    monkeypatch.setitem(commands.TARGET_CONFIG_PATHS, "mcp", config)
    _use_handler(monkeypatch, _skill_handler("Verified"))
    installed = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert installed.exit_code == 0, _combined(installed)

    # `_build_mcp_entry` turned the npmjs repo into `npx -y demo`; check must
    # recover the package identity "demo" from it.
    _use_handler(monkeypatch, _registry_handler([_row("demo", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(config), "--json"])
    payload = _json_payload(result)
    assert [a["name"] for a in payload["artifacts"]] == ["demo-skill"]
    assert payload["artifacts"][0]["status"] == "unsafe"
    assert result.exit_code == 1


def test_check_matches_on_package_name_not_just_config_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The mcpServers key is user-chosen ("todoist"); the registry knows the
    # package ("@abhiz123/todoist-mcp-server"). Identity comes from the args.
    root = _project(tmp_path, {"todoist": _npx("@abhiz123/todoist-mcp-server@1.2.3")})
    _use_handler(monkeypatch, _registry_handler([_row("@abhiz123/todoist-mcp-server", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unsafe"  # version pin stripped


def test_check_matches_on_repository_url_for_non_package_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `_build_mcp_entry` writes {"repository": ...} when there is no runnable
    # package; that URL is the identity, normalised across .git/case/slash.
    root = _project(
        tmp_path,
        {"whatever": {"repository": "https://GitHub.com/o/r.git", "nerlo_badge": "Verified"}},
    )
    _use_handler(
        monkeypatch,
        _registry_handler([_row("unrelated-name", "Unsafe", repo="https://github.com/o/r/")]),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unsafe"


def test_check_does_not_accept_a_fuzzy_search_hit_as_a_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The registry's q= is fuzzy. Attaching some other project's Verified badge
    # to a local artifact is the same collapse as scoring an unknown green.
    root = _project(tmp_path, {"my-private-server": _npx("my-private-server")})
    _use_handler(monkeypatch, _registry_handler([_row("some-other-server", "Verified")]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unknown"
    assert payload["artifacts"][0]["server_id"] is None


def test_check_duplicate_registry_rows_take_the_worst_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Duplicate submissions of one name are real in live data. A Verified
    # duplicate must never launder an Unsafe one into a pass.
    root = _project(tmp_path, {"dup": _npx("dup")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("dup", "Verified", row_id="id-ok"), _row("dup", "Unsafe", row_id="id-bad")]
        ),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unsafe"
    assert payload["artifacts"][0]["duplicate_matches"] == 1
    assert result.exit_code == 1


def test_check_discovers_installed_claude_skills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `_install_claude_skill` copies skill dirs under <root>/.claude/skills/;
    # discovery reads that same layout back.
    root = tmp_path / "proj"
    skill = root / ".claude" / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    staging = root / ".claude" / "skills" / ".demo-skill.tmp.nerlo-tmp"
    staging.mkdir()
    (staging / "SKILL.md").write_text("# staging leftover\n", encoding="utf-8")
    _use_handler(monkeypatch, _registry_handler([_row("demo-skill", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    # The installer's dot-prefixed staging dir is not a discovered artifact.
    assert [a["name"] for a in payload["artifacts"]] == ["demo-skill"]
    assert payload["artifacts"][0]["artifact_type"] == "claude_skill"
    assert payload["artifacts"][0]["status"] == "unsafe"


def test_check_reads_project_scoped_dot_mcp_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Claude Code's project-scoped `.mcp.json` is a discovery-only extra; a CI
    # gate that cannot see it is a gate with a hole.
    root = _project(tmp_path, {"bad": _npx("bad")}, filename=".mcp.json")
    _use_handler(monkeypatch, _registry_handler([_row("bad", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 1


def test_check_with_a_path_does_not_scan_the_home_locations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # CI runs in a checkout where $HOME belongs to an ephemeral runner. A
    # project scan must depend on the repo, not on the machine.
    home = _fake_home(monkeypatch, tmp_path)
    monkeypatch.setattr(
        commands,
        "TARGET_CONFIG_PATHS",
        {"claude-code": home / ".claude.json", "mcp": home / "mcp.json"},
    )
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"runner-global": _npx("runner-global")}}), encoding="utf-8"
    )
    root = _project(tmp_path, {"in-repo": _npx("in-repo")})
    _use_handler(
        monkeypatch,
        _registry_handler([_row("in-repo", "Verified"), _row("runner-global", "Unsafe")]),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert [a["name"] for a in payload["artifacts"]] == ["in-repo"]
    assert result.exit_code == 0  # the runner's Unsafe global did not leak in


def test_check_without_a_path_scans_the_standard_locations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The positive control for the test above: with no PATH, the same home
    # config IS scanned (so its absence there is isolation, not blindness).
    home = _fake_home(monkeypatch, tmp_path)
    monkeypatch.setattr(commands, "TARGET_CONFIG_PATHS", {"claude-code": home / ".claude.json"})
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"runner-global": _npx("runner-global")}}), encoding="utf-8"
    )
    _use_handler(monkeypatch, _registry_handler([_row("runner-global", "Unsafe")]))
    result = CliRunner().invoke(commands.check, ["--json"])
    payload = _json_payload(result)
    assert [a["name"] for a in payload["artifacts"]] == ["runner-global"]
    assert result.exit_code == 1


def test_check_json_carries_the_exit_code_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _project(tmp_path, {"bad": _npx("bad"), "good": _npx("good")})
    _use_handler(
        monkeypatch, _registry_handler([_row("bad", "Unsafe"), _row("good", "Verified")])
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["exit_code"] == result.exit_code == 1
    assert payload["fail_on"] == "unsafe"
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["unsafe"] == 1
    assert payload["summary"]["verified"] == 1


def test_check_unknown_badge_string_is_not_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A badge value a future API grows must fail closed, not read as a pass.
    root = _project(tmp_path, {"new": _npx("new")})
    _use_handler(monkeypatch, _registry_handler([_row("new", "Sparkling")]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unscored"
    assert CliRunner().invoke(commands.check, [str(root), "--fail-on", "any"]).exit_code == 1


def test_check_does_not_invent_an_identity_for_opaque_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `node server.js` names no package. Guessing "server.js" could match some
    # unrelated registry row; the entry resolves on its config key alone.
    root = _project(tmp_path, {"local-thing": {"command": "node", "args": ["server.js"]}})
    _use_handler(monkeypatch, _registry_handler([_row("server.js", "Verified")]))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unknown"


def test_check_match_without_a_usable_id_is_error_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A matched row whose id is missing cannot be resolved to an authoritative
    # verdict. Reading the badge off the search row instead would be the silent
    # fallback that turns "could not verify" into "verified".
    root = _project(tmp_path, {"good": _npx("good")})
    row = _row("good", "Verified")
    row["id"] = ""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"results": [row]})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "error"
    assert result.exit_code == 3


def test_check_reports_a_zero_composite_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 0.0 is falsy but is a real — and maximally alarming — score. Picking the
    # score with `a or b` would discard it and report the rosier list value.
    root = _project(tmp_path, {"bad": _npx("bad")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("bad", "Unsafe", score=99.0)],
            detail_overrides={"id-bad": {"composite_score": 0.0}},
        ),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["score"] == 0.0

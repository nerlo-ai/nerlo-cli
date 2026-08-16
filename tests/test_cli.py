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
  * badge-based install gating (Req 11.2) — a Flagged badge is refused, an
    Unrated artifact is refused, Caution prompts, and a Clean badge proceeds
    and writes the mcpServers entry. The handlers serve the WIRE badges
    (`Unsafe`/`Caution`/`Verified`) because that is what the API sends; the
    assertions are on the DISPLAY words, which is the split under test
  * `--json` machine output
  * authentication handling (Req 11.10) — missing token refused before any
    network call; a 401/403 from the API aborts with no action taken
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from nerlo_cli import _update, commands

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

    # The PyPI update check hangs off a `call_on_close` hook on EVERY command
    # (see `commands.UpdateNoticeCommand`), so without this it would run — and
    # try to reach pypi.org — in every test in this file. Off by env, and the
    # fetcher additionally replaced so a regression in the env handling still
    # cannot produce a real request. `tests/test_update_check.py` is the only
    # place the notice is deliberately switched on.
    monkeypatch.setenv("NERLO_UPDATE_CHECK", "0")

    def _no_pypi() -> str | None:
        raise RuntimeError("update check not stubbed for this test")

    monkeypatch.setattr(_update, "fetch_latest", _no_pypi)


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
    # The handler still serves the WIRE badge `Unsafe` — that is what the API
    # sends and what the gate compares against. Only the printed word changed.
    _use_handler(monkeypatch, _skill_handler("Unsafe"))
    result = CliRunner().invoke(commands.install, ["demo", "--target", "mcp", "--token", "t"])
    assert result.exit_code == 1
    assert "Flagged badge" in _combined(result)


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
    # POSIX ONLY, and this is a platform fact rather than a flaky test.
    # Windows has no POSIX mode bits: `os.chmod` there toggles a read-only flag
    # and the file lands 0o666, so asserting 0600 tests the platform, not our
    # code. Found by the first cross-platform CI run on 2026-08-12 — all three
    # Windows cells failed here while all six macOS/Linux cells passed, which is
    # exactly the isolation `fail-fast: false` exists to give.
    #
    # SKIPPED, NOT DELETED: the guarantee is real on macOS and Linux and worth
    # keeping pinned. Deleting the assertion would quietly stop covering them
    # too, which is how a control becomes decorative.
    if os.name != "nt":
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
    assert "Flagged badge" in _combined(result)
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
    max_page_size: int = 100,
    queries: list[httpx.URL] | None = None,
) -> Handler:
    """Stub registry.

    Search returns EVERY row regardless of `q` — the live `q=` is fuzzy (a
    query for "server" returns "@4everland/hosting-mcp"), so returning
    everything is both faithful and the strongest possible test that `check`
    does its own exact-identity matching rather than trusting the search hit.

    IT PAGINATES, and that is load-bearing. The previous version answered every
    search with the WHOLE row list and `total_count = len(rows)`, so it could
    not express "there are more rows than this response carries" — which is
    exactly the state the live registry was in when eight Unsafe rows named
    `app` sat on page 2 and `check` reported "not in registry", EXIT 0. A
    harness that cannot represent a truncated page cannot catch that bug, and
    28 check tests duly did not. So this handler honours `page`/`page_size`
    like the real endpoint (clamping page_size at `max_page_size`, as the live
    API clamps at 100) and reports `total_count`/`total_pages` for the FULL set.
    Drive `max_page_size` down to force multi-page reads; drive it down further
    than `commands.CHECK_MAX_PAGES` can consume to force a genuine truncation.

    `queries` collects every search URL issued, so a test can assert what was
    SEARCHED FOR rather than only what was matched — the difference between
    proving a term is in the identity set and proving a query was sent.
    """
    by_id = {str(r["id"]): r for r in rows}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/servers":
            if queries is not None:
                queries.append(request.url)
            if search_status != 200:
                return _json_response(request, search_status, {})
            params = request.url.params
            page = max(1, int(params.get("page", 1)))
            page_size = min(max(1, int(params.get("page_size", 20))), max_page_size)
            start = (page - 1) * page_size
            total_pages = (len(rows) + page_size - 1) // page_size
            return _json_response(
                request,
                200,
                {
                    "page": page,
                    "page_size": page_size,
                    "total_count": len(rows),
                    "total_pages": total_pages,
                    "results": rows[start : start + page_size],
                },
            )
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
    assert "CLEAN" not in combined
    assert "VERIFIED" not in combined  # the retired word, gone from output entirely
    assert "NOT in the Nerlo registry listing" in combined
    assert "Unknown is not safe" in combined
    assert "nerlo submit" in combined  # the submission funnel
    # The claim is scoped to the LISTING, because the list endpoint documents
    # that `undistributed` artifacts are never listed. "We did not find it" is
    # a true statement about a search; "it is not in the registry" is not.
    assert "undistributed" in combined


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


# --------------------------------------------------------------------------- #
# nerlo check — an unresolved artifact is never an absent one                  #
#                                                                             #
# Everything below this line exists because an adversarial run against the     #
# LIVE registry found two ways to make `check` print EXIT 0 over a row the     #
# registry had already marked Unsafe. Both were the same defect: an inability  #
# to determine a verdict rendering as a pass.                                  #
# --------------------------------------------------------------------------- #


def _decoys(count: int) -> list[dict[str, Any]]:
    """Filler rows that match nothing, to push a real row off page 1."""
    return [_row(f"decoy-{i}", "Verified") for i in range(count)]


def test_check_finds_a_match_that_is_not_on_the_first_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # BLOCKER 1, the "search harder" half. Live: eight rows named exactly `app`,
    # every one composite_badge Unsafe, all on page 2 of `q=app&page_size=50`
    # (total_count 780) — one-page resolution called that "not in registry" and
    # exited 0. `check` must page until the listing is exhausted.
    root = _project(tmp_path, {"app": _npx("app")})
    _use_handler(
        monkeypatch,
        # page_size clamped to 3, so the Unsafe row is on page 3 of 3.
        _registry_handler([*_decoys(6), _row("app", "Unsafe")], max_page_size=3),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unsafe"
    assert result.exit_code == 1


def test_check_truncated_search_is_unresolved_not_not_in_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # BLOCKER 1, the "report honestly" half — the invariant is that AN
    # UNRESOLVED ARTIFACT MUST NEVER BE INDISTINGUISHABLE FROM AN ABSENT ONE.
    # Here the budget genuinely runs out with rows unread. The old code
    # answered "not in registry" + EXIT 0 in exactly this state.
    monkeypatch.setattr(commands, "CHECK_MAX_PAGES", 1)
    root = _project(tmp_path, {"app": _npx("app")})
    _use_handler(
        monkeypatch,
        _registry_handler([*_decoys(6), _row("app", "Unsafe")], max_page_size=2),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "unresolved"
    assert artifact["status"] != "unknown"  # NOT the same fact as absence
    assert "2 of 7 rows read" in artifact["note"]  # says how much it did not read
    assert result.exit_code == 3  # and it is not a pass


def test_check_truncated_search_is_not_a_pass_at_any_fail_on_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The gate's exit code is its product, so "we could not determine" must be
    # non-zero at EVERY level — including the laxest one, which is the level a
    # default CI job actually runs.
    root = _project(tmp_path, {"app": _npx("app")})
    for level in sorted(commands.FAIL_ON_STATUSES):
        monkeypatch.setattr(commands, "CHECK_MAX_PAGES", 1)
        _use_handler(
            monkeypatch,
            _registry_handler([*_decoys(6), _row("app", "Unsafe")], max_page_size=2),
        )
        result = CliRunner().invoke(commands.check, [str(root), "--fail-on", level])
        assert result.exit_code != 0, f"--fail-on {level} passed an unresolved artifact"
        assert "UNRESOLVED" in _combined(result)


def test_check_search_response_without_a_results_list_is_error_not_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A malformed body read as "no results" is the same collapse one level down:
    # a broken response must not be able to say "not in the registry".
    root = _project(tmp_path, {"ghost": _npx("ghost")})

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, 200, {"total_count": 12})

    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "error"
    assert result.exit_code == 3


def test_check_stops_paging_once_the_listing_is_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive control for the two tests above: when the registry says the
    # listing IS exhausted, `check` believes it, reports `unknown`, and does not
    # burn the whole page budget doing so. Without this, "always truncated"
    # would pass the truncation tests.
    root = _project(tmp_path, {"ghost": _npx("ghost")})
    queries: list[httpx.URL] = []
    _use_handler(monkeypatch, _registry_handler(_decoys(3), max_page_size=2, queries=queries))
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unknown"
    assert result.exit_code == 0
    # 3 rows at 2/page = 2 pages, then it stops — not CHECK_MAX_PAGES pages.
    assert [q.params.get("page") for q in queries] == ["1", "2"]


def test_check_resolves_an_entry_that_has_only_a_repository_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # BLOCKER 2. `_match_rows` matches on repository_url, but a row that was
    # never RETRIEVED can never be matched, and the registry's q= does not index
    # repository URLs (live: q=<the full URL> -> 0 results, q=alloydb -> the
    # row). Live repro: an entry keyed `byrepo` with repository
    # https://GitHub.com/gemini-cli-extensions/alloydb.git resolved UNKNOWN/exit
    # 0 while the identical entry keyed `alloydb` resolved UNSAFE/exit 1 — the
    # same row, two answers, decided by the config key.
    root = _project(
        tmp_path,
        {"byrepo": {"repository": "https://GitHub.com/gemini-cli-extensions/alloydb.git"}},
    )
    queries: list[httpx.URL] = []
    handler = _registry_handler(
        [_row("alloydb", "Unsafe", repo="https://github.com/gemini-cli-extensions/alloydb")],
        queries=queries,
    )
    _use_handler(monkeypatch, handler)
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    payload = _json_payload(result)
    assert payload["artifacts"][0]["status"] == "unsafe"
    assert result.exit_code == 1
    # The repo's own path segment must be ISSUED as a query. Asserting only on
    # the status would pass on a stub that returns every row for every term.
    assert "alloydb" in [q.params.get("q") for q in queries]


def test_check_issues_the_package_name_as_a_query_not_just_as_a_matcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Mutation survivor: deleting the package name from the search terms left
    # all 62 tests green, because the stub returns every row for every query —
    # so the package-name test proved only that the name was in the MATCH set,
    # never that it was SEARCHED FOR. Against the real fuzzy API that is the
    # difference between resolving and not.
    root = _project(tmp_path, {"todoist": _npx("@abhiz123/todoist-mcp-server@1.2.3")})
    queries: list[httpx.URL] = []
    _use_handler(
        monkeypatch,
        _registry_handler([_row("@abhiz123/todoist-mcp-server", "Unsafe")], queries=queries),
    )
    result = CliRunner().invoke(commands.check, [str(root), "--json"])
    assert result.exit_code == 1
    issued = [q.params.get("q") for q in queries]
    assert issued[0] == "@abhiz123/todoist-mcp-server"  # first, before the key


def test_an_unsearchable_name_is_unresolved_not_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # THE THIRD ROUTE to "could not determine" being rendered as a pass, found
    # by adversarial review after the other two were closed.
    #
    # `_search_terms` drops candidates under the registry's 2-character `q=`
    # floor. A server keyed "a" whose command yields no package name therefore
    # reaches `_resolve_one` with terms == (), the search loop never runs, and
    # the fall-through reported STATUS_UNKNOWN "not in the registry listing" —
    # a claim about a search that was never issued — which does not fail
    # `--fail-on unsafe`, so it exited 0.
    #
    # THE QUERY ASSERTION IS THE POINT. Asserting only the status would pass
    # against an implementation that searched and legitimately found nothing;
    # `issued == []` is what pins "we never asked".
    root = _project(tmp_path, {"a": {"command": "node", "args": ["server.js"]}})
    queries: list[httpx.URL] = []
    _use_handler(monkeypatch, _registry_handler([_row("a", "Unsafe")], queries=queries))
    result = CliRunner().invoke(commands.check, [str(root)])

    issued = [q.params.get("q") for q in queries]
    assert issued == [], f"expected no query to be issuable, got {issued}"
    assert result.exit_code == commands.EXIT_INCOMPLETE
    assert "UNRESOLVED" in result.output
    assert "not in the registry listing" not in result.output
    # And it must never be a pass at ANY level — the invariant the status exists for.
    for level in ("unsafe", "caution", "any"):
        r = CliRunner().invoke(commands.check, [str(root), "--fail-on", level])
        assert r.exit_code != 0, f"--fail-on {level} passed an artifact it never searched for"


def test_fail_on_ladder_is_monotonic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Mutation survivor: narrowing FAIL_ON_STATUSES["caution"] to drop
    # STATUS_UNSAFE left all 62 tests green — nothing asserted that a laxer
    # -sounding level still catches everything the stricter one does. A gate
    # whose severity ordering is not monotonic lies about what it enforces.
    levels = ["unsafe", "caution", "any"]
    assert sorted(commands.FAIL_ON_STATUSES) == sorted(levels)
    for lower, higher in pairwise(levels):
        assert commands.FAIL_ON_STATUSES[lower] < commands.FAIL_ON_STATUSES[higher], (
            f"--fail-on {higher} must be a strict superset of --fail-on {lower}"
        )
    # And end to end, not just as set algebra: an Unsafe artifact fails at every
    # level, which is the property a user actually depends on.
    root = _project(tmp_path, {"bad": _npx("bad")})
    for level in levels:
        _use_handler(monkeypatch, _registry_handler([_row("bad", "Unsafe")]))
        result = CliRunner().invoke(commands.check, [str(root), "--fail-on", level])
        assert result.exit_code == 1, f"Unsafe passed --fail-on {level}"


def test_check_never_treats_could_not_ask_as_a_policy_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The other half of the ladder invariant: the statuses that mean "no answer"
    # are in none of the --fail-on sets (they drive exit 3 instead), so a future
    # edit cannot quietly reclassify "we could not ask" as a verdict.
    for statuses in commands.FAIL_ON_STATUSES.values():
        assert not (statuses & commands.INCOMPLETE_STATUSES)
    assert commands.STATUS_UNRESOLVED in commands.INCOMPLETE_STATUSES
    assert commands.STATUS_ERROR in commands.INCOMPLETE_STATUSES


def test_check_non_numeric_score_does_not_manufacture_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `float(r.score)` in the renderer raised ValueError on "N/A", which escaped
    # the command as exit 1 — and the contract reads exit 1 as "policy
    # violated". An unparseable decoration must not be able to invent a verdict.
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("good", "Verified")],
            detail_overrides={"id-good": {"composite_score": "N/A"}},
        ),
    )
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exception is None, result.exception
    assert result.exit_code == 0
    assert "CLEAN" in _combined(result)
    payload = _json_payload(
        CliRunner().invoke(commands.check, [str(root), "--json"]),
    )
    assert payload["artifacts"][0]["score"] is None  # rendered as "-", not 1.0


def test_check_numeric_string_score_is_still_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive control for the guard above: it must coerce, not blanket-discard.
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("good", "Verified")],
            detail_overrides={"id-good": {"composite_score": "88.5"}},
        ),
    )
    payload = _json_payload(CliRunner().invoke(commands.check, [str(root), "--json"]))
    assert payload["artifacts"][0]["score"] == 88.5


def _claude_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def test_check_discovers_servers_nested_under_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Claude Code nests per-project servers under `projects.<path>.mcpServers`
    # in ~/.claude.json, and on a real machine that is where most entries live —
    # confirmed against a ~/.claude.json carrying a 10-entry `projects` map.
    # Reading only the top-level key means default-mode `check` silently skips
    # them, which is a gate with a hole in the shape of most real installs.
    config = tmp_path / ".claude.json"
    _claude_json(
        config,
        {
            "mcpServers": {"top": _npx("top")},
            "projects": {"/home/dev/work": {"mcpServers": {"nested": _npx("nested")}}},
        },
    )
    _use_handler(
        monkeypatch, _registry_handler([_row("top", "Verified"), _row("nested", "Unsafe")])
    )
    result = CliRunner().invoke(commands.check, [str(config), "--json"])
    payload = _json_payload(result)
    statuses = {a["name"]: a["status"] for a in payload["artifacts"]}
    assert statuses == {"top": "verified", "nested": "unsafe"}
    assert result.exit_code == 1
    nested = next(a for a in payload["artifacts"] if a["name"] == "nested")
    assert "/home/dev/work" in nested["source"]  # says WHICH project it came from


def test_check_keeps_same_named_servers_from_different_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two projects may each configure a server called `github`. Merging the
    # nested maps into one name-keyed dict would drop one of them — an unchecked
    # artifact looking exactly like an absent one, one layer lower down.
    config = tmp_path / ".claude.json"
    _claude_json(
        config,
        {
            "projects": {
                "/a": {"mcpServers": {"github": _npx("github-a")}},
                "/b": {"mcpServers": {"github": _npx("github-b")}},
            }
        },
    )
    _use_handler(
        monkeypatch,
        _registry_handler([_row("github-a", "Verified"), _row("github-b", "Unsafe")]),
    )
    payload = _json_payload(CliRunner().invoke(commands.check, [str(config), "--json"]))
    assert len(payload["artifacts"]) == 2
    assert {a["status"] for a in payload["artifacts"]} == {"verified", "unsafe"}


def test_check_malformed_nested_mcp_servers_is_incomplete_not_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A present-but-wrong-shaped nested `mcpServers` may be hiding entries, so
    # it is unreadable — not empty.
    config = tmp_path / ".claude.json"
    _claude_json(config, {"projects": {"/a": {"mcpServers": ["not", "an", "object"]}}})
    _use_handler(monkeypatch, _registry_handler([]))
    result = CliRunner().invoke(commands.check, [str(config)])
    assert result.exit_code == 3
    assert "could not read config" in _combined(result)


def test_check_ignores_a_projects_entry_that_cannot_hold_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive control for the test above: a scalar project value genuinely
    # cannot contain an `mcpServers` key, so skipping it hides nothing and must
    # not red the whole config.
    config = tmp_path / ".claude.json"
    _claude_json(
        config,
        {"projects": {"/a": "not-an-object", "/b": {"mcpServers": {"real": _npx("real")}}}},
    )
    _use_handler(monkeypatch, _registry_handler([_row("real", "Unsafe")]))
    result = CliRunner().invoke(commands.check, [str(config), "--json"])
    payload = _json_payload(result)
    assert [a["name"] for a in payload["artifacts"]] == ["real"]
    assert result.exit_code == 1


def test_check_unparseable_only_project_does_not_claim_nothing_to_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # It exited 3 and warned on stderr while stdout said "No AI artifacts
    # configured ... nothing to check." — the summary line contradicting the
    # verdict. A reader who trusts stdout reads a blind spot as a clean scope.
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text("{ this is not json", encoding="utf-8")
    _use_handler(monkeypatch, _registry_handler([]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 3
    assert "nothing to check" not in result.output
    assert "INCOMPLETE" in result.output
    assert "could not be parsed" in result.output


def test_check_empty_scope_still_says_nothing_to_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Positive control: the genuinely-empty case keeps its plain-words pass.
    root = tmp_path / "bare"
    root.mkdir()
    _use_handler(monkeypatch, _registry_handler([]))
    result = CliRunner().invoke(commands.check, [str(root)])
    assert result.exit_code == 0
    assert "nothing to check" in result.output
    assert "INCOMPLETE" not in result.output


def test_search_pages_keeps_a_reported_total_count_of_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `total = reported or total` looks harmless and is not: total_count 0 is a
    # real answer ("the registry matched nothing"), and `or` discards it as
    # falsy, leaving the count unknown. Same class as the `composite_score` 0.0
    # bug already pinned above — a falsy number is still a number.
    def handler(request: httpx.Request) -> httpx.Response:
        # Deliberately inconsistent: a body that reports 0 while carrying a row.
        # It is what the parser does with the 0 that is under test.
        return _json_response(
            request,
            200,
            {"results": [_row("x", "Verified")], "total_count": 0, "total_pages": 0},
        )

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x") as client:
        search = commands._search_pages(client, "q")
    assert search.total == 0
    assert search.total is not None
    assert search.truncated is False


def test_check_boolean_score_is_not_read_as_a_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `float(True)` is 1.0. A JSON `true` in composite_score must not become a
    # score of 1.0 — that is a number invented out of a non-number, and 1.0
    # reads as catastrophic.
    root = _project(tmp_path, {"good": _npx("good")})
    _use_handler(
        monkeypatch,
        _registry_handler(
            [_row("good", "Verified")],
            detail_overrides={"id-good": {"composite_score": True}},
        ),
    )
    payload = _json_payload(CliRunner().invoke(commands.check, [str(root), "--json"]))
    assert payload["artifacts"][0]["score"] is None

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

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from nerlo_cli import commands

Handler = Callable[[httpx.Request], httpx.Response]


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

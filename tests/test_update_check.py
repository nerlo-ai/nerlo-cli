"""The PyPI update notice — `nerlo_cli/_update.py`.

WHAT IS ACTUALLY AT RISK HERE, and therefore what these tests are for:

  1. A LEAK INTO `--json`. `nerlo check --json` is a CI gate other people's
     pipelines parse; a stray line on stdout breaks them, and a notice that
     "only" lands on stderr under `--json` still changes what a job logs. The
     surface test below runs EVERY command in `commands.ALL_COMMANDS` — the
     table itself, not a hand-written list, so a seventh command is covered the
     day it lands — four times each (plain/`--json` x notice available/off) and
     compares stdout BYTE FOR BYTE. Each negative ("no notice in this stream")
     is paired with the positive control that the notice fired at all in that
     exact configuration; without the pairing every one of them passes
     vacuously the moment the feature stops working.
  2. A CHANGED EXIT CODE. The notice hangs off a `call_on_close` hook, which
     runs on the `sys.exit` paths too. `test_a_notice_that_explodes_...`
     asserts that even a broken notice cannot move an exit code or touch
     output.
  3. A WRONG COMPARISON. "0.10.0" < "0.9.0" as strings. The table in
     `test_is_newer` is that trap plus the pre-release/dev/post/local cases a
     git checkout produces.
  4. A REAL NETWORK CALL FROM A TEST. The autouse fixture replaces the HTTP
     client factory with one that raises, so a test that forgets to stub gets
     an exception rather than an outbound request to pypi.org.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from nerlo_cli import _update, commands, main

# Bound at import time, BEFORE the autouse fixture below replaces the module
# attribute — the only way to reach the real client factory from a test.
_REAL_HTTP_CLIENT = _update._http_client

# The substring every assertion below keys on. It is the part of the message
# that carries the fact — "a newer one exists, here is its number".
NOTICE = "is available"

INSTALLED = "0.3.0"
LATEST = "9.9.9"

SERVER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No real network, no real ~/.nerlo, no ambient CI/opt-out env."""
    monkeypatch.setenv("NERLO_HOME", str(tmp_path / ".nerlo"))
    monkeypatch.delenv("NERLO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("NERLO_TELEMETRY", raising=False)
    # A developer's shell will not have CI set; a CI runner's will, and this
    # module's tests are almost all about behaviour with the check ENABLED.
    monkeypatch.delenv("CI", raising=False)

    def _no_network() -> httpx.Client:
        raise AssertionError("a test reached the real pypi.org HTTP client")

    monkeypatch.setattr(_update, "_http_client", _no_network)


def _no_config() -> dict[str, str]:
    return {}


# --------------------------------------------------------------------------- #
# version comparison (requirement 5: a string compare is wrong)               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("candidate", "installed", "expected"),
    [
        # The trap. Lexicographically "0.10.0" < "0.9.0"; numerically it is not.
        ("0.10.0", "0.9.0", True),
        ("0.9.0", "0.10.0", False),
        ("1.0.0", "0.99.99", True),
        ("0.4.0", "0.3.0", True),
        ("0.3.1", "0.3.0", True),
        # Same version, however it is spelled.
        ("0.3.0", "0.3.0", False),
        ("0.3", "0.3.0", False),
        ("0.3.0", "0.3", False),
        ("v0.3.0", "0.3.0", False),
        ("0.3.1", "0.3", True),
        # Never backwards (requirement 6).
        ("0.2.0", "0.3.0", False),
        # A pre-release IS behind the release it precedes, so 0.3.0 is genuinely
        # an upgrade from 0.3.0rc1.
        ("0.3.0", "0.3.0rc1", True),
        ("0.3.0", "0.3.0a1", True),
        ("0.3.0", "0.3.0b2", True),
        ("0.3.0", "0.3.0.dev5", True),
        # ...and a dev build of a FUTURE release is ahead of the published one.
        # This is the git-checkout case, and telling it to "upgrade" backwards
        # is the defect requirement 6 names.
        ("0.3.0", "0.4.0.dev1", False),
        ("0.3.0", "0.4.0rc1", False),
        # A local segment is metadata, not precedence: +g1a2b3c is still 0.3.0.
        ("0.3.0", "0.3.0+g1a2b3c", False),
        ("0.4.0", "0.3.0+local", True),
        # A post-release is AHEAD of the plain release it follows.
        ("0.3.0", "0.3.0.post1", False),
        ("0.3.0.post1", "0.3.0", True),
        # Fails closed. Anything unorderable is silence, never a guess.
        ("garbage", "0.3.0", False),
        ("0.3.0", "garbage", False),
        ("", "0.3.0", False),
        ("0.3.0", "", False),
        ("0.3.0", "0.3.0zzz9", False),
        ("0.3.0zzz9", "0.3.0", False),
    ],
)
def test_is_newer(candidate: str, installed: str, expected: bool) -> None:
    assert _update.is_newer(candidate, installed) is expected


def test_installed_version_is_none_when_the_package_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The git-checkout case: no distribution metadata, so nothing to say."""

    def _missing(name: str) -> str:
        raise _update.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_update.metadata, "version", _missing)
    assert _update.installed_version() is None


def test_installed_version_reads_the_distribution_metadata() -> None:
    """Positive control for the test above — the real lookup does resolve."""
    assert _update.installed_version() == _update.metadata.version("nerlo")


# --------------------------------------------------------------------------- #
# opt-out (requirement 3)                                                     #
# --------------------------------------------------------------------------- #


def test_the_check_runs_by_default() -> None:
    """Positive control for every opt-out assertion below."""
    assert _update.opted_out(_no_config) is None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_the_env_var_switches_it_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_update.ENV_VAR, value)
    assert _update.opted_out(_no_config) == "env"


def test_the_config_file_switches_it_off() -> None:
    assert _update.opted_out(lambda: {"update_check": "false"}) == "config"


def test_an_unrelated_config_key_does_not_switch_it_off() -> None:
    assert _update.opted_out(lambda: {"telemetry": "false"}) is None


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_ci_switches_it_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CI", value)
    assert _update.opted_out(_no_config) == "ci"


@pytest.mark.parametrize("value", ["", "0", "false"])
def test_a_falsey_ci_variable_is_not_ci(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CI", value)
    assert _update.opted_out(_no_config) is None


def test_an_explicit_env_opt_in_beats_the_ci_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Somebody who wants the notice in their pipeline has said so."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv(_update.ENV_VAR, "1")
    assert _update.opted_out(_no_config) is None


def test_the_env_var_beats_the_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_update.ENV_VAR, "1")
    assert _update.opted_out(lambda: {"update_check": "false"}) is None


# --------------------------------------------------------------------------- #
# cache (requirement 2)                                                       #
# --------------------------------------------------------------------------- #


def _counting_fetch(monkeypatch: pytest.MonkeyPatch, answer: str | None) -> list[int]:
    calls: list[int] = []

    def _fetch() -> str | None:
        calls.append(1)
        return answer

    monkeypatch.setattr(_update, "fetch_latest", _fetch)
    return calls


def _notify(state_dir: Path, *, now: float, as_json: bool = False) -> None:
    _update.maybe_notify(state_dir, as_json=as_json, read_config=_no_config, now=now)


def _cache_file(state_dir: Path) -> Path:
    return state_dir / _update.CACHE_FILENAME


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(_update, "installed_version", lambda: INSTALLED)
    yield tmp_path / "state"


def test_a_second_run_inside_the_interval_does_not_ask_pypi_again(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0)
    _notify(state, now=1000.0 + _update.CHECK_INTERVAL_SECONDS - 1)
    assert len(calls) == 1
    # ...and the notice still prints on the cached run. Caching the ANSWER, not
    # the notice: a user must keep being told until they upgrade.
    assert capsys.readouterr().err.count(NOTICE) == 2


def test_the_interval_expires(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control for the test above."""
    calls = _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0)
    _notify(state, now=1000.0 + _update.CHECK_INTERVAL_SECONDS)
    assert len(calls) == 2


def test_the_interval_is_one_day() -> None:
    assert _update.CHECK_INTERVAL_SECONDS == 24 * 60 * 60


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[]",
        '{"latest": "9.9.9"}',  # no timestamp
        '{"checked_at": "yesterday", "latest": "9.9.9"}',
        '{"checked_at": true, "latest": "9.9.9"}',
        '{"checked_at": 1000.0, "latest": 4}',
        "\x00\x01\x02",
    ],
)
def test_an_unusable_cache_file_costs_one_request_and_nothing_else(
    state: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    calls = _counting_fetch(monkeypatch, LATEST)
    state.mkdir(parents=True)
    _cache_file(state).write_text(content, encoding="utf-8")
    _notify(state, now=1000.0)
    assert len(calls) == 1
    assert json.loads(_cache_file(state).read_text(encoding="utf-8"))["latest"] == LATEST


def test_a_cache_timestamp_from_the_future_is_re_checked(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restored backup or a moved clock must not silence the check for years."""
    calls = _counting_fetch(monkeypatch, LATEST)
    state.mkdir(parents=True)
    _cache_file(state).write_text(
        json.dumps({"checked_at": 10_000.0, "latest": LATEST}), encoding="utf-8"
    )
    _notify(state, now=1000.0)
    assert len(calls) == 1


def test_a_state_dir_that_cannot_be_created_still_notifies(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The read-only-HOME case: no cache, no crash, notice still delivered."""
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("i am a file, not a directory", encoding="utf-8")
    calls = _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0)
    assert len(calls) == 1
    assert NOTICE in capsys.readouterr().err


def test_no_temp_file_is_left_behind_when_the_cache_cannot_be_written(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _counting_fetch(monkeypatch, LATEST)
    state.mkdir(parents=True)

    def _boom(source: Any, target: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(_update.os, "replace", _boom)
    _notify(state, now=1000.0)
    assert list(state.iterdir()) == []


def test_a_failed_fetch_is_remembered_so_offline_is_not_taxed_every_run(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No network: one bounded attempt per day, not one per invocation."""
    calls: list[int] = []

    def _offline() -> str | None:
        calls.append(1)
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(_update, "fetch_latest", _offline)
    _notify(state, now=1000.0)
    _notify(state, now=1001.0)
    assert len(calls) == 1
    assert json.loads(_cache_file(state).read_text(encoding="utf-8"))["latest"] is None


def test_an_offline_run_prints_nothing_and_raises_nothing(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _offline() -> str | None:
        raise httpx.ConnectTimeout("too slow")

    monkeypatch.setattr(_update, "fetch_latest", _offline)
    _notify(state, now=1000.0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_the_cache_lives_under_the_cli_state_dir(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0)
    assert _cache_file(state).is_file()
    assert _cache_file(state).name == "update-check.json"


# --------------------------------------------------------------------------- #
# what gets said, and when                                                    #
# --------------------------------------------------------------------------- #


def test_the_notice_names_the_versions_and_the_upgrade_command(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0)
    err = capsys.readouterr().err
    assert LATEST in err
    assert INSTALLED in err
    # A version number alone is not actionable — the command to type is.
    assert _update.upgrade_command() in err
    # ...and so is the way to make it stop (requirement 3, privacy posture).
    assert f"{_update.ENV_VAR}=0" in err


def test_nothing_is_said_when_the_installed_version_is_current(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _counting_fetch(monkeypatch, INSTALLED)
    _notify(state, now=1000.0)
    assert capsys.readouterr().err == ""


def test_nothing_is_said_when_pypi_is_behind_the_installed_version(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _counting_fetch(monkeypatch, "0.1.0")
    _notify(state, now=1000.0)
    assert capsys.readouterr().err == ""


def test_nothing_is_said_when_nerlo_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_update, "installed_version", lambda: None)
    calls = _counting_fetch(monkeypatch, LATEST)
    _notify(tmp_path / "state", now=1000.0)
    assert calls == []
    assert capsys.readouterr().err == ""


def test_json_suppresses_the_notice_and_the_request(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _counting_fetch(monkeypatch, LATEST)
    _notify(state, now=1000.0, as_json=True)
    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("/home/u/.local/pipx/venvs/nerlo", "pipx upgrade nerlo"),
        ("/home/u/.local/share/uv/tools/nerlo", "uv tool upgrade nerlo"),
        ("/usr", "pip install --upgrade nerlo"),
        ("/home/u/project/.venv", "pip install --upgrade nerlo"),
        (r"C:\Users\u\AppData\Local\pipx\venvs\nerlo", "pipx upgrade nerlo"),
    ],
)
def test_upgrade_command_follows_the_install_layout(
    monkeypatch: pytest.MonkeyPatch, prefix: str, expected: str
) -> None:
    monkeypatch.setattr(_update.sys, "prefix", prefix)
    assert _update.upgrade_command() == expected


# --------------------------------------------------------------------------- #
# the HTTP call itself — never over the real network                          #
# --------------------------------------------------------------------------- #


def _stub_pypi(monkeypatch: pytest.MonkeyPatch, status: int, body: Any) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body, request=request)

    monkeypatch.setattr(
        _update, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(_handler))
    )
    return seen


def test_fetch_reads_pypis_latest_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_pypi(monkeypatch, 200, {"info": {"version": "1.2.3"}})
    assert _update.fetch_latest() == "1.2.3"
    assert str(seen[0].url) == "https://pypi.org/pypi/nerlo/json"


def test_fetch_is_a_bodyless_get(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_pypi(monkeypatch, 200, {"info": {"version": "1.2.3"}})
    _update.fetch_latest()
    assert seen[0].method == "GET"
    assert seen[0].content == b""


def test_the_pypi_client_sends_no_identifier_and_cannot_hang() -> None:
    """Privacy (hard rule 5) + requirement 1, asserted on the REAL client.

    Not through the mock: the mock replaces this factory, so a header or a
    timeout removed here would be invisible to every other test in this file.
    """
    with _REAL_HTTP_CLIENT() as client:
        assert client.headers["user-agent"] == "nerlo-cli"
        # No version suffix — the user agent is a constant, not a fingerprint.
        assert not any(char.isdigit() for char in client.headers["user-agent"])
        assert "cookie" not in client.headers
        assert "authorization" not in client.headers
        assert client.timeout.connect is not None and client.timeout.connect <= 2.0
        assert client.timeout.read is not None and client.timeout.read <= 2.0
        assert client.follow_redirects is False


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (404, {}),
        (500, {}),
        (200, {}),
        (200, {"info": None}),
        (200, {"info": {}}),
        (200, {"info": {"version": None}}),
        (200, {"info": {"version": ""}}),
        (200, {"info": {"version": "1.2.3", "yanked": True}}),
        (200, []),
    ],
)
def test_fetch_answers_none_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch, status: int, body: Any
) -> None:
    _stub_pypi(monkeypatch, status, body)
    assert _update.fetch_latest() is None


# --------------------------------------------------------------------------- #
# THE COMMAND SURFACE — every command, both output modes                      #
# --------------------------------------------------------------------------- #
#
# Enumerated from `commands.ALL_COMMANDS` rather than from a list written here,
# so a command added later is covered without anyone remembering to add it.


def _registry_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    skill = {
        "skill_id": "demo-skill",
        "name": "demo",
        "artifact_type": "mcp_server",
        "current_badge": "Verified",
        "current_security_score": 91.0,
        "repository_url": "https://www.npmjs.com/package/demo",
        "mcp_server_id": SERVER_ID,
    }
    if path.startswith("/api/v1/skills/"):
        return httpx.Response(200, json=skill, request=request)
    if path == "/api/v1/servers" and request.method == "POST":
        return httpx.Response(
            201, json={"mcp_server_id": SERVER_ID, "scan_job_id": "job-1"}, request=request
        )
    if path == "/api/v1/servers":
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": SERVER_ID,
                        "name": "demo",
                        "artifact_type": "mcp_server",
                        "current_badge": "Verified",
                        "current_security_score": 91.0,
                    }
                ],
                "total": 1,
            },
            request=request,
        )
    if path == f"/api/v1/servers/{SERVER_ID}/installation-stats":
        return httpx.Response(200, json={"total": 3, "last_30d": 1}, request=request)
    if path == f"/api/v1/servers/{SERVER_ID}/rescan":
        return httpx.Response(
            202, json={"scan_job_id": "job-2", "dispatch": "queued"}, request=request
        )
    if path == f"/api/v1/servers/{SERVER_ID}":
        return httpx.Response(
            200,
            json={
                "id": SERVER_ID,
                "name": "demo",
                "composite_badge": "Verified",
                "composite_score": 91.0,
                "scanner_reports": [{"scanner_name": "semgrep", "score": 91.0, "findings": []}],
            },
            request=request,
        )
    return httpx.Response(404, json={}, request=request)


def _args_for(name: str, workdir: Path) -> list[str]:
    """Arguments that drive each command to a normal, successful finish."""
    project = workdir / "project"
    project.mkdir(parents=True, exist_ok=True)
    return {
        "search": ["alpha"],
        "info": ["demo"],
        "install": ["demo", "--target", "mcp", "--token", "t"],
        "submit": ["https://github.com/o/r", "--token", "t"],
        "rescan": [SERVER_ID, "--token", "t"],
        "check": [str(project)],
        "version": [],
    }[name]


_ALL_BY_NAME = {command.name: command for command in commands.ALL_COMMANDS}


def _run(
    workdir: Path, name: str, *, notice: bool, as_json: bool, calls: list[int] | None = None
) -> Result:
    """Invoke one command in a sandboxed home/config, resetting it first.

    Two runs of the same command MUST use the same `workdir`, because `install`
    and `check` print the paths they touched — a different directory per run
    would make stdout differ for a reason that has nothing to do with the
    notice, which is the only thing this comparison is allowed to measure.
    Hence the reset: `install` WRITES, and a second run against a config it had
    already written would refuse with "already installed".
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(workdir / ".nerlo", ignore_errors=True)
    (workdir / "mcp.json").unlink(missing_ok=True)
    command = _ALL_BY_NAME.get(name) or main.version
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NERLO_HOME", str(workdir / ".nerlo"))
        mp.setitem(commands.TARGET_CONFIG_PATHS, "mcp", workdir / "mcp.json")
        mp.setattr(
            commands,
            "_client",
            lambda api_url, token=None: httpx.Client(
                base_url=api_url, transport=httpx.MockTransport(_registry_handler)
            ),
        )
        mp.setattr(
            commands,
            "_telemetry_client",
            lambda api_url: httpx.Client(
                base_url=api_url,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(202, json={}, request=request)
                ),
            ),
        )
        mp.setattr(_update, "installed_version", lambda: INSTALLED)

        def _fetch() -> str | None:
            if calls is not None:
                calls.append(1)
            return LATEST

        if notice:
            mp.delenv(_update.ENV_VAR, raising=False)
            mp.delenv("CI", raising=False)
            mp.setattr(_update, "fetch_latest", _fetch)
        else:
            mp.setenv(_update.ENV_VAR, "0")
        args = [*_args_for(name, workdir), *(["--json"] if as_json else [])]
        return CliRunner().invoke(command, args)


@pytest.mark.parametrize("name", sorted(_ALL_BY_NAME))
def test_the_notice_reaches_stderr_and_never_stdout(tmp_path: Path, name: str) -> None:
    """The `--json` contract, per command, with its own positive control.

    Four runs: plain and `--json`, each with the notice available and with it
    switched off. stdout is compared byte for byte, which is the property a
    parsing pipeline actually depends on.
    """
    calls: list[int] = []
    plain = tmp_path / f"{name}-plain"
    machine = tmp_path / f"{name}-json"
    plain_quiet = _run(plain, name, notice=False, as_json=False)
    plain_loud = _run(plain, name, notice=True, as_json=False, calls=calls)
    json_quiet = _run(machine, name, notice=False, as_json=True)
    json_loud = _run(machine, name, notice=True, as_json=True, calls=calls)

    # POSITIVE CONTROL. Without this line every assertion below passes when the
    # notice never fires at all — which is exactly how a suppression test rots.
    assert NOTICE in plain_loud.stderr, plain_loud.output
    assert LATEST in plain_loud.stderr
    assert len(calls) == 1, "the --json run asked PyPI; it must not"

    # The notice never touches stdout, in either mode...
    assert NOTICE not in plain_loud.stdout
    assert NOTICE not in json_loud.stdout
    # ...and stdout is byte-identical to the run with the feature switched off.
    assert plain_loud.stdout == plain_quiet.stdout
    assert json_loud.stdout == json_quiet.stdout
    # Under --json it is suppressed EVERYWHERE, stderr included.
    assert NOTICE not in json_loud.stderr
    # And it never moves an exit code.
    assert plain_loud.exit_code == plain_quiet.exit_code
    assert json_loud.exit_code == json_quiet.exit_code


@pytest.mark.parametrize("name", sorted(_ALL_BY_NAME))
def test_json_output_still_parses_with_the_notice_available(tmp_path: Path, name: str) -> None:
    """stdout under `--json` is a JSON document and nothing else."""
    result = _run(tmp_path / f"{name}-parse", name, notice=True, as_json=True)
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)


def test_nerlo_version_notifies_too(tmp_path: Path) -> None:
    """`nerlo version` is where "…and a newer one exists" belongs most."""
    loud = _run(tmp_path / "version-loud", "version", notice=True, as_json=False)
    quiet = _run(tmp_path / "version-quiet", "version", notice=False, as_json=False)
    assert NOTICE in loud.stderr
    assert NOTICE not in loud.stdout
    assert loud.stdout == quiet.stdout == f"nerlo {_update.metadata.version('nerlo')}\n"


def test_the_notice_fires_on_a_failing_command_too(tmp_path: Path) -> None:
    """A close hook, not a success hook — `_fail` and `sys.exit` still notify."""
    workdir = tmp_path / "failing"
    workdir.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NERLO_HOME", str(workdir / ".nerlo"))
        mp.delenv(_update.ENV_VAR, raising=False)
        mp.delenv("CI", raising=False)
        mp.setattr(_update, "installed_version", lambda: INSTALLED)
        mp.setattr(_update, "fetch_latest", lambda: LATEST)
        # No token: refused before any network call, exit 1.
        result = CliRunner().invoke(commands.submit, ["https://github.com/o/r"])
    assert result.exit_code == 1
    assert "authentication required" in result.stderr
    assert NOTICE in result.stderr


def test_a_notice_that_explodes_changes_neither_output_nor_exit_code(tmp_path: Path) -> None:
    """Rule 2: an upgrade nicety must never become a failure mode of the tool.

    The same `nerlo check` run twice against the same directory — once with the
    notice sabotaged mid-flight, once with it switched off — must be
    indistinguishable in exit code, stdout AND stderr. Not even a traceback.
    """
    exploded: list[int] = []

    def _boom() -> str | None:
        exploded.append(1)
        raise RuntimeError("pypi ate my homework")

    project = tmp_path / "boom" / "project"
    project.mkdir(parents=True)

    def _invoke(*, sabotage: bool) -> Result:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("NERLO_HOME", str(tmp_path / "boom" / ".nerlo"))
            mp.delenv("CI", raising=False)
            if sabotage:
                mp.delenv(_update.ENV_VAR, raising=False)
                mp.setattr(_update, "installed_version", _boom)
            else:
                mp.setenv(_update.ENV_VAR, "0")
            return CliRunner().invoke(commands.check, [str(project)])

    broken = _invoke(sabotage=True)
    healthy = _invoke(sabotage=False)
    # The sabotage really ran — otherwise everything below passes vacuously.
    assert exploded == [1]
    assert broken.exit_code == healthy.exit_code == 0
    assert broken.stdout == healthy.stdout
    assert broken.stderr == healthy.stderr


# --------------------------------------------------------------------------- #
# structural: the control is on the command table, not on six memories        #
# --------------------------------------------------------------------------- #


def test_every_command_carries_the_notice_class_and_the_json_flag() -> None:
    """A seventh command cannot quietly opt out of either half of the contract.

    This is the enumeration that makes the `--json` suppression a property of
    the CLI rather than of the six commands somebody happened to think about.
    """
    for command in [*commands.ALL_COMMANDS, main.version]:
        assert isinstance(command, commands.UpdateNoticeCommand), command.name
    for command in commands.ALL_COMMANDS:
        assert "as_json" in {param.name for param in command.params}, command.name


def test_the_group_exposes_every_command_that_notifies() -> None:
    """Positive control: the table under test is the one the CLI actually runs."""
    registered = set(main.cli.commands)
    assert {command.name for command in commands.ALL_COMMANDS} <= registered
    assert "version" in registered

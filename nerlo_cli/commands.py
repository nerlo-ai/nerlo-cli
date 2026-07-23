"""Nerlo registry CLI commands — search, info, install, submit, rescan.

Every command talks HTTP to the public Nerlo registry API (`NERLO_API_BASE_URL`,
default https://api.nerlo.ai); write operations (`submit`, `rescan`)
authenticate with a Bearer token (`--token` / `NERLO_API_TOKEN`) and exit
non-zero without acting when the credential is missing or rejected.

Every command supports `--json` for machine output; the default is a
human-readable table.

`install` writes an `mcpServers` entry into the target platform config with
badge gating: Verified proceeds, Caution prompts for confirmation, Unsafe
refuses. For npm-hosted packages the entry is runnable (`npx -y <package>`);
for other sources the entry records the repository and the user finishes the
command wiring — Nerlo verifies code, it does not (yet) ship a package runtime.
"""

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import uuid as uuid_mod
from importlib import metadata
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import click
import httpx

from nerlo_cli._logging import get_logger

logger = get_logger(__name__)

DEFAULT_API_BASE_URL = "https://api.nerlo.ai"
SEARCH_LIMIT = 50  # Req 11.3
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
# Telemetry (Ticket 30.5) is best-effort and must never delay an install — keep
# the timeout short and swallow every failure.
TELEMETRY_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# Artifact types the backend recognises (Ticket 33.9). `nerlo install` only has
# a defined local install action for `mcp_server` (writing an mcpServers config
# entry); the others land in platform-specific locations this thin client does
# not manage yet, so it refuses rather than guessing a path.
SUBMIT_ARTIFACT_TYPES = ("mcp_server", "claude_skill", "gemini_extension", "cursor_rule")
# TODO(nerlo): teach `install` to place claude_skill (skills dir),
# gemini_extension (extensions), and cursor_rule (rules dir) artifacts once
# those install paths are specified; until then only mcp_server is installable.
MCP_INSTALLABLE_ARTIFACT_TYPES = frozenset({"mcp_server"})

# Req 11.1 target platforms -> local MCP config file (entries land under
# the file's "mcpServers" object in every case).
TARGET_CONFIG_PATHS: dict[str, Path] = {
    "claude-code": Path.home() / ".claude.json",
    "cursor": Path.home() / ".cursor" / "mcp.json",
    "gemini": Path.home() / ".gemini" / "settings.json",
    "mcp": Path.cwd() / "mcp.json",
}

_api_url_option = click.option(
    "--api-url",
    envvar="NERLO_API_BASE_URL",
    default=DEFAULT_API_BASE_URL,
    show_default=True,
    help="Nerlo registry API base URL.",
)
_token_option = click.option(
    "--token",
    envvar="NERLO_API_TOKEN",
    default=None,
    help="API bearer token (or set NERLO_API_TOKEN).",
)
_json_option = click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")


def _client(api_url: str, token: str | None = None) -> httpx.Client:
    headers = {"User-Agent": "nerlo-cli"}
    if token:
        parsed = urlparse(api_url)
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
            click.secho(
                "warning: sending API token over plain HTTP to a non-local host — use https.",
                fg="yellow",
                err=True,
            )
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=api_url, headers=headers, timeout=HTTP_TIMEOUT)


def _fail(message: str, code: int = 1) -> None:
    """Req 11.11/11.12: error to stderr, non-zero exit, no action taken."""
    click.secho(f"error: {message}", fg="red", err=True)
    sys.exit(code)


def _require_token(token: str | None) -> str:
    if not token:
        _fail("authentication required — pass --token or set NERLO_API_TOKEN (Req 11.10)")
    assert token is not None  # _fail exits; this narrows for the type checker
    return token


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        _fail(f"cannot reach registry API: {type(exc).__name__}")
        raise AssertionError from exc  # unreachable; _fail exits
    if response.status_code in (401, 403):
        _fail(f"authentication failed (HTTP {response.status_code}) — no action taken")
    return response


def _echo_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.upper().ljust(widths[c]) for c in columns)
    click.secho(header, bold=True)
    for row in rows:
        click.echo("  ".join(str(row.get(c, "") or "").ljust(widths[c]) for c in columns))


# --------------------------------------------------------------------- #
# nerlo search (Req 11.3, 11.4)                                            #
# --------------------------------------------------------------------- #


@click.command()
@click.argument("query")
@_api_url_option
@_json_option
def search(query: str, api_url: str, as_json: bool) -> None:
    """Search the registry by name/description/author keyword."""
    if not 2 <= len(query) <= 100:
        _fail("query must be 2-100 characters")
    with _client(api_url) as client:
        response = _request(
            client, "GET", "/api/v1/servers", params={"q": query, "page_size": SEARCH_LIMIT}
        )
    if response.status_code != 200:
        logger.debug("cli.search_error_body", body=response.text[:1000])
        _fail(f"search failed (HTTP {response.status_code})")
    payload = response.json()
    results = payload.get("results", [])[:SEARCH_LIMIT]
    if as_json:
        _echo_json(results)
        return
    if not results:
        click.echo(f"No results found for '{query}'.")  # Req 11.4: exit 0
        return
    _table(
        [
            {
                "name": r.get("name"),
                "score": r.get("current_security_score"),
                "badge": r.get("current_badge"),
                "author": r.get("author"),
                "id": r.get("id"),
            }
            for r in results
        ],
        ["name", "score", "badge", "author", "id"],
    )


# --------------------------------------------------------------------- #
# nerlo info (Req 11.9)                                                    #
# --------------------------------------------------------------------- #


@click.command()
@click.argument("skill_name")
@_api_url_option
@_json_option
def info(skill_name: str, api_url: str, as_json: bool) -> None:
    """Show score, badge, and per-scanner scoresheets for a skill."""
    with _client(api_url) as client:
        skill = _resolve_skill(client, skill_name)
        server_id = _resolve_server_id(client, skill_name, skill)
        detail: dict[str, Any] | None = None
        install_stats: dict[str, Any] | None = None
        if server_id is not None:
            response = _request(client, "GET", f"/api/v1/servers/{server_id}")
            if response.status_code == 200:
                detail = response.json()
            # Req 29.10: display-only install engagement signal (CLI installs).
            stats_resp = _request(client, "GET", f"/api/v1/servers/{server_id}/installation-stats")
            if stats_resp.status_code == 200:
                install_stats = stats_resp.json()
    if as_json:
        _echo_json({"skill": skill, "detail": detail, "install_stats": install_stats})
        return
    click.secho(f"{skill.get('name')} ({skill.get('skill_id')})", bold=True)
    click.echo(f"  repository: {skill.get('repository_url', '-')}")
    click.echo(f"  badge:      {skill.get('current_badge', '-')}")
    click.echo(f"  score:      {skill.get('current_security_score', '-')}")
    if install_stats is not None:
        total = install_stats.get("total", 0)
        last_30d = install_stats.get("last_30d", 0)
        # Req 29.5: labelled "Installed via Nerlo", counts CLI installs only —
        # deliberately NOT "popular"/"trusted"; a raw engagement signal.
        click.echo(
            f"  installed via Nerlo: {total} total ({last_30d} in last 30d, CLI installs only)"
        )
    # Req 11.9 / aggregator stance: per-scanner scoresheets are the
    # primary view; the composite above is the summary.
    scanner_reports = cast(list[dict[str, Any]], (detail or {}).get("scanner_reports") or [])
    if scanner_reports:
        click.echo("")
        click.secho("  per-scanner scoresheets:", bold=True)
        _table(
            [
                {
                    "scanner": s.get("scanner_name") or s.get("tool_name"),
                    "score": s.get("score"),
                    "badge": s.get("badge"),
                    "findings": len(s.get("findings", [])),
                }
                for s in scanner_reports
            ],
            ["scanner", "score", "badge", "findings"],
        )


def _resolve_skill(client: httpx.Client, skill_name: str) -> dict[str, Any]:
    response = _request(client, "GET", f"/api/v1/skills/{skill_name}")
    if response.status_code == 200:
        return response.json()
    if response.status_code in (404, 422):
        _fail(f"skill not found: {skill_name!r} (Req 11.12)")
    _fail(f"lookup failed (HTTP {response.status_code})")
    raise AssertionError  # unreachable


def _resolve_server_id(client: httpx.Client, skill_name: str, skill: dict[str, Any]) -> str | None:
    """Skill detail doesn't expose the server UUID; match it via search."""
    if "mcp_server_id" in skill:
        return str(skill["mcp_server_id"])
    name = str(skill.get("name", skill_name))[:100]
    if len(name) < 2:
        return None
    response = _request(client, "GET", "/api/v1/servers", params={"q": name, "page_size": 50})
    if response.status_code != 200:
        return None
    for item in response.json().get("results", []):
        if item.get("name") == skill.get("name"):
            return str(item.get("id"))
    return None


# --------------------------------------------------------------------- #
# install telemetry (Ticket 30.5)                                          #
# --------------------------------------------------------------------- #


def _nerlo_home() -> Path:
    """Per-user Nerlo state dir. `NERLO_HOME` overrides it (used by tests)."""
    override = os.environ.get("NERLO_HOME")
    return Path(override) if override else Path.home() / ".nerlo"


def _read_config() -> dict[str, str]:
    """Parse ~/.nerlo/config — simple `key=value` lines, `#` comments ignored.

    This is the CLI's only persisted settings store; there is no other, so
    telemetry opt-out (`telemetry=false`) lives here.
    """
    path = _nerlo_home() / "config"
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _telemetry_enabled() -> bool:
    """Honour both `NERLO_TELEMETRY=0` (env) and `telemetry=false` (config)."""
    if os.environ.get("NERLO_TELEMETRY") == "0":
        return False
    value = _read_config().get("telemetry")
    if value is not None and value.strip().lower() in ("false", "0", "no", "off"):
        return False
    return True


def _anonymous_installer_id() -> str:
    """Stable anonymous installer id from ~/.nerlo/installer-id (uuid4, 0600).

    Created on first use and reused thereafter, so the derived hash is stable
    across runs for the same machine/user.
    """
    path = _nerlo_home() / "installer-id"
    with contextlib.suppress(OSError):
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    installer_id = str(uuid_mod.uuid4())
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create 0600 from the start (don't briefly expose the id world-readable).
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(installer_id + "\n")
        os.chmod(path, 0o600)
    return installer_id


def _installer_token_hash(token: str | None) -> str:
    """SHA-256 hex of the installer identity (64 lowercase hex chars).

    Identity is the authenticated credential when logged in, else the anonymous
    installer id. The hash is one-way and stable across runs for the same
    installer.
    """
    # TODO(nerlo): the CLI has no user-id lookup, so we hash the bearer token as
    # a stand-in for the authenticated user id. Swap to the real user id if the
    # API grows a `/me` endpoint. The hash is one-way, so the token never leaves
    # the machine in a recoverable form.
    identity = token if token else _anonymous_installer_id()
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cli_version() -> str:
    """This CLI's version string, clamped to the contract's 1–50 chars."""
    try:
        version = metadata.version("nerlo")
    except metadata.PackageNotFoundError:
        version = "0.0.0+unknown"
    return (version[:50] or "0.0.0+unknown")


def _telemetry_client(api_url: str) -> httpx.Client:
    """Unauthenticated client for the telemetry POST (no Bearer token sent)."""
    return httpx.Client(
        base_url=api_url,
        headers={"User-Agent": "nerlo-cli"},
        timeout=TELEMETRY_TIMEOUT,
    )


def _maybe_print_telemetry_notice() -> None:
    """One-time notice that telemetry is on and how to opt out."""
    marker = _nerlo_home() / "telemetry-notice-shown"
    if marker.exists():
        return
    click.secho(
        "note: nerlo sends anonymous install telemetry. Opt out with "
        "NERLO_TELEMETRY=0 or `telemetry=false` in ~/.nerlo/config.",
        fg="yellow",
        err=True,
    )
    with contextlib.suppress(OSError):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("1\n", encoding="utf-8")


def _emit_install_event(api_url: str, target: str, token: str | None) -> None:
    """Best-effort install telemetry (Ticket 30.5).

    POSTs to the unauthenticated `/api/v1/installations`. Every failure is
    swallowed and logged at debug — telemetry must never fail or delay install.
    """
    try:
        if not _telemetry_enabled():
            return
        _maybe_print_telemetry_notice()
        body = {
            "installer_token_hash": _installer_token_hash(token),
            "target_platform": target,
            "cli_version": _cli_version(),
        }
        with _telemetry_client(api_url) as client:
            client.post("/api/v1/installations", json=body)
        logger.debug("cli.install", telemetry="sent", target=target)
    except Exception as exc:  # noqa: BLE001 — best-effort, never break install
        logger.debug("cli.install", telemetry="failed", error=type(exc).__name__)


# --------------------------------------------------------------------- #
# nerlo install (Req 11.1, 11.2)                                           #
# --------------------------------------------------------------------- #


@click.command()
@click.argument("skill_name")
@click.option(
    "--target",
    required=True,
    type=click.Choice(sorted(TARGET_CONFIG_PATHS)),
    help="AI platform whose local config receives the entry.",
)
@click.option("--force", is_flag=True, help="Replace an existing mcpServers entry for this skill.")
@_api_url_option
@_token_option
@_json_option
def install(
    skill_name: str,
    target: str,
    force: bool,
    api_url: str,
    token: str | None,
    as_json: bool,
) -> None:
    """Install a verified skill into a platform's MCP configuration.

    Writes an mcpServers config entry (runnable for npm/PyPI-hosted
    packages; a repository reference otherwise — finish the command
    wiring manually for those). Authenticated per Req 11.10.
    """
    auth = _require_token(token)
    with _client(api_url, auth) as client:
        skill = _resolve_skill(client, skill_name)

    # Ticket 33.9: type-aware install. Only `mcp_server` artifacts have a
    # defined local install action here (writing an mcpServers entry). For any
    # other classified type, refuse with a clear message rather than guessing a
    # path (a skills/extension/rules artifact is NOT an mcpServers entry).
    artifact_type = skill.get("artifact_type")
    if artifact_type is not None and artifact_type not in MCP_INSTALLABLE_ARTIFACT_TYPES:
        _fail(
            f"{skill_name!r} is a {artifact_type!r} artifact — `nerlo install` "
            "can only write MCP server config entries so far. Install it "
            "manually per your platform's docs (install support for "
            f"{artifact_type} is planned)."
        )

    badge = skill.get("current_badge")
    # Req 11.2 badge gate.
    if badge == "Unsafe":
        _fail(f"{skill_name!r} carries an Unsafe badge — installation refused")
    if badge == "Caution":
        click.secho(
            f"WARNING: {skill_name!r} carries a Caution badge — its scan "
            "found issues worth reviewing before use.",
            fg="yellow",
        )
        if not click.confirm("Install anyway?"):
            click.echo("Aborted.")
            sys.exit(1)
    elif badge != "Verified":
        _fail(f"{skill_name!r} has no badge yet (status: {badge!r}) — not installable")

    config_path = TARGET_CONFIG_PATHS[target]
    entry = _build_mcp_entry(skill)
    _write_mcp_entry(config_path, str(skill.get("skill_id", skill_name)), entry, force=force)

    logger.info(
        "cli.install",
        skill_id=skill.get("skill_id"),
        target=target,
        badge=badge,
        config_path=str(config_path),
    )
    # Ticket 30.5: best-effort install telemetry — never raises, never delays.
    _emit_install_event(api_url, target, token)
    if as_json:
        _echo_json(
            {
                "installed": skill.get("skill_id"),
                "target": target,
                "config_path": str(config_path),
                "entry": entry,
            }
        )
        return
    click.secho(f"Installed {skill.get('skill_id')} -> {config_path}", fg="green")
    if "command" not in entry:
        click.secho(
            "  note: no runnable package source detected — entry records the "
            "repository; finish the command wiring for your platform.",
            fg="yellow",
        )


def _build_mcp_entry(skill: dict[str, Any]) -> dict[str, Any]:
    repo = str(skill.get("repository_url", ""))
    # Exact host match — suffix matching would let `evilnpmjs.com` produce
    # a runnable `npx` entry (arbitrary code execution at platform start).
    host = urlparse(repo).hostname or ""
    if host in ("www.npmjs.com", "npmjs.com"):
        package = urlparse(repo).path.split("/package/")[-1].strip("/")
        if package:
            return {"command": "npx", "args": ["-y", package]}
    if host == "pypi.org":
        package = urlparse(repo).path.split("/project/")[-1].strip("/")
        if package:
            return {"command": "uvx", "args": [package]}
    return {"repository": repo, "nerlo_badge": skill.get("current_badge")}


def _write_mcp_entry(
    config_path: Path, skill_id: str, entry: dict[str, Any], *, force: bool
) -> None:
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded: object = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read {config_path}: {type(exc).__name__}")
            raise AssertionError from exc  # unreachable
        if not isinstance(loaded, dict):
            _fail(f"{config_path} does not contain a JSON object — refusing to overwrite")
        config = cast(dict[str, Any], loaded)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    servers = cast(dict[str, Any], config.setdefault("mcpServers", {}))
    if skill_id in servers and not force:
        _fail(
            f"an mcpServers entry for {skill_id!r} already exists in "
            f"{config_path} — re-run with --force to replace it"
        )
    servers[skill_id] = entry
    # Atomic replace: this file can be the user's live Claude Code state
    # (~/.claude.json); a torn write must never destroy it.
    fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".nerlo-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(config, indent=2) + "\n")
        os.replace(tmp_path, config_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# --------------------------------------------------------------------- #
# nerlo submit / rescan (Req 11.6, 11.7, 11.10, 11.11)                     #
# --------------------------------------------------------------------- #


@click.command()
@click.argument("url")
@click.option(
    "--type",
    "artifact_type",
    type=click.Choice(SUBMIT_ARTIFACT_TYPES),
    default=None,
    help="Artifact type. Omit to let the server infer it.",
)
@_api_url_option
@_token_option
@_json_option
def submit(
    url: str, artifact_type: str | None, api_url: str, token: str | None, as_json: bool
) -> None:
    """Submit a repository URL for ingestion + scanning (authenticated)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        _fail(f"malformed repository URL: {url!r} (Req 11.12)")
    auth = _require_token(token)
    body: dict[str, Any] = {"repository_url": url}
    # Ticket 33.9: only send artifact_type when the caller set --type; omitting
    # it preserves the existing server-side inference behaviour.
    if artifact_type is not None:
        body["artifact_type"] = artifact_type
    with _client(api_url, auth) as client:
        response = _request(client, "POST", "/api/v1/servers", json=body)
    if response.status_code not in (200, 201, 202):
        logger.debug("cli.submit_error_body", body=response.text[:1000])
        _fail(f"submit failed (HTTP {response.status_code})")
    payload = response.json()
    if as_json:
        _echo_json(payload)
        return
    click.secho("Submitted.", fg="green")
    click.echo(f"  server:   {payload.get('mcp_server_id')}")
    click.echo(f"  scan job: {payload.get('scan_job_id')}")


@click.command()
@click.argument("identifier")
@_api_url_option
@_token_option
@_json_option
def rescan(identifier: str, api_url: str, token: str | None, as_json: bool) -> None:
    """Queue a re-scan for a server by UUID or skill slug (authenticated)."""
    auth = _require_token(token)
    with _client(api_url, auth) as client:
        server_id = identifier
        try:
            uuid_mod.UUID(identifier)
        except ValueError:
            skill = _resolve_skill(client, identifier)
            resolved = _resolve_server_id(client, identifier, skill)
            if resolved is None:
                _fail(f"cannot resolve {identifier!r} to a server id (Req 11.12)")
            assert resolved is not None
            server_id = resolved
        response = _request(client, "POST", f"/api/v1/servers/{server_id}/rescan")
    if response.status_code not in (200, 202):
        logger.debug("cli.rescan_error_body", body=response.text[:1000])
        _fail(f"rescan failed (HTTP {response.status_code})")
    payload = response.json()
    if as_json:
        _echo_json(payload)
        return
    click.secho("Re-scan queued.", fg="green")
    click.echo(f"  scan job: {payload.get('scan_job_id')} ({payload.get('dispatch')})")


# Public consumer commands only. Operator/service commands (jobs, verify, serve,
# discovery-scheduler, monitor) stay in the backend — they need DB/pipeline
# internals and are not part of the installable CLI.
ALL_COMMANDS: list[click.Command] = [search, info, install, submit, rescan]

__all__ = ["ALL_COMMANDS"]

"""Nerlo registry CLI commands — search, info, install, submit, rescan, check.

Every command talks HTTP to the public Nerlo registry API (`NERLO_API_BASE_URL`,
default https://api.nerlo.ai); write operations (`submit`, `rescan`)
authenticate with a Bearer token (`--token` / `NERLO_API_TOKEN`) and exit
non-zero without acting when the credential is missing or rejected.

Every command supports `--json` for machine output; the default is a
human-readable table.

`install` routes by the resolved artifact's `artifact_type` (Ticket 33.9):
`mcp_server` writes an `mcpServers` entry into the target platform config;
`claude_skill` copy-installs the skill directory (the one containing SKILL.md,
materialised via a shallow `git clone` of the repository) into
`~/.claude/skills/<skill-slug>/`; `gemini_extension` is a placeholder (install
path pending Google runtime API); `cursor_rule` is refused. All installs are
badge gated: Verified proceeds, Caution prompts for confirmation, Unsafe
refuses. For npm-hosted packages the mcpServers entry is runnable
(`npx -y <package>`); for other sources the entry records the repository and
the user finishes the command wiring — Nerlo verifies code, it does not (yet)
ship a package runtime.

`check` is the CI gate and runs the same map in reverse: it READS the platform
configs `install` WRITES, resolves each entry against the public
(unauthenticated) registry, and exits non-zero on policy violation. Its exit
code is its product — see the `nerlo check` section for the contract, and for
why "not in the registry" is reported as its own outcome rather than as a pass.
"""

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid as uuid_mod
from dataclasses import dataclass
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

# Artifact types the backend recognises (Ticket 33.9). `nerlo install` routes
# by type: mcp_server writes an mcpServers config entry, claude_skill
# copy-installs into ~/.claude/skills/, gemini_extension is a placeholder
# (install path pending Google runtime API). cursor_rule lands in a
# platform-specific location this thin client does not manage yet, so it
# refuses rather than guessing a path.
SUBMIT_ARTIFACT_TYPES = ("mcp_server", "claude_skill", "gemini_extension", "cursor_rule")
# TODO(nerlo): teach `install` to place cursor_rule (rules dir) artifacts once
# that install path is specified.
MCP_INSTALLABLE_ARTIFACT_TYPES = frozenset({"mcp_server"})
INSTALL_ROUTABLE_ARTIFACT_TYPES = frozenset({"mcp_server", "claude_skill", "gemini_extension"})
# Shallow clone used to materialise claude_skill sources — best-effort, bounded.
GIT_CLONE_TIMEOUT_SECONDS = 120.0
# Directory names under ~/.claude/skills/ come from the API's skill slug; keep
# them strictly path-safe (no separators, no traversal) before touching disk.
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Req 11.1 target platforms -> local MCP config file (entries land under
# the file's "mcpServers" object in every case).
#
# ONE table, two directions. `install` WRITES the path built here; `check`
# READS the same relative parts, re-rooted at a project directory when the user
# passes one. Keeping the relative parts in a single place is what stops the
# reader from parsing a layout the writer no longer produces.
#   platform -> (path parts relative to its root, which root the writer uses)
PLATFORM_CONFIG_LAYOUT: dict[str, tuple[tuple[str, ...], str]] = {
    "claude-code": ((".claude.json",), "home"),
    "cursor": ((".cursor", "mcp.json"), "home"),
    "gemini": ((".gemini", "settings.json"), "home"),
    "mcp": (("mcp.json",), "cwd"),
}
# Discovery-only extras (`check` reads them; `install` never writes them). A
# repo checked out in CI commonly carries Claude Code's project-scoped
# `.mcp.json`; a gate that cannot see it is a gate with a hole, and reading a
# file that does not exist costs nothing. Writers deliberately still emit
# `mcp.json` — this list only widens what the reader looks at.
PROJECT_CONFIG_EXTRAS: tuple[tuple[str, tuple[str, ...]], ...] = (("mcp", (".mcp.json",)),)
# Claude Code's per-user skills directory, relative to its root. Shared by the
# claude_skill installer (`_claude_skills_dir`) and by `check` discovery.
CLAUDE_SKILLS_PARTS: tuple[str, ...] = (".claude", "skills")


def _platform_root(kind: str) -> Path:
    return Path.home() if kind == "home" else Path.cwd()


TARGET_CONFIG_PATHS: dict[str, Path] = {
    name: _platform_root(root).joinpath(*parts)
    for name, (parts, root) in PLATFORM_CONFIG_LAYOUT.items()
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


class RegistryUnreachable(Exception):
    """A registry call did not produce an answer.

    `_request(..., fatal=False)` raises this instead of exiting, so a caller
    that must distinguish "the registry says this is fine" from "the registry
    never answered" can. `nerlo check` is that caller: collapsing an
    unanswered request into a pass is precisely the failure the gate exists to
    prevent.
    """


def _request(
    client: httpx.Client, method: str, path: str, *, fatal: bool = True, **kwargs: Any
) -> httpx.Response:
    """Issue a registry request.

    `fatal=True` (the default, and every pre-existing caller) keeps the
    original behaviour: a transport error or a 401/403 prints an error and
    exits. `fatal=False` raises `RegistryUnreachable` instead.
    """
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        if not fatal:
            raise RegistryUnreachable(type(exc).__name__) from exc
        _fail(f"cannot reach registry API: {type(exc).__name__}")
        raise AssertionError from exc  # unreachable; _fail exits
    if response.status_code in (401, 403):
        if not fatal:
            raise RegistryUnreachable(f"HTTP {response.status_code}")
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
                # Ticket 33.9: surface the artifact type alongside every result.
                "type": r.get("artifact_type"),
                "score": r.get("current_security_score"),
                "badge": r.get("current_badge"),
                "author": r.get("author"),
                "id": r.get("id"),
            }
            for r in results
        ],
        ["name", "type", "score", "badge", "author", "id"],
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
    # Ticket 33.9: artifact type is part of the human summary (and the raw
    # `--json` skill object already carries it through unmodified).
    click.echo(f"  type:       {skill.get('artifact_type') or '-'}")
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
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing install (mcpServers entry or skills directory) for this skill.",
)
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
    """Install a verified skill, routed by its artifact type.

    mcp_server artifacts get an mcpServers config entry (runnable for
    npm/PyPI-hosted packages; a repository reference otherwise — finish
    the command wiring manually for those). claude_skill artifacts are
    copy-installed into ~/.claude/skills/<skill-slug>/. Authenticated
    per Req 11.10.
    """
    auth = _require_token(token)
    with _client(api_url, auth) as client:
        skill = _resolve_skill(client, skill_name)

    # Ticket 33.9: type-aware install, routed on the resolved artifact_type.
    # Legacy rows without a type are mcp_server (matches the backend default).
    artifact_type = str(skill.get("artifact_type") or "mcp_server")
    if artifact_type not in INSTALL_ROUTABLE_ARTIFACT_TYPES:
        # cursor_rule (and any future unknown type) lands in a location this
        # thin client does not manage — refuse rather than guessing a path.
        _fail(
            f"{skill_name!r} is a {artifact_type!r} artifact — `nerlo install` "
            "can only write MCP server config entries so far. Install it "
            "manually per your platform's docs (install support for "
            f"{artifact_type} is planned)."
        )

    if artifact_type == "gemini_extension":
        # Ticket 33.9: placeholder route — Google has not published a local
        # runtime install location yet, so this exits 0 without writing
        # anything. @NERLO-REVIEW: runs before the badge gate on purpose — the
        # ticket mandates exit 0 and nothing is written, so the gate has
        # nothing to protect; re-place the gate when the real install lands.
        logger.info(
            "cli.install",
            skill_id=skill.get("skill_id"),
            artifact_type=artifact_type,
            status="placeholder",
            reason="install path pending Google runtime API",
        )
        if as_json:
            _echo_json(
                {
                    "installed": None,
                    "artifact_type": artifact_type,
                    "status": "install path pending Google runtime API",
                }
            )
            return
        click.secho(
            f"{skill_name!r} is a gemini_extension — install path pending "
            "Google runtime API; nothing was written.",
            fg="yellow",
        )
        return

    if artifact_type == "claude_skill" and target != "claude-code":
        _fail(
            f"{skill_name!r} is a claude_skill artifact — it installs into "
            "Claude Code's skills directory. Re-run with --target claude-code."
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

    if artifact_type == "claude_skill":
        # Ticket 33.9: copy-install the skill directory into
        # ~/.claude/skills/<skill-slug>/ (materialised via shallow git clone —
        # the API carries only the repository URL, not the file tree).
        skill_slug = str(skill.get("skill_id") or skill_name)
        dest = _install_claude_skill(skill, skill_slug, force=force)
        logger.info(
            "cli.install",
            skill_id=skill.get("skill_id"),
            artifact_type=artifact_type,
            target="claude-code",
            badge=badge,
            path=str(dest),
        )
        # Ticket 30.5 telemetry — claude_skill installs report target_platform
        # "claude-code" (the only runtime that consumes ~/.claude/skills).
        _emit_install_event(api_url, "claude-code", token)
        if as_json:
            _echo_json(
                {
                    "installed": skill.get("skill_id"),
                    "artifact_type": artifact_type,
                    "target": "claude-code",
                    "path": str(dest),
                }
            )
            return
        click.secho(f"Installed {skill.get('skill_id')} -> {dest}", fg="green")
        return

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
# claude_skill copy-install (Ticket 33.9)                                  #
# --------------------------------------------------------------------- #


def _claude_skills_dir() -> Path:
    """Claude Code's per-user skills directory.

    Plain `Path.home()` — the CLI has no env override pattern for `~/.claude`
    (`NERLO_HOME` governs only `~/.nerlo`), matching TARGET_CONFIG_PATHS.
    """
    return Path.home().joinpath(*CLAUDE_SKILLS_PARTS)


def _git_shallow_clone(repo_url: str, dest: Path) -> None:
    """Best-effort `git clone --depth 1` of `repo_url` into `dest`.

    `git` is invoked via the stdlib subprocess (not a package dependency);
    every failure mode surfaces as a clear CLI error, never a traceback.
    """
    parsed = urlparse(repo_url)
    # http(s) only — refuses git's other transports (ssh, file, ext::) so a
    # registry-supplied URL can never smuggle a local command or path.
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        _fail(f"cannot clone {repo_url!r} — only http(s) repository URLs are supported")
    try:
        # Fixed argv (no shell); `--` stops a URL from being parsed as an option.
        # `-c core.symlinks=false`: an untrusted skill repo must never check out
        # a committed symlink as a real link — otherwise a `x -> ~/.ssh/id_rsa`
        # (or `-> /`) would be dereferenced when the skill dir is copied into
        # ~/.claude/skills/, copying an outside file's content into the install
        # (and rglob could traverse a dir symlink out of the clone). With this,
        # git writes each symlink as a plain placeholder file — inert.
        completed = subprocess.run(
            ["git", "-c", "core.symlinks=false", "clone", "--depth", "1", "--", repo_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=GIT_CLONE_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # never hang on a prompt
            check=False,  # non-zero handled below with a clear CLI error
        )
    except FileNotFoundError:
        _fail(
            "installing a claude_skill needs `git` to fetch the skill source, "
            "but `git` was not found on PATH — install git and retry"
        )
        raise AssertionError from None  # unreachable; _fail exits
    except subprocess.TimeoutExpired:
        _fail(f"git clone of {repo_url} timed out after {GIT_CLONE_TIMEOUT_SECONDS:.0f}s")
        raise AssertionError from None  # unreachable; _fail exits
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        _fail(
            f"git clone of {repo_url} failed (exit {completed.returncode})"
            + (f": {detail[-1][:200]}" if detail else "")
        )


def _find_skill_dir(root: Path, slug: str, name: str) -> Path:
    """Locate the skill directory (the one containing SKILL.md) under `root`.

    Refuses (clear error, no guessing) when no SKILL.md exists, or when
    several exist and none of their directories matches the skill's slug/name.
    """
    if (root / "SKILL.md").is_file():
        return root
    candidates = sorted(
        {p.parent for p in root.rglob("SKILL.md") if p.is_file() and ".git" not in p.parts},
        key=lambda d: (len(d.parts), str(d)),
    )
    if not candidates:
        _fail(
            "no SKILL.md found in the repository — cannot identify a skill "
            "directory to install (refusing to guess)"
        )
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate.name in (slug, name):
            return candidate
    _fail(
        f"multiple SKILL.md files found and none of their directories is named "
        f"{slug!r} — cannot identify which skill to install (refusing to guess)"
    )
    raise AssertionError  # unreachable; _fail exits


def _install_claude_skill(skill: dict[str, Any], slug: str, *, force: bool) -> Path:
    """Copy-install a claude_skill into `~/.claude/skills/<slug>/`.

    The registry API carries only the repository URL (no file tree), so the
    source is materialised via a shallow clone into a temp dir, then the
    directory containing SKILL.md is copied into place.
    """
    if not _SAFE_SLUG.match(slug):
        _fail(f"skill slug {slug!r} is not a safe directory name — refusing to install")
    repo_url = str(skill.get("repository_url") or "")
    if not repo_url:
        _fail("skill record carries no repository_url — nothing to install from")
    skills_root = _claude_skills_dir()
    dest = skills_root / slug
    if dest.exists() and not force:
        _fail(f"{dest} already exists — re-run with --force to replace it")
    with tempfile.TemporaryDirectory(prefix="nerlo-skill-") as tmp:
        clone_dir = Path(tmp) / "repo"
        _git_shallow_clone(repo_url, clone_dir)
        skill_dir = _find_skill_dir(clone_dir, slug, str(skill.get("name") or ""))
        skills_root.mkdir(parents=True, exist_ok=True)
        # Stage next to the destination, then swap — never leave a half-copied
        # skill dir where Claude Code would load it.
        staging = Path(
            tempfile.mkdtemp(prefix=f".{slug}.", suffix=".nerlo-tmp", dir=skills_root)
        )
        try:
            staged = staging / slug
            # symlinks=True PRESERVES any symlink as a link instead of
            # dereferencing it (defense in depth behind core.symlinks=false on
            # the clone): the install never READS an out-of-tree file's content.
            shutil.copytree(
                skill_dir, staged, symlinks=True, ignore=shutil.ignore_patterns(".git")
            )
            if dest.exists():  # only reachable with --force
                shutil.rmtree(dest)
            os.replace(staged, dest)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return dest


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


# --------------------------------------------------------------------- #
# nerlo check — the CI gate                                                #
# --------------------------------------------------------------------- #
#
# `check` discovers the AI artifacts configured on this machine (or in a
# checked-out project) by READING the same config files `install` writes, then
# resolves each one against the public registry and EXITS NON-ZERO when policy
# is violated. The exit code is the product: a web dashboard is visited when
# somebody remembers, a non-zero exit blocks the merge whether anybody
# remembered or not.

# The outcomes `check` reports. These are deliberately NOT collapsible into a
# pass/fail boolean, because three of them are different facts:
#   verified  — in the registry, aggregate verdict Verified
#   caution   — in the registry, aggregate verdict Caution
#   unsafe    — in the registry, aggregate verdict Unsafe
#   withheld  — in the registry, and the registry is REFUSING to publish an
#               aggregate verdict (`aggregate_verdict_withheld`). Live data
#               shows this is common; it means coverage was insufficient.
#   unscored  — in the registry, no aggregate verdict yet (never scanned, or a
#               badge string this CLI does not recognise)
#   unknown   — NOT IN THE REGISTRY AT ALL
#   error     — could not be resolved (registry unreachable / bad response)
#
# `unknown` IS NOT `verified`. Rendering "nobody has ever looked at this" as a
# green check is the exact tri-state collapse this product exists to stop — a
# scanner that did not run must never score as a pass. Same for `withheld` and
# `error`: an absent answer is not a good answer. Every non-verified status
# gets its own visibly distinct row, and `unknown` additionally gets a submit
# funnel pointing at `nerlo submit`.
STATUS_VERIFIED = "verified"
STATUS_CAUTION = "caution"
STATUS_UNSAFE = "unsafe"
STATUS_WITHHELD = "withheld"
STATUS_UNSCORED = "unscored"
STATUS_UNKNOWN = "unknown"
STATUS_ERROR = "error"

# Exit codes. THIS IS THE CONTRACT — CI reads it, humans read the table.
#   0  every discovered artifact satisfied the policy (including "nothing
#      installed", which is a legitimate pass)
#   1  policy violated — at least one artifact is at or worse than --fail-on
#   2  usage error (Click's own; listed here so nothing else claims it)
#   3  incomplete — the check could not determine an answer for at least one
#      artifact (registry unreachable, bad response, unparseable local config)
#      and found no outright violation. A check that could not reach the
#      registry has NOT passed.
# Violation outranks incomplete: if we already know something is Unsafe, exit 1
# is the more actionable signal, and the incomplete rows are still printed.
EXIT_OK = 0
EXIT_POLICY_VIOLATION = 1
EXIT_INCOMPLETE = 3

# --fail-on level -> the statuses that trip it.
#
# `unsafe` and `caution` are VERDICT thresholds: they fire on a published
# verdict at or worse than the named level, and deliberately do NOT fire on
# unknown/withheld/unscored. Reason, stated plainly: the registry holds a few
# hundred artifacts and the ecosystem holds many thousands, so a default that
# failed on unknown would red-build essentially every repo on day one, and a
# gate that red-builds everything on day one gets deleted in week one. A
# deleted gate protects nothing. Unknowns are instead made loud in the output
# and funnelled to `nerlo submit`.
#
# `any` DOES include unknown (and withheld, and unscored). That is the whole
# point of the strict level: it means "fail unless the registry affirmatively
# verified this", which is the correct policy once a team has submitted its
# dependency set and wants to keep it that way. Choosing `any` is choosing to
# treat absence of evidence as failure — available, not default.
FAIL_ON_STATUSES: dict[str, frozenset[str]] = {
    "unsafe": frozenset({STATUS_UNSAFE}),
    "caution": frozenset({STATUS_UNSAFE, STATUS_CAUTION}),
    "any": frozenset(
        {STATUS_UNSAFE, STATUS_CAUTION, STATUS_WITHHELD, STATUS_UNSCORED, STATUS_UNKNOWN}
    ),
}
# STATUS_ERROR is in none of the sets above on purpose: "we could not ask" is
# reported as EXIT_INCOMPLETE at every level, never as a policy verdict and
# never as a pass.

# Command wrappers whose first non-flag argument names a package, so a
# hand-written mcpServers entry still yields something searchable. The Nerlo
# writer only ever emits `npx` / `uvx` (see `_build_mcp_entry`); the rest are
# here because `check` reads configs this CLI did not write.
_PACKAGE_RUNNERS: dict[str, frozenset[str]] = {
    "npx": frozenset(),
    "bunx": frozenset(),
    "uvx": frozenset(),
    "pnpm": frozenset({"dlx", "exec"}),
    "uv": frozenset({"tool", "run"}),
}


@dataclass(frozen=True)
class _Discovered:
    """One locally-configured artifact, before registry resolution."""

    name: str  # as configured locally (mcpServers key, or skill dir name)
    platform: str  # which platform config it came from
    source: str  # the file/dir on disk it was read out of
    kind: str  # mcp_server | claude_skill
    terms: tuple[str, ...]  # registry search terms, best first
    names: frozenset[str]  # lowercased names that count as an exact match
    repos: frozenset[str]  # normalised repository URLs that count as a match


@dataclass
class _Checked:
    """A discovered artifact plus whatever the registry could tell us."""

    found: _Discovered
    status: str
    badge: str | None = None
    score: float | None = None
    server_id: str | None = None
    scanners: int | None = None
    note: str = ""
    duplicates: int = 0


def _normalise_repo(url: str) -> str:
    """host+path, lowercased, `.git` and trailing slash stripped.

    So `https://GitHub.com/o/r.git` and `https://github.com/o/r/` compare
    equal. Scheme is dropped; host is NOT, because the host is the part that
    makes two URLs genuinely different projects.
    """
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{host}{path.lower()}"


def _package_from_command(command: str, args: Any) -> str | None:
    """The package name a runnable mcpServers entry launches, if we can tell.

    Inverse of `_build_mcp_entry`: `{"command": "npx", "args": ["-y", pkg]}`
    -> `pkg`. Returns None for `node server.js`, `docker run ...` and anything
    else whose first argument is not a package — guessing there would invent an
    identity, and a wrong identity resolving to some other project's Verified
    badge is worse than reporting the entry under its config key alone.
    """
    runner = Path(str(command or "")).name.lower()
    if runner.endswith(".exe"):
        runner = runner[:-4]
    subcommands = _PACKAGE_RUNNERS.get(runner)
    if subcommands is None or not isinstance(args, list):
        return None
    for arg in args:
        if not isinstance(arg, str) or not arg or arg.startswith("-"):
            continue
        if arg in subcommands:
            continue
        # Strip a version pin (`pkg@1.2.3`) while preserving an npm scope
        # (`@scope/pkg`, whose leading `@` is at index 0).
        at = arg.find("@", 1)
        return arg[:at] if at > 0 else arg
    return None


def _discovered_from_entry(name: str, entry: Any, platform: str, source: Path) -> _Discovered:
    """Build the search/match identity for one `mcpServers` entry."""
    names = {name.strip().lower()}
    repos: set[str] = set()
    terms: list[str] = [name]
    if isinstance(entry, dict):
        # `_build_mcp_entry` writes {"repository": ...} for non-package sources.
        repo = _normalise_repo(str(entry.get("repository") or ""))
        if repo:
            repos.add(repo)
        package = _package_from_command(entry.get("command", ""), entry.get("args"))
        if package:
            names.add(package.lower())
            # Package name first: it is the registry's own naming for anything
            # published to npm/PyPI, while the config key is user-chosen.
            terms.insert(0, package)
    return _Discovered(
        name=name,
        platform=platform,
        source=str(source),
        kind="mcp_server",
        terms=_search_terms(terms),
        names=frozenset(n for n in names if n),
        repos=frozenset(repos),
    )


def _search_terms(candidates: list[str]) -> tuple[str, ...]:
    """Dedupe, clamp to the API's 2-100 char query window, cap at two.

    Two queries per artifact is the bound: enough to try the package name and
    then the config key, few enough that a 30-server config stays quick.
    """
    out: list[str] = []
    for candidate in candidates:
        term = candidate.strip()[:100]
        if len(term) >= 2 and term not in out:
            out.append(term)
    return tuple(out[:2])


def _read_mcp_servers(config_path: Path) -> dict[str, Any]:
    """The `mcpServers` object from a platform config file.

    Raises OSError/ValueError to the caller: a config we could not parse must
    surface as `incomplete`, never as "no artifacts configured here".
    """
    loaded: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config root is not a JSON object")
    servers = cast(dict[str, Any], loaded).get("mcpServers")
    if servers is None:
        return {}
    if not isinstance(servers, dict):
        raise ValueError("mcpServers is not a JSON object")
    return cast(dict[str, Any], servers)


def _discover_claude_skills(skills_root: Path) -> list[_Discovered]:
    """Skill directories installed under a `.claude/skills` root.

    Mirrors `_install_claude_skill`, which copies a directory containing
    SKILL.md to `<skills_root>/<slug>/`. Dot-prefixed names are skipped, which
    also skips that installer's `.<slug>.*.nerlo-tmp` staging directories.
    """
    if not skills_root.is_dir():
        return []
    found: list[_Discovered] = []
    for child in sorted(skills_root.iterdir()):
        if child.name.startswith(".") or not (child / "SKILL.md").is_file():
            continue
        found.append(
            _Discovered(
                name=child.name,
                platform="claude-code",
                source=str(child),
                kind="claude_skill",
                terms=_search_terms([child.name]),
                names=frozenset({child.name.lower()}),
                repos=frozenset(),
            )
        )
    return found


def _discover(project: Path | None) -> tuple[list[_Discovered], list[str]]:
    """Find locally-configured artifacts. Returns (artifacts, unreadable).

    `project is None` scans the standard per-user locations (TARGET_CONFIG_PATHS
    plus `~/.claude/skills`). An explicit PATH scans a PROJECT instead: the same
    relative layouts rooted at that directory, and NOT the home locations.
    That split is the CI use case — a workflow runs inside a checkout where
    $HOME belongs to an ephemeral runner, so mixing runner-global config into a
    repo's gate would make the gate's result depend on the machine rather than
    on the repo.
    """
    sources: list[tuple[str, Path]] = []
    skills_roots: list[Path] = []
    if project is None:
        sources = sorted(TARGET_CONFIG_PATHS.items())
        skills_roots = [_claude_skills_dir()]
    elif project.is_file():
        # An explicit file is read as a platform config directly, so
        # `nerlo check .cursor/mcp.json` works without a directory dance.
        sources = [("mcp", project)]
    else:
        seen: set[Path] = set()
        for name, (parts, _root) in sorted(PLATFORM_CONFIG_LAYOUT.items()):
            sources.append((name, project.joinpath(*parts)))
            seen.add(project.joinpath(*parts))
        for name, parts in PROJECT_CONFIG_EXTRAS:
            path = project.joinpath(*parts)
            if path not in seen:
                sources.append((name, path))
        skills_roots = [project.joinpath(*CLAUDE_SKILLS_PARTS)]

    artifacts: list[_Discovered] = []
    unreadable: list[str] = []
    for platform, config_path in sources:
        if not config_path.is_file():
            continue
        try:
            servers = _read_mcp_servers(config_path)
        except (OSError, ValueError) as exc:
            # NOT silently empty: an unparseable config is an unchecked config.
            unreadable.append(f"{config_path}: {type(exc).__name__}")
            continue
        for name, entry in servers.items():
            artifacts.append(_discovered_from_entry(str(name), entry, platform, config_path))
    for skills_root in skills_roots:
        artifacts.extend(_discover_claude_skills(skills_root))
    return artifacts, unreadable


# Rank used ONLY to pick which row wins when the registry holds several exact
# matches for one local artifact (duplicate submissions are real — live data
# shows repeated names). Lower wins, so a definite bad verdict always beats a
# Verified duplicate: a security gate must never let a duplicate row launder a
# bad verdict into a pass.
_MATCH_PREFERENCE: dict[str, int] = {
    STATUS_UNSAFE: 0,
    STATUS_CAUTION: 1,
    STATUS_WITHHELD: 2,
    STATUS_UNSCORED: 3,
    STATUS_VERIFIED: 4,
}


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    """First key whose value is not None.

    Explicitly not `a or b`: a composite_score of 0.0 is falsy but is a real,
    and maximally alarming, score — `or` would silently discard it.
    """
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _status_from_record(record: dict[str, Any]) -> str:
    """Map a registry record onto one of the check statuses.

    Order matters: the withheld flag is checked BEFORE the badge, because a
    withheld aggregate verdict is the registry stating it will not vouch —
    that must not be read off as "no badge, probably fine".
    """
    if record.get("aggregate_verdict_withheld"):
        return STATUS_WITHHELD
    badge = _first_present(record, "composite_badge", "current_badge")
    if badge == "Unsafe":
        return STATUS_UNSAFE
    if badge == "Caution":
        return STATUS_CAUTION
    if badge == "Verified":
        return STATUS_VERIFIED
    # Anything else — null, or a badge string a future API grows that this CLI
    # has never heard of — is NOT verified. Unrecognised must fail closed.
    return STATUS_UNSCORED


def _match_rows(rows: list[Any], found: _Discovered) -> list[dict[str, Any]]:
    """Rows that are EXACTLY this artifact (name or repository URL).

    The registry's `q=` is a fuzzy search — querying "server" returns
    "@4everland/hosting-mcp". Accepting a fuzzy hit would attach some other
    project's Verified badge to an unknown local artifact, which is the same
    collapse as scoring an unknown green. Only exact identity counts; anything
    else stays `unknown`.
    """
    matched: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        record = cast(dict[str, Any], row)
        name = str(record.get("name") or "").strip().lower()
        repo = _normalise_repo(str(record.get("repository_url") or ""))
        if (name and name in found.names) or (repo and repo in found.repos):
            matched.append(record)
    return matched


def _resolve_one(
    client: httpx.Client,
    found: _Discovered,
    search_cache: dict[str, list[Any]],
    detail_cache: dict[str, dict[str, Any]],
) -> _Checked:
    """Resolve one discovered artifact against the registry."""
    matches: list[dict[str, Any]] = []
    try:
        for term in found.terms:
            if term not in search_cache:
                response = _request(
                    client,
                    "GET",
                    "/api/v1/servers",
                    params={"q": term, "page_size": SEARCH_LIMIT},
                    fatal=False,
                )
                if response.status_code != 200:
                    raise RegistryUnreachable(f"search HTTP {response.status_code}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RegistryUnreachable("malformed search response")
                results = cast(dict[str, Any], payload).get("results")
                search_cache[term] = results if isinstance(results, list) else []
            matches = _match_rows(search_cache[term], found)
            if matches:
                break
    except (RegistryUnreachable, ValueError) as exc:
        # ValueError covers a body that is not JSON. Either way we have no
        # answer, and no answer is not a pass.
        return _Checked(found=found, status=STATUS_ERROR, note=f"registry: {exc}")

    if not matches:
        # (c) NOT FOUND. Unknown, not safe. The submit funnel below says so.
        return _Checked(found=found, status=STATUS_UNKNOWN, note="not in registry")

    matches.sort(key=lambda r: _MATCH_PREFERENCE.get(_status_from_record(r), 9))
    row = matches[0]
    server_id = str(row.get("id") or "")

    # The search row already carries a badge, but the authoritative aggregate
    # lives on the detail endpoint (`composite_badge` + the scanner reports
    # that back it). Fetch it — and if that fetch fails, report `error` rather
    # than quietly falling back to the list row, because a silent fallback is
    # how "we could not verify" turns into "verified".
    #
    # An id that is missing, or that carries path separators (which would let a
    # malformed response steer the request at another endpoint), is treated the
    # same way: unusable id -> error. Deliberately NOT "use the search row
    # instead" — that would be the silent fallback this whole block avoids.
    if not server_id or "/" in server_id or "?" in server_id:
        return _Checked(
            found=found,
            status=STATUS_ERROR,
            note="registry: unusable server id in search result",
            duplicates=len(matches) - 1,
        )
    try:
        if server_id not in detail_cache:
            response = _request(client, "GET", f"/api/v1/servers/{server_id}", fatal=False)
            if response.status_code != 200:
                raise RegistryUnreachable(f"detail HTTP {response.status_code}")
            body = response.json()
            if not isinstance(body, dict):
                raise RegistryUnreachable("malformed detail response")
            detail_cache[server_id] = cast(dict[str, Any], body)
        detail = detail_cache[server_id]
    except (RegistryUnreachable, ValueError) as exc:
        return _Checked(
            found=found,
            status=STATUS_ERROR,
            server_id=server_id,
            note=f"registry: {exc}",
        )

    record = detail
    status = _status_from_record(record)
    reports = record.get("scanner_reports")
    note = ""
    if status == STATUS_WITHHELD:
        note = str(record.get("aggregate_verdict_withheld_reason") or "verdict withheld")
    elif record.get("unscannable_reason"):
        note = str(record["unscannable_reason"])
    return _Checked(
        found=found,
        status=status,
        badge=_first_present(record, "composite_badge", "current_badge"),
        score=_first_present(record, "composite_score", "current_security_score"),
        server_id=server_id or None,
        scanners=len(reports) if isinstance(reports, list) else None,
        note=note,
        duplicates=len(matches) - 1,
    )


_STATUS_COLOURS: dict[str, str] = {
    STATUS_UNSAFE: "red",
    STATUS_CAUTION: "yellow",
    STATUS_VERIFIED: "green",
    STATUS_WITHHELD: "magenta",
    STATUS_UNSCORED: "magenta",
    STATUS_UNKNOWN: "magenta",
    STATUS_ERROR: "red",
}
# Printed in this order so the things that matter are read first.
_SUMMARY_ORDER = (
    STATUS_UNSAFE,
    STATUS_CAUTION,
    STATUS_WITHHELD,
    STATUS_UNSCORED,
    STATUS_UNKNOWN,
    STATUS_ERROR,
    STATUS_VERIFIED,
)


@click.command()
@click.argument(
    "path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--fail-on",
    "fail_on",
    type=click.Choice(sorted(FAIL_ON_STATUSES)),
    default="unsafe",
    show_default=True,
    help="Exit non-zero at this severity or worse ('any' also fails on unknown).",
)
@_api_url_option
@_json_option
def check(path: Path | None, fail_on: str, api_url: str, as_json: bool) -> None:
    """Check locally-configured AI artifacts against the registry (CI gate).

    Reads the same platform configs `nerlo install` writes, resolves each
    entry against the public registry, and exits non-zero when policy is
    violated. With no PATH the standard per-user locations are scanned; give a
    PATH to scan a project checkout instead (the CI case).

    Exit codes: 0 pass, 1 policy violated, 2 usage error, 3 could not
    determine (registry unreachable or a local config could not be parsed).
    """
    artifacts, unreadable = _discover(path)
    scope = "project" if path is not None else "standard locations"
    for problem in unreadable:
        click.secho(f"warning: could not read config — {problem}", fg="yellow", err=True)

    results: list[_Checked] = []
    if artifacts:
        search_cache: dict[str, list[Any]] = {}
        detail_cache: dict[str, dict[str, Any]] = {}
        with _client(api_url) as client:
            results = [_resolve_one(client, a, search_cache, detail_cache) for a in artifacts]

    counts = {status: sum(1 for r in results if r.status == status) for status in _SUMMARY_ORDER}
    violations = [r for r in results if r.status in FAIL_ON_STATUSES[fail_on]]
    # Unreadable local configs are as unresolved as an unreachable registry:
    # in both cases something that should have been checked was not.
    unresolved = [r for r in results if r.status == STATUS_ERROR]
    if violations:
        exit_code = EXIT_POLICY_VIOLATION
    elif unresolved or unreadable:
        exit_code = EXIT_INCOMPLETE
    else:
        exit_code = EXIT_OK

    logger.info(
        "cli.check",
        scope=scope,
        artifacts=len(artifacts),
        fail_on=fail_on,
        violations=len(violations),
        unresolved=len(unresolved),
        exit_code=exit_code,
    )

    if as_json:
        _echo_json(
            {
                "scope": scope,
                "path": str(path) if path is not None else None,
                "fail_on": fail_on,
                "exit_code": exit_code,
                "summary": {"total": len(results), **counts},
                "unreadable_configs": unreadable,
                "artifacts": [
                    {
                        "name": r.found.name,
                        "platform": r.found.platform,
                        "artifact_type": r.found.kind,
                        "source": r.found.source,
                        "status": r.status,
                        "badge": r.badge,
                        "score": r.score,
                        "server_id": r.server_id,
                        "scanners": r.scanners,
                        "duplicate_matches": r.duplicates,
                        "note": r.note,
                    }
                    for r in results
                ],
            }
        )
        sys.exit(exit_code)

    if not artifacts:
        # A legitimate pass — but say it in words. An empty table would read as
        # "checked, all good" when the truth is "there was nothing to check".
        click.echo(f"No AI artifacts configured in {scope} — nothing to check.")
        sys.exit(exit_code)

    _table(
        [
            {
                "status": r.status.upper(),
                "artifact": r.found.name,
                "platform": r.found.platform,
                "score": "-" if r.score is None else f"{float(r.score):.1f}",
                "scanners": "-" if r.scanners is None else str(r.scanners),
                "source": r.found.source,
            }
            for r in results
        ],
        ["status", "artifact", "platform", "score", "scanners", "source"],
    )
    click.echo("")
    click.echo(
        "  ".join(
            click.style(f"{counts[s]} {s}", fg=_STATUS_COLOURS[s])
            for s in _SUMMARY_ORDER
            if counts[s]
        )
    )

    unknowns = [r for r in results if r.status == STATUS_UNKNOWN]
    if unknowns:
        click.echo("")
        click.secho(
            f"{len(unknowns)} artifact(s) are NOT in the Nerlo registry. "
            "Unknown is not safe — nobody has scanned these:",
            fg="magenta",
            bold=True,
        )
        for r in unknowns:
            click.echo(f"  - {r.found.name}  ({r.found.source})")
        click.echo("  Get them scanned:  nerlo submit <repository-url>")

    if unresolved:
        click.echo("")
        click.secho(
            f"{len(unresolved)} artifact(s) could not be resolved — this run did NOT "
            "verify them:",
            fg="red",
            bold=True,
        )
        for r in unresolved:
            click.echo(f"  - {r.found.name}: {r.note}")

    click.echo("")
    if exit_code == EXIT_POLICY_VIOLATION:
        click.secho(
            f"FAIL: {len(violations)} artifact(s) violate --fail-on {fail_on}.", fg="red", bold=True
        )
    elif exit_code == EXIT_INCOMPLETE:
        click.secho("INCOMPLETE: some artifacts could not be checked.", fg="red", bold=True)
    else:
        click.secho(f"PASS: no artifact violates --fail-on {fail_on}.", fg="green", bold=True)
    sys.exit(exit_code)


# Public consumer commands only. Operator/service commands (jobs, verify, serve,
# discovery-scheduler, monitor) stay in the backend — they need DB/pipeline
# internals and are not part of the installable CLI.
ALL_COMMANDS: list[click.Command] = [search, info, install, submit, rescan, check]

__all__ = ["ALL_COMMANDS"]

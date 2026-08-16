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
badge gated: Clean proceeds, Caution prompts for confirmation, Flagged
refuses. For npm-hosted packages the mcpServers entry is runnable
(`npx -y <package>`); for other sources the entry records the repository and
the user finishes the command wiring — Nerlo verifies code, it does not (yet)
ship a package runtime.

`check` is the CI gate and runs the same map in reverse: it READS the platform
configs `install` WRITES, resolves each entry against the public
(unauthenticated) registry, and exits non-zero on policy violation. Its exit
code is its product — see the `nerlo check` section for the contract, and for
why a listing miss is reported as its own outcome rather than as a pass.

The governing invariant of `check`, which every status and exit code below
serves: AN UNRESOLVED ARTIFACT MUST NEVER BE INDISTINGUISHABLE FROM AN ABSENT
ONE, and neither may pass a gate whose whole purpose is to fail on a flagged
artifact.

THE VOCABULARY IS THREE THINGS, NOT ONE — DO NOT "SIMPLIFY" THIS BACK
========================================================================
The registry's badge has a wire spelling and a display spelling, and this CLI
additionally lets a user TYPE one at `--fail-on`. Those are three different
audiences with three different compatibility stories, so they are three
different values and must stay that way:

  1. WIRE — `"Verified"` / `"Caution"` / `"Unsafe"` in the API, and the
     lowercase `verified` / `caution` / `unsafe` in `--json`'s `status` field
     and `summary` keys. THESE DO NOT CHANGE. CI pipelines parse them today;
     renaming them strands every existing consumer mid-run. Every comparison in
     this module (`badge == "Unsafe"`, `_status_from_record`) keys on the wire
     value and nothing else.
  2. DISPLAY — what a human reads: `Clean` / `Caution` / `Flagged`, plus
     `Unrated` for an absent badge and `Scan Halted` for the badge-present /
     score-absent pairing. Produced ONLY by `badge_label_or_unrated`,
     `verdict_label` and `STATUS_LABELS`. Never send one of these anywhere.
  3. TYPED TOKENS — the `--fail-on` words. Both vocabularies are accepted
     (`flagged` and `unsafe` both mean the same level); only the new one is
     documented in `--help`. The old one keeps working silently and
     permanently, because a user's `.github/workflows/*.yml` is not something
     we get to break.

WHY THE DISPLAY WORDS CHANGED. "Unsafe" does not mean unsafe — it means "at
least one scanner of eight-to-eleven scored below 60", which at measurement sat
on ~90% of badged artifacts, most of them scoring well. "Verified" fails the
other way: Nerlo verifies nothing, it runs tools and publishes what they said.
"Caution" is unchanged and deliberately so — it is advice to a reader rather
than an assertion about someone's code. The canonical copy of this table is the
registry's own `web/src/lib/badge-label.ts`; this module is the CLI's twin of
it, and if the two ever disagree the registry's is the behaviour that wins.
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

from nerlo_cli import _update
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


class UpdateNoticeCommand(click.Command):
    """A command that may print "a newer nerlo exists" to STDERR when it ends.

    WHY A COMMAND CLASS AND NOT SIX CALLS. A notice added to the command
    bodies is a control applied to the paths somebody happened to think of:
    the seventh command lands without it, and nothing says so. Carrying it on
    the class makes "which commands notify" a property of the command table
    rather than of anyone's diligence — and `tests/test_update_check.py`
    asserts that every entry in `ALL_COMMANDS` is one of these AND takes
    `--json`, so a command that opts out has to do it visibly.

    WHY `call_on_close` AND NOT A RETURN-VALUE HOOK. `check` (and every
    `_fail`) ends in `sys.exit`, which no result callback survives. Click
    closes the command's context on the way out of `MultiCommand.invoke`
    regardless — SystemExit included — so a close callback is the one hook
    that fires on all six commands' success AND failure paths. It also runs
    after the command's own output, which is where a footnote belongs.

    The notice cannot reach stdout and cannot change an exit code; see the
    three rules in `nerlo_cli/_update.py`.
    """

    def invoke(self, ctx: click.Context) -> Any:
        # Registered BEFORE the body runs, so it still fires when the body
        # exits non-zero — "your nerlo is old" is most useful next to a
        # failure, not least.
        ctx.call_on_close(lambda: self._notify(ctx))
        return super().invoke(ctx)

    @staticmethod
    def _notify(ctx: click.Context) -> None:
        # `as_json` is this CLI's one machine-output switch (`_json_option`).
        # A command without it — `nerlo version` — reads False and notifies.
        _update.maybe_notify(
            _nerlo_home(),
            as_json=bool(ctx.params.get("as_json")),
            read_config=_read_config,
        )


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
# BADGE DISPLAY LABELS — item 2 of the three-way split in the module        #
# docstring. Read that first if you are here to change something.          #
# --------------------------------------------------------------------- #

#: Wire badge value -> the word a human reads. The keys are the API's spelling
#: and are the ONLY thing this module ever compares against; the values never
#: travel anywhere. Mirrors `BADGE_LABELS` in the registry's badge-label.ts.
BADGE_LABELS: dict[str, str] = {
    "Verified": "Clean",
    "Caution": "Caution",
    "Unsafe": "Flagged",
}
#: No badge recorded. Never a grade and never a zero — an artifact nobody has
#: rated is not thereby a bad one, nor a good one.
UNRATED_LABEL = "Unrated"
#: A scan that stopped on a critical finding before the later phases ran. On the
#: wire this is the pairing "badge present, score ABSENT", and it is a DIFFERENT
#: CLAIM from "the scanners disagreed" — most rows carrying the `Unsafe` wire
#: badge are this case rather than the other one, which is precisely why
#: rendering them all as "Flagged" would be the conflation this vocabulary
#: exists to undo. "Absent" means the field was null/missing, NOT that it was
#: present and unparseable; `_coerce_score` collapses those two and callers
#: therefore test the RAW value, not its coercion.
HALTED_LABEL = "Scan Halted"


def badge_label_or_unrated(badge: Any) -> str:
    """The word to SHOW for a badge that may be absent. Never send it onward.

    An unrecognised badge string (one a future API grows) falls through to
    `UNRATED_LABEL` rather than being echoed raw: this function's whole job is
    that no wire spelling reaches a human, and a passthrough would defeat it
    for exactly the values nobody has reviewed.
    """
    if badge is None:
        return UNRATED_LABEL
    return BADGE_LABELS.get(str(badge), UNRATED_LABEL)


def verdict_label(badge: Any, raw_score: Any) -> str:
    """Full display state for a (badge, score) pair, halt case included.

    `raw_score` is the score EXACTLY as the API handed it over — do not pass a
    coerced float, because `None` out of `_coerce_score` also means "present but
    unreadable", and calling that a halt would invent a stop that never
    happened. Halt is resolved before the badge lookup, mirroring the registry's
    own `TrustBadge`.
    """
    if badge is not None and raw_score is None:
        return HALTED_LABEL
    return badge_label_or_unrated(badge)


# --------------------------------------------------------------------- #
# nerlo search (Req 11.3, 11.4)                                            #
# --------------------------------------------------------------------- #


@click.command(cls=UpdateNoticeCommand)
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
                # DISPLAY label, not the wire badge. `--json` above returns the
                # API rows untouched, so a machine consumer still gets
                # `current_badge` verbatim.
                "badge": verdict_label(r.get("current_badge"), r.get("current_security_score")),
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


@click.command(cls=UpdateNoticeCommand)
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
    # DISPLAY labels. `--json` above hands back the API objects unmodified, so
    # `current_badge` is still available verbatim to anything parsing this.
    # Both lines read the RAW score: an absent one beside a present badge is the
    # halt case, and `.get(key, '-')` cannot see that — it returns the stored
    # `None` (rendering the literal "None") whenever the key exists and is null,
    # which `_resolve_skill`'s UUID and search fallbacks both produce.
    raw_score = skill.get("current_security_score")
    click.echo(f"  badge:      {verdict_label(skill.get('current_badge'), raw_score)}")
    click.echo(f"  score:      {'-' if raw_score is None else raw_score}")
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
                    # `badge_label_or_unrated`, not `verdict_label`: halting is
                    # a JOB-level event, so "Scan Halted" is a statement about
                    # the artifact's scan and not about one scanner's opinion.
                    # A per-scanner row with no score is a scanner that did not
                    # score, which is Unrated.
                    "badge": badge_label_or_unrated(s.get("badge")),
                    "findings": len(s.get("findings", [])),
                }
                for s in scanner_reports
            ],
            ["scanner", "score", "badge", "findings"],
        )


def _resolve_skill(client: httpx.Client, skill_name: str) -> dict[str, Any]:
    response = _request(client, "GET", f"/api/v1/skills/{skill_name}", fatal=False)
    if response.status_code == 200:
        return response.json()

    # Fallback 1: Direct UUID lookup on /api/v1/servers/{id}
    try:
        uuid_mod.UUID(skill_name)
        is_uuid = True
    except ValueError:
        is_uuid = False

    if is_uuid:
        server_resp = _request(client, "GET", f"/api/v1/servers/{skill_name}", fatal=False)
        if server_resp.status_code == 200:
            srv = server_resp.json()
            return {
                "skill_id": srv.get("id"),
                "name": srv.get("name"),
                "repository_url": srv.get("repository_url"),
                "artifact_type": srv.get("artifact_type"),
                "current_badge": srv.get("composite_badge"),
                "current_security_score": srv.get("composite_score"),
                "mcp_server_id": srv.get("id"),
            }

    # Fallback 2: Name match on /api/v1/servers search
    search_resp = _request(
        client, "GET", "/api/v1/servers", params={"q": skill_name, "page_size": 10}, fatal=False
    )
    if search_resp.status_code == 200:
        for item in search_resp.json().get("results", []):
            if item.get("name") == skill_name or str(item.get("id")) == skill_name:
                return {
                    "skill_id": item.get("id"),
                    "name": item.get("name"),
                    "repository_url": item.get("repository_url"),
                    "artifact_type": item.get("artifact_type"),
                    "current_badge": item.get("current_badge"),
                    "current_security_score": item.get("current_security_score"),
                    "mcp_server_id": item.get("id"),
                }

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
    """Stable anonymous installer id from ~/.nerlo/installer-id (uuid4).

    Created on first use and reused thereafter, so the derived hash is stable
    across runs for the same machine/user.

    THE 0600 BELOW IS BEST-EFFORT AND IS A NO-OP ON WINDOWS. This docstring said
    "0600" flatly until 2026-08-12, when the first cross-platform CI run refuted
    it: Windows has no POSIX mode bits, `os.chmod` there only toggles a read-only
    flag, and the file lands 0o666. Stated plainly because a comment claiming a
    permission the platform cannot grant is worse than no comment — someone would
    reasonably store something sensitive here on its authority.

    Nothing sensitive IS stored here: the contents are a random uuid4 used to
    make anonymous install telemetry stable per machine. Another local user
    reading it learns an anonymous id they could largely infer from usage anyway.
    That is why this is a documentation fix and not a redesign. If a real secret
    ever needs to live in ~/.nerlo, it needs a Windows-appropriate mechanism
    (DPAPI or an ACL), not this chmod.
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


@click.command(cls=UpdateNoticeCommand)
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
    """Install a registry-listed skill, routed by its artifact type.

    Badge gated: a Clean badge installs, Caution prompts for confirmation, a
    Flagged badge is refused, and an Unrated artifact is not installable.

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
    # Req 11.2 badge gate. EVERY COMPARISON HERE IS AGAINST THE WIRE VALUE and
    # must stay that way — the display words below come out of `BADGE_LABELS`
    # and are never compared against. Swapping a comparison to a label would
    # make the gate stop matching what the API actually sends.
    if badge == "Unsafe":
        _fail(f"{skill_name!r} carries a Flagged badge — installation refused")
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
        # Unreachable for the three known badges (each is handled above), so
        # the raw value echoed here can never be one of them. It is labelled as
        # a registry value because that is what it is: an unrecognised wire
        # string is worth showing verbatim when someone has to report it.
        _fail(
            f"{skill_name!r} is Unrated — it has no badge yet "
            f"(registry value: {badge!r}) — not installable"
        )

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
    # `nerlo_badge` keeps the WIRE value — it is a record written into a
    # machine-readable config, in the same class as `--json`, and nothing should
    # have to re-map it to compare against the API. `nerlo_badge_label` is its
    # display sibling, because this file is also one a human opens: it is the
    # only place the vocabulary would otherwise be missing. Neither key is read
    # back by `check` (which resolves on `repository` / `command`), so adding
    # one cannot change discovery.
    badge = skill.get("current_badge")
    return {
        "repository": repo,
        "nerlo_badge": badge,
        "nerlo_badge_label": badge_label_or_unrated(badge),
    }


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


@click.command(cls=UpdateNoticeCommand)
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


@click.command(cls=UpdateNoticeCommand)
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

# The outcomes `check` reports. THESE STRINGS ARE WIRE VALUES: they are what
# `--json` puts in `status` and in the `summary` keys, and CI parses them. The
# words a human sees come from `STATUS_LABELS` further down and are a separate
# table on purpose — see the three-way split in the module docstring.
#
# They are deliberately NOT collapsible into a pass/fail boolean, because three
# of them are different facts (display label in brackets):
#   verified  — in the registry, aggregate verdict `Verified`   [Clean]
#   caution   — in the registry, aggregate verdict `Caution`    [Caution]
#   unsafe    — in the registry, aggregate verdict `Unsafe`     [Flagged, or
#               Scan Halted when the score is absent — see HALTED_LABEL]
#   withheld  — in the registry, and the registry is REFUSING to publish an
#               aggregate verdict (`aggregate_verdict_withheld`). Live data
#               shows this is common; it means coverage was insufficient.
#   unscored  — in the registry, no aggregate verdict yet (never scanned, or a
#               badge string this CLI does not recognise)
#   unknown   — the registry's listing was searched to exhaustion and this
#               artifact is not in it
#   unresolved— the registry holds MORE candidate rows than this command is
#               willing to read, and none of the rows it did read matched. We
#               do not know whether the artifact is listed. NOT the same fact
#               as `unknown`, and that distinction is the whole point (see
#               `_search_pages`).
#   error     — could not be resolved (registry unreachable / bad response)
#
# `unknown` IS NOT `verified`. Rendering "nobody has ever looked at this" as a
# green check is the exact tri-state collapse this product exists to stop — a
# scanner that did not run must never score as a pass. Same for `withheld`,
# `unresolved` and `error`: an absent answer is not a good answer. Every
# non-verified status gets its own visibly distinct row, and `unknown`
# additionally gets a submit funnel pointing at `nerlo submit`.
#
# Even `unknown` is stated carefully. The OpenAPI schema for the list endpoint
# says `undistributed` artifacts "are never listed; they remain retrievable by
# direct id", so a miss against the LISTING is "we did not find it", not proof
# the registry has never heard of it. The output says listing, not existence.
STATUS_VERIFIED = "verified"
STATUS_CAUTION = "caution"
STATUS_UNSAFE = "unsafe"
STATUS_WITHHELD = "withheld"
STATUS_UNSCORED = "unscored"
STATUS_UNKNOWN = "unknown"
STATUS_UNRESOLVED = "unresolved"
STATUS_ERROR = "error"

# Status wire value -> the words a human reads. Only two entries differ from
# their key, and they are the two the vocabulary change is about; the rest are
# here so that ONE table produces every rendered status and none can be missed
# by a future edit that touches only the interesting ones.
STATUS_LABELS: dict[str, str] = {
    STATUS_VERIFIED: BADGE_LABELS["Verified"],
    STATUS_CAUTION: BADGE_LABELS["Caution"],
    STATUS_UNSAFE: BADGE_LABELS["Unsafe"],
    STATUS_WITHHELD: "Withheld",
    STATUS_UNSCORED: "Unscored",
    STATUS_UNKNOWN: "Unknown",
    STATUS_UNRESOLVED: "Unresolved",
    STATUS_ERROR: "Error",
}
# The statuses that were read off a BADGE, and so are the only ones the halt
# case can apply to. `withheld` is excluded because `_status_from_record`
# resolves it before it ever looks at a badge: "the registry declines to vouch"
# is a stronger and more specific statement than "the scan stopped", and
# overwriting it with the halt label would lose the reason the registry gave.
_BADGE_DERIVED_STATUSES: frozenset[str] = frozenset(
    {STATUS_VERIFIED, STATUS_CAUTION, STATUS_UNSAFE}
)

# Exit codes. THIS IS THE CONTRACT — CI reads it, humans read the table.
#   0  every discovered artifact satisfied the policy (including "nothing
#      installed", which is a legitimate pass)
#   1  policy violated — at least one artifact is at or worse than --fail-on
#   2  usage error (Click's own; listed here so nothing else claims it)
#   3  incomplete — the check could not determine an answer for at least one
#      artifact (registry unreachable, bad response, unparseable local config,
#      or a search too broad to read to the end) and found no outright
#      violation. A check that could not reach the registry has NOT passed.
# Violation outranks incomplete: if we already know something is flagged, exit 1
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
# rated this clean", which is the correct policy once a team has submitted its
# dependency set and wants to keep it that way. Choosing `any` is choosing to
# treat absence of evidence as failure — available, not default.
#
# THE LADDER IS MONOTONIC and `test_fail_on_ladder_is_monotonic` pins it: each
# level is a superset of the one before, so raising --fail-on can only ever add
# failures. Narrowing `caution` to just {caution} would let a laxer-sounding
# level miss a flagged artifact that the stricter-sounding level catches, which
# is a gate that lies about its own severity ordering.
#
# THE KEYS ARE WIRE TOKENS. They are what `--json` reports back in `fail_on`,
# and they keep their original spelling for the same reason `status` does. What
# a user TYPES is a separate, wider vocabulary — see `FAIL_ON_TOKENS`.
FAIL_ON_STATUSES: dict[str, frozenset[str]] = {
    "unsafe": frozenset({STATUS_UNSAFE}),
    "caution": frozenset({STATUS_UNSAFE, STATUS_CAUTION}),
    "any": frozenset(
        {STATUS_UNSAFE, STATUS_CAUTION, STATUS_WITHHELD, STATUS_UNSCORED, STATUS_UNKNOWN}
    ),
}

# Item 3 of the three-way split: the words a user may TYPE at `--fail-on`.
#
# `flagged` and `unsafe` are the SAME LEVEL and both are permanently accepted.
# Only `flagged` is documented, because `unsafe` is the word we are retiring;
# but removing it would break every `--fail-on unsafe` already committed to
# somebody's workflow file, and a security gate that fails closed on its own
# CLI upgrade is a gate people rip out. So: new word in the help, old word
# still works, no deprecation warning to spam a build log with.
FAIL_ON_TOKENS: dict[str, str] = {
    "any": "any",
    "caution": "caution",
    "flagged": "unsafe",
    "unsafe": "unsafe",  # retired spelling, still accepted — do not remove
}
#: The tokens `--help` advertises. Deliberately a subset of `FAIL_ON_TOKENS`.
FAIL_ON_DOCUMENTED: tuple[str, ...] = ("any", "caution", "flagged")
#: Canonical level -> the word to print when a human is told which level ran.
FAIL_ON_LABELS: dict[str, str] = {"unsafe": "flagged", "caution": "caution", "any": "any"}


class _FailOnLevel(click.ParamType):
    """`--fail-on`, accepting both vocabularies and normalising to the wire one.

    Not a `click.Choice`, and that is the entire point: `Choice` renders every
    accepted value into `--help`, so keeping `unsafe` working would also keep it
    printed. This type accepts the wider set and advertises the narrower one.
    """

    name = "level"

    def get_metavar(self, param: click.Parameter, ctx: click.Context | None = None) -> str:
        # `ctx` is positional in Click >= 8.2 and absent in 8.1; defaulted so
        # both call shapes work, since pyproject only floors at click>=8.1.
        return "[" + "|".join(FAIL_ON_DOCUMENTED) + "]"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        level = FAIL_ON_TOKENS.get(str(value).strip().lower())
        if level is None:
            self.fail(f"{value!r} is not one of {', '.join(FAIL_ON_DOCUMENTED)}.", param, ctx)
        return level


# Statuses that mean "we did not get an answer", as opposed to "the answer was
# bad". These drive EXIT_INCOMPLETE and are deliberately in NONE of the
# FAIL_ON_STATUSES sets above: "we could not ask" is never a policy verdict —
# and, critically, never a pass either, at any --fail-on level.
INCOMPLETE_STATUSES: frozenset[str] = frozenset({STATUS_ERROR, STATUS_UNRESOLVED})

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
# Executable suffixes a Windows-authored config records on the runner.
#
# `.cmd` FIRST BECAUSE IT IS THE COMMON ONE: npm installs its runners on Windows
# as batch shims (`npx.cmd`, `pnpm.cmd`), and MCP config examples for Windows
# spell them that way. This list stripped only `.exe` until a coverage audit
# reached the line, which meant the commonest Windows spelling fell through to
# `_PACKAGE_RUNNERS.get("npx.cmd") -> None`: `check` could not name the package,
# resolved the entry by its user-chosen config key alone, found no match, and
# reported a registry-listed artifact as `unknown`. `unknown` does not fail
# `--fail-on flagged`, so a Windows gate silently degraded — the same
# "could-not-determine renders as fine" shape this module exists to prevent,
# reached through the platform instead of through the network.
#
# Widening this cannot create a false identity: the stripped name must still be
# a key of `_PACKAGE_RUNNERS` above, so `evil.cmd` -> `evil` still resolves to
# None and the entry keeps being reported under its config key.
_RUNNER_EXECUTABLE_SUFFIXES: tuple[str, ...] = (".cmd", ".exe", ".bat")


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
    #: Badge present, score ABSENT — the scan stopped before it produced one.
    #: Display-only: the `status` beside it is unchanged and still decides the
    #: gate, so a halted artifact fails exactly the levels it always did. This
    #: is a flag rather than a ninth status precisely so it CANNOT change the
    #: exit code; a halt that stopped failing `--fail-on` would be a security
    #: regression wearing a vocabulary change as a disguise.
    halted: bool = False


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
    identity, and a wrong identity resolving to the badge some OTHER project
    earned is worse than reporting the entry under its config key alone.
    """
    runner = Path(str(command or "")).name.lower()
    for suffix in _RUNNER_EXECUTABLE_SUFFIXES:
        if runner.endswith(suffix):
            runner = runner[: -len(suffix)]
            break
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


def _repo_search_terms(repo: str) -> list[str]:
    """Search terms derived from an already-normalised repository URL.

    THIS IS WHAT MAKES THE REPOSITORY MATCH ARM REACHABLE. `_match_rows` will
    happily match a row on `repository_url`, but a row you never retrieved
    cannot be matched — and the registry's `q=` is a keyword search over
    name/description/author that does NOT index repository URLs. Measured
    against the live API: `q=https://github.com/gemini-cli-extensions/alloydb`
    -> 0 results, `q=gemini-cli-extensions/alloydb` -> 0 results,
    `q=alloydb` -> the row. So the only way to RETRIEVE a repository-keyed
    entry is to query the repo path's own segments.

    The LAST path segment only — it is the one that names the artifact. The
    host is dropped (`q=github.com` is noise) and so is the owner: live,
    `q=gemini-cli-extensions` returns 0 results, so an owner query buys nothing
    and costs a full paginated search per unmatched artifact. Returned as a
    list so the caller splices it into the term list positionally.
    """
    segments = [s for s in repo.split("/")[1:] if s]
    return segments[-1:]


def _discovered_from_entry(name: str, entry: Any, platform: str, source: str) -> _Discovered:
    """Build the search/match identity for one `mcpServers` entry."""
    names = {name.strip().lower()}
    repos: set[str] = set()
    package: str | None = None
    repo_terms: list[str] = []
    if isinstance(entry, dict):
        # `_build_mcp_entry` writes {"repository": ...} for non-package sources.
        repo = _normalise_repo(str(entry.get("repository") or ""))
        if repo:
            repos.add(repo)
            repo_terms = _repo_search_terms(repo)
        package = _package_from_command(entry.get("command", ""), entry.get("args"))
        if package:
            names.add(package.lower())
    # Priority order, best identity first: the package name is the registry's
    # own naming for anything published to npm/PyPI; the repository path names
    # the project itself; the config key is merely user-chosen.
    terms = [t for t in [package, *repo_terms, name] if t]
    return _Discovered(
        name=name,
        platform=platform,
        source=source,
        kind="mcp_server",
        terms=_search_terms(terms),
        names=frozenset(n for n in names if n),
        repos=frozenset(repos),
    )


# Queries issued per artifact. Three, not two: package name, repository path
# segment, config key. The middle one is what resurrected the repository-match
# path (see `_repo_search_terms`), and searching stops at the first term that
# yields an exact match, so the common case is still one query.
_MAX_SEARCH_TERMS = 3


def _search_terms(candidates: list[str]) -> tuple[str, ...]:
    """Dedupe, clamp to the API's 2-100 char query window, cap the count."""
    out: list[str] = []
    for candidate in candidates:
        term = candidate.strip()[:100]
        if len(term) >= 2 and term not in out:
            out.append(term)
    return tuple(out[:_MAX_SEARCH_TERMS])


def _mcp_servers_object(container: dict[str, Any], where: str) -> dict[str, Any]:
    """The `mcpServers` object out of one JSON container, or `{}` if absent.

    A present-but-wrong-shaped `mcpServers` raises rather than reading as
    empty: it may well be hiding entries, and an unread entry must never look
    like an absent one.
    """
    servers = container.get("mcpServers")
    if servers is None:
        return {}
    if not isinstance(servers, dict):
        raise ValueError(f"mcpServers is not a JSON object ({where})")
    return cast(dict[str, Any], servers)


def _read_mcp_servers(config_path: Path) -> list[tuple[str, str, Any]]:
    """Every configured MCP server in a platform config, as (source, name, entry).

    TWO shapes, both real, and reading only the first is how a default-mode
    `nerlo check` misses most Claude Code installs:

      1. the TOP-LEVEL `mcpServers` object — what `nerlo install` writes and
         what every platform documents; and
      2. `projects.<absolute-path>.mcpServers` — Claude Code ALSO nests
         per-project servers under a `projects` map in `~/.claude.json`, and on
         a working developer machine that is where most entries actually live.
         Verified against a real `~/.claude.json` carrying a 10-entry
         `projects` map.

    Returned as a LIST of triples rather than merged into one dict on purpose:
    two projects may each configure a server called `github`, and a dict keyed
    by name would silently drop one of them — the same "unchecked looks like
    absent" collapse this command exists to stop. The `source` string carries
    the owning project path so the table can say which config an entry came
    from.

    Raises OSError/ValueError to the caller: a config we could not parse must
    surface as `incomplete`, never as "no artifacts configured here".
    """
    loaded: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config root is not a JSON object")
    root = cast(dict[str, Any], loaded)
    source = str(config_path)
    entries: list[tuple[str, str, Any]] = [
        (source, str(name), entry) for name, entry in _mcp_servers_object(root, source).items()
    ]

    projects = root.get("projects")
    if projects is None:
        return entries
    if not isinstance(projects, dict):
        raise ValueError("projects is not a JSON object")
    for project_path, project in sorted(cast(dict[str, Any], projects).items()):
        # A non-object project value cannot be hiding an `mcpServers` key, so
        # skipping it hides nothing — this is not the "read it as empty" case
        # `_mcp_servers_object` refuses.
        if not isinstance(project, dict):
            continue
        nested = f"{source}#projects[{project_path}]"
        entries.extend(
            (nested, str(name), entry)
            for name, entry in _mcp_servers_object(cast(dict[str, Any], project), nested).items()
        )
    return entries


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
        for source, name, entry in servers:
            artifacts.append(_discovered_from_entry(name, entry, platform, source))
    for skills_root in skills_roots:
        artifacts.extend(_discover_claude_skills(skills_root))
    return artifacts, unreadable


# Rank used ONLY to pick which row wins when the registry holds several exact
# matches for one local artifact (duplicate submissions are real — live data
# shows repeated names). Lower wins, so a definite bad verdict always beats a
# clean duplicate: a security gate must never let a duplicate row launder a
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


def _coerce_score(value: Any) -> float | None:
    """A score we can safely render, or None.

    Every other API field in this module is isinstance-guarded; this one was
    not, and the renderer's `float(r.score)` raised ValueError on a
    `"composite_score": "N/A"` — which escapes the command as a non-zero exit
    that the contract reads as "policy violated". A field we could not parse
    must never be able to manufacture a verdict, in either direction: it
    renders as `-` and the BADGE, which is what the gate actually keys on,
    still decides.

    `bool` is rejected before the numeric check because `float(True)` is 1.0,
    and a score of "1.0" invented out of a JSON `true` would be a lie.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
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
    # has never heard of — is NOT a clean verdict. Unrecognised must fail closed.
    return STATUS_UNSCORED


def _match_rows(rows: list[Any], found: _Discovered) -> list[dict[str, Any]]:
    """Rows that are EXACTLY this artifact (name or repository URL).

    The registry's `q=` is a fuzzy search — querying "server" returns
    "@4everland/hosting-mcp". Accepting a fuzzy hit would attach some other
    the badge some OTHER project earned to an unknown local artifact, which is
    the same collapse as scoring an unknown green. Only exact identity counts; anything
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


# How `check` reads the paginated list endpoint.
#
# THE BUG THIS EXISTS TO KILL: reading page 1 and reporting a miss as "not in
# registry". Live, a config entry named `app` produced exactly that — the
# registry holds EIGHT rows named exactly `app`, every one of them carrying the
# `Unsafe` wire badge, and at page_size=50 all eight sit on page 2 of 16. One
# page, no match, "not in registry", EXIT 0. A known-bad artifact walked through
# the gate whose only job is to stop known-bad artifacts.
#
# CHECK_PAGE_SIZE is the API's documented maximum (openapi: page_size <= 100).
# CHECK_MAX_PAGES bounds the read at 1000 rows per term, which at the registry's
# current size (total_count ~787 for the broadest term measured, 2026-08-12)
# exhausts even a term that matches everything, with headroom. It is a budget,
# not an assumption: when the budget runs out with rows still unread we report
# `unresolved` rather than inventing an answer.
CHECK_PAGE_SIZE = 100
CHECK_MAX_PAGES = 10


@dataclass(frozen=True)
class _Search:
    """The outcome of searching one term, INCLUDING what we did not read."""

    rows: tuple[Any, ...]
    total: int | None  # the registry's own total_count, when it gave one
    truncated: bool  # rows remain that we did not read

    @property
    def scanned(self) -> int:
        return len(self.rows)


def _int_or_none(value: Any) -> int | None:
    """An int field from the API, or None. `bool` is not an int here."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _search_pages(client: httpx.Client, term: str) -> _Search:
    """Read up to CHECK_MAX_PAGES pages of `q=term`, reporting any remainder.

    `truncated` is set unless we have POSITIVE evidence the listing was
    exhausted — `total_pages` says we read the last page, `total_count` says we
    hold every row, or the server handed back an empty page. Absent that
    evidence we assume rows remain. Failing closed here is the point: the
    caller turns `truncated` into its own status, and an unread remainder must
    never be reported as an absence.
    """
    rows: list[Any] = []
    total: int | None = None
    for page in range(1, CHECK_MAX_PAGES + 1):
        response = _request(
            client,
            "GET",
            "/api/v1/servers",
            params={"q": term, "page_size": CHECK_PAGE_SIZE, "page": page},
            fatal=False,
        )
        if response.status_code != 200:
            raise RegistryUnreachable(f"search HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RegistryUnreachable("malformed search response")
        body = cast(dict[str, Any], payload)
        results = body.get("results")
        if not isinstance(results, list):
            # A search response with no results LIST is malformed, not empty.
            # Reading it as empty is how a broken response becomes "not in the
            # registry", which is the whole failure mode of this module.
            raise RegistryUnreachable("malformed search response")
        page_rows = cast(list[Any], results)
        rows.extend(page_rows)
        reported = _int_or_none(body.get("total_count"))
        # Explicitly not `reported or total`: total_count 0 is a real answer
        # ("nothing matches"), and `or` would discard it as falsy.
        if reported is not None:
            total = reported
        total_pages = _int_or_none(body.get("total_pages"))
        exhausted = (
            not page_rows
            or (total_pages is not None and page >= total_pages)
            or (total is not None and len(rows) >= total)
        )
        if exhausted:
            return _Search(rows=tuple(rows), total=total, truncated=False)
    return _Search(rows=tuple(rows), total=total, truncated=True)


def _resolve_one(
    client: httpx.Client,
    found: _Discovered,
    search_cache: dict[str, _Search],
    detail_cache: dict[str, dict[str, Any]],
) -> _Checked:
    """Resolve one discovered artifact against the registry."""
    # ZERO SEARCHABLE TERMS IS "WE NEVER ASKED", NOT "WE ASKED AND IT IS ABSENT".
    #
    # `_search_terms` drops candidates under the registry's 2-character `q=`
    # floor. An artifact whose every candidate identity is 1 character — an
    # mcpServers key of "a" with a command we cannot derive a package from —
    # therefore arrives here with `terms == ()`. The loop below would not
    # execute, `unread` would stay empty, and the function would fall through to
    # STATUS_UNKNOWN: "not in the registry listing". That is a claim about a
    # search that was never issued (proved with a recording transport:
    # QUERIES ISSUED: []), and STATUS_UNKNOWN does not fail the gate at
    # `--fail-on flagged`, so the artifact passes.
    #
    # This is the SAME defect as the pagination one below, reached through the
    # term-length gate instead — the third distinct route to "could not
    # determine" being rendered as "fine" in this one function. Both now land on
    # STATUS_UNRESOLVED, which is in INCOMPLETE_STATUSES and in no FAIL_ON set,
    # so it exits 3 and is never a pass and never a verdict.
    #
    # It also keeps two shipped claims true: the README's definition of UNKNOWN
    # as "searched the registry listing to exhaustion and did not find it"
    # (`grep -n 'to exhaustion' README.md`) and this module's "could not
    # determine is never a pass, at any --fail-on level".
    if not found.terms:
        return _Checked(
            found=found,
            status=STATUS_UNRESOLVED,
            note="no searchable identity: every candidate name is under the registry's "
            "2-character query minimum, so no search was issued",
        )

    matches: list[dict[str, Any]] = []
    unread: list[str] = []
    try:
        for term in found.terms:
            if term not in search_cache:
                search_cache[term] = _search_pages(client, term)
            result = search_cache[term]
            matches = _match_rows(list(result.rows), found)
            if matches:
                break
            if result.truncated:
                seen = "?" if result.total is None else str(result.total)
                unread.append(f"{term!r} ({result.scanned} of {seen} rows read)")
    except (RegistryUnreachable, ValueError) as exc:
        # ValueError covers a body that is not JSON. Either way we have no
        # answer, and no answer is not a pass.
        return _Checked(found=found, status=STATUS_ERROR, note=f"registry: {exc}")

    if not matches:
        if unread:
            # WE DO NOT KNOW. The term is too common to resolve at a sane cost,
            # so say that — as its own status, with its own exit behaviour —
            # instead of reporting the unread remainder as an absence.
            return _Checked(
                found=found,
                status=STATUS_UNRESOLVED,
                note="search too broad to read to the end: " + "; ".join(unread),
            )
        # (c) NOT IN THE LISTING. Unknown, not safe. The submit funnel says so.
        return _Checked(found=found, status=STATUS_UNKNOWN, note="not in the registry listing")

    matches.sort(key=lambda r: _MATCH_PREFERENCE.get(_status_from_record(r), 9))
    row = matches[0]
    server_id = str(row.get("id") or "")

    # The search row already carries a badge, but the authoritative aggregate
    # lives on the detail endpoint (`composite_badge` + the scanner reports
    # that back it). Fetch it — and if that fetch fails, report `error` rather
    # than quietly falling back to the list row, because a silent fallback is
    # how "we could not verify" turns into a clean verdict.
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
    badge = _first_present(record, "composite_badge", "current_badge")
    # The RAW score, before coercion. `_coerce_score` returns None for both
    # "the field was null" and "the field was present but unreadable", and only
    # the first of those is a halt — see HALTED_LABEL.
    raw_score = _first_present(record, "composite_score", "current_security_score")
    return _Checked(
        found=found,
        status=status,
        badge=badge,
        score=_coerce_score(raw_score),
        server_id=server_id or None,
        scanners=len(reports) if isinstance(reports, list) else None,
        note=note,
        duplicates=len(matches) - 1,
        halted=status in _BADGE_DERIVED_STATUSES and badge is not None and raw_score is None,
    )


# Printed in this order so the things that matter are read first. Also the key
# set of `--json`'s `summary` object, which is why it is spelled in WIRE values.
_SUMMARY_ORDER = (
    STATUS_UNSAFE,
    STATUS_CAUTION,
    STATUS_WITHHELD,
    STATUS_UNSCORED,
    STATUS_UNKNOWN,
    STATUS_UNRESOLVED,
    STATUS_ERROR,
    STATUS_VERIFIED,
)


def _display_label(checked: _Checked) -> str:
    """The word to SHOW for one checked artifact. Never send it anywhere."""
    return HALTED_LABEL if checked.halted else STATUS_LABELS[checked.status]


# The human summary line tallies DISPLAY labels, not statuses, so that the
# counts add up against the rows printed directly above them. A halted artifact
# is one row saying "SCAN HALTED", so the summary must not call it a flagged
# one — while `--json`'s `summary` keeps counting by status, because that is the
# machine contract and the gate really does treat a halt as the status it has.
_DISPLAY_ORDER: tuple[str, ...] = (
    STATUS_LABELS[STATUS_UNSAFE],
    HALTED_LABEL,
    STATUS_LABELS[STATUS_CAUTION],
    STATUS_LABELS[STATUS_WITHHELD],
    STATUS_LABELS[STATUS_UNSCORED],
    STATUS_LABELS[STATUS_UNKNOWN],
    STATUS_LABELS[STATUS_UNRESOLVED],
    STATUS_LABELS[STATUS_ERROR],
    STATUS_LABELS[STATUS_VERIFIED],
)
_DISPLAY_COLOURS: dict[str, str] = {
    STATUS_LABELS[STATUS_UNSAFE]: "red",
    HALTED_LABEL: "red",
    STATUS_LABELS[STATUS_CAUTION]: "yellow",
    STATUS_LABELS[STATUS_WITHHELD]: "magenta",
    STATUS_LABELS[STATUS_UNSCORED]: "magenta",
    STATUS_LABELS[STATUS_UNKNOWN]: "magenta",
    STATUS_LABELS[STATUS_UNRESOLVED]: "red",
    STATUS_LABELS[STATUS_ERROR]: "red",
    STATUS_LABELS[STATUS_VERIFIED]: "green",
}


@click.command(cls=UpdateNoticeCommand)
@click.argument(
    "path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--fail-on",
    "fail_on",
    type=_FailOnLevel(),
    # The DOCUMENTED spelling of the unchanged default level. `_FailOnLevel`
    # converts it to the wire token `unsafe` before the command body sees it,
    # so `--json`'s `fail_on` is byte-identical to what it always was.
    default="flagged",
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
    determine (registry unreachable, a local config could not be parsed, or a
    search too broad to read to the end). "Could not determine" is never a
    pass, at any --fail-on level.
    """
    artifacts, unreadable = _discover(path)
    scope = "project" if path is not None else "standard locations"
    for problem in unreadable:
        click.secho(f"warning: could not read config — {problem}", fg="yellow", err=True)

    results: list[_Checked] = []
    if artifacts:
        search_cache: dict[str, _Search] = {}
        detail_cache: dict[str, dict[str, Any]] = {}
        with _client(api_url) as client:
            results = [_resolve_one(client, a, search_cache, detail_cache) for a in artifacts]

    counts = {status: sum(1 for r in results if r.status == status) for status in _SUMMARY_ORDER}
    violations = [r for r in results if r.status in FAIL_ON_STATUSES[fail_on]]
    # Unreadable local configs are as unresolved as an unreachable registry or
    # a search we could not read to the end: in every one of those cases
    # something that should have been checked was not.
    unresolved = [r for r in results if r.status in INCOMPLETE_STATUSES]
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
                # WIRE token, always — a caller that passed `--fail-on flagged`
                # still reads back `unsafe` here, because this field is the one
                # existing pipelines compare against. `fail_on_label` carries
                # the documented spelling for anyone migrating.
                "fail_on": fail_on,
                "fail_on_label": FAIL_ON_LABELS[fail_on],
                "exit_code": exit_code,
                "summary": {"total": len(results), **counts},
                "unreadable_configs": unreadable,
                "artifacts": [
                    {
                        "name": r.found.name,
                        "platform": r.found.platform,
                        "artifact_type": r.found.kind,
                        "source": r.found.source,
                        # `status` and `badge` are WIRE values and do not
                        # change — CI parses them. `status_label` is the new
                        # sibling carrying the word the table prints, halt case
                        # included, so a consumer can migrate its own output
                        # without re-deriving the mapping.
                        "status": r.status,
                        "status_label": _display_label(r),
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
        if unreadable:
            # NOT "nothing to check" — there was something to check and we
            # could not read it. Saying "nothing to check" here contradicted
            # the stderr warning and the exit-3 this same run produces.
            click.secho(
                f"No AI artifacts could be read in {scope}: "
                f"{len(unreadable)} config(s) could not be parsed. "
                "Nothing was checked.",
                fg="red",
                bold=True,
            )
            click.secho("INCOMPLETE: some configs could not be checked.", fg="red", bold=True)
        else:
            # A legitimate pass — but say it in words. An empty table would read
            # as "checked, all good" when the truth is "nothing to check".
            click.echo(f"No AI artifacts configured in {scope} — nothing to check.")
        sys.exit(exit_code)

    _table(
        [
            {
                "status": _display_label(r).upper(),
                "artifact": r.found.name,
                "platform": r.found.platform,
                # `r.score` is already a float or None — `_coerce_score` made it
                # so at the point the API handed it over, precisely so that this
                # renderer cannot raise on a junk field and turn "we could not
                # read a number" into a non-zero exit meaning "policy violated".
                "score": "-" if r.score is None else f"{r.score:.1f}",
                "scanners": "-" if r.scanners is None else str(r.scanners),
                "source": r.found.source,
            }
            for r in results
        ],
        ["status", "artifact", "platform", "score", "scanners", "source"],
    )
    click.echo("")
    display_counts = {
        label: sum(1 for r in results if _display_label(r) == label) for label in _DISPLAY_ORDER
    }
    click.echo(
        "  ".join(
            click.style(f"{display_counts[label]} {label.lower()}", fg=_DISPLAY_COLOURS[label])
            for label in _DISPLAY_ORDER
            if display_counts[label]
        )
    )

    unknowns = [r for r in results if r.status == STATUS_UNKNOWN]
    if unknowns:
        click.echo("")
        click.secho(
            f"{len(unknowns)} artifact(s) are NOT in the Nerlo registry listing. "
            "Unknown is not safe — nobody has scanned these:",
            fg="magenta",
            bold=True,
        )
        for r in unknowns:
            click.echo(f"  - {r.found.name}  ({r.found.source})")
        click.echo("  Get them scanned:  nerlo submit <repository-url>")
        # Stated as a listing miss, not as proof of absence: the list endpoint
        # documents that `undistributed` artifacts "are never listed; they
        # remain retrievable by direct id", so there is a class of row this
        # search cannot surface at all.
        click.echo(
            "  (Searched the registry listing; `undistributed` artifacts are never listed.)"
        )

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
    # The DOCUMENTED spelling of the level, not the wire token, so the word we
    # print back is the word we told the user to type.
    level = FAIL_ON_LABELS[fail_on]
    if exit_code == EXIT_POLICY_VIOLATION:
        click.secho(
            f"FAIL: {len(violations)} artifact(s) violate --fail-on {level}.", fg="red", bold=True
        )
    elif exit_code == EXIT_INCOMPLETE:
        click.secho("INCOMPLETE: some artifacts could not be checked.", fg="red", bold=True)
    else:
        click.secho(f"PASS: no artifact violates --fail-on {level}.", fg="green", bold=True)
    sys.exit(exit_code)


# Public consumer commands only. Operator/service commands (jobs, verify, serve,
# discovery-scheduler, monitor) stay in the backend — they need DB/pipeline
# internals and are not part of the installable CLI.
ALL_COMMANDS: list[click.Command] = [search, info, install, submit, rescan, check]

__all__ = ["ALL_COMMANDS"]

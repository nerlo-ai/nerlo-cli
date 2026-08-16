"""Best-effort "a newer nerlo is on PyPI" notice.

WHY THIS EXISTS. 0.2.0 shipped a broken Windows console-script shim; 0.3.0
fixed it. Everybody still on 0.2.0 had no way to learn that 0.3.0 existed short
of visiting PyPI on a hunch — which means the upgrade notice is the mechanism
by which somebody running a knowingly-broken build finds out. That is the bar
this feature is held to: it is not a growth surface, it is a defect-notification
channel.

THE THREE RULES THIS MODULE MUST NEVER BREAK, in priority order:

  1. IT NEVER WRITES TO STDOUT. Every command in this CLI supports `--json`,
     and `nerlo check` is a CI gate whose exit code and `--json` body other
     people's pipelines parse. So the notice goes to STDERR, and under
     `--json` it is suppressed entirely — machine output is byte-identical
     with and without this feature. `tests/test_update_check.py` proves that
     by running each command twice and comparing stdout byte-for-byte.
  2. IT NEVER CHANGES AN EXIT CODE. `maybe_notify` catches every exception it
     can catch and turns it into a debug log line. A version notice that can
     fail a build is strictly worse than no version notice.
  3. IT NEVER BLOCKS. One attempt, a hard 1.5s ceiling, and at most once per
     24h — never on the hot path of a user with no network or a DNS hole.

PRIVACY (repo hard rule 5). The check GETs `https://pypi.org/pypi/nerlo/json`,
which discloses this machine's IP address and the user agent `nerlo-cli` to
PyPI, at most once a day. It sends NO identifier, NO usage data, NO install id,
and nothing about what you scanned — deliberately not even the installed
version, which is why the user agent carries no version suffix. It is not
telemetry: nothing about it reaches Nerlo, and there is nothing in the request
to correlate. It is documented in the README and switched off with
`NERLO_UPDATE_CHECK=0`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import click
import httpx

from nerlo_cli._logging import get_logger

logger = get_logger(__name__)

#: The distribution this CLI is published as. `nerlo-cli` on PyPI is a redirect
#: stub with no releases of its own, so it is the wrong thing to poll.
PACKAGE_NAME = "nerlo"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"

#: One day. Justified, not picked: releases here are event-driven (a version
#: bump on main ships itself via `publish.yml`), so there is no schedule for a
#: shorter interval to track. A day bounds the request volume at one per user
#: per day, which is nothing to PyPI and near-nothing to the user; a week — the
#: obvious alternative — would have left a 0.2.0 user running the broken
#: Windows shim for up to seven more days after 0.3.0 landed, and this notice
#: exists precisely to shorten that window.
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

#: 1.5s total, 1.0s to connect, ONE attempt (httpx does not retry by default).
#: The ceiling is deliberately tighter than the registry client's 30s: this
#: request is a nicety appended to a command whose real work is already done and
#: printed, so every millisecond it costs is a millisecond the user did not ask
#: for. It is also the worst case for a machine that cannot write the cache
#: file — see `_write_cache`.
FETCH_TIMEOUT = httpx.Timeout(1.5, connect=1.0)

#: Cache file, relative to the CLI's per-user state dir (`~/.nerlo`, or
#: `NERLO_HOME`). It is a CACHE: deleting it is always safe and costs exactly
#: one extra PyPI request.
CACHE_FILENAME = "update-check.json"

#: Opt-out. `NERLO_UPDATE_CHECK=0` disables; `NERLO_UPDATE_CHECK=1` force-ENABLES
#: even under CI, which is the escape hatch for anyone who wants the notice in
#: their pipeline logs.
ENV_VAR = "NERLO_UPDATE_CHECK"
#: The same switch, persisted, mirroring how `telemetry=false` already works.
CONFIG_KEY = "update_check"

_FALSEY = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class _Cached:
    """A cache entry that is still inside `CHECK_INTERVAL_SECONDS`.

    `latest` is None when the last attempt FAILED. That is recorded on purpose:
    without it, a machine with no network would pay the (bounded) timeout on
    every single invocation instead of once a day.
    """

    latest: str | None


# --------------------------------------------------------------------------- #
# version comparison                                                          #
# --------------------------------------------------------------------------- #
#
# NO NEW DEPENDENCY. `packaging.version` is the right tool and this file does
# not use it, because `packaging` is not installed by `pip install nerlo`: this
# package declares exactly `click` and `httpx` (verified with
# `python -c "import packaging"` in a clean install of this project — it raises
# ModuleNotFoundError), that leanness is a selling point for a supply-chain
# security tool, and adding a top-level dependency needs the owner's go-ahead.
# The stdlib has no version parser at all since `distutils` was removed in 3.12.
#
# So this is a deliberate SUBSET of PEP 440, sized to the only question asked
# here — "is the release on PyPI newer than the one installed?" — and it fails
# CLOSED: anything it cannot confidently order returns None and no notice is
# printed. A string compare would be the wrong answer ("0.10.0" < "0.9.0"), and
# a wrong "upgrade available" is a worse defect than a missing one.

_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)(.*)$", re.ASCII)

_RANK_PRE = 0  # 1.2.0rc1 / a1 / b1 / .dev3 — older than the plain release
_RANK_RELEASE = 1  # 1.2.0
_RANK_POST = 2  # 1.2.0.post1 — newer than the plain release

_PRE_MARKERS = ("rc", "alpha", "beta", "pre", "dev", "a", "b", "c")
# "rev"/"r" before the "rc" check would swallow release candidates, so ordering
# inside `_suffix_rank` matters — pre-release markers are tested first.
_POST_MARKERS = ("post", "rev", "r")


def _suffix_rank(suffix: str) -> int | None:
    """Order a version's trailing segment against the plain release, or None."""
    if not suffix:
        return _RANK_RELEASE
    if suffix.startswith(_PRE_MARKERS):
        return _RANK_PRE
    if suffix.startswith(_POST_MARKERS):
        return _RANK_POST
    # A spelling nobody here has reasoned about. Silence beats a guess.
    return None


def _sort_key(version: str) -> tuple[tuple[int, ...], int] | None:
    """`(release numbers, rank)` for a version string, or None if unorderable."""
    match = _VERSION_RE.match(version.strip().lower())
    if match is None:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    # The local segment ("+g1a2b3c", "+ubuntu1") is metadata, not precedence:
    # an editable checkout reporting `0.3.0+local` holds 0.3.0.
    suffix = match.group(2).split("+", 1)[0].strip(" .-_")
    rank = _suffix_rank(suffix)
    if rank is None:
        return None
    return release, rank


def is_newer(candidate: str, installed: str) -> bool:
    """Is `candidate` a later version than `installed`?

    False whenever either side cannot be ordered — including the dev-checkout
    case, where the installed version is ahead of anything published and must
    never be told to "upgrade" backwards.
    """
    left, right = _sort_key(candidate), _sort_key(installed)
    if left is None or right is None:
        return False
    width = max(len(left[0]), len(right[0]))
    return _padded(left, width) > _padded(right, width)


def _padded(key: tuple[tuple[int, ...], int], width: int) -> tuple[tuple[int, ...], int]:
    """Zero-extend the release tuple so 1.2 and 1.2.0 compare equal."""
    release, rank = key
    return release + (0,) * (width - len(release)), rank


def installed_version() -> str | None:
    """The installed distribution's version, or None when it is not installed.

    None is the git-checkout / `python -m nerlo_cli.main` case: there is no
    installed version to compare, so there is nothing honest to say.
    """
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# opt-out                                                                     #
# --------------------------------------------------------------------------- #


def _is_ci() -> bool:
    """The conventional `CI` signal, as GitHub Actions and friends set it."""
    raw = os.environ.get("CI")
    return raw is not None and raw.strip().lower() not in _FALSEY | {""}


def opted_out(read_config: Callable[[], Mapping[str, str]]) -> str | None:
    """Why the check is off, or None if it may run.

    Precedence is explicit-env, then CI, then config file: someone who exports
    `NERLO_UPDATE_CHECK=1` in a pipeline has said what they want, and the CI
    heuristic must not overrule them.

    CI IS OFF BY DEFAULT, and the argument for that is not "notices are noisy".
    It is that a CI run cannot act on the notice — the version there is whatever
    the pipeline pinned, and a machine reading the log is not going to type
    `pipx upgrade` — while the cost is real: unrequested network egress from a
    security tool's build step, on a host whose egress policy the tool's author
    does not get to decide. The check is also, deliberately, the only network
    call `nerlo check` would make beyond the registry it was pointed at.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is not None:
        return "env" if raw.strip().lower() in _FALSEY else None
    if _is_ci():
        return "ci"
    value = read_config().get(CONFIG_KEY)
    if value is not None and value.strip().lower() in _FALSEY:
        return "config"
    return None


# --------------------------------------------------------------------------- #
# cache                                                                       #
# --------------------------------------------------------------------------- #


def _read_cache(path: Path, now: float) -> _Cached | None:
    """The cached answer if it is fresh, else None (meaning: go ask PyPI).

    Every unreadable, unparseable, wrong-shaped or wrong-typed file is None —
    a corrupt cache costs one HTTP request, never an error. A timestamp in the
    FUTURE is also None: a restored backup or a clock that moved must not be
    able to suppress the check for years.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        cached = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(cached, dict):
        return None
    checked_at = cached.get("checked_at")
    if isinstance(checked_at, bool) or not isinstance(checked_at, int | float):
        return None
    age = now - float(checked_at)
    if age < 0 or age >= CHECK_INTERVAL_SECONDS:
        return None
    latest = cached.get("latest")
    # `null` is a legitimate entry — it is how a FAILED attempt is recorded.
    # Any other non-string is a file somebody or something else corrupted, and
    # trusting its timestamp would suppress the check for a day on the strength
    # of a value we just rejected.
    if latest is not None and not isinstance(latest, str):
        return None
    return _Cached(latest or None)


def _write_cache(path: Path, latest: str | None, now: float) -> None:
    """Record the attempt. Silent on every failure — a read-only HOME is fine.

    Write-then-replace so a reader never sees a half-written file, and the temp
    name carries the pid so two concurrent `nerlo` processes cannot land on the
    same one. A machine that cannot write here simply re-asks PyPI on the next
    invocation; that is the case `FETCH_TIMEOUT` is kept tight for.
    """
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps({"checked_at": now, "latest": latest}) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp.unlink()


# --------------------------------------------------------------------------- #
# fetch                                                                       #
# --------------------------------------------------------------------------- #


def _http_client() -> httpx.Client:
    """The client used to ask PyPI. A seam, so no test can reach the network."""
    return httpx.Client(
        timeout=FETCH_TIMEOUT,
        # No version suffix, no identifier: see the module docstring's privacy
        # paragraph. This is the entire content of what PyPI learns, besides
        # the IP any HTTPS request discloses.
        headers={"User-Agent": "nerlo-cli"},
        follow_redirects=False,
    )


def fetch_latest() -> str | None:
    """The latest published version on PyPI, or None if we could not tell.

    `info.version` is PyPI's own "latest" pointer, which excludes pre-releases —
    so a user on a stable version is never nudged onto an rc. A yanked latest is
    treated as no answer: telling somebody to install a release its own
    maintainer pulled is worse than saying nothing.

    May raise (a transport error, a body that is not JSON). `maybe_notify` is
    where that is turned into "no answer" — see the comment on the call.
    """
    with _http_client() as client:
        response = client.get(PYPI_JSON_URL)
    if response.status_code != 200:
        return None
    payload = response.json()
    info = payload.get("info") if isinstance(payload, dict) else None
    if not isinstance(info, dict) or info.get("yanked") is True:
        return None
    version = info.get("version")
    return version if isinstance(version, str) and version else None


# --------------------------------------------------------------------------- #
# the notice                                                                  #
# --------------------------------------------------------------------------- #


def upgrade_command() -> str:
    """The upgrade line to print, guessed from where this interpreter lives.

    A guess, and only ever advice — pipx installs into `~/.local/pipx/venvs/…`
    and `uv tool` into `…/uv/tools/…`, both of which break if you `pip install`
    into them, so naming the wrong tool is a real (if recoverable) annoyance.
    Anything unrecognised gets the pip line, and the README lists all three.

    Split on both separators rather than with `Path`, so this reads a Windows
    prefix identically on every OS — otherwise the Linux CI cell would be
    testing something the Windows cell does not do.
    """
    parts = {part.lower() for part in re.split(r"[\\/]+", sys.prefix) if part}
    if "pipx" in parts:
        return "pipx upgrade nerlo"
    if "uv" in parts and "tools" in parts:
        return "uv tool upgrade nerlo"
    return "pip install --upgrade nerlo"


def _emit(installed: str, latest: str) -> None:
    """STDERR ONLY. Read rule 1 in the module docstring before touching this."""
    click.secho(
        f"note: nerlo {latest} is available (you have {installed}). Upgrade: {upgrade_command()}",
        fg="yellow",
        err=True,
    )
    click.secho(
        "      This check hits pypi.org at most once a day and sends no "
        f"identifier; turn it off with {ENV_VAR}=0.",
        fg="yellow",
        err=True,
    )


def maybe_notify(
    state_dir: Path,
    *,
    as_json: bool,
    read_config: Callable[[], Mapping[str, str]],
    now: float | None = None,
) -> None:
    """Print an upgrade notice to stderr if one is warranted. Never raises.

    `state_dir` is the CLI's per-user directory (`commands._nerlo_home()`); it
    is passed in rather than imported so this module stays free of any
    dependency on `commands`, which imports it. `read_config` is likewise a
    callable so the config file is only read when it can still change the
    outcome.
    """
    try:
        # Rule 1, and the first line of it deliberately: under `--json` this
        # function does nothing at all — no notice, no network, no cache write.
        if as_json:
            return
        reason = opted_out(read_config)
        if reason is not None:
            logger.debug("cli.update_check", skipped=reason)
            return
        installed = installed_version()
        if installed is None:
            logger.debug("cli.update_check", skipped="not-installed")
            return

        moment = time.time() if now is None else now
        cache_path = state_dir / CACHE_FILENAME
        cached = _read_cache(cache_path, moment)
        if cached is None:
            # The fetch is caught HERE rather than by the outer handler, so
            # that a FAILED attempt is still recorded in the cache. Without
            # this, a laptop with no network would pay the (bounded) timeout on
            # every single `nerlo` invocation instead of one per day — the
            # offline user being exactly the one who must not be taxed for a
            # feature they cannot benefit from.
            try:
                latest = fetch_latest()
            except Exception as exc:  # noqa: BLE001 — offline is not an error
                logger.debug("cli.update_check", fetch_failed=type(exc).__name__)
                latest = None
            _write_cache(cache_path, latest, moment)
            source = "pypi"
        else:
            latest = cached.latest
            source = "cache"

        if latest is None:
            logger.debug("cli.update_check", source=source, result="no-answer")
            return
        newer = is_newer(latest, installed)
        logger.info(
            "cli.update_check",
            source=source,
            installed=installed,
            latest=latest,
            update_available=newer,
        )
        if newer:
            _emit(installed, latest)
    except Exception as exc:  # noqa: BLE001 — rule 2: never change an exit code
        logger.debug("cli.update_check", error=type(exc).__name__)

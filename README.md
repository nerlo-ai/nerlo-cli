# Nerlo CLI

`nerlo` — search, inspect, and install MCP servers from the [Nerlo](https://nerlo.ai) security registry, straight from your terminal.

Nerlo continuously scans, scores, and publishes Model Context Protocol (MCP) servers with per-scanner security scoresheets. This CLI is a thin, dependency-light client (just `click` + `httpx`) over the public Nerlo registry API — it never touches a database or the scan pipeline.

## Install

```sh
pip install nerlo
```

Or, to keep a CLI out of your project environments — either works:

```sh
pipx install nerlo
```

```sh
uv tool install nerlo
```

## Upgrading

Use the line that matches how you installed it:

```sh
pip install --upgrade nerlo
```

```sh
pipx upgrade nerlo
```

```sh
uv tool upgrade nerlo
```

`nerlo version` prints what you are running now. In GitHub Actions there is
nothing to upgrade: `pipx run nerlo check .` resolves the latest release on
every run.

### "a new version is available"

When the version you are running is behind the one on PyPI, `nerlo` says so on
**stderr** and names the command to fix it:

```
note: nerlo 0.4.0 is available (you have 0.3.0). Upgrade: pip install --upgrade nerlo
```

This is how you find out about a fix you are missing — 0.2.0, for instance,
shipped a broken Windows console-script shim that 0.3.0 repaired. Exactly what
it does, so you can decide whether you want it:

- **It asks PyPI, at most once a day.** One `GET https://pypi.org/pypi/nerlo/json`, a 1.5s ceiling, no retry. The answer is cached in `~/.nerlo/update-check.json` (or under `NERLO_HOME`) and not asked for again for 24 hours. A failed attempt is cached too, so being offline costs one bounded attempt a day rather than one per command. Deleting the cache file is always safe.
- **It sends nothing about you.** No identifier, no install id, no usage data, nothing about what you scanned — not even which version you are on. PyPI sees an IP address and the user agent `nerlo-cli`; that is the entire request. It is not telemetry, and nothing about it reaches Nerlo. (Separately, `nerlo install` sends anonymous install telemetry to the registry — a different thing, with its own opt-out below.)
- **It cannot touch machine output.** Under `--json` it is suppressed completely — no notice, and no request either — so `nerlo check --json` produces byte-identical output with the notice available or switched off. It never changes an exit code, and a failure inside the check is silent.
- **It is off in CI.** When `CI` is set, the check does not run at all: a pipeline cannot act on the notice, and a security tool should not make unrequested network calls from your build. `NERLO_UPDATE_CHECK=1` forces it on if you want it in your logs.

To switch it off everywhere:

```sh
export NERLO_UPDATE_CHECK=0
```

Or permanently, as a line in `~/.nerlo/config`:

```
update_check=false
```

## Usage

```sh
nerlo check [PATH]                # audit what's installed and exit non-zero on risk
nerlo search <query>              # search the registry by name/description/author
nerlo info <skill>                # score, badge, and per-scanner scoresheets
nerlo install <skill> --target claude-code   # install into a platform's MCP config
nerlo submit <repo-url>           # submit a repo for ingestion + scanning (auth)
nerlo rescan <id-or-slug>         # queue a re-scan (auth)
```

Every command supports `--json` for machine-readable output.

## `nerlo check` — the CI gate

A dashboard gets looked at when somebody remembers. A non-zero exit blocks the
merge whether anybody remembered or not.

`nerlo check` finds the AI artifacts configured where it runs — by reading the
same platform config files `nerlo install` writes — resolves each one against
the public registry, and exits non-zero when your policy is violated. It needs
no token; the registry read path is public.

```sh
nerlo check                       # audit this machine's standard locations
nerlo check .                     # audit a project checkout (the CI case)
nerlo check --fail-on caution     # stricter
nerlo check . --json              # machine-readable, carries the exit code
```

```
STATUS       ARTIFACT                   PLATFORM  SCORE  SCANNERS  SOURCE
CLEAN        todoist                    mcp       95.9   8         ./mcp.json
CAUTION      acb-tax-mcp                mcp       93.4   9         ./mcp.json
FLAGGED      accessibility-agents       mcp       41.0   7         ./mcp.json
SCAN HALTED  abap-adt-mcp-server        mcp       -      3         ./mcp.json
WITHHELD     abap-adt-mcp-server        mcp       -      0         ./mcp.json
UNKNOWN      totally-made-up-thing-xyz  mcp       -      -         ./mcp.json
UNRESOLVED   app                        mcp       -      -         ./mcp.json
```

### Unknown is not safe

The three outcomes are reported distinctly and are never collapsed:

| Status | `--json` `status` | Meaning |
|--------|-------------------|---------|
| `CLEAN` | `verified` | In the registry, and no scanner scored it below the threshold |
| `CAUTION` | `caution` | In the registry, and a scanner found something worth reviewing |
| `FLAGGED` | `unsafe` | In the registry, and at least one scanner scored it below the threshold |
| `SCAN HALTED` | `unsafe` | In the registry, and the scan **stopped on a critical finding** before it produced a score. A different fact from `FLAGGED`, and the more common one |
| `WITHHELD` | `withheld` | In the registry — and the registry is **declining to publish a verdict** (insufficient scanner coverage) |
| `UNSCORED` | `unscored` | In the registry, not yet scored |
| `UNKNOWN` | `unknown` | **Searched the registry listing to exhaustion and did not find it.** Nobody has scanned this |
| `UNRESOLVED` | `unresolved` | **We do not know.** The search matched more rows than `check` is willing to read, and none of the ones it read were this artifact |
| `ERROR` | `error` | Could not be resolved — the registry did not answer |

The middle column is what `--json` puts in each artifact's `status` field, and it
is **unchanged** — see [Words vs. wire values](#words-vs-wire-values). `SCAN
HALTED` has no wire value of its own: it is `unsafe` beside an absent score, and
it fails every `--fail-on` level that `FLAGGED` does. `--json` also carries a
`status_label` field with the word from the first column.

`UNKNOWN` is not a pass. Rendering "nobody has looked at this" as a green check
is the failure this tool exists to prevent, so unknown artifacts get their own
status, their own callout, and a pointer to `nerlo submit`. The same applies to
`WITHHELD`, `UNRESOLVED` and `ERROR`: an absent answer is not a good answer.

`UNRESOLVED` is deliberately a different status from `UNKNOWN`, because they are
different facts and only one of them is safe to act on. `check` reads the
registry's listing endpoint a page at a time; when it runs out of budget with
rows still unread it reports what it did not read (`'app' (100 of 787 rows
read)`) and exits `3`. It does **not** report an unread remainder as an
absence — that is precisely how eight registry rows named `app`, every one of
them flagged, once produced a green `EXIT 0`.

`UNKNOWN` is also stated as a miss against the **listing**, not as proof of
absence: the API documents that `undistributed` artifacts "are never listed;
they remain retrievable by direct id".

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Every discovered artifact satisfied the policy. Nothing installed is also a pass — and says so in words rather than printing an empty table |
| `1` | Policy violated — something is at or worse than `--fail-on` |
| `2` | Usage error |
| `3` | **Incomplete** — at least one artifact could not be resolved (registry unreachable, a local config could not be parsed, or a search too broad to read to the end) and nothing outright violated the policy. A check that could not reach the registry has not passed |

A violation outranks an incomplete: if something is already known to be flagged,
you get `1`, and the unresolved rows are still printed.

### `--fail-on`

| Level | Fails on |
|-------|----------|
| `flagged` (default) | `FLAGGED`, `SCAN HALTED` |
| `caution` | `FLAGGED`, `SCAN HALTED`, `CAUTION` |
| `any` | anything not `CLEAN`, **including `UNKNOWN`** |

`UNRESOLVED` and `ERROR` are in **none** of these levels: "we could not ask" is
never a policy verdict. They exit `3` at every level — never `0`.

`flagged` and `caution` are verdict thresholds and deliberately do not fail on
unknowns — most of the ecosystem is not in the registry yet, and a gate that
red-builds every repo on day one gets deleted in week one. Use `--fail-on any`
once you have submitted your dependency set: it means "fail unless the registry
affirmatively rated this clean".

**`--fail-on unsafe` still works and always will.** It is the previous spelling
of `--fail-on flagged` and means exactly the same thing. Nothing in an existing
pipeline needs editing; the new spelling is simply the one `--help` documents.

### Words vs. wire values

The words changed. The machine contract did not.

`--json` is unchanged: each artifact's `status` is still `verified` / `caution` /
`unsafe` / `withheld` / `unscored` / `unknown` / `unresolved` / `error`, `badge`
is still the registry's `Verified` / `Caution` / `Unsafe`, and `summary` is still
keyed by those same status values. A pipeline parsing any of them keeps working
with no edit. Two additive fields are new: `status_label` on each artifact and
`fail_on_label` at the top level, both carrying the displayed word, so you can
migrate your own output when you choose to.

Why the words changed: `Unsafe` did not mean unsafe — it meant "at least one
scanner of eight-to-eleven scored below 60", which sat on the large majority of
badged artifacts, most of them scoring well overall. `Verified` overclaimed in
the other direction: Nerlo verifies nothing, it runs independent scanners and
publishes what they said. `Caution` is unchanged and deliberately so — it is
advice to a reader rather than an assertion about someone's code.

### In GitHub Actions

```yaml
- run: pipx run nerlo check .
```

### What it scans

With no `PATH`, the per-user locations `nerlo install` writes to: `~/.claude.json`,
`~/.cursor/mcp.json`, `~/.gemini/settings.json`, `./mcp.json`, and skills under
`~/.claude/skills/`. With a `PATH`, those same layouts rooted at that directory
instead (plus a project-scoped `.mcp.json`) — and **not** the home locations,
because CI runs in a checkout where `$HOME` belongs to an ephemeral runner and a
repo's gate should depend on the repo, not on the machine. `PATH` may also be a
single config file.

In every config it reads both MCP server shapes: the top-level `mcpServers`
object, and Claude Code's per-project `projects.<path>.mcpServers` nesting in
`~/.claude.json` — which on a working machine is where most entries actually
live. Two projects configuring a server under the same name are reported as two
rows, not one, and each row's `SOURCE` names the project it came from.

Identity for each entry comes from the package name in its `command`/`args`, the
`repository` URL, and the config key — in that order. The repository URL is
searched by its path segments because the registry's keyword search does not
index URLs; without that, a repository-only entry can never be retrieved and so
can never be matched.

### Badge-gated install

`nerlo install` respects the composite security badge:

- **Clean** → installs
- **Caution** → warns and asks for confirmation
- **Flagged** → refused
- **Unrated** (no badge published yet) → refused

The registry aggregates evidence from multiple independent scanners; you make the trust decision.

## Configuration

| Setting | Flag | Env var | Default |
|---------|------|---------|---------|
| Registry API base URL | `--api-url` | `NERLO_API_BASE_URL` | `https://api.nerlo.ai` |
| API token (write ops) | `--token` | `NERLO_API_TOKEN` | — |
| [Update notice](#a-new-version-is-available) | — | `NERLO_UPDATE_CHECK` | on, except when `CI` is set |
| Anonymous install telemetry | — | `NERLO_TELEMETRY` | on |
| State/cache directory | — | `NERLO_HOME` | `~/.nerlo` |

`search`, `info` and `check` are unauthenticated — no token needed.

The two network calls you did not ask for are both switchable, by env var
(`NERLO_UPDATE_CHECK=0`, `NERLO_TELEMETRY=0`) or by a line in `~/.nerlo/config`
(`update_check=false`, `telemetry=false`).

Set `NERLO_DEBUG=1` for verbose diagnostic logging on stderr.

## License

Apache-2.0. <!-- @NERLO-REVIEW: confirm MIT vs Apache-2.0 before first publish -->

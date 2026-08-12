# Nerlo CLI

`nerlo` — search, inspect, and install MCP servers from the [Nerlo](https://nerlo.ai) security registry, straight from your terminal.

Nerlo continuously scans, scores, and publishes Model Context Protocol (MCP) servers with per-scanner security scoresheets. This CLI is a thin, dependency-light client (just `click` + `httpx`) over the public Nerlo registry API — it never touches a database or the scan pipeline.

## Install

```sh
pip install nerlo
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
STATUS    ARTIFACT                   PLATFORM  SCORE  SCANNERS  SOURCE
VERIFIED  todoist                    mcp       95.9   8         ./mcp.json
CAUTION   acb-tax-mcp                mcp       93.4   9         ./mcp.json
UNSAFE    accessibility-agents       mcp       -      7         ./mcp.json
WITHHELD  abap-adt-mcp-server        mcp       -      0         ./mcp.json
UNKNOWN   totally-made-up-thing-xyz  mcp       -      -         ./mcp.json
UNRESOLVED app                       mcp       -      -         ./mcp.json
```

### Unknown is not safe

The three outcomes are reported distinctly and are never collapsed:

| Status | Meaning |
|--------|---------|
| `VERIFIED` | In the registry, aggregate verdict Verified |
| `CAUTION` | In the registry, aggregate verdict Caution |
| `UNSAFE` | In the registry, aggregate verdict Unsafe |
| `WITHHELD` | In the registry — and the registry is **declining to publish a verdict** (insufficient scanner coverage) |
| `UNSCORED` | In the registry, not yet scored |
| `UNKNOWN` | **Searched the registry listing to exhaustion and did not find it.** Nobody has scanned this |
| `UNRESOLVED` | **We do not know.** The search matched more rows than `check` is willing to read, and none of the ones it read were this artifact |
| `ERROR` | Could not be resolved — the registry did not answer |

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
them Unsafe, once produced a green `EXIT 0`.

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

A violation outranks an incomplete: if something is already known to be Unsafe,
you get `1`, and the unresolved rows are still printed.

### `--fail-on`

| Level | Fails on |
|-------|----------|
| `unsafe` (default) | `UNSAFE` |
| `caution` | `UNSAFE`, `CAUTION` |
| `any` | anything not `VERIFIED`, **including `UNKNOWN`** |

`UNRESOLVED` and `ERROR` are in **none** of these levels: "we could not ask" is
never a policy verdict. They exit `3` at every level — never `0`.

`unsafe` and `caution` are verdict thresholds and deliberately do not fail on
unknowns — most of the ecosystem is not in the registry yet, and a gate that
red-builds every repo on day one gets deleted in week one. Use `--fail-on any`
once you have submitted your dependency set: it means "fail unless the registry
affirmatively verified this".

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

- **Verified** → installs
- **Caution** → warns and asks for confirmation
- **Unsafe** → refused

The registry aggregates evidence from multiple independent scanners; you make the trust decision.

## Configuration

| Setting | Flag | Env var | Default |
|---------|------|---------|---------|
| Registry API base URL | `--api-url` | `NERLO_API_BASE_URL` | `https://api.nerlo.ai` |
| API token (write ops) | `--token` | `NERLO_API_TOKEN` | — |

`search`, `info` and `check` are unauthenticated — no token needed.

Set `NERLO_DEBUG=1` for verbose diagnostic logging on stderr.

## License

Apache-2.0. <!-- @NERLO-REVIEW: confirm MIT vs Apache-2.0 before first publish -->

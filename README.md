# Nerlo CLI

`nerlo` — search, inspect, and install MCP servers from the [Nerlo](https://nerlo.ai) security registry, straight from your terminal.

Nerlo continuously scans, scores, and publishes Model Context Protocol (MCP) servers with per-scanner security scoresheets. This CLI is a thin, dependency-light client (just `click` + `httpx`) over the public Nerlo registry API — it never touches a database or the scan pipeline.

## Install

```sh
pip install nerlo
```

## Usage

```sh
nerlo search <query>              # search the registry by name/description/author
nerlo info <skill>                # score, badge, and per-scanner scoresheets
nerlo install <skill> --target claude-code   # install into a platform's MCP config
nerlo submit <repo-url>           # submit a repo for ingestion + scanning (auth)
nerlo rescan <id-or-slug>         # queue a re-scan (auth)
```

Every command supports `--json` for machine-readable output.

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

Set `NERLO_DEBUG=1` for verbose diagnostic logging on stderr.

## License

Apache-2.0. <!-- @NERLO-REVIEW: confirm MIT vs Apache-2.0 before first publish -->

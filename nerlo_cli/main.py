"""Nerlo CLI — search, inspect, and install MCP servers from the Nerlo registry.

`nerlo <command>`. All commands talk to the public Nerlo registry API over HTTPS
(override with `--api-url` or `NERLO_API_BASE_URL`); write operations
authenticate with `--token` / `NERLO_API_TOKEN`. This is a thin, dependency-light
client (click + httpx) — it never touches a database or the scan pipeline.
"""

from __future__ import annotations

from importlib import metadata

import click

from nerlo_cli.commands import ALL_COMMANDS, UpdateNoticeCommand


@click.group()
def cli() -> None:
    """Nerlo — MCP server security registry."""


# `cls=` here too, and for the obvious reason: "what version am I on" is the
# single most likely moment for "…and a newer one exists" to be useful. The
# notice goes to stderr, so `nerlo version` still prints exactly one line on
# stdout for anything parsing it.
@cli.command(cls=UpdateNoticeCommand)
def version() -> None:
    """Print the installed nerlo CLI version."""
    try:
        ver = metadata.version("nerlo")
    except metadata.PackageNotFoundError:
        ver = "unknown (not installed)"
    click.echo(f"nerlo {ver}")


for _command in ALL_COMMANDS:
    cli.add_command(_command)


if __name__ == "__main__":
    cli()

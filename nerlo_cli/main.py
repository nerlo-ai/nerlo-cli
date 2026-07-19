"""Nerlo CLI — search, inspect, and install MCP servers from the Nerlo registry.

`nerlo <command>`. All commands talk to the public Nerlo registry API over HTTPS
(override with `--api-url` or `NERLO_API_BASE_URL`); write operations
authenticate with `--token` / `NERLO_API_TOKEN`. This is a thin, dependency-light
client (click + httpx) — it never touches a database or the scan pipeline.
"""

from __future__ import annotations

from importlib import metadata

import click

from nerlo_cli.commands import ALL_COMMANDS


@click.group()
def cli() -> None:
    """Nerlo — MCP server security registry."""


@cli.command()
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

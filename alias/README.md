# `nerlo-cli` — use [`nerlo`](https://pypi.org/project/nerlo/) instead

This package exists only so the name cannot be taken by someone else. It ships no
code: installing it installs [`nerlo`](https://pypi.org/project/nerlo/), which
provides the `nerlo` command.

```console
pip install nerlo        # ← what you want
pipx install nerlo
```

**Why the mismatch?** `nerlo-ai/nerlo-cli` is the GitHub repository; `nerlo` is
the published distribution. People reasonably guess the repo name, so this alias
makes that guess resolve to the real package rather than to a 404 — or, worse, to
a package published by someone else.

Source, issues and releases: <https://github.com/nerlo-ai/nerlo-cli>

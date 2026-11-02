# tau-schedule-prompt

Schedule prompts that fire into a Tau session — a cron/interval/one-shot timer the
agent itself can set, list, and cancel.

**Status: experimental, unofficial, out-of-tree.** This is a port of
[`pi-schedule-prompt`](https://github.com/badlogic/pi-schedule-prompt) v0.4.1 to Tau's
extension API. It is not part of Tau, not an official example under
`examples/extensions/`, and carries no compatibility promise. Pi's interactive
add-flow and jobs-view UIs are not ported yet; `/schedule-prompt` is text output only.

## What you get

| Surface | Name | Notes |
| --- | --- | --- |
| Tool | `schedule_prompt` | actions `add`, `list`, `remove`, `enable`, `disable`, `update`, `cleanup` |
| Command | `/schedule-prompt` | `list`, `pause <job>`, `resume <job>`, `remove <job>`, `cleanup`, `show`, `hide` |
| Widget | below the prompt | job table (status, name, schedule, prompt, next, last, runs, model) |
| Custom message | `scheduled_prompt` | one transcript marker per firing, rendered by the extension |

## Install

From a Tau checkout:

```bash
tau install /path/to/tau-schedule-prompt   # -> ~/.tau/extensions/tau-schedule-prompt
tau install --force /path/to/tau-schedule-prompt   # re-install after edits
```

Or run it without installing:

```bash
tau -e /path/to/tau-schedule-prompt
```

The package declares its entry point in `pyproject.toml`:

```toml
[tool.tau]
extensions = ["src/tau_schedule_prompt/extension.py"]
```

`tau install` copies the tree, so re-run it with `--force` after every edit — the installed copy is
independent of this source tree.

## Use

Ask the agent: *"remind me to check the build every morning at 9"* — it calls
`schedule_prompt` with a 6-field cron expression. Jobs are stored per scope:

- `session` (default): the job dies with the session that created it
- `workdir`: `<cwd>/.tau/schedule-prompts.json`, shared across sessions in that directory

Schedules are `cron` (6 fields: `second minute hour dom month dow`), `once`
(`+10m`, `+2h`, or an ISO timestamp) or `interval` (`5m`, `90s`). Cron fields are
matched in **host-local** time, like Vixie cron and Pi.

Jobs without `model` re-enter the owning conversation with the prompt. Jobs with
`model` run in a background child session; `notify: true` additionally wakes the
parent with the result when the child finishes.

Settings (`<cwd>/.tau/schedule-prompts-settings.json`, or `~/.tau/…`):
`widget_visible`, `default_job_scope`.

## Differences from Pi

- **One delivery per inline firing.** Pi injects an empty visible marker plus a
  separate user message; this port sends the prompt once as a `scheduled_prompt`
  custom message, with the structure the renderer needs carried in `details`.
  Route is `steer` while the session is running, `follow_up` otherwise.
- **One asyncio wake-up loop** instead of one timer per job, and it repaints the
  widget on every wake so relative labels ("in 5m") stay honest.
- **Widget content is Rich-markup strings**, so the extension never imports Textual
  and stays functional (widget-less) in print mode.
- **Hand-ported cron parser** — Pi depends on `croner`; this package has no runtime
  dependency beyond Tau.

## Develop

```bash
uv sync
uv run pytest -q          # unit + wiring tests
uv run ruff check .
uv run ruff format .
uv run mypy               # strict
```

`tests/test_extension.py` drives the entry point against a faked host (fake
component bridge, fake bound session), so registration, lifecycle, firing, the
recursion guard, and the slash command are covered without a TUI or a provider.

## License

MIT

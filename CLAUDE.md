# CLAUDE.md

Project context, decisions and progress live in `memory.md`:

@memory.md

## Working rules

- **Read `memory.md` before editing anything.** It records the scope, data semantics and what is
  (and is not) verified.
- **Raw data is immutable.** Never write to imported files or to `sources/` inside a review
  workspace. Annotations, view state and derived summaries are separate objects. No filtering,
  resampling, normalization or artifact removal of the recorded measurements.
- **Separate-repository boundary.** This repo (`~/erdem/emg-data-viewer`) is independent of the
  live recorder `~/erdem/gforce-emg-ui`. Treat that repo as read-only reference material; do not
  edit it, import from it, or depend on its installation. No sensor SDK / Bluetooth dependency here.
- **Local environment only.** Use the project venv: `.venv/bin/python` (create it per README if
  missing). No system package installs, no global Python changes.
- **Scoped credentials.** This is a shared Linux account: never modify global git/gh config.
  GitHub access uses the repo-local credential helper with `GH_CONFIG_DIR=$HOME/erdem/.config/gh`
  set per process (`GH_CONFIG_DIR=$HOME/erdem/.config/gh gh ...`). Never print tokens.
  Never force-push.
- **Privacy.** Never commit real recordings, review workspaces, logs or credentials. Only
  explicitly synthetic demo/test data may be committed. Inspect staged files before every commit.
- **Update `memory.md`** after each meaningful milestone and before ending a session (dated,
  append-only progress entry: changes, tests actually run with results, commit references).
- **Honesty about verification.** Never mark proposed, simulated or unrun work as verified.
  Distinguish automated tests, offscreen GUI checks and manual checks. Do not claim real-device
  acquisition, timing, or camera/robot synchronization was tested unless it actually was.

# Project Memory — EMG Review (emg-data-viewer)

Shared, version-controlled record of scope, decisions and progress.
**Rule:** read this before editing; append a dated entry to the *Progress log* after each
milestone and before ending a session. Never record unrun work as verified.

## Purpose and MVP scope

Import-first, **offline** desktop viewer for our sEMG research recordings (made by the live
recorder `~/erdem/gforce-emg-ui`, which this repo does not depend on).

Workflow: launch empty → drag in recordings → explore original signals → inspect exact samples →
select / compare / label intervals → save and reopen the review.

Out of scope for the first draft: automatic synchronization, camera player, robot integration,
machine learning, training-data pipeline, filtering/resampling/normalization of EMG.

## Locations

- Local checkout: `~/erdem/emg-data-viewer`
- GitHub: `ErdemTel/emg-data-viewer` (**private**) — see progress log for push status.
- Git identity is repo-local (copied from the sibling repo's local config); credentials via the
  repo-local helper `GH_CONFIG_DIR=$HOME/erdem/.config/gh gh auth git-credential`. Global git/gh
  config is never touched.
- Reference (read-only): `~/erdem/gforce-emg-ui` (recorder; schema v1 committed there, v2
  additive columns in progress as uncommitted work on 2026-10-13 by another session).

## Architecture (src/emg_review/)

| Module | Role (GUI-free unless noted) |
|---|---|
| `snapshot.py` | SHA-256, verified byte copies (`copy_verified`), `snapshot_file` (stat+rehash before/after, retry → `SnapshotError`), `WorkArea` (`/tmp/emg-review-*/sources/<source_id>/<name>`, read-only copies), `atomic_write_text` |
| `csvtable.py` | strict CSV → `CsvTable`: file order, 0-based data rows, per-row byte offsets for exact original text (`raw_cells`), typed columns (float NaN / int + known mask / text), bad-cell + blank-line reporting, field-count errors |
| `schema.py` | column sets, recognition by columns (`classify_csv_columns`, `classify_json`) |
| `recording.py` | `SourceFile`, `EmgRecording` (read-only arrays, segment runs, eligibility, contiguity, axis modes, nearest/row-range lookups, receipt groups, `step`), `ContactTrack`/`ImuTrack`/`MarkerTable` with segment-aware receipt-based `asof` + `placement` |
| `stats.py` | `channel_stats` (AC RMS, exclusion policy), `trailing_ac_rms` envelope, `interval_wave`/`compare_intervals`, `display_series` (display-only min/max reduction) |
| `diagnostics.py` | integrity, per-run timing (`run_timing`, estimator text), timing warnings, contact summary, session.json helpers |
| `annotations.py` | `Annotation` (half-open rows), `make_annotation`/`validate_range` (no segment crossing), store, CSV round-trip, `export_annotations` (refuses source paths) |
| `workspace.py` | `import_paths` (expand folder top-level only, sniff, snapshot, recognise, parse, ambiguity checks), `consistency_issues`, `Review`, `save_review` / `open_review` |
| `demo.py` | deterministic SYNTHETIC demo writer (`python -m emg_review.demo OUTDIR`) |
| `gui/` (Qt) | `viewmodel.py` (cursor/selection/display state + signals, view_state), `plots.py` (8 plots, manual guarded X-sync, overview, companions), `inspector.py` (Sample/Selection/Annotations/Compare/Recording tabs), `dialogs.py`, `worker.py` (QThread task), `main_window.py` (flows, shortcuts, drag-drop, `run`) |

Key decisions:
- X-axes are synced manually with a guard flag (not `setXLink`), because hidden channel plots
  have no screen geometry and pyqtgraph's link then stops updating. Hidden GraphicsViews get
  `resizeEvent(None)` after being shown (stale scene geometry bug found in the offscreen check).
- Plotted y = recorded value where eligible else NaN; lines connect only contiguous rows
  (same run, sample_index +1) — so lost/masked/empty samples and gaps are never bridged.
- Default axis = sample index (row if indices are not strictly increasing within a run).
- One EMG session per review; a second EMG, contradictory session IDs or duplicate companion
  kinds are refused; consistency issues (segments, receipt-time overlap ±10 s, session id,
  session.json row count, unmatched markers) require explicit confirmation.
- Saved review layout: `manifest.json`, `sources/<id>/<name>`, `annotations.csv`,
  `view_state.json`; manifest written last; save target must be new, empty, or same review_id.

## Source immutability and timestamp semantics

- Import = byte-for-byte snapshot into an app-managed temp work area, SHA-256 fingerprinted;
  parsing reads the snapshot. Supplied files are only opened for reading.
- Rows keep their original order and a zero-based data-row number (header excluded). Exact
  original cell text is re-read from the snapshot for the inspector and annotation endpoints.
- `device_time` in the current recorder is the SDK **nominal** timeline (stream anchor +
  sample_index / nominal rate) — not an independently verified acquisition clock.
- Observed on our band (recorder memory, 2026-10-11/13): ~1000 Hz nominal vs ~502–509 received
  rows/s, delivered in ~150 ms bursts. Neither the physical ADC rate nor sensor→host latency is
  verified. The viewer never corrects timestamps or assumes 500 Hz; mismatches are warnings.
- Identical receipt timestamps = "receipt groups"; called SDK callbacks only with `callback_id`.
- contact.csv = electrode–skin impedance (Ohm; kOhm shown). Association = backward as-of on a
  shared recorded receipt clock (host_mono preferred, else host_time), same segment only; age =
  time since LOGGED observation (recorder logs on change → measurement age unknown).

## Implemented (v0.1.0) vs planned

Implemented: everything in README "The window", "Statistics and derived views", "Annotations",
"Saved reviews"; CLI `python -m emg_review [paths] [--demo] [--review DIR]`.

Planned / not done:
- Manual desktop checks (docs/MANUAL_TESTS.md) — **not run yet** (only offscreen).
- Editing an annotation's row range (currently: create a new one), undo/redo.
- Selected-annotation highlight in plots; annotation label collision avoidance.
- Multiple sessions per review / cross-session comparison (deliberately out of scope).
- Large-file performance beyond ~70k rows not measured (while parsing, the whole decoded file
  text plus one 64k-row chunk of split cells are held in memory; arrays are float64).
- Camera/robot matching, sync, ML — out of scope for v1.

## Run / test commands

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m emg_review            # empty start
.venv/bin/python -m emg_review --demo     # SYNTHETIC demo
.venv/bin/python -m pytest                # 72 tests incl. offscreen GUI (sets QT_QPA_PLATFORM=offscreen)
.venv/bin/python -m emg_review.demo OUT   # demo files for manual drag-and-drop
```

## Known limitations, open questions, next steps

- Manual GUI checks pending (real mouse, file-manager drag-and-drop, native dialogs).
- Receipt-axis view of batched data shows vertical stacks (honest, but busy); consider dots.
- Annotation/marker text labels can overlap when close together.
- Impedance dots use channel colours; policy colours are only in the inspector.
- A crash (not SIGTERM/SIGINT) can leave `/tmp/emg-review-*`; it is never auto-deleted by other
  instances (shared machine).
- Open question: which recorder v2 columns (observation, measurement_age_s, block ids) should get
  dedicated views rather than raw display in the inspector.
- Next: run manual checks; review with the lab; decide annotation label vocabulary and
  review_status workflow; then plan camera/hand-pose matching using the exported row references.

## Progress log

### 2026-10-13
- Surveyed `~/erdem/gforce-emg-ui` read-only (README, memory.md, recorder schema v1 + uncommitted
  v2 extras: callback_id, block_id, block_offset, invalid_reasons, contact host_mono/source/...,
  markers event_type/...). Real recordings exist there (`data/…test2` has 70,032 EMG rows) —
  NOT copied or committed; only header/metadata keys inspected.
- Verified scoped gh login = `ErdemTel` (GH_CONFIG_DIR=~/erdem/.config/gh, process-scoped);
  target repo name was free. Created local repo with repo-local identity copied from the
  sibling repo's local config and the same scoped credential helper. Created `.venv`
  (PyQt5 5.15.11, pyqtgraph 0.14.0, numpy 2.5.4, pytest 9.1.1).
- Core milestone (commit 3eee55c): snapshots, CSV parsing, recording model, stats,
  annotations, workspace, demo; 68 tests passing. Found+fixed during testing: `.gitignore`
  pattern `*_review/` hid `src/emg_review/` (removed); test helper used `repr(np.float64)`.
- GUI milestone: main window, plots, inspector, dialogs, worker, CLI; offscreen GUI tests.
  Bugs found via offscreen screenshots and fixed: companion plots kept stale scene width after
  being shown (→ `resizeEvent(None)` refit); numpy bool passed to `setVisible`; crowded log
  impedance axis (→ explicit kΩ ticks). Temp work area now also removed on SIGTERM/SIGINT.
- Test results (this machine, automated): `pytest` → **72 passed** (68 core + 4 offscreen GUI).
- Measured (synthetic demo, 70,032 rows, offscreen, this machine): core import+parse 0.24 s;
  GUI generate+import+first draw ~1.0 s; render() data prep 4 ms for the full 59,527-row
  segment (2,028 drawn points/channel); render+offscreen paint 13–17 ms full segment,
  21–24 ms for a 3,000-row zoom. Offscreen paint is not a measure of on-screen frame rate.
- Read-only compatibility check on the two real recordings in `gforce-emg-ui/data` (parsed in
  place, no copy, hashes+mtimes unchanged): all files recognised; 12,928 / 70,032 EMG rows,
  0 lost/gaps/bad cells; received 503.8 / 502.3 rows/s, receipt groups of 16 rows, receipt gap
  p95 150.5 ms, nominal/receipt elapsed 0.504 / 0.502 → flagged as mismatch (unverified timing);
  27 / 141 impedance observations; IMU 2,578 / 13,956 rows. No real data committed.
- NOT done / not verified: manual desktop GUI checks; any real-device acquisition; any
  camera/robot synchronization.
- Pushed to private GitHub repo https://github.com/ErdemTel/emg-data-viewer (created with
  `gh repo create --private` under the scoped `ErdemTel` login; commits 3eee55c, 6c2f1c9).
- Live launch on this machine's desktop (DISPLAY=:0, XWayland): `python -m emg_review` opened the
  empty welcome screen; verified by an xwd capture of the window (review actions disabled).
  Interactive mouse/keyboard checks (docs/MANUAL_TESTS.md) handed to the user — results not yet
  recorded here.
- Hardening (change found uncommitted in the working tree, kept): `jump_to` refuses annotations
  that do not match the loaded EMG source; `sys.excepthook` shows unexpected slot errors in a
  dialog instead of aborting, so unsaved annotations are not lost. `pytest` → 72 passed.

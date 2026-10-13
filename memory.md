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
- GitHub: `ErdemTel/emg-data-viewer` (private) — see progress log for actual creation/push status.
- Reference (read-only): `~/erdem/gforce-emg-ui` (recorder; schema v1 committed, v2 columns in
  progress there as uncommitted work on 2026-10-13).

## Architecture / file decisions

(filled in as implemented — see progress log)

## Source immutability and timestamp semantics

- Import = byte-for-byte snapshot into an app-managed temp work area, SHA-256 fingerprinted;
  parsing reads the snapshot. Supplied files are never written.
- Rows keep their original order and a zero-based data-row number (header excluded).
- `device_time` in the current recorder is the SDK **nominal** timeline (stream anchor +
  sample_index / nominal rate) — not an independently verified acquisition clock.
- Observed on our band (recorder memory, 2026-10-11): ~1000 Hz nominal vs ~502–509 received
  rows/s, delivered in ~150 ms bursts. Neither the physical ADC rate nor sensor→host latency is
  verified. The viewer must never "correct" timestamps or assume 500 Hz.

## Implemented vs planned

(filled in as implemented)

## Run / test commands

(filled in as implemented)

## Known limitations, open questions, next steps

(filled in as implemented)

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

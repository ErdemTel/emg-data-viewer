# EMG Review

A desktop viewer for our 8-channel sEMG research recordings
(the CSV session folders written by the live recorder `gforce-emg-ui`). It is a separate
application: no sensor SDK, no Bluetooth, no live acquisition.

**Workflow:** launch empty → drag in a recording → explore the original signals → inspect
exact samples → select / compare / label intervals → save the review → reopen it later.

The recorded measurements are never modified. Display summaries, statistics, the optional
envelope and all annotations are separate, derived objects.

> First working draft (v0.1.0). Not included on purpose: automatic synchronization, a
> camera player, robot integration, machine learning or a training-data pipeline.

## Install

Requires Python ≥ 3.10 (developed on Ubuntu, Python 3.12). Everything goes into a
project-local virtual environment:

```bash
git clone https://github.com/ErdemTel/emg-data-viewer.git
cd emg-data-viewer
python3 -m venv .venv                 # Ubuntu: sudo apt install python3-venv if missing
.venv/bin/pip install -e ".[dev]"     # PyQt5, pyqtgraph, numpy (+ pytest)
```

If Qt fails to start with an `xcb` plugin error on Linux, the system needs
`libxcb-cursor0 libxkbcommon-x11-0` (ask an administrator on shared machines).

## Run

```bash
.venv/bin/python -m emg_review                    # normal launch: empty welcome screen
.venv/bin/python -m emg_review --demo             # clearly labelled SYNTHETIC demo
.venv/bin/python -m emg_review path/to/session/   # import a recording folder at start
.venv/bin/python -m emg_review --review my_review # open a saved review folder
.venv/bin/python -m emg_review.demo OUTDIR        # write the synthetic demo files (for drag-and-drop tests)
```

Normal launch never scans the computer, connects hardware, reopens a previous
recording or shows synthetic data. The demo is generated deterministically and is
marked **SYNTHETIC DEMO** in the title bar, the banner and the Recording tab.

## Importing

Drop onto the window (or use *Import / Add Files…*, *Import Folder…*):

- an EMG CSV alone,
- several companion files together,
- a recording folder — only the files directly inside it are inspected (no subfolders,
  no other locations),
- later, additional companion files into the open review.

Files are recognised by their **columns**, not by their names:

| Kind | Required columns | Notes |
|---|---|---|
| EMG | `sample_index`, `ch1`…`ch8` | recorder schema: `segment,sample_index,device_time,host_time,host_mono,lost,valid_mask,ch1..ch8`; other columns are optional, extra columns are kept |
| Electrode–skin impedance | `imp1`…`imp8` + `host_time` or `host_mono` | `contact.csv`: `host_time,segment,imp1..imp8` in **Ohm** (shown in kΩ — a unit conversion only) |
| IMU | `acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z` + a timing/index column | |
| Recorder markers | `label` + `emg_sample_index` / `host_time` / `host_mono` | read-only source annotations |
| Session metadata | a JSON object with session keys (`session_id`, `device`, `emg_rate`, …) | |

Missing optional fields and files are shown as **unknown**; nothing is fabricated
(no timestamps, units, rates or validity flags). Recorder schema v2 columns
(`callback_id`, `block_id`, …) are accepted and preserved.

Import runs in a background thread. Every file is first copied **byte-for-byte** into a
temporary app work area and fingerprinted (SHA-256); parsing reads that snapshot. A file
that changes while it is copied is retried, then rejected. Clear errors are shown for
unreadable, binary, non-UTF-8 or malformed files and for EMG files without the required
columns; malformed rows are never dropped silently and rows are never reordered.

**One recording session per review.** Supplying two EMG recordings together, two
session files with different session IDs, or two files of the same companion kind is
refused. Companions whose segments, receipt-time range, session ID or row counts do not
match the loaded EMG trigger a warning, and nothing is merged unless you confirm.

## The window

- **Toolbar:** Import / Add Files, Open Review, Save Review, Export Annotations,
  recording name, **Axis** mode, **Segment** selector, Fit Recording, Zoom to Selection,
  Go to Sample, Select mode, Help.
- **Main area:** eight stacked EMG channels with a shared horizontal axis and one sample
  cursor. **Equal vertical scale** by default; an *Independent scales* mode is labelled as
  not comparable. Channels can be hidden without removing data.
- **Companion tracks** (when supplied): electrode–skin impedance (log kΩ) and IMU.
- **Overview:** the whole selected segment with a draggable blue window for the visible
  region (click to centre). Red triangles mark lost/masked rows, orange diamonds index gaps.
- **Inspector tabs:** *Sample* (exact original row), *Selection* (statistics + AC RMS bar
  chart), *Annotations*, *Compare A/B*, *Recording* (sources, integrity, timing, impedance
  summary, session metadata).

### Controls

| Action | Input |
|---|---|
| Zoom time axis around the pointer | mouse wheel (`+` / `−` keys around the cursor) |
| Pan | drag; `PgUp` / `PgDn` |
| Select original rows | **Shift + drag**, or toggle **Select mode** (`S`) and drag |
| Inspect a sample | click (nearest original row) |
| Step through ORIGINAL rows | `←` / `→` (Shift ×10, Ctrl ×100), `Home` / `End` |
| Selection start / inclusive end at cursor | `[` / `]` |
| Clear selection, zoom to it, fit segment | `Esc`, `Z`, `F` |
| Label selection / bookmark cursor row | `L` / `B` |
| Store selection as A / B | `1` / `2` |
| Go to row or sample_index | `Ctrl+G` (ambiguous matches are listed for you to choose) |
| Import, open review, save, export | `Ctrl+O`, `Ctrl+Shift+O`, `Ctrl+S`, `Ctrl+E` |
| Help | `F1` |

Lines are never drawn across segments, sample-index discontinuities (dashed orange
markers) or excluded samples (lost / masked / empty). For wide views each channel is
drawn as per-pixel min/max (**display only**); zoomed-in views draw every original
sample, and the inspector and statistics always read the original rows.

## Time axes and timing limitations

| Axis mode | Meaning |
|---|---|
| **Sample index** (default) | the recorded `sample_index`; falls back to *Original row* if indices are not strictly increasing |
| **Original row** | zero-based data-row number in the file (header excluded) |
| **Host receipt time** | elapsed `host_mono` (monotonic) from the segment start — when the host *received* the batch carrying the row; rows of one batch share it. If only `host_time` exists, the axis says **WALL-CLOCK fallback** |
| **SDK nominal time** | elapsed `device_time`, labelled **NOMINAL — acquisition timing unverified** |

In the current recorder `device_time` is the SDK's **nominal** timeline
(stream anchor + `sample_index` / nominal rate), not an independently verified
acquisition clock. Our recordings show ~1000 Hz nominal against ~502 received rows/s,
delivered in ~150 ms bursts. Those observations do **not** verify the physical ADC rate
or the sensor-to-host latency. The viewer therefore never doubles timestamps, never
assumes 500 Hz and never substitutes receipt time for acquisition time; a mismatch is
reported as a warning, and the absolute original timestamps are always shown in the
Sample inspector.

The Recording tab reports, per segment run: rows, index gaps/duplicates, receipt-group
statistics (identical receipt timestamps — called SDK callbacks only if a recorded
`callback_id` column supports it), receipt-gap median / p95 / max, received throughput
(estimator: rows after the first receipt group / receipt span; not estimated for spans
< 10 s; missing indices are not counted), SDK nominal span and its ratio to the receipt
span, plus file integrity (lost flags, masks, non-finite values, unparsed cells).

There is **no** signal-quality score and no SNR. A successful import does not show
physiological quality or synchronization.

## Statistics and derived views

- **Selection statistics** per channel: original row count, eligible count, mean,
  **AC RMS = sqrt(mean((x − mean x)²))**, min, max, |peak|, and the excluded counts.
  *Eligible* = finite value, row not flagged `lost`, channel bit set in `valid_mask`.
  Rows without recorded `lost`/`valid_mask` are not excluded; their count is shown as
  "validity unknown". A channel with no eligible samples shows **unknown**, not 0.
- **AC RMS bar chart** — same computation as the table.
- **Derived envelope** (optional overlay, off by default): trailing AC RMS over a
  **sample-count** window (default 128 samples — not milliseconds). It restarts after
  every segment boundary, index gap or excluded sample and is undefined until a full
  window exists. The original waveform stays visible.
- **A/B comparison** within the recording: two stored selections, per-channel statistics
  with identical settings, grouped bar chart and overlaid waveforms on a relative
  sample-index offset (display coordinate only). No stretching, interpolation,
  resampling or normalization.

No filtering, rectification, denoising, resampling, normalization or artifact removal
is applied in this release. Amplitude alone is not interpreted as gesture, force,
stiffness or signal quality, and no interval is labelled automatically.

## Electrode–skin impedance and markers

`contact.csv` holds **electrode–skin impedance**, not object contact. Logged observations
are plotted as dots. They are placed on non-receipt axes at the first EMG row received
at/after each observation (**receipt-based association**). The Sample inspector shows the
last observation **logged in the same segment** at or before the cursor row's receipt
time (backward as-of on a shared recorded receipt clock), with the time since it was
**logged**. The recorder logs cached values when they change, so this is not the age of
the measurement (unknown). Values that are empty, non-finite or ≤ 0 are shown as unknown,
never as good contact. The 500 / 999 kΩ colours are a configurable **display policy**,
not a validity rule; waveform inspection does not depend on them.

Recorder markers are read-only. Their `emg_sample_index` is the **last received** EMG
sample when the marker was logged — not an acquisition-time event match.

## Annotations

Your own interval annotations and bookmarks are stored separately from the data, as
**half-open original row ranges `[start_row, end_row_exclusive)`**; the inclusive last
selected row is `end_row_exclusive − 1` and is shown explicitly. A selection can never
cross a segment. Fields:

`annotation_id, annotation_type (interval|bookmark), source_id, source_sha256, segment,
start_row, end_row_exclusive, start_sample_index, end_sample_index_inclusive,
start_/end_…_inclusive device_time / host_time / host_mono (exact original text),
selected_channels, label, notes, annotation_source, review_status, created, modified`

Labels such as “pre-grasp candidate” are user assessments, not verified events. The
row references and original timing fields prepare later matching with camera / hand-pose
or VR / robot recordings; **no cross-modal alignment is implemented**. *Export
Annotations* writes this CSV and refuses to overwrite any imported source file.

## Saved reviews

*Save Review* writes a self-contained folder:

```
review_folder/
  manifest.json                         review id, sources (sha256, size, names, rows)
  sources/<source_id>/<original_name>   unchanged byte copies of the imported files
  annotations.csv                       user annotations
  view_state.json                       axis mode, segment, visible rows, cursor, selection,
                                        A/B, channel visibility, scale, envelope, policy
```

- Sources are byte copies (never CSV re-serializations), verified by SHA-256 when written
  and again when reopened; identically named files get separate `source_id` folders.
- Metadata files are written atomically (temporary file + rename), `manifest.json` last.
- Saving only goes to a new folder, an empty folder, or this review's own folder;
  unrelated or other reviews' folders are refused.
- Reopening never needs the original import paths. A modified EMG source makes the review
  refuse to open; a modified companion is skipped with a warning.
- *Open Review* is explicit; you are warned before unsaved annotations are discarded.
- The temporary import work area (`/tmp/emg-review-*`) is removed when the app exits.

## Tests

```bash
.venv/bin/python -m pytest                          # all tests (GUI tests run offscreen)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_smoke.py -s
```

- **Automated (core, no GUI):** parsing and recognition, malformed input, duplicates,
  gaps, segments; byte-for-byte preservation through import / edit / save / reopen,
  basename collisions, deleted originals, sources changing during import, unsafe save
  targets, tampered workspaces; stable row references, stepping, half-open boundaries and
  export; statistics, invalid / all-invalid windows, envelope resets, display reduction;
  timing mismatch as a warning only; segment-aware impedance association; A/B without
  alteration; the synthetic demo (70,032 rows).
- **Offscreen GUI:** the real window driven with the synthetic demo — empty launch,
  click/step/select, statistics, A/B, annotations, all axis modes, hide channels,
  envelope, save and reopen with restored view settings, adding a companion, refusals.
- **Manual:** see [docs/MANUAL_TESTS.md](docs/MANUAL_TESTS.md) (real mouse drag-and-drop,
  wheel/drag feel, dialogs). These need a desktop session.

Project context, decisions and progress: [memory.md](memory.md).

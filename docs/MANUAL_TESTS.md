# Manual GUI checks

Automated tests cover the core logic and drive the real window offscreen
(`tests/test_gui_smoke.py`). The checks below need a real desktop session (mouse,
file manager, native dialogs). Record the date, machine and result in `memory.md`
when you run them; until then they are **not run**.

Prepare synthetic files for drag-and-drop:

```bash
.venv/bin/python -m emg_review.demo /tmp/emg_demo_files      # SYNTHETIC data
.venv/bin/python -m emg_review
```

| # | Check | Expected |
|---|---|---|
| 1 | Launch normally | Empty welcome screen “Drop an EMG recording here”; no data, no scanning |
| 2 | Drag `/tmp/emg_demo_files` (folder) from the file manager onto the window | Progress dialog, then 8 channels, impedance + IMU tracks, SYNTHETIC banner |
| 3 | Drag `emg.csv` alone onto a fresh window | Loads; companions absent; Recording tab says no session.json (units unknown) |
| 4 | Then drag `contact.csv` onto the open review | Added as companion; impedance track appears; title shows `*` |
| 5 | Drag two `emg.csv` from different folders together | Refused: one recording at a time |
| 6 | Wheel over a plot | Zooms the time axis around the pointer; all channels, companions and the overview window follow |
| 7 | Drag in a plot; drag the blue overview window | Pans; no jitter or feedback loop between plot and overview |
| 8 | Shift+drag; press `S` and drag | Selects rows; Selection tab shows `[start, end)`, inclusive last row, stats, bars |
| 9 | Click a plot, then `←`/`→` repeatedly in *Host receipt time* axis | Cursor visits every original row even where 16 rows share one receipt time |
| 10 | `Ctrl+G`, sample_index `5` | Two matches (segments 1 and 2) are listed for you to choose |
| 11 | `[` at one row, move, `]` | Selection ends at the cursor row inclusive |
| 12 | `L` → label “pre-grasp candidate”; `B` on a row | Appear in Annotations tab and as bands / dashed line |
| 13 | `1`, select another interval, `2` | Compare A/B tab: table, bars, overlaid waveforms |
| 14 | Toggle channels, Independent scales, Derived envelope | Plots hide/show; independent-scale warning in the info line; black envelope with gaps at the index gap |
| 15 | Save Review to a new folder; try saving to a non-empty unrelated folder | First succeeds; second is refused with an explanation |
| 16 | Close with unsaved annotations | “Discard unsaved changes?” prompt |
| 17 | Delete `/tmp/emg_demo_files`, then *Open Review* on the saved folder | Opens; annotations and view settings restored |
| 18 | Export Annotations | CSV with half-open row ranges and original timestamp text |

"""Recognition of recorder files by their columns (never by file name alone).

Supported (gforce-emg-ui recorder, schema v1; v2 adds columns, which are accepted and
preserved):

  EMG      segment,sample_index,device_time,host_time,host_mono,lost,valid_mask,ch1..ch8
  contact  host_time,segment,imp1..imp8          (electrode–skin impedance, Ohm)
  IMU      timing fields + acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
  markers  host_time,host_mono,segment,emg_sample_index,label
  session  JSON object with session/device/settings metadata
"""

import json
from pathlib import Path

N_CHANNELS = 8
CHANNELS = [f"ch{i + 1}" for i in range(N_CHANNELS)]
IMPEDANCES = [f"imp{i + 1}" for i in range(N_CHANNELS)]
IMU_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TIME_COLUMNS = ["device_time", "host_time", "host_mono"]

KIND_EMG = "emg"
KIND_CONTACT = "contact"
KIND_IMU = "imu"
KIND_MARKERS = "markers"
KIND_SESSION = "session"
KIND_LABELS = {
    KIND_EMG: "EMG",
    KIND_CONTACT: "electrode–skin impedance",
    KIND_IMU: "IMU",
    KIND_MARKERS: "recorder markers",
    KIND_SESSION: "session metadata",
}

# What each kind needs: (required columns, column types to convert)
EMG_REQUIRED = ("sample_index", *CHANNELS)
EMG_TYPES = {"segment": "int", "sample_index": "int", "lost": "int", "valid_mask": "int",
             "callback_id": "int",
             **{c: "float" for c in TIME_COLUMNS}, **{c: "float" for c in CHANNELS}}
CONTACT_TYPES = {"segment": "int", "host_time": "float", "host_mono": "float",
                 **{c: "float" for c in IMPEDANCES}}
IMU_TYPES = {"segment": "int", "sample_index": "int", "valid_mask": "int",
             **{c: "float" for c in TIME_COLUMNS}, **{c: "float" for c in IMU_AXES}}
MARKER_TYPES = {"segment": "int", "emg_sample_index": "int", "host_time": "float",
                "host_mono": "float", "label": "text", "event_type": "text"}
SESSION_KEYS = {"session_id", "device", "emg_rate", "started", "schema_version", "counts",
                "onboard_filters", "rates", "app_version"}
CSV_LIKE_SUFFIXES = {".csv", ".txt", ".tsv", ""}


def classify_csv_columns(columns: list) -> tuple:
    """Return (kind or None, message). message explains near misses, e.g. an EMG-like file
    missing required columns, so the user gets a clear reason."""
    cols = set(columns)
    has_ch = [c for c in CHANNELS if c in cols]
    timing = cols & {"host_time", "host_mono", "sample_index", "device_time"}
    if len(has_ch) == N_CHANNELS and "sample_index" in cols:
        extra = ""
        if set(IMPEDANCES) <= cols:
            extra = " (it also has imp1..imp8 columns; treated as EMG)"
        return KIND_EMG, "EMG recording" + extra
    if has_ch:
        missing = [c for c in EMG_REQUIRED if c not in cols]
        return None, ("looks like EMG but is missing required column(s): " + ", ".join(missing))
    if set(IMU_AXES) <= cols:
        if timing:
            return KIND_IMU, "IMU track"
        return None, "IMU-like columns but no timing/index column"
    if set(IMPEDANCES) <= cols:
        if cols & {"host_time", "host_mono"}:
            return KIND_CONTACT, "electrode–skin impedance observations"
        return None, "impedance-like columns but no host_time/host_mono receipt time"
    if "label" in cols and cols & {"emg_sample_index", "host_time", "host_mono"}:
        return KIND_MARKERS, "recorder markers"
    return None, "columns not recognized (" + ", ".join(columns[:12]) + \
        (", …" if len(columns) > 12 else "") + ")"


def load_session_json(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{Path(path).name}: not valid UTF-8 JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{Path(path).name}: JSON top level is not an object")
    return doc


def classify_json(doc: dict) -> tuple:
    keys = set(doc)
    if keys & SESSION_KEYS:
        return KIND_SESSION, "session metadata"
    return None, "JSON object without recognized session keys"

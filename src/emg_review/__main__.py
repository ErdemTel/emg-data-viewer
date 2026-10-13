"""Command line entry point: python -m emg_review [paths…] [--demo] [--review FOLDER]"""

import argparse
import sys

from . import APP_NAME, __version__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m emg_review",
        description=f"{APP_NAME}: import-first, offline viewer for 8-channel sEMG recordings. "
                    "Starts empty unless you pass files, --review or --demo.")
    ap.add_argument("paths", nargs="*",
                    help="recording files or one recording folder to import at start (optional)")
    ap.add_argument("--demo", action="store_true",
                    help="open the clearly labelled SYNTHETIC demo (generated data, not a recording)")
    ap.add_argument("--review", metavar="FOLDER", help="open a saved review folder")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = ap.parse_args(argv)
    from .gui.main_window import run
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

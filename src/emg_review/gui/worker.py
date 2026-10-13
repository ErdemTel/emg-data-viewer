"""Run import/parse/open work off the GUI thread so the interface stays responsive."""

import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class Task(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)          # message, traceback

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args

    def run(self):
        try:
            result = self._fn(*self._args)
        except Exception as e:             # reported to the user, never swallowed
            self.failed.emit(f"{type(e).__name__}: {e}" if not str(e) else str(e),
                             traceback.format_exc())
            return
        self.succeeded.emit(result)

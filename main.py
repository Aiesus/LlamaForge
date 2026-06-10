"""
llama-gui v2 — entry point.
"""
from __future__ import annotations
import sys
import threading
import traceback
from pathlib import Path

LOG_FILE = Path(__file__).parent / "crash.log"


def _log_exception(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{msg}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _log_exception
threading.excepthook = lambda args: _log_exception(
    args.exc_type, args.exc_value, args.exc_traceback
)

if __name__ == "__main__":
    from gui.app import LlamaApp
    app = LlamaApp()
    app.run()

"""Make the repository root importable when scripts are run as ``python scripts/<name>.py``."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; force UTF-8 so unicode in logs never crashes a run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys
import numpy as np
import math
from dataclasses import dataclass, is_dataclass, asdict
from collections.abc import Mapping

# === 1. Reporting Utilities ===

@dataclass
class FailureReport:
    source: str
    summary: str
    details: dict

def formatFailure(report: FailureReport) -> str:
    """Helper to stringify a failure report."""
    lines = [f"Source={report.source}", f"Summary={report.summary}"]
    for k, v in (report.details or {}).items():
        lines.append(f"{k}={v}")
    return " | ".join(lines)

class TeeStdout:
    """Mirror all prints to both console and a file."""
    def __init__(self, logfile_path):
        self._file = open(logfile_path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# === 2. Serialization Utilities ===

def serializeValue(v):
    """Helper to serialize numpy/tuple data for JSON/ADIOS2."""

    if isinstance(v, str):
        return v
        

    if np.isscalar(v):
        if hasattr(v, "item"): 
            return v.item() # numpy -> python native
        return v


    if isinstance(v, tuple) and hasattr(v, "_fields"):
        return {name: serializeValue(getattr(v, name)) for name in v._fields}
        

    if isinstance(v, (list, np.ndarray, tuple)):
        arr = np.atleast_1d(v)
        if arr.size < 64:
            return arr.tolist()
        else:
            return f"<array shape={arr.shape}>"
            

    return str(v)

def toMapping(obj):
    """Best-effort conversion of arbitrary objects to {key: value} mapping."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return {name: getattr(obj, name) for name in obj._fields}
    if hasattr(obj, "__dict__") and isinstance(obj.__dict__, dict):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"<value>": obj}

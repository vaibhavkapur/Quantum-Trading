"""Explicit optional C++20 adapter. No compilation or library loading on import."""
import ctypes
from pathlib import Path

import numpy as np

from .measures import validate_alpha
from .models import finite_vector


def native_risk(losses, alpha, library):
    validate_alpha(alpha)
    values = np.ascontiguousarray(finite_vector(losses, "losses"), dtype=np.float64)
    handle = ctypes.CDLL(str(Path(library).resolve()))
    function = handle.empirical_tail_risk
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer, ctypes.c_size_t, ctypes.c_double, pointer, pointer]
    function.restype = ctypes.c_int
    var, es = ctypes.c_double(), ctypes.c_double()
    status = function(values.ctypes.data_as(pointer), len(values), alpha, ctypes.byref(var), ctypes.byref(es))
    if status:
        raise RuntimeError(f"native risk calculator failed with status {status}")
    return {"alpha": alpha, "var": var.value, "cvar": es.value, "expected_shortfall": es.value}

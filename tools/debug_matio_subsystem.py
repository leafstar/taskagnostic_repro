from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from matio.matio5 import read_subsystem


def main() -> None:
    path = (
        Path.home()
        / "Documents"
        / "taskagnostic_repro_data"
        / "raw"
        / "AB06"
        / "10_09_18"
        / "levelground"
        / "ik"
        / "levelground_cw_fast_01_01.mat"
    )
    mat = loadmat(path, chars_as_strings=True, verify_compressed_data_integrity=True)
    ssdata = mat["__function_workspace__"]
    byte_order = "<" if ssdata[0, 2] == b"I"[0] else ">"
    ss = read_subsystem(ssdata, byte_order, False, True)
    print("ss shape", ss.shape, "dtype names", ss.dtype.names)
    for idx in np.ndindex(ss.shape):
        item = ss[idx]
        print("idx", idx)
        if item.dtype.names:
            for name in item.dtype.names:
                val = item[name]
                print(" field", name, "shape", getattr(val, "shape", None), "dtype", getattr(val, "dtype", None))
                if getattr(val, "dtype", None) is not None and val.dtype.names:
                    print("  nested names", val.dtype.names)
                    nested = val.flat[0]
                    for n in val.dtype.names:
                        vv = nested[n]
                        print("   ", n, "shape", getattr(vv, "shape", None), "dtype", getattr(vv, "dtype", None), "sample", str(vv)[:120])


if __name__ == "__main__":
    main()

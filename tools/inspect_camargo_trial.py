from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main() -> None:
    root = (
        Path.home()
        / "Documents"
        / "taskagnostic_repro_data"
        / "raw"
        / "AB06"
        / "10_09_18"
        / "levelground"
    )
    for modality in ["ik", "id", "imu", "fp", "gon", "jp"]:
        path = root / modality / "levelground_cw_fast_01_01.mat"
        print(f"\n=== {modality} {path.name} exists={path.exists()} size={path.stat().st_size if path.exists() else None}")
        try:
            mat = loadmat(path, squeeze_me=False, struct_as_record=True)
        except Exception as exc:
            print("ERROR", type(exc).__name__, exc)
            continue
        print("keys:", [repr(k) for k in mat.keys()])
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            print("key", repr(key), "type", type(value), "shape", getattr(value, "shape", None), "dtype", getattr(value, "dtype", None))
            if arr.dtype.names:
                print(" dtype.names", arr.dtype.names)
                elem = arr.flat[0]
                for name in arr.dtype.names:
                    val = elem[name]
                    sample = str(val).replace("\n", " ")[:120]
                    print("  field", name, "shape", getattr(val, "shape", None), "dtype", getattr(val, "dtype", None), "sample", sample)
            elif arr.dtype == object:
                first = arr.flat[0]
                print(" object first", type(first), getattr(first, "shape", None), getattr(first, "dtype", None), str(first)[:120])


if __name__ == "__main__":
    main()

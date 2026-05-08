from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat

from matio.matio5 import load_opaque_object, read_subsystem
from matio.subsystem import get_matio_context, set_file_wrapper


def read_table(path: Path):
    mat = loadmat(path, chars_as_strings=True, verify_compressed_data_integrity=True)
    ssdata = mat.pop("__function_workspace__")
    byte_order = "<" if ssdata[0, 2] == b"I"[0] else ">"
    ss = read_subsystem(ssdata, byte_order, False, True)
    mcos = ss[0, 0]["MCOS"]
    fwrap_data = mcos[0]["arr"]
    with get_matio_context():
        fw = set_file_wrapper()
        fw.init_load(fwrap_data, byte_order, raw_data=False, add_table_attrs=True)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if value.dtype.names and set(value.dtype.names) == {"s0", "s1", "s2", "arr"}:
                item = value.flat[0]
                classname = item["s2"].decode() if isinstance(item["s2"], bytes) else str(item["s2"])
                type_system = item["s1"].decode() if isinstance(item["s1"], bytes) else str(item["s1"])
                return key, load_opaque_object(item["arr"], classname, type_system)
            return key, value
    raise RuntimeError("No table")


def main() -> None:
    root = Path.home() / "Documents" / "taskagnostic_repro_data" / "raw" / "AB06" / "10_09_18" / "levelground"
    for modality in ["ik", "id", "imu", "fp", "gon", "jp"]:
        path = root / modality / "levelground_cw_fast_01_01.mat"
        key, obj = read_table(path)
        print("\n===", modality, "key", repr(key), "type", type(obj))
        print("shape", getattr(obj, "shape", None))
        print("columns", list(getattr(obj, "columns", []))[:40])
        print(obj.head(2).to_string() if hasattr(obj, "head") else repr(obj)[:1000])


if __name__ == "__main__":
    main()

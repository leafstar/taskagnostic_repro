from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import compile_patterns, finite_or_zero, infer_side, normalize_moment_by_mass, resolve_path


@dataclass
class TrialData:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    sample_rate_hz: float
    participant: str
    trial_id: str
    task: str
    side: str | None = None
    mass_kg: float | None = None


def _is_numeric_matrix(value: Any) -> bool:
    arr = np.asarray(value)
    return arr.ndim in (1, 2) and arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def _safe_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return None
    if arr.shape[0] < arr.shape[1] and arr.shape[1] > 20:
        arr = arr.T
    if arr.shape[0] < 20:
        return None
    return finite_or_zero(arr.astype(np.float32))


def _decode_bytes(arr: Any) -> str | None:
    try:
        value = np.asarray(arr)
        if value.dtype.kind in {"S", "U"}:
            return "".join(value.astype(str).ravel()).strip()
    except Exception:
        return None
    return None


def _flatten_scipy(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.startswith("__"):
                continue
            name = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_scipy(value, name))
        return out
    if hasattr(obj, "_fieldnames"):
        for key in obj._fieldnames:
            name = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_scipy(getattr(obj, key), name))
        return out
    arr = np.asarray(obj)
    if arr.dtype == object and arr.size == 1:
        return _flatten_scipy(arr.item(), prefix)
    out[prefix] = obj
    return out


def _read_hdf5_node(node: h5py.Dataset | h5py.Group, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(node, h5py.Dataset):
        data = node[()]
        if isinstance(data, bytes):
            out[prefix] = data.decode(errors="ignore")
        else:
            out[prefix] = np.array(data)
        return
    for key, child in node.items():
        name = f"{prefix}.{key}" if prefix else key
        _read_hdf5_node(child, name, out)


def loadmat_any(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        mat = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return _flatten_scipy(mat)
    except NotImplementedError:
        pass
    try:
        import mat73

        mat = mat73.loadmat(path)
        return _flatten_scipy(mat)
    except Exception:
        pass
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        _read_hdf5_node(f, "", out)
    return out


def _is_mcos_opaque(value: Any) -> bool:
    arr = np.asarray(value)
    return bool(arr.dtype.names and set(arr.dtype.names) == {"s0", "s1", "s2", "arr"})


def _decode_mcos_field(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode_mcos_field(value.item())
    return str(value)


def read_matlab_table(path: str | Path) -> pd.DataFrame:
    """Read Camargo MATLAB table files saved as MCOS opaque objects.

    Camargo's public release stores most modality files as MATLAB `table`
    objects. SciPy exposes these as old-style MatlabOpaque records with fields
    s0/s1/s2/arr. The mat-io package can convert the table once its FileWrapper
    metadata is initialized; this helper adapts the old opaque layout.
    """

    try:
        from matio.matio5 import load_opaque_object, read_subsystem
        from matio.subsystem import get_matio_context, set_file_wrapper
    except Exception as exc:
        raise ImportError("Install mat-io>=0.4.2 to read Camargo MATLAB table files.") from exc

    mat = scipy_io.loadmat(path, chars_as_strings=True, verify_compressed_data_integrity=True)
    ssdata = mat.pop("__function_workspace__", None)
    if ssdata is None:
        raise ValueError(f"No MATLAB MCOS function workspace found in {path}")
    byte_order = "<" if ssdata[0, 2] == b"I"[0] else ">"
    ss_array = read_subsystem(ssdata, byte_order, False, True)
    mcos = ss_array[0, 0]["MCOS"]
    if not _is_mcos_opaque(mcos) or _decode_mcos_field(mcos[0]["s2"]) != "FileWrapper__":
        raise ValueError(f"Unsupported MCOS FileWrapper layout in {path}")
    fwrap_data = mcos[0]["arr"]
    with get_matio_context():
        file_wrapper = set_file_wrapper()
        file_wrapper.init_load(fwrap_data, byte_order, raw_data=False, add_table_attrs=True)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if not _is_mcos_opaque(value):
                continue
            item = value.flat[0]
            classname = _decode_mcos_field(item["s2"])
            type_system = _decode_mcos_field(item["s1"])
            obj = load_opaque_object(item["arr"], classname, type_system)
            if isinstance(obj, pd.DataFrame):
                return obj
    raise ValueError(f"No MATLAB table object found in {path}")


def _try_read_dataframe(path: Path) -> pd.DataFrame:
    try:
        return read_matlab_table(path)
    except Exception:
        flat = loadmat_any(path)
        arrays = []
        names = []
        for key, value in flat.items():
            arr = _safe_array(value)
            if arr is None:
                continue
            arrays.append(arr)
            names.extend([key] if arr.shape[1] == 1 else [f"{key}[{i}]" for i in range(arr.shape[1])])
        if not arrays:
            raise
        length = min(a.shape[0] for a in arrays)
        data = np.concatenate([a[:length] for a in arrays], axis=1)
        return pd.DataFrame(data, columns=names)


def inspect_mat(path: str | Path, max_items: int = 200) -> None:
    flat = loadmat_any(path)
    rows = []
    for key, value in sorted(flat.items()):
        arr = np.asarray(value)
        text = _decode_bytes(value)
        if text:
            rows.append((key, "str", text[:80]))
        else:
            rows.append((key, str(arr.shape), str(arr.dtype)))
    for key, shape, dtype in rows[:max_items]:
        print(f"{key:80s} {shape:20s} {dtype}")
    if len(rows) > max_items:
        print(f"... {len(rows) - max_items} more fields")


def _match_patterns(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns)


def _collect_named_columns(
    flat: dict[str, Any],
    patterns: Iterable[str],
    exclude_patterns: Iterable[str] = (),
) -> tuple[np.ndarray | None, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    excludes = list(exclude_patterns)
    for key, value in sorted(flat.items()):
        if excludes and _match_patterns(key, excludes):
            continue
        if not _match_patterns(key, patterns):
            continue
        arr = _safe_array(value)
        if arr is None:
            continue
        if arr.shape[1] > 128:
            continue
        arrays.append(arr)
        if arr.shape[1] == 1:
            names.append(key)
        else:
            names.extend([f"{key}[{i}]" for i in range(arr.shape[1])])
    if not arrays:
        return None, []
    length = min(a.shape[0] for a in arrays)
    x = np.concatenate([a[:length] for a in arrays], axis=1)
    return x, names


def _collect_dataframe_columns(
    df: pd.DataFrame,
    patterns: Iterable[str],
    *,
    prefix: str = "",
    exclude_patterns: Iterable[str] = (),
) -> tuple[np.ndarray | None, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    excludes = list(exclude_patterns)
    for col in df.columns:
        col_name = str(col)
        full_name = f"{prefix}{col_name}" if prefix else col_name
        if excludes and _match_patterns(full_name, excludes):
            continue
        if not _match_patterns(full_name, patterns):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        arr = series.to_numpy(dtype=np.float32)[:, None]
        if np.isfinite(arr).sum() == 0:
            continue
        arrays.append(arr.astype(np.float32))
        names.append(full_name)
    if not arrays:
        return None, []
    length = min(a.shape[0] for a in arrays)
    return np.concatenate([a[:length] for a in arrays], axis=1), names


def _find_sample_rate(flat: dict[str, Any], default_hz: float) -> float:
    for key, value in flat.items():
        if re.search(r"(sample|sampling|rate|freq|frequency|fs)\b", key, flags=re.IGNORECASE):
            arr = np.asarray(value).squeeze()
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                hz = float(arr)
                if 10 <= hz <= 5000:
                    return hz
    return default_hz


def _find_mass(flat: dict[str, Any]) -> float | None:
    for key, value in flat.items():
        if re.search(r"(mass|weight)", key, flags=re.IGNORECASE):
            arr = np.asarray(value).squeeze()
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                v = float(arr)
                if 30 <= v <= 200:
                    return v
                if 300 <= v <= 2000:
                    return v / 9.80665
    return None


def _find_camargo_mass(ik_path: Path) -> float | None:
    subject_dir = None
    for parent in ik_path.parents:
        if re.match(r"AB\d+", parent.name, flags=re.IGNORECASE):
            subject_dir = parent
            break
    if subject_dir is None:
        return None
    osim = subject_dir / "osimxml" / f"{subject_dir.name}.osim"
    if not osim.exists():
        return None
    try:
        tree = ET.parse(osim)
    except Exception:
        return None
    masses: list[float] = []
    for node in tree.iter():
        if node.tag.split("}")[-1] != "mass" or node.text is None:
            continue
        try:
            value = float(node.text.strip())
        except ValueError:
            continue
        if value > 0:
            masses.append(value)
    total = float(sum(masses)) if masses else None
    if total is not None and 30 <= total <= 200:
        return total
    return None


def _resample_pair(x: np.ndarray, y: np.ndarray, source_hz: float, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    length = min(x.shape[0], y.shape[0])
    x = x[:length]
    y = y[:length]
    if abs(source_hz - target_hz) < 1e-6:
        return x.astype(np.float32), y.astype(np.float32)
    target_len = int(round(length * target_hz / source_hz))
    if target_len < 2:
        raise ValueError(f"Resampling would create too few samples: {target_len}")
    ratio = Fraction(float(target_hz) / float(source_hz)).limit_denominator(1000)
    x_res = signal.resample_poly(x, up=ratio.numerator, down=ratio.denominator, axis=0)
    y_res = signal.resample_poly(y, up=ratio.numerator, down=ratio.denominator, axis=0)
    target_len = min(target_len, x_res.shape[0], y_res.shape[0])
    return x_res[:target_len].astype(np.float32), y_res[:target_len].astype(np.float32)


def _resample_array(arr: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if abs(source_hz - target_hz) < 1e-6:
        return arr.astype(np.float32)
    target_len = int(round(arr.shape[0] * target_hz / source_hz))
    if target_len < 2:
        raise ValueError(f"Resampling would create too few samples: {target_len}")
    ratio = Fraction(float(target_hz) / float(source_hz)).limit_denominator(1000)
    res = signal.resample_poly(arr, up=ratio.numerator, down=ratio.denominator, axis=0)
    return res[:target_len].astype(np.float32)


def _estimate_hz_from_header(df: pd.DataFrame, default_hz: float) -> float:
    if "Header" not in df.columns or len(df) < 3:
        return default_hz
    t = pd.to_numeric(df["Header"], errors="coerce").to_numpy(dtype=np.float64)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size == 0:
        return default_hz
    hz = 1.0 / float(np.median(diffs))
    if 10 <= hz <= 5000:
        return hz
    return default_hz


def _camargo_modality_path(ik_path: Path, modality: str) -> Path:
    return ik_path.parent.parent / modality / ik_path.name


def _add_joint_velocities(ik_df: pd.DataFrame, sample_hz: float) -> pd.DataFrame:
    out = ik_df.copy()
    for col in list(ik_df.columns):
        if re.search(r"(hip_flexion|knee_angle|ankle_angle)_[rl]$", str(col), flags=re.IGNORECASE):
            values = pd.to_numeric(ik_df[col], errors="coerce").to_numpy(dtype=np.float32)
            out[f"{col}_vel"] = np.gradient(values, 1.0 / sample_hz).astype(np.float32)
    return out


def _load_camargo_grouped_trial(
    ik_path: Path,
    *,
    feature_patterns: Iterable[str],
    target_patterns: dict[str, str],
    target_hz: float,
    data_root: str | Path | None,
    normalize_moments_by_mass: bool,
) -> TrialData:
    ik_df = _try_read_dataframe(ik_path)
    id_path = _camargo_modality_path(ik_path, "id")
    imu_path = _camargo_modality_path(ik_path, "imu")
    if not id_path.exists():
        raise FileNotFoundError(f"Missing matching inverse-dynamics file: {id_path}")
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing matching IMU file: {imu_path}")
    id_df = _try_read_dataframe(id_path)
    imu_df = _try_read_dataframe(imu_path)

    ik_hz = _estimate_hz_from_header(ik_df, target_hz)
    id_hz = _estimate_hz_from_header(id_df, ik_hz)
    imu_hz = _estimate_hz_from_header(imu_df, ik_hz)
    ik_df = _add_joint_velocities(ik_df, ik_hz)

    target_regexes = list(target_patterns.values())
    x_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for prefix, df in [("ik.", ik_df), ("imu.", imu_df)]:
        arr, names = _collect_dataframe_columns(df, feature_patterns, prefix=prefix, exclude_patterns=target_regexes)
        if arr is None:
            continue
        hz = ik_hz if prefix == "ik." else imu_hz
        x_parts.append(finite_or_zero(_resample_array(arr, hz, target_hz)))
        feature_names.extend(names)
    if not x_parts:
        raise ValueError(f"No feature channels matched for grouped Camargo trial {ik_path}")

    targets: list[np.ndarray] = []
    target_names: list[str] = []
    mass_kg = _find_camargo_mass(ik_path)
    for name, pattern in target_patterns.items():
        arr, names = _collect_dataframe_columns(id_df, [pattern], prefix="id.")
        if arr is None:
            raise ValueError(f"No target channel for {name!r} matched pattern {pattern!r} in {id_path}")
        channel = _resample_array(arr[:, :1], id_hz, target_hz)
        if normalize_moments_by_mass:
            channel = normalize_moment_by_mass(channel, mass_kg)
        targets.append(channel)
        target_names.append(names[0] if names else name)
    y = np.concatenate(targets, axis=1)
    length = min([p.shape[0] for p in x_parts] + [y.shape[0]])
    x = np.concatenate([p[:length] for p in x_parts], axis=1)
    y = y[:length]

    root = resolve_path(data_root) if data_root else None
    return TrialData(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        feature_names=feature_names,
        target_names=target_names,
        sample_rate_hz=target_hz,
        participant=_participant_from_path(ik_path, root),
        trial_id=ik_path.stem,
        task=_task_from_path(ik_path),
        side="right",
        mass_kg=mass_kg,
    )


def _participant_from_path(path: Path, data_root: Path | None) -> str:
    if data_root is not None:
        try:
            rel = path.relative_to(data_root)
            return rel.parts[0]
        except ValueError:
            pass
    for part in path.parts[::-1]:
        if re.search(r"(ab|sub|subject|participant)?\d+", part, flags=re.IGNORECASE):
            return part
    return path.parent.name


def _task_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "stair" in text:
        return "stair"
    if "ramp" in text or "incline" in text or "slope" in text:
        return "ramp"
    if "treadmill" in text or "level" in text or "walk" in text or "ground" in text:
        return "cyclic"
    return "unknown"


def load_trial(
    path: str | Path,
    *,
    feature_patterns: Iterable[str],
    target_patterns: dict[str, str],
    target_hz: float = 200.0,
    data_root: str | Path | None = None,
    normalize_moments_by_mass: bool = True,
) -> TrialData:
    path = Path(path)
    if path.parent.name.lower() == "ik" and _camargo_modality_path(path, "id").exists():
        return _load_camargo_grouped_trial(
            path,
            feature_patterns=feature_patterns,
            target_patterns=target_patterns,
            target_hz=target_hz,
            data_root=data_root,
            normalize_moments_by_mass=normalize_moments_by_mass,
        )

    root = resolve_path(data_root) if data_root else None
    flat = loadmat_any(path)
    source_hz = _find_sample_rate(flat, target_hz)
    mass_kg = _find_mass(flat)

    target_regexes = list(target_patterns.values())
    x, feature_names = _collect_named_columns(flat, feature_patterns, exclude_patterns=target_regexes)
    if x is None:
        raise ValueError(
            f"No feature channels matched in {path}. Run `python data/camargo_loader.py --inspect {path}` "
            "and refine data.feature_patterns in the config."
        )

    targets: list[np.ndarray] = []
    target_names: list[str] = []
    for name, pattern in target_patterns.items():
        arr, names = _collect_named_columns(flat, [pattern])
        if arr is None:
            raise ValueError(f"No target channel for {name!r} matched pattern {pattern!r} in {path}")
        channel = arr[:, :1]
        if normalize_moments_by_mass:
            channel = normalize_moment_by_mass(channel, mass_kg)
        targets.append(channel)
        target_names.append(names[0] if names else name)
    y = np.concatenate(targets, axis=1)

    x, y = _resample_pair(x, y, source_hz, target_hz)
    return TrialData(
        x=x,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        sample_rate_hz=target_hz,
        participant=_participant_from_path(path, root),
        trial_id=path.stem,
        task=_task_from_path(path),
        side=infer_side(path.as_posix()),
        mass_kg=mass_kg,
    )


def discover_trials(
    data_root: str | Path,
    participant_glob: str = "*",
    trial_glob: str = "**/*.mat",
    max_subjects: int | None = None,
    exclude_path_patterns: Iterable[str] = (),
) -> list[Path]:
    root = resolve_path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    subject_dirs = sorted([p for p in root.glob(participant_glob) if p.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[: int(max_subjects)]
    files: list[Path] = []
    if trial_glob in {"**/ik/*.mat", "**/ik/*.MAT", "**/ik/*.[mM][aA][tT]"}:
        search_roots = subject_dirs if subject_dirs else [root]
        for subject in search_roots:
            for ik_dir in subject.rglob("ik"):
                files.extend(sorted(p for p in ik_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mat"))
    if not files:
        for subject in subject_dirs:
            files.extend(sorted(subject.glob(trial_glob)))
    if not files:
        files = sorted(root.glob(trial_glob))
    excludes = compile_patterns(exclude_path_patterns)
    if excludes:
        files = [p for p in files if not any(pattern.search(p.as_posix()) for pattern in excludes)]
    return sorted(set(files))


def load_trials_from_config(config: dict[str, Any]) -> list[TrialData]:
    data_cfg = config["data"]
    trial_paths = discover_trials(
        data_cfg["root"],
        participant_glob=data_cfg.get("participant_glob", "*"),
        trial_glob=data_cfg.get("trial_glob", "**/*.mat"),
        max_subjects=data_cfg.get("max_subjects"),
        exclude_path_patterns=data_cfg.get("exclude_path_patterns", []),
    )
    trials = []
    for path in trial_paths:
        try:
            trials.append(
                load_trial(
                    path,
                    feature_patterns=data_cfg["feature_patterns"],
                    target_patterns=data_cfg["target_patterns"],
                    target_hz=float(data_cfg.get("sample_rate_hz", 200)),
                    data_root=data_cfg["root"],
                )
            )
        except Exception as exc:
            print(f"[warn] skipping {path}: {exc}", file=sys.stderr)
    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=str, help="Print the field tree for one Camargo .mat trial.")
    args = parser.parse_args()
    if args.inspect:
        inspect_mat(args.inspect)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

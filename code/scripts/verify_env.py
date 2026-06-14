"""Environment verification for Aurora-DA.

Single-shot diagnostic that confirms the host has everything we need to run
the project. Prints one line per check with [ OK ] / [WARN] / [FAIL] prefix
and exits 0 iff zero FAILs (WARNs are tolerated).

Run:
    uv run python scripts/verify_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Small helpers for uniform output
# ---------------------------------------------------------------------------

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

_results: list[tuple[str, str, str]] = []  # (status, label, detail)


def _record(status: str, label: str, detail: str) -> None:
    _results.append((status, label, detail))
    print(f"{status} {label:<22} {detail}")


def _summary_and_exit() -> None:
    n_ok = sum(1 for s, _, _ in _results if s == OK)
    n_warn = sum(1 for s, _, _ in _results if s == WARN)
    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL")
    sys.exit(0 if n_fail == 0 else 1)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 11) <= (v.major, v.minor) < (3, 13):
        _record(OK, "python version", detail)
    else:
        _record(FAIL, "python version", f"{detail} (need 3.11.x or 3.12.x)")


def check_torch() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _record(FAIL, "torch", f"import failed: {exc}")
        return

    version = torch.__version__
    # Compare on the major.minor part only (avoid string ordering issues).
    major_minor = tuple(int(x) for x in version.split("+")[0].split(".")[:2])
    if major_minor >= (2, 3):
        _record(OK, "torch", version)
    else:
        _record(FAIL, "torch", f"{version} (need >= 2.3)")


def check_cuda() -> None:
    try:
        import torch
    except Exception:
        _record(FAIL, "cuda available", "torch import failed earlier")
        return

    if torch.cuda.is_available():
        _record(OK, "cuda available", "True")
    else:
        _record(FAIL, "cuda available", "False")


def check_gpu_identity() -> None:
    try:
        import torch
    except Exception:
        _record(FAIL, "gpu", "torch import failed earlier")
        return

    if not torch.cuda.is_available():
        _record(FAIL, "gpu", "CUDA not available")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _record(OK, "gpu", f"{name} (cap {cap[0]}.{cap[1]}, {total_gib:.2f} GiB)")


def check_cuda_runtime() -> None:
    try:
        import torch
    except Exception:
        _record(FAIL, "cuda runtime", "torch import failed earlier")
        return

    runtime = torch.version.cuda
    if runtime:
        _record(OK, "cuda runtime", runtime)
    else:
        _record(FAIL, "cuda runtime", "torch has no CUDA build")


def check_bf16_native() -> None:
    try:
        import torch
    except Exception:
        _record(FAIL, "bf16 native", "torch import failed earlier")
        return

    if not torch.cuda.is_available():
        _record(WARN, "bf16 native", "no GPU to test")
        return

    if torch.cuda.is_bf16_supported():
        _record(OK, "bf16 native", "supported")
    else:
        # FP16 autocast fallback is fine for our use case; not a hard failure.
        _record(WARN, "bf16 native", "not supported; FP16 fallback will be used")


def check_aurora_import() -> None:
    """Import only; do NOT instantiate the model (avoids downloading the checkpoint)."""
    try:
        from aurora import AuroraPretrained  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _record(FAIL, "aurora import", f"failed: {exc}")
        return
    _record(OK, "aurora import", "AuroraPretrained")


def check_data_stack_imports() -> None:
    try:
        import gcsfs  # noqa: F401
        import xarray  # noqa: F401
        import zarr  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _record(FAIL, "zarr/xarray/gcsfs", f"failed: {exc}")
        return

    import xarray as _xr
    import zarr as _zr
    import gcsfs as _gc
    _record(OK, "zarr/xarray/gcsfs", f"{_zr.__version__} / {_xr.__version__} / {_gc.__version__}")


def check_config_readable() -> None:
    cfg_path = Path("configs/data.yaml")
    if not cfg_path.exists():
        _record(FAIL, "config readable", f"{cfg_path} not found")
        return

    try:
        import yaml
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _record(FAIL, "config readable", f"yaml parse error: {exc}")
        return

    required_keys = {
        "evaluation", "aurora", "variables", "observations", "paths", "era5_source",
    }
    missing = required_keys - set(cfg.keys())
    if missing:
        _record(FAIL, "config readable", f"missing top-level keys: {sorted(missing)}")
        return
    _record(OK, "config readable", str(cfg_path))


def check_data_dirs() -> None:
    try:
        import yaml
        with open("configs/data.yaml") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _record(FAIL, "data dirs", f"could not read config: {exc}")
        return

    paths_to_check = [
        cfg["paths"]["era5_ic_dir"],
        cfg["paths"]["background_dir"],
    ]
    problems = []
    for p in paths_to_check:
        path = Path(p)
        if not path.exists():
            problems.append(f"{p} does not exist")
        elif not os.access(p, os.W_OK):
            problems.append(f"{p} not writable")
    if problems:
        _record(FAIL, "data dirs", "; ".join(problems))
    else:
        data_root = cfg["paths"]["data_root"]
        _record(OK, "data dirs", f"{data_root} ok")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    check_python_version()
    check_torch()
    check_cuda()
    check_gpu_identity()
    check_cuda_runtime()
    check_bf16_native()
    check_aurora_import()
    check_data_stack_imports()
    check_config_readable()
    check_data_dirs()
    _summary_and_exit()


if __name__ == "__main__":
    main()
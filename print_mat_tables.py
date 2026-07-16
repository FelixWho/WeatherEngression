#!/usr/bin/env python
"""Print all top-level variables ("tables") in the ENA weather .mat file.

``weather_data.mat`` is a MATLAB v7.3 file, which is HDF5 underneath, so we read
it with ``h5py``. This lists each top-level variable with its shape, dtype, and
MATLAB class -- WITHOUT loading the full data. MATLAB structs (HDF5 groups) are
expanded to show their members, and the ``var_name`` cell of strings is decoded.

Usage:
    .venv/bin/python print_mat_tables.py [--mat-path PATH]
"""

from __future__ import annotations

import argparse

DEFAULT_MAT_PATH = "/storage3/fs1/myu/Active/felixhu/weather_data.mat"


def _decode_matlab_string(f, ref) -> str:
    """Decode a MATLAB char array (referenced by ``ref``) into a Python string."""
    import numpy as np

    codes = np.asarray(f[ref][()]).flatten()
    return "".join(chr(int(c)) for c in codes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", default=DEFAULT_MAT_PATH)
    args = parser.parse_args()

    try:
        import h5py
    except ModuleNotFoundError:
        raise SystemExit(
            "needs h5py (in requirements.txt): .venv/bin/pip install -r requirements.txt"
        )

    with h5py.File(args.mat_path, "r") as f:
        print(f"MAT file: {args.mat_path}\n")
        header = f"{'variable':14s} {'kind':7s} {'shape':>18s}  dtype / members"
        print(header)
        print("-" * len(header))
        for name in f:
            if name == "#refs#":
                continue  # MATLAB internal reference store, not a user variable
            item = f[name]
            cls = item.attrs.get("MATLAB_class", b"")
            cls = cls.decode() if isinstance(cls, (bytes, bytearray)) else str(cls)
            if isinstance(item, h5py.Group):  # MATLAB struct
                members = ", ".join(item.keys())
                print(f"{name:14s} {'struct':7s} {'':>18s}  members: {members}")
            else:
                print(f"{name:14s} {cls or 'array':7s} {str(item.shape):>18s}  {item.dtype}")

        # Bonus: decode the variable names (the feature list) if present.
        if "var_name" in f:
            vn = f["var_name"]
            names = [_decode_matlab_string(f, vn[0, i]) for i in range(vn.shape[1])]
            print("\nvar_name (decoded feature list):")
            print("  " + ", ".join(names))


if __name__ == "__main__":
    main()

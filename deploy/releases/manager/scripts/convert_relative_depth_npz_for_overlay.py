#!/usr/bin/env python3
"""Convert a relative inverse-depth NPZ to the `depth` key expected by overlay rendering."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    z = np.load(args.input, allow_pickle=True)
    if "depth" in z:
        depth = z["depth"]
    elif "relative_inverse_depth" in z:
        rel = np.asarray(z["relative_inverse_depth"], dtype=np.float32)
        valid = np.isfinite(rel) & (rel > 0)
        depth = np.zeros_like(rel, dtype=np.float32)
        depth[valid] = 1.0 / np.maximum(rel[valid], 1e-6)
    else:
        raise RuntimeError(f"no depth or relative_inverse_depth key in {args.input}")
    frame_idx = z["frame_idx"] if "frame_idx" in z else np.arange(depth.shape[0], dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, depth=depth.astype(np.float32), frame_idx=frame_idx.astype(np.int32), source_npz=str(args.input), value_semantics="converted_relative_inverse_depth_for_overlay_only")


if __name__ == "__main__":
    main()

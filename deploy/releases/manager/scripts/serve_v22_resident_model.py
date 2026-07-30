#!/usr/bin/env python3
"""Launch one long-lived resident API service for one V22 model family."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.model_adapters import HaWoRAdapter, UniDepthAdapter, VGGTAdapter, WiLoRAdapter
from services.resident_model_service import serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unidepth", "wilor", "hawor", "vggt"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=20.0)
    parser.add_argument("--pending-limit", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-batch-cap", type=int, default=None)
    parser.add_argument("--unidepth-repo", type=Path, default=REPO_ROOT / "third_party" / "algorithms" / "unidepth")
    parser.add_argument("--unidepth-model", type=Path, default=Path("/home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected"))
    parser.add_argument("--wilor-root", type=Path, default=REPO_ROOT / "third_party" / "algorithms" / "wilor")
    parser.add_argument("--hawor-root", type=Path, default=REPO_ROOT / "third_party" / "algorithms" / "hawor")
    parser.add_argument("--hawor-checkpoint", type=Path, default=Path("/home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt"))
    parser.add_argument("--hawor-model-config", type=Path, default=Path("/home/zjh/ego-annation-checkpoints/hawor/model_config.yaml"))
    parser.add_argument("--mano-root", type=Path, default=Path("/home/zjh/ego-annation-checkpoints/mano"))
    parser.add_argument("--hawor-detector-checkpoint", type=Path, default=Path("/home/zjh/ego-annation-checkpoints/wilor/detector.pt"))
    parser.add_argument("--vggt-repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--vggt-checkpoint", type=Path, default=Path("/home/zjh/ego_annotation_checkpoint/vggt/model.pt"))
    parser.add_argument("--sequence-length", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "unidepth":
        adapter = UniDepthAdapter(repo=args.unidepth_repo, model_dir=args.unidepth_model, device=args.device, native_batch_cap=int(args.native_batch_cap or 32))
    elif args.model == "wilor":
        adapter = WiLoRAdapter(wilor_root=args.wilor_root, device=args.device, crop_batch_cap=int(args.native_batch_cap or 128))
    elif args.model == "hawor":
        adapter = HaWoRAdapter(hawor_root=args.hawor_root, checkpoint=args.hawor_checkpoint, model_config=args.hawor_model_config, mano_root=args.mano_root, detector_checkpoint=args.hawor_detector_checkpoint, device=args.device, native_batch_cap=int(args.native_batch_cap or 8))
    else:
        adapter = VGGTAdapter(repo_root=args.vggt_repo_root, checkpoint=args.vggt_checkpoint, device=args.device, native_batch_cap=int(args.native_batch_cap or 2), sequence_length=args.sequence_length)
    serve(adapter, host=args.host, port=args.port, artifact_root=args.artifact_root, wait_s=args.wait_s, pending_limit=args.pending_limit)


if __name__ == "__main__":
    main()

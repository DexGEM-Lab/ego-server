from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.check_vggt_camera_backend_readiness import evaluate, parse_args


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def environment_probe(*, symbols_ok: bool = True, checkpoint_ok: bool = True, checkpoint_requested: bool = True) -> dict:
    imports = {
        "torch": {"imported": True, "module": "torch"},
        "vggt.models.vggt.VGGT": {"imported": symbols_ok, "module": "vggt.models.vggt" if symbols_ok else None},
        "vggt.utils.pose_enc.pose_encoding_to_extri_intri": {"imported": symbols_ok, "module": "vggt.utils.pose_enc" if symbols_ok else None},
    }
    checkpoint_load = {"status": "not_requested"}
    if checkpoint_requested:
        checkpoint_load = {"status": "ok", "type": "dict", "top_level_key_count": 1} if checkpoint_ok else {"status": "failed", "error": "invalid load key"}
    status = "ok" if symbols_ok and checkpoint_load["status"] != "failed" else "failed"
    return {
        "status": status,
        "python": sys.executable,
        "third_party_hint": "/tmp/third_party/vggt",
        "imports": imports,
        "checkpoint_load": checkpoint_load,
    }


def args_for(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return parse_args(["--backend", "vggt", "--repo-root", str(Path.cwd()), "--python", sys.executable, *extra])


def test_readiness_probe_imports_exact_symbols_and_loads_checkpoint_without_real_torch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    vggt_root = repo_root / "third_party" / "vggt"
    write_text(vggt_root / "torch.py", "def load(path, map_location=None):\n    return {'path': path, 'map_location': map_location}\n")
    write_text(vggt_root / "vggt" / "__init__.py", "")
    write_text(vggt_root / "vggt" / "models" / "__init__.py", "")
    write_text(vggt_root / "vggt" / "models" / "vggt.py", "class VGGT:\n    pass\n")
    write_text(vggt_root / "vggt" / "utils" / "__init__.py", "")
    write_text(
        vggt_root / "vggt" / "utils" / "pose_enc.py",
        "def pose_encoding_to_extri_intri(*args, **kwargs):\n    return None\n",
    )
    for rel in [
        "scripts/run_v22_resident_vggt_camera_batch.py",
        "scripts/run_v22_camera_trajectory_stage.py",
        "scripts/v22_model_request_helpers.py",
    ]:
        write_text(repo_root / rel, "# readiness fixture\n")
    checkpoint = tmp_path / "vggt.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")

    report = evaluate(parse_args(["--backend", "vggt", "--repo-root", str(repo_root), "--python", sys.executable, "--checkpoint", str(checkpoint)]))

    assert report["status"] == "ok"
    assert report["blocked_reasons"] == []
    assert report["python_probe"]["imports"]["vggt.models.vggt.VGGT"]["imported"] is True
    assert report["python_probe"]["imports"]["vggt.utils.pose_enc.pose_encoding_to_extri_intri"]["imported"] is True
    assert report["python_probe"]["checkpoint_load"]["status"] == "ok"
    assert report["python_probe"]["checkpoint_load"]["type"] == "dict"


def test_readiness_ok_with_loadable_checkpoint_and_exact_symbols(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vggt.pt"
    checkpoint.write_bytes(b"unit-test-placeholder")
    with patch("scripts.check_vggt_camera_backend_readiness.probe_python_environment", return_value=environment_probe()):
        report = evaluate(args_for(tmp_path, "--checkpoint", str(checkpoint)))

    assert report["status"] == "ok"
    assert report["checkpoint"]["status"] == "checkpoint_present"
    assert report["python_probe"]["checkpoint_load"]["status"] == "ok"
    assert report["blocked_reasons"] == []


def test_readiness_blocks_unimportable_exact_symbols_and_missing_checkpoint(tmp_path: Path) -> None:
    with patch("scripts.check_vggt_camera_backend_readiness.probe_python_environment", return_value=environment_probe(symbols_ok=False, checkpoint_requested=False)):
        report = evaluate(args_for(tmp_path))

    assert report["status"] == "blocked"
    assert "missing_or_unimportable_symbol:vggt.models.vggt.VGGT" in report["blocked_reasons"]
    assert "missing_or_unimportable_symbol:vggt.utils.pose_enc.pose_encoding_to_extri_intri" in report["blocked_reasons"]
    assert "blocked_missing_checkpoint_or_download_permission" in report["blocked_reasons"]


def test_readiness_blocks_existing_but_unloadable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vggt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    with patch("scripts.check_vggt_camera_backend_readiness.probe_python_environment", return_value=environment_probe(checkpoint_ok=False)):
        report = evaluate(args_for(tmp_path, "--checkpoint", str(checkpoint)))

    assert report["status"] == "blocked"
    assert "checkpoint_not_torch_loadable" in report["blocked_reasons"]


def test_readiness_allows_remote_download_only_when_explicit(tmp_path: Path) -> None:
    with patch("scripts.check_vggt_camera_backend_readiness.probe_python_environment", return_value=environment_probe(checkpoint_requested=False)):
        report = evaluate(args_for(tmp_path, "--allow-remote-model-download"))

    assert report["status"] == "ok"
    assert report["checkpoint"]["status"] == "remote_download_explicitly_allowed"
    assert report["checkpoint"]["network_or_cache_required"] is True

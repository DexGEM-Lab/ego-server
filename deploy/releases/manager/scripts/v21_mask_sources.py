from __future__ import annotations

from pathlib import Path


class MaskSourceError(RuntimeError):
    pass


def resolve_current_object_mask_dir(run_root: Path, object_id: str) -> Path:
    """Return the active V21 object mask directory from OWLv2 bbox-prompt SAM2."""
    candidate = run_root / "measurements" / "object_tracks" / "sam2_proper" / object_id / "sam2_masks"
    if candidate.exists() and candidate.is_dir() and any(candidate.glob("*.png")):
        return candidate
    raise MaskSourceError(
        "current_object_mask_dir_missing: expected active V21 sam2_proper OWLv2 bbox-prompt masks. searched="
        + str(candidate)
    )

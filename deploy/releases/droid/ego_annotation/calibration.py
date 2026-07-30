"""Calibration resolver with explicit uncertainty and no silent intrinsics fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    intrinsics_fx_fy_cx_cy: list[float] | None
    distortion: dict[str, Any]
    rectification: dict[str, Any]
    source: str
    uncertainty: dict[str, Any]
    errors: list[dict[str, Any]]
    provenance: list[dict[str, Any]]

    def to_contract(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "intrinsics_fx_fy_cx_cy": self.intrinsics_fx_fy_cx_cy,
            "distortion": self.distortion,
            "rectification": self.rectification,
            "source": self.source,
            "uncertainty": self.uncertainty,
            "coordinate_frame": "image_px",
            "contract": "one canonical camera model per clip/session; all consumers must cite this contract",
        }


def _finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out > 0 and out < float("inf"):
        return out
    return None


def _parse_k(calibration: dict[str, Any]) -> list[float] | None:
    if "intrinsics_fx_fy_cx_cy" in calibration:
        values = calibration["intrinsics_fx_fy_cx_cy"]
        if isinstance(values, (list, tuple)) and len(values) == 4:
            parsed = [_finite_positive(values[0]), _finite_positive(values[1]), _finite_positive(values[2]), _finite_positive(values[3])]
            if all(v is not None for v in parsed):
                return [float(v) for v in parsed if v is not None]
    if "K" in calibration:
        matrix = calibration["K"]
        if isinstance(matrix, (list, tuple)) and len(matrix) == 3:
            try:
                fx = _finite_positive(matrix[0][0])
                fy = _finite_positive(matrix[1][1])
                cx = _finite_positive(matrix[0][2])
                cy = _finite_positive(matrix[1][2])
            except (TypeError, IndexError):
                return None
            if fx is not None and fy is not None and cx is not None and cy is not None:
                return [float(fx), float(fy), float(cx), float(cy)]
    return None


def resolve_calibration(
    calibration: dict[str, Any],
    *,
    media_width: int | None,
    media_height: int | None,
    allow_estimated: bool,
) -> CalibrationResult:
    errors: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    k = _parse_k(calibration)
    source = str(calibration.get("source") or calibration.get("intrinsics_source") or "customer_metadata")
    distortion = calibration.get("distortion") if isinstance(calibration.get("distortion"), dict) else {}
    rectification = calibration.get("rectification") if isinstance(calibration.get("rectification"), dict) else {"model": "identity"}
    if k is not None:
        provenance.append(
            {
                "stage": "calibration_resolver",
                "event": "canonical_intrinsics_from_metadata",
                "source": source,
                "intrinsics_fx_fy_cx_cy": k,
            }
        )
        uncertainty = dict(calibration.get("uncertainty") or {}) if isinstance(calibration.get("uncertainty"), dict) else {}
        uncertainty.setdefault("intrinsics_px_std", 0.0)
        uncertainty.setdefault("scale_gauge", "provided_metric_camera_contract")
        return CalibrationResult(
            status="resolved",
            intrinsics_fx_fy_cx_cy=k,
            distortion=distortion,
            rectification=rectification,
            source=source,
            uncertainty=uncertainty,
            errors=errors,
            provenance=provenance,
        )

    if allow_estimated and media_width and media_height and media_width > 0 and media_height > 0:
        focal = float(max(media_width, media_height))
        estimated = [focal, focal, float(media_width) / 2.0, float(media_height) / 2.0]
        errors.append(
            {
                "code": "calibration_estimated_from_image_size",
                "severity": "degraded",
                "message": "No supplied camera calibration; estimated pinhole K from frame dimensions for projection bookkeeping only.",
                "mechanism": "image-center focal heuristic cannot establish fixed metric gauge or distortion.",
            }
        )
        provenance.append(
            {
                "stage": "calibration_resolver",
                "event": "estimated_intrinsics_low_confidence",
                "source": "image_size_heuristic",
                "intrinsics_fx_fy_cx_cy": estimated,
            }
        )
        return CalibrationResult(
            status="estimated_low_confidence",
            intrinsics_fx_fy_cx_cy=estimated,
            distortion={"model": "unknown", "coefficients": []},
            rectification={"model": "identity", "validity": "unverified"},
            source="image_size_heuristic",
            uncertainty={
                "intrinsics_px_std": float(max(media_width, media_height) * 0.25),
                "scale_gauge": "unresolved_without_metric_calibration_or_pose_metadata",
                "use_for_metric_error": False,
            },
            errors=errors,
            provenance=provenance,
        )

    errors.append(
        {
            "code": "calibration_unresolved",
            "severity": "error",
            "message": "No canonical camera intrinsics could be resolved for this clip/session.",
            "mechanism": "Calibration resolver requires supplied K or enough image metadata to make an explicit low-confidence estimate; silent fallback intrinsics are invalid.",
        }
    )
    provenance.append({"stage": "calibration_resolver", "event": "unresolved", "source": "none"})
    return CalibrationResult(
        status="unresolved",
        intrinsics_fx_fy_cx_cy=None,
        distortion={},
        rectification={},
        source="none",
        uncertainty={"scale_gauge": "unresolved", "use_for_metric_error": False},
        errors=errors,
        provenance=provenance,
    )

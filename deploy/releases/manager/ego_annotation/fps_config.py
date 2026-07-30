"""Built-in UniDepth/DROID sampling configurations for the API-Ify manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class FpsCondition:
    name: str
    unidepth_fps: float | None
    droid_fps: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FPS condition name is required")
        for field_name, value in (("unidepth_fps", self.unidepth_fps), ("droid_fps", self.droid_fps)):
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when configured")


FPS_CONDITIONS: Final[tuple[FpsCondition, ...]] = (
    FpsCondition("unidepth_full__droid_full", None, None),
    FpsCondition("unidepth_10fps__droid_full", 10.0, None),
    FpsCondition("unidepth_full__droid_10fps", None, 10.0),
    FpsCondition("unidepth_10fps__droid_10fps", 10.0, 10.0),
    FpsCondition("unidepth_15fps__droid_full", 15.0, None),
    FpsCondition("unidepth_full__droid_15fps", None, 15.0),
    FpsCondition("unidepth_15fps__droid_15fps", 15.0, 15.0),
    FpsCondition("unidepth_20fps__droid_full", 20.0, None),
    FpsCondition("unidepth_full__droid_20fps", None, 20.0),
    FpsCondition("unidepth_20fps__droid_20fps", 20.0, 20.0),
)

FPS_CONDITION_BY_NAME: Final[dict[str, FpsCondition]] = {condition.name: condition for condition in FPS_CONDITIONS}
DEFAULT_FPS_CONDITION: Final[str] = "unidepth_full__droid_full"


def get_fps_condition(name: str) -> FpsCondition:
    try:
        return FPS_CONDITION_BY_NAME[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown built-in FPS condition {name!r}; choose one of {sorted(FPS_CONDITION_BY_NAME)}") from exc


__all__ = ["DEFAULT_FPS_CONDITION", "FPS_CONDITIONS", "FPS_CONDITION_BY_NAME", "FpsCondition", "get_fps_condition"]

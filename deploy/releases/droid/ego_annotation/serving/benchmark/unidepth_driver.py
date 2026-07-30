"""Backward-compatible import wrapper for the generic detached experiment driver."""
from ego_annotation.serving.benchmark.experiment_driver import _load_application, main, run_driver

__all__ = ["_load_application", "main", "run_driver"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

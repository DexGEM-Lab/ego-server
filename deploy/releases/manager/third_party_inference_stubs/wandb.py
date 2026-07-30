"""Minimal WandB inference stub for third-party model imports.

UniDepth imports wandb from a training visualization module during model
construction. V22 inference does not use training logging, so this stub fails
loudly if logging is actually invoked while allowing inference-only imports.
"""


class Image:  # noqa: D101
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def log(*args, **kwargs):
    raise RuntimeError("wandb logging is unavailable in the V22 inference stub")

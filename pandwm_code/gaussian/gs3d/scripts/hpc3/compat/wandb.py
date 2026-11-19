"""Evaluation-only compatibility shim for the optional wandb dependency."""

run = None


def log(*args, **kwargs):
    """No-op: GS evaluation never emits training logs."""

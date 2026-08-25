"""Reproducible partially observable 3-D capture-radius pursuit benchmark."""

__all__ = ["CaptureRadiusPursuit3DEnv"]


def __getattr__(name: str):
    if name == "CaptureRadiusPursuit3DEnv":
        from .pursuit_env import CaptureRadiusPursuit3DEnv

        return CaptureRadiusPursuit3DEnv
    raise AttributeError(name)

"""IceCache package.

Runtime adapters depend on the native DCI/CUDA stack, so they are loaded only
when requested. This keeps metadata and control-plane modules importable in
CPU-only analysis environments while preserving ``from icecache import
adapter``.
"""

from importlib import import_module

__all__ = ["adapter"]


def __getattr__(name: str):
    if name == "adapter":
        module = import_module(".adapter", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

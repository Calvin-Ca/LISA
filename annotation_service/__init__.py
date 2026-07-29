"""Construction-safety annotation service.

The API process intentionally does not import or load GroundingDINO, SAM,
Qwen, or LISA model weights. Model execution is delegated to independent GPU
worker processes.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)

"""Rasterizer backend selection for mesh projection workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RasterizerConfig:
    backend: str = "nvdiffrast"
    fallback_backend: str = "pytorch3d"


class RasterizerBackend:
    """Backend descriptor for mesh rasterization.

    This round provides backend wiring and clear fallback semantics.
    """

    def __init__(self, config: RasterizerConfig) -> None:
        self.config = config

    def describe(self) -> str:
        return (
            f"primary={self.config.backend}, "
            f"fallback={self.config.fallback_backend}"
        )

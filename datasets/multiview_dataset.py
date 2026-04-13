"""Multi-view or multi-frame dataset placeholder implementation."""

from __future__ import annotations

from typing import Any, Dict

from .base_dataset import BaseAvatarDataset


class MultiViewAvatarDataset(BaseAvatarDataset):
    """Placeholder dataset for sparse multi-view / multi-frame head inputs."""

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError("TODO: implement according to confirmed data fields.")

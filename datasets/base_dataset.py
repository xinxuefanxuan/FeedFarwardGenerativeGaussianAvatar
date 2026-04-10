"""Dataset abstractions for confirmed FastAvatar-style schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import Dataset


class BaseAvatarDataset(Dataset, ABC):
    """Base dataset with explicit schema contract.

    The primary supported schema is the processed FastAvatar-style layout.
    """

    def __init__(self, split: str = "train") -> None:
        self.split = split

    @abstractmethod
    def __len__(self) -> int:
        """Return number of frames/samples in this dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one parsed sample dictionary.

        Required keys for current geometry/debug workflow:
        - rgb
        - mask
        - intrinsics
        - transform_matrix
        - flame_params
        - frame_meta
        """

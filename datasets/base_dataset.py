"""Base dataset abstractions."""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Dataset


class BaseAvatarDataset(Dataset):
    """Minimal dataset base class for avatar research.

    TODO: Confirm exact sample dictionary schema with real data protocol.
    """

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one sample.

        Expected keys are intentionally left unresolved until data contract is confirmed.
        """
        raise NotImplementedError("TODO: implement sample parsing after data schema confirmation.")

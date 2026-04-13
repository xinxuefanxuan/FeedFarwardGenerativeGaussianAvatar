"""Evaluation entry point (placeholder)."""

from __future__ import annotations

import argparse

import torch

from evaluation.metrics import psnr
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    _ = load_yaml(args.config)
    pred = torch.zeros(1, 3, 64, 64)
    target = torch.ones(1, 3, 64, 64)
    print("PSNR placeholder:", float(psnr(pred, target)))


if __name__ == "__main__":
    main()

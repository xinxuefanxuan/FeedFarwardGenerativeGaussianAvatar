"""Entry script for Stage 2 skeleton training."""

from __future__ import annotations

import argparse

import torch

from models.stage2_gaussian.stage2_pipeline import Stage2GaussianAvatar
from trainers.stage2_trainer import Stage2Trainer
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _ = cfg

    model = Stage2GaussianAvatar(in_dim=224)
    trainer = Stage2Trainer(model)

    dummy_stage1 = {"canonical_uv": torch.randn(2, 128, 224)}
    out = trainer.training_step(dummy_stage1)
    print("Stage2 skeleton run OK:", out.keys())


if __name__ == "__main__":
    main()

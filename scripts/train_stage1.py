"""Entry script for Stage 1 skeleton training."""

from __future__ import annotations

import argparse

import torch

from models.stage1_prior.stage1_pipeline import Stage1CanonicalPrior
from trainers.stage1_trainer import Stage1Trainer
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _ = cfg

    model = Stage1CanonicalPrior()
    trainer = Stage1Trainer(model)

    dummy = {"images": torch.randn(2, 3, 224, 224)}
    out = trainer.training_step(dummy)
    print("Stage1 skeleton run OK:", out.keys())


if __name__ == "__main__":
    main()

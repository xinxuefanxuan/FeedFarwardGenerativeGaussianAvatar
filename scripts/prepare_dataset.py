"""Dataset preparation entry point (placeholder)."""

from __future__ import annotations

import argparse

from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print("Dataset preparation placeholder loaded config keys:", list(cfg.keys()))
    print("TODO: implement dataset conversion/validation once schema is confirmed.")


if __name__ == "__main__":
    main()

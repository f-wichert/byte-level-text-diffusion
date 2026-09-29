import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_train import train_diffusion
from src.utils import load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to diffusion YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_diffusion(load_diffusion_config(args.config))
    print(f"final_step={result.final_step}")
    print(f"trainable_parameters={result.trainable_parameters}")
    print(f"checkpoints={result.checkpoint_paths}")
    print(f"validation_losses={result.validation_losses}")
    print(f"logger_url={result.logger_url}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train
from src.utils import load_phase1_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to phase1 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(load_phase1_config(args.config))
    print(f"final_step={result.final_step}")
    print(f"trainable_parameters={result.trainable_parameters}")
    print(f"checkpoints={result.checkpoint_paths}")
    print(f"validation_losses={result.validation_losses}")
    print(f"latent_shuffle={result.latent_shuffle}")
    print(f"logger_url={result.logger_url}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid occasional native-extension shutdown aborts after W&B/datasets cleanup.
    os._exit(0)

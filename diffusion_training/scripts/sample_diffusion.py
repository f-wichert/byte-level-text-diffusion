import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sampling import (  # noqa: F401  (re-exports for probe scripts)
    _clip_boundaries,
    decode_latents,
    generate_bytes,
    sample_latents,
    sample_latents_conditional,
)


def main() -> None:
    print("note: sample_diffusion.py now delegates to scripts/generate.py", file=sys.stderr)
    from scripts.generate import main as generate_main

    generate_main()


if __name__ == "__main__":
    main()

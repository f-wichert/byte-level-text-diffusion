import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TextCollator
from src.model import DiffusionAutoencoder
from src.train import _dtype_from_name, move_batch_to_device
from src.utils import load_phase1_config


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="phase1.yaml", help="Phase 1 YAML config (bare name resolves against configs/training/).")
    parser.add_argument("--checkpoint", required=True, help="Path to a Phase 1 checkpoint .pt file.")
    parser.add_argument("--text", help="Text to reconstruct. If omitted, stdin is used.")
    parser.add_argument("--device", help="Override device from config, for example cuda:0.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download/check remote model files. By default only cached files are used.",
    )
    parser.add_argument(
        "--show-bytes",
        action="store_true",
        help="Print raw predicted byte ids in addition to decoded text.",
    )
    parser.add_argument(
        "--free-run",
        action="store_true",
        help=(
            "Disable teacher forcing for reconstruction: seed with the first byte, then feed each "
            "predicted byte back into the decoder. The original text is still encoded once to provide latents."
        ),
    )
    return parser.parse_args()


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    text = sys.stdin.read()
    if not text:
        raise SystemExit("No input text provided. Pass --text or pipe text on stdin.")
    return text


def _clip_boundaries(boundaries: torch.Tensor, length: int) -> torch.Tensor:
    clipped = boundaries[boundaries < length]
    if clipped.numel() == 0:
        clipped = boundaries[:1]
    if int(clipped[-1].item()) != length:
        clipped = torch.cat([clipped, clipped.new_tensor([length])])
    return clipped


def _teacher_forced_reconstruction(logits: torch.Tensor, original_ids: list[int]) -> list[int]:
    predicted_next_ids = logits[: max(0, len(original_ids) - 1)].argmax(dim=-1).tolist()
    return original_ids[:1] + predicted_next_ids


@torch.no_grad()
def _free_running_reconstruction(
    model: DiffusionAutoencoder,
    original_ids: list[int],
    full_z_words: torch.Tensor,
    full_boundaries: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> list[int]:
    if not original_ids:
        return []

    generated = [original_ids[0]]
    target_length = min(len(original_ids), sequence_length)
    while len(generated) < target_length:
        current_length = len(generated)
        byte_ids = torch.zeros((1, current_length), dtype=torch.long, device=device)
        byte_ids[0] = torch.tensor(generated, dtype=torch.long, device=device)

        boundaries = _clip_boundaries(full_boundaries, current_length).to(device=device, dtype=torch.int32)
        word_count = boundaries.numel() - 1
        z_words = full_z_words[:, :word_count, :]

        logits = model.decode(byte_ids, [z_words], [boundaries])
        next_id = int(logits[0, current_length - 1].argmax(dim=-1).item())
        generated.append(next_id)

    return generated


@torch.no_grad()
def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    log(f"Loading config: {args.config}")
    config = load_phase1_config(args.config)
    device = torch.device(args.device or config.training.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "This project model cannot run on CPU because the vendored HAT modules allocate CUDA tensors."
        )

    log(f"Building model on {device} from {config.model.pretrained_model_id}")
    model = DiffusionAutoencoder.from_pretrained(
        config.model,
        torch_dtype=_dtype_from_name(config.training.dtype),
        device=device,
        freeze_encoder=True,
        local_files_only=not args.allow_download,
    )
    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    log("Encoding input text")
    collator = TextCollator(sequence_length=config.data.sequence_length)
    batch = collator([_read_text(args)])
    batch = move_batch_to_device(batch, device)

    log("Running reconstruction forward pass")
    output = model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    log(f"Finished in {time.perf_counter() - started_at:.1f}s")

    logits = output["logits"][0]
    length = int(batch["lengths"][0].item())
    original_ids = batch["byte_ids"][0, :length].tolist()

    if args.free_run:
        log("Running free-running reconstruction")
        reconstructed_ids = _free_running_reconstruction(
            model=model,
            original_ids=original_ids,
            full_z_words=output["z_words"][0],
            full_boundaries=batch["word_boundaries"][0],
            sequence_length=config.data.sequence_length,
            device=device,
        )
    else:
        # The Phase 1 objective predicts byte[t + 1] from byte[:t] and word latents.
        reconstructed_ids = _teacher_forced_reconstruction(logits, original_ids)

    original = collator.decode_bytes(original_ids)
    reconstructed = collator.decode_bytes(reconstructed_ids)
    step = checkpoint.get("step")

    print(f"checkpoint={args.checkpoint}")
    print(f"step={step}")
    print(f"input_length_bytes={length}")
    print(f"mode={'free-run' if args.free_run else 'teacher-forced'}")
    print()
    print("INPUT")
    print(original)
    print()
    print("RECONSTRUCTION")
    print(reconstructed)

    if args.show_bytes:
        print()
        print(f"predicted_byte_ids={reconstructed_ids}")


if __name__ == "__main__":
    main()

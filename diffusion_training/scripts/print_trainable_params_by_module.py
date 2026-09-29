import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="phase1.yaml", help="Phase 1 YAML config (bare name resolves against configs/training/).")
    parser.add_argument("--checkpoint", required=True, help="Path to a Phase 1 checkpoint .pt file.")
    parser.add_argument("--device", help="Override device from config, for example cuda:0.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum module name depth to print. Use 0 for only the whole model.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include modules with zero trainable parameters.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download/check remote model files. By default only cached files are used.",
    )
    return parser.parse_args()


def _module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def _direct_count(module: object, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters(recurse=False)
        if not trainable_only or parameter.requires_grad
    )


def _subtree_count(module: object, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _format_int(value: int) -> str:
    return f"{value:,}"


def _print_table(rows: list[tuple[str, str, int, int, int]]) -> None:
    headers = ("module", "type", "direct_trainable", "subtree_trainable", "subtree_total")
    if not rows:
        print("No modules matched the requested filters.")
        return

    formatted_rows = [
        (name, class_name, _format_int(direct), _format_int(trainable), _format_int(total))
        for name, class_name, direct, trainable, total in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted_rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in formatted_rows:
        print(
            f"{row[0].ljust(widths[0])}  "
            f"{row[1].ljust(widths[1])}  "
            f"{row[2].rjust(widths[2])}  "
            f"{row[3].rjust(widths[3])}  "
            f"{row[4].rjust(widths[4])}"
        )


def main() -> None:
    args = parse_args()

    import torch

    from src.model import DiffusionAutoencoder
    from src.train import _dtype_from_name, count_trainable_parameters
    from src.utils import load_phase1_config

    config = load_phase1_config(args.config)
    device = torch.device(args.device or config.training.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "This project model cannot run on CPU because the vendored HAT modules allocate CUDA tensors."
        )

    model = DiffusionAutoencoder.from_pretrained(
        config.model,
        torch_dtype=_dtype_from_name(config.training.dtype),
        device=device,
        freeze_encoder=config.training.freeze_encoder,
        local_files_only=not args.allow_download,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    rows = []
    for name, module in model.named_modules():
        display_name = name or "<root>"
        if _module_depth(name) > args.max_depth:
            continue
        direct_trainable = _direct_count(module, trainable_only=True)
        subtree_trainable = _subtree_count(module, trainable_only=True)
        subtree_total = _subtree_count(module, trainable_only=False)
        if subtree_trainable == 0 and not args.include_zero:
            continue
        rows.append((display_name, module.__class__.__name__, direct_trainable, subtree_trainable, subtree_total))

    step = checkpoint.get("step")
    extra_trainable = (checkpoint.get("extra") or {}).get("trainable_parameters")
    print(f"checkpoint={args.checkpoint}")
    print(f"step={step}")
    print(f"config={args.config}")
    print(f"freeze_encoder={config.training.freeze_encoder}")
    print(f"total_trainable_parameters={_format_int(count_trainable_parameters(model))}")
    if extra_trainable is not None:
        print(f"checkpoint_reported_trainable_parameters={_format_int(int(extra_trainable))}")
    print()
    _print_table(rows)


if __name__ == "__main__":
    main()

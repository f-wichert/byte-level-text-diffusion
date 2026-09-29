import contextlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


INPUT = "When was Rome founded?"
MODEL_ID = "Aleph-Alpha/tfree-hat-pretrained-7b-base"


def run() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        trust_remote_code=True,
        pretrained_model_name_or_path=MODEL_ID,
    ).to("cuda", torch.bfloat16)
    input_ids, cumulative_word_lengths = model._prepare_input(INPUT, add_llama_template=True)
    model_output = model.generate(
        input_ids,
        cumulative_seq_lengths_per_word=cumulative_word_lengths,
        max_new_tokens=300,
        use_cache=False,
    )
    print("Prompt:", INPUT)
    print("Completion:", model_output.completion_text)


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle):
                run()
        print(f"Wrote inference output to {output_path}")
    else:
        run()


if __name__ == "__main__":
    main()

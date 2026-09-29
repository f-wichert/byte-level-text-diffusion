from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"
bolmo = AutoModelForCausalLM.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True)

# message = ["Language modeling is "]
# input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to(device)

# # `max_new_tokens` is the amount of bytes to generate
# # response = bolmo.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.1)
# # print(tokenizer.decode(response[0], skip_special_tokens=True))


# print("=" * 30 + " Finding the Encoder " + "=" * 30)
# print(bolmo)




# message = ["Language modeling is "]
# input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to(device)

# with torch.no_grad():
#     h_byte, patch_embeddings, boundary_logprobs, boundary_mask = bolmo.model.local_encoder(input_ids)

# # patch_embeddings: (batch, n_patches, 4096) — the pooled latent representations
# # boundary_mask:    (batch, seq_len)          — True at each patch boundary position
# print(patch_embeddings.shape)
# print(f"Number of patches: {boundary_mask.sum(-1)}")


import importlib.util

spec = importlib.util.spec_from_file_location(
    "utils_bolmo",
    "/home/fwichert/.cache/huggingface/modules/transformers_modules/allenai/Bolmo_hyphen_7B/2b307590fce19dea1d79a775ec7275c69d11ac6a/utils_bolmo.py"
)
utils_bolmo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_bolmo)
pad_right = utils_bolmo.pad_right

message = ["TEXT"]
input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to(device)

# Replicate what BolmoModel.forward() does internally
with torch.no_grad():
    expanded_input_ids_list = []
    for i in range(input_ids.shape[0]):
        expanded = bolmo.model.tokenizer.expand_byte_ids(input_ids[i].tolist())
        expanded_input_ids_list.append(
            torch.tensor(expanded, dtype=torch.long, device=device)
        )

    # pad_right is imported inside the model's module, so access it via the model
    expanded_input_ids = pad_right(
        expanded_input_ids_list,
        value=bolmo.model.tokenizer.pad_token_id,
        multiple_of=1
    )

    h_byte, patch_embeddings, boundary_logprobs, boundary_mask = bolmo.model.local_encoder(
        input_ids,
        expanded_input_ids=expanded_input_ids,
    )

print(patch_embeddings.shape)
print(f"Number of patches: {boundary_mask.sum(-1)}")
import gc
import torch
import pandas as pd
import numpy as np

import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM

from abc import ABC, abstractmethod
from tqdm import tqdm

import sys
from pathlib import Path

# `diffusion_training` is the sibling project consumed by TFreeHatFineTunedEncoder.
# Its code uses top-level `from src.*` imports, so put its root on sys.path here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFUSION_TRAINING = _REPO_ROOT / "diffusion_training"
if str(_DIFFUSION_TRAINING) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_TRAINING))

class Encoder(ABC):
    encoder_model = None

    def __init__(self):
        print("Encoder initialized...")
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def encode_text(self):
        if self.encoder_model is None:
            raise ValueError("Encoder model has not been initialized.")
    
    @abstractmethod
    def delete_model(self):
        if self.encoder_model is None:
            raise ValueError("No encoder model present to delete")

class BLTEncoder(Encoder):
    def __init__(self):
        super()
        print("Encoder type is BLT")
    
    def load_model(self, device="cuda"):
        from bytelatent.transformer import LMTransformer
        from bytelatent.model.blt import ByteLatentTransformer
        from bytelatent.hf import BltTokenizerAndPatcher
        from bytelatent.model.utils import downsample
        from bytelatent.model.blt import (
            get_blt_input,
            patch_ids_from_lengths,
            compute_hash_embeddings,
            cross_attn_mask,
        )
        from bytelatent.data.patcher import PatchingModeEnum

        entropy_repo = "facebook/blt-entropy"
        blt_repo = "facebook/blt-1b"
        # entropy_repo = "blt/blt-entropy"
        # blt_repo = "blt/blt-1b"

        entropy_model = LMTransformer.from_pretrained(entropy_repo)
        entropy_model = entropy_model.to(device)
        # entropy_model.eval()

        blt_model = ByteLatentTransformer.from_pretrained(blt_repo)
        blt_model = blt_model.to(device)
        blt_model.eval()
        blt_model.patcher.entropy_model = entropy_model
        # self.blt_model.patcher.patching_mode = PatchingModeEnum.space

        self.blt_model = blt_model

        # tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(blt_repo)
        # tokenizer = tok_and_patcher.tokenizer_args.build()
        # patcher = tok_and_patcher.patcher_args.build()
    
    def encode_text(self, text, device="cuda"):
        """
        Encodes the given text. 
        Mostly taken from  blt/bytlatent/model/blt.py ByteLatentTransformer.forward(...)
        """
        from bytelatent.transformer import LMTransformer
        from bytelatent.model.blt import ByteLatentTransformer
        from bytelatent.hf import BltTokenizerAndPatcher
        from bytelatent.model.utils import downsample
        from bytelatent.model.blt import (
            get_blt_input,
            patch_ids_from_lengths,
            compute_hash_embeddings,
            cross_attn_mask,
        )
        from bytelatent.data.patcher import PatchingModeEnum
        
        byte_list = list(text.encode("utf-8"))
        tokens = torch.tensor([byte_list], dtype=torch.long, device=device)
        
        bs, N = tokens.shape # Batch size and sequence length
        
        # Get megabyte inputs
        nb_boe = int(0 if self.blt_model.patching_mode != "" else self.blt_model.patch_size - 1)
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.blt_model.patch_size,
            boe_id=self.blt_model.boe_id,
        )
        
        # Patching
        patch_lengths, tok_scores = self.blt_model.patcher.patch(
            local_encoder_tokens,
            include_next_token=True,
            threshold=self.blt_model.patcher.threshold,
        )
        
        if nb_boe > 0:
            patch_lengths[:, 0] += nb_boe
        
        # Generate patch IDs from patch_lengths
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )
        
        # Cross-attention encoder
        # cross_attn_mask_enc = None
        # if self.blt_model.cross_attn_encoder:
        #     cross_attn_mask_enc = cross_attn_mask(
        #         patch_ids,
        #         patch_lengths,
        #         N,
        #         patches_as_queries=True,
        #         cross_attn_k=self.blt_model.cross_attn_k,
        #         window=self.blt_model.cross_attn_window_encoder,
        #         block_mask=self.blt_model.cross_attn_use_flex_attention,
        #     )
        
        # Hashing and embedding
        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=self.blt_model.local_encoder,
            encoder_hash_tok_embedding=self.blt_model.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.blt_model.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.blt_model.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.blt_model.encoder_hash_byte_group_vocab,
        )
        
        # Local encoder
        (h_encoder, h_cross), cache_encoder = self.blt_model.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            # cross_mask=cross_attn_mask_enc,
            cross_mask=None,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        
        # Downsampling
        if not self.blt_model.cross_attn_encoder:
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.blt_model.downsampling_by_pooling,
                patch_size=self.blt_model.patch_size,
            )
        else:
            h = h_cross.view(bs, patch_lengths.shape[1], -1)
        
        return {
            "byte_embeddings": h_encoder,      # Byte-level representations
            "patch_embeddings": h,              # Patch-level representations (downsampled)
            "patch_lengths": patch_lengths,
            "patch_ids": patch_ids,
        }
    
    def delete_model(self):
        del self.blt_model

        torch.cuda.empty_cache()
        gc.collect()

class BolmoEncoder(Encoder):
    def __init__(self):
        super()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Encoder type is Bolmo")

    def load_model(self, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import importlib.util
        self.bolmo = AutoModelForCausalLM.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True).to(device)
        self.bolmo.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True)

        spec = importlib.util.spec_from_file_location(
            "utils_bolmo",
            "/home/fwichert/.cache/huggingface/modules/transformers_modules/allenai/Bolmo_hyphen_7B/2b307590fce19dea1d79a775ec7275c69d11ac6a/utils_bolmo.py"
        )
        self.utils_bolmo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utils_bolmo)
        self.pad_right = self.utils_bolmo.pad_right

    def encode_text(self, text, device="cuda"):
        message = [text]
        input_ids = self.tokenizer(
            message,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].to(device)

        # Replicate what BolmoModel.forward() does internally
        with torch.no_grad():
            expanded_input_ids_list = []
            for i in range(input_ids.shape[0]):
                expanded = self.bolmo.model.tokenizer.expand_byte_ids(input_ids[i].tolist())
                expanded_input_ids_list.append(
                    torch.tensor(expanded, dtype=torch.long, device=device)
                )

            # pad_right is imported inside the model's module, so access it via the model
            expanded_input_ids = self.pad_right(
                expanded_input_ids_list,
                value=self.bolmo.model.tokenizer.pad_token_id,
                multiple_of=1
            )

            h_byte, patch_embeddings, boundary_logprobs, boundary_mask = self.bolmo.model.local_encoder(
                input_ids,
                expanded_input_ids=expanded_input_ids,
            )

            return {
                "byte_embeddings": h_byte,
                "patch_embeddings": patch_embeddings,
            }

    def delete_model(self):
        del self.bolmo
        del self.tokenizer
        del self.utils_bolmo
        del self.pad_right

        torch.cuda.empty_cache()
        gc.collect()

class NeoBERTEncoder(Encoder):
    def __init__(self):
        super()
        print("Encoder type is NeoBERT")
    
    def load_model(self, device="cuda"):
        model_name = "chandar-lab/NeoBERT"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NeoBERT on {device}...")
    
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        model.to(device)
    
        print(f"NeoBERT loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer, device

    def encode_text(self, 
        text, 
        device="cuda",
        pooling: str = "cls",       # "cls" | "mean" | "max"
        max_length: int = 512,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into NeoBERT embeddings.
    
        Args:
            texts:                List of input strings.
            model:                Loaded NeoBERT model.
            tokenizer:            Corresponding tokenizer.
            device:               "cuda" or "cpu".
            pooling:              How to pool token embeddings.
                                - "cls"  : [CLS] token (index 0) — classic BERT style
                                - "mean" : average of all non-padding token states
                                - "max"  : element-wise max across non-padding tokens
            max_length:           Max token length (up to 4096 for NeoBERT).
            batch_size:           Number of texts processed per forward pass.
            normalize_embeddings: L2-normalize outputs (recommended for similarity tasks).
    
        Returns:
            np.ndarray of shape (len(texts), hidden_size)  — hidden_size = 768
        """
        from sklearn.preprocessing import normalize

        all_embeddings = []
        texts = [text]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
    
            with torch.no_grad():
                outputs = self.model(**encoded)
    
            hidden = outputs.last_hidden_state          # (B, T, H)
            attention_mask = encoded["attention_mask"]  # (B, T)
    
            if pooling == "cls":
                embeddings = hidden[:, 0, :]            # [CLS] token
    
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).float()
                embeddings = (hidden * mask).sum(1) / mask.sum(1)
    
            elif pooling == "max":
                mask = attention_mask.unsqueeze(-1).bool()
                hidden = hidden.masked_fill(~mask, float("-inf"))
                embeddings = hidden.max(dim=1).values
    
            else:
                raise ValueError(f"Unknown pooling method: {pooling!r}. Choose cls/mean/max.")
    
            all_embeddings.append(embeddings.cpu().float().numpy())
            # print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts", end="\r")
    
        embeddings_np = np.vstack(all_embeddings)          # (N, 768)
    
        if normalize_embeddings:
            embeddings_np = normalize(embeddings_np)
    
        return {
            "patch_embeddings": torch.from_numpy(embeddings_np),              # Patch-level representations (downsampled)
        }

    def delete_model(self):
        del self.bolmo
        del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()

class GeminiEncoder(Encoder):
    def __init__(self):
        super()
        print("Encoder type is Gemini / EmbeddingGemma")

    def load_model(self, device="cuda"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Gemini encoder requires `sentence-transformers`."
            ) from exc

        model_name = "google/embeddinggemma-300m"
        self.encoder_model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": torch.float32},
        )
        self.encoder_model.to(device)

    def encode_text(self, text, device="cuda"):
        if hasattr(self.encoder_model, "encode_document"):
            embeddings = self.encoder_model.encode_document(
                [text],
                convert_to_tensor=True,
            )
        else:
            embeddings = self.encoder_model.encode(
                [text],
                convert_to_tensor=True,
            )

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        return {
            "patch_embeddings": embeddings.to(dtype=torch.float32).cpu(),
        }

    def delete_model(self):
        del self.encoder_model

        torch.cuda.empty_cache()
        gc.collect()

class LangFlowEncoder(Encoder):
    def __init__(self):
        super()
        print("Encoder type is LangFlow")

    def load_model(self, device="cuda"):
        model_name = "Continuous-Rivals-Discrete/langflow-owt"
        revision = "a08f933dd337d52762fec5ef7d60c131896cc341"

        class BackboneOutputPassthrough(torch.nn.Module):
            def forward(self, x, c):
                return x

        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.gamma_min = model.proposal.gamma_min
        model.backbone.output_layer = BackboneOutputPassthrough()
        self.encoder_model = model.backbone.to(device)
        self.encoder_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode_text(self, text, device="cuda", max_length=1024):
        encoded = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)

        if input_ids.shape[1] == 0:
            input_ids = torch.tensor(
                [[self.tokenizer.eos_token_id]],
                dtype=torch.long,
                device=device,
            )

        with torch.no_grad():
            input_embeddings = self.encoder_model.vocab_embed(input_ids)
            sigma = torch.full(
                (input_ids.shape[0],),
                self.gamma_min,
                device=input_ids.device,
            )
            _, hidden_states = self.encoder_model(
                input_embeddings,
                sigma,
                output_hidden_states=True,
            )

        if not hidden_states:
            raise RuntimeError("LangFlow did not return hidden states.")

        return {
            "patch_embeddings": hidden_states[-1].squeeze(0).to(dtype=torch.float32).cpu(),
        }

    def delete_model(self):
        del self.encoder_model
        del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()
    
_DEFAULT_AE_CONFIG = "configs/training/phase2-compress.yaml"
# Pinned (not latest-of-dir) so the default run stays reproducible even if the
# checkpoint directory ever grows.
_DEFAULT_AE_CHECKPOINT = "checkpoints/phase2-compress256l4-lr1.0e-4-uniform-la1.0e-3-t2.0-skip/step_040000.pt"


class TFreeHatFineTunedEncoder(Encoder):
    def __init__(self, ae_config_path=None, ae_checkpoint_path=None):
        super()
        self.ae_config_path = ae_config_path
        self.ae_checkpoint_path = ae_checkpoint_path
        # Set by load_model for non-default config/checkpoint combinations;
        # used to tag cache files and reports so AE variants never share them.
        self.variant_tag = None
        print("Encoder type is a fine tuned version of TFreeHat")

    @staticmethod
    def _resolve_repo_path(path):
        path = Path(path)
        return path if path.is_absolute() else _REPO_ROOT / path

    def load_model(self, device="cuda"):
        from src.data import TextCollator
        from src.model import DiffusionAutoencoder
        from src.sampling import find_latest_checkpoint
        from src.train import _dtype_from_name, move_batch_to_device
        from src.utils import load_phase1_config

        is_default = self.ae_config_path is None and self.ae_checkpoint_path is None
        config_path = self._resolve_repo_path(self.ae_config_path or _DEFAULT_AE_CONFIG)
        config = load_phase1_config(str(config_path))
        # Fall back to the historical pin only when no device was passed at all.
        device = torch.device(device or "cuda:0")

        self.encoder_model = DiffusionAutoencoder.from_pretrained(
            config.model,
            torch_dtype=_dtype_from_name(config.training.dtype),
            device=device,
            freeze_encoder=True,
            local_files_only=False,  # set True if base HAT model is already cached
        )

        if self.ae_checkpoint_path is not None:
            checkpoint_path = self._resolve_repo_path(self.ae_checkpoint_path)
        elif is_default:
            checkpoint_path = self._resolve_repo_path(_DEFAULT_AE_CHECKPOINT)
        else:
            checkpoint_path = find_latest_checkpoint(config.training.checkpoint_dir)

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        self.encoder_model.load_state_dict(checkpoint["model"])
        self.encoder_model.eval()

        if not is_default:
            self.variant_tag = f"{checkpoint_path.parent.name}-{checkpoint_path.stem}"
        print(f"Loaded AE checkpoint: {checkpoint_path}")

        self.collator = TextCollator(sequence_length=config.data.sequence_length)


    def encode_text(self, text, device="cuda"):
        from src.train import move_batch_to_device

        batch = self.collator([text])
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            output = self.encoder_model(**batch)

        logits = output["logits"]
        z_words = output["z_words"]

        return {
            "patch_embeddings": z_words[0].cpu(),
        }


    def delete_model(self):
        del self.encoder_model

        torch.cuda.empty_cache()
        gc.collect()

class TFreeHatEncoder(Encoder):
    def __init__(self):
        super()
        print("Encoder type is TFreeHat")

    def load_model(self, device="cuda"):

        MODEL_ID = "Aleph-Alpha/tfree-hat-pretrained-7b-base"
        MODEL_REVISION = "c4797ee9ad934afd5adb27b490d0a3b1d84e4895"
        INCLUDE_BACKBONE_LATENTS = False
        DEVICE = "cuda:0"


        def _identity_decorator(fn):
            return fn


        # The model's remote code imports `dynamic_rope_update` from the transformers
        # top-level package, but transformers 4.46.3 doesn't export it. For this model's
        # config, rope_type is always "default", so a no-op decorator is sufficient.
        if not hasattr(transformers, "dynamic_rope_update"):
            transformers.dynamic_rope_update = _identity_decorator

        if not torch.cuda.is_available():
            raise RuntimeError("tfree-hat requires a CUDA GPU for FlashAttention-based encoder analysis.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,   # required — uses custom HATForCausalLM class
            dtype=torch.bfloat16,
            device_map={"": DEVICE},
        )

        # Keep the analysis path on one explicit device. Some remote-code submodules can
        # still end up on CPU after loading, so we move the parts we actually call.
        model.encoder.to(DEVICE)
        model.encoder_connector.to(DEVICE)
        if INCLUDE_BACKBONE_LATENTS:
            model.backbone.to(DEVICE)

        self.encoder_model = model

    def encode_text(self, text, device="cuda", include_backbone=False):
        import itertools
        self.encoder_model.eval()

        words = self.encoder_model.splitter.encode(text)  # list[list[int]], one byte-list per word/chunk
        word_texts = [self.encoder_model.splitter.decode(w) for w in words]

        flat_ids = list(itertools.chain.from_iterable(words))
        cum_lens = [0]
        for w in words:
            cum_lens.append(cum_lens[-1] + len(w))

        encoder_device = self.encoder_model.encoder.embedding_layer.weight.device
        input_ids = torch.tensor([flat_ids], dtype=torch.long, device=encoder_device)
        byte_position_ids = torch.arange(len(flat_ids), dtype=torch.int32, device=encoder_device).unsqueeze(0)
        word_position_ids = torch.arange(len(words), dtype=torch.int32, device=encoder_device).unsqueeze(0)
        cumulative_seq_lengths_per_word = torch.tensor(cum_lens, dtype=torch.int32, device=encoder_device)

        with torch.no_grad():
            encoder_out = self.encoder_model.encoder(
                input_ids=input_ids,
                cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
                byte_position_ids=byte_position_ids,
                word_position_ids=word_position_ids,
            )
            byte_latents = encoder_out.hidden_states  # [1, num_bytes, encoder_hidden]

            connector_device = self.encoder_model.encoder_connector.latent_query.device
            word_latents = self.encoder_model.encoder_connector(
                byte_latents.to(connector_device),
                cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word.to(connector_device),
                word_position_ids=word_position_ids.to(connector_device),
                byte_position_ids=byte_position_ids.to(connector_device),
            )  # [1, num_words, backbone_hidden]

            backbone_latents = None
            if include_backbone:
                backbone_device = next(self.encoder_model.backbone.parameters()).device
                backbone_latents = self.encoder_model.backbone(
                    hidden_states=word_latents.to(backbone_device),
                    position_ids=word_position_ids.to(backbone_device),
                ).hidden_states  # [1, num_words, backbone_hidden]

        # return {
        #     "words": word_texts,
        #     "byte_ids": flat_ids,
        #     "byte_latents": byte_latents.squeeze(0).cpu(),
        #     "word_latents": word_latents.squeeze(0).cpu(),
        #     "backbone_latents": None if backbone_latents is None else backbone_latents.squeeze(0).cpu(),
        # }
        return {
            "patch_embeddings": word_latents.squeeze(0).to(dtype=torch.float32).cpu(),              # Patch-level representations (downsampled)
        }

    def delete_model(self):
        del self.encoder_model

        torch.cuda.empty_cache()
        gc.collect()

import torch
import torch.nn as nn


class LengthPredictorV1(nn.Module):
    """Word latent -> byte-length class (1..max_length, clamped), one word at a time."""

    arch = "mlp"

    def __init__(self, input_dim: int, hidden_dim: int = 512, max_length: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.space: str | None = None
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_length),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def predict_lengths(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """z: [..., input_dim] latents -> integer byte lengths [...].

        Element-wise: accepts ``[W, D]`` (one sequence) or ``[B, W, D]`` and returns the
        same shape minus the latent dim. ``mask`` is accepted for a uniform interface with
        v2 but ignored (each word is scored independently).
        """

        return self.net(z.float()).argmax(dim=-1) + 1

    @classmethod
    def load(cls, path: str, device: torch.device) -> "LengthPredictorV1":
        artifact = torch.load(path, map_location=device)
        model = cls(artifact["input_dim"], artifact["hidden_dim"], artifact["max_length"])
        model.load_state_dict(artifact["state_dict"])
        model.space = artifact.get("space")
        return model.to(device).eval()


class LengthPredictorV2(nn.Module):

    arch = "context"

    def __init__(
        self,
        input_dim: int,
        max_words: int,
        max_length: int = 32,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_words = max_words
        self.max_length = max_length
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.space: str | None = None
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(max_words, model_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first makes the nested-tensor fast path unused; disable it to avoid the warning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(model_dim, max_length)

    def forward(self, z: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """z: [B, W, D] -> logits [B, W, max_length]. key_padding_mask: True = ignore."""

        num_words = z.size(1)
        hidden = self.input_proj(z.float()) + self.pos_embed[:num_words]
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.head(hidden)

    @torch.no_grad()
    def predict_lengths(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """z: ``[W, D]`` or ``[B, W, D]`` -> integer byte lengths, same shape minus D.

        ``mask`` (True = valid word) marks real vs padding positions for the attention;
        omit it for a single unpadded sequence.
        """

        squeeze = z.ndim == 2
        if squeeze:
            z = z.unsqueeze(0)
        key_padding_mask = None
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            key_padding_mask = ~mask.bool()
        logits = self.forward(z, key_padding_mask=key_padding_mask)
        lengths = logits.argmax(dim=-1) + 1
        return lengths[0] if squeeze else lengths


def load_length_predictor(path: str, device: torch.device) -> nn.Module:

    artifact = torch.load(path, map_location=device)
    arch = artifact.get("arch", "mlp")
    if arch == "mlp":
        model: nn.Module = LengthPredictorV1(
            artifact["input_dim"], artifact["hidden_dim"], artifact["max_length"]
        )
        model.load_state_dict(artifact["state_dict"])
    elif arch == "context":
        max_words = int(artifact["state_dict"]["pos_embed"].shape[0])
        model = LengthPredictorV2(
            input_dim=artifact["input_dim"],
            max_words=max_words,
            max_length=artifact["max_length"],
            model_dim=artifact["model_dim"],
            num_layers=artifact["num_layers"],
            num_heads=artifact["num_heads"],
        )
        model.load_state_dict(artifact["state_dict"])
    else:
        raise ValueError(f"Unknown length-predictor arch {arch!r} in {path}")
    model.space = artifact.get("space")
    return model.to(device).eval()


# Back-compat alias: old imports (`from scripts.train_length_predictor import LengthPredictor`)
# and old code paths keep working.
LengthPredictor = LengthPredictorV1

"""
model.py — Stage 1 model: RETFound + Phase Embeddings + Attention MIL + Classifier.

Architecture recap:
  For each image in a patient bag:
    1. RETFound ViT-Large → 1024-d feature vector
    2. Phase embedding lookup → 32-d vector
    3. Concatenate → 1056-d combined vector

  Across all images in the bag:
    4. Attention module → scalar weight per image (sum to 1)
    5. Weighted sum → single 1056-d patient vector
    6. FC classifier → sigmoid → probability of RV present
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL, TRAIN, DEVICE


# ──────────────────────────────────────────────────────────────
# BACKBONE LOADER
# ──────────────────────────────────────────────────────────────

def load_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """
    Load the image encoder backbone.
    Returns (model, output_feature_dim).

    Supported:
      "retfound"       — ViT-Large pretrained on 1.6M retinal images (best)
      "vit_base"       — Vanilla ViT-Base pretrained on ImageNet (ablation)
      "efficientnet_b3"— EfficientNet-B3 (CNN baseline)
    """

    if backbone_name == "retfound":
        return _load_retfound()

    elif backbone_name == "vit_base":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # num_classes=0 removes the classification head → raw feature output
        feat_dim = model.num_features  # 768 for ViT-Base
        return model, feat_dim

    elif backbone_name == "efficientnet_b3":
        import timm
        model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0,
                                  global_pool="avg")
        feat_dim = model.num_features  # 1536 for EfficientNet-B3
        return model, feat_dim

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Choose from: retfound, vit_base, efficientnet_b3")


def _load_retfound() -> tuple[nn.Module, int]:
    """
    Load RETFound (ViT-Large, MAE pretrained on retinal images).

    RETFound uses the MAE (Masked Autoencoder) ViT-Large architecture.
    Weights are downloaded from the official RETFound GitHub release.

    If weights are not downloaded yet, we fall back to a ViT-Large with
    ImageNet weights and print a clear message.
    """
    import timm

    RETFOUND_WEIGHTS = "retfound_weights/RETFound_mae_natureCFP.pth"

    try:
        import os
        if not os.path.exists(RETFOUND_WEIGHTS):
            raise FileNotFoundError("RETFound weights not found locally.")

        # RETFound uses a standard ViT-Large architecture
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        state = torch.load(RETFOUND_WEIGHTS, map_location="cpu")
        # HuggingFace checkpoint may store directly or under "model" key
        weights = state.get("model", state)
        # Also handle if it's stored under "state_dict"
        if "state_dict" in state:
            weights = state["state_dict"]
        # Remove the classification head weights if present
        weights = {k: v for k, v in weights.items()
                   if not k.startswith("head.")}
        msg = model.load_state_dict(weights, strict=False)
        print(f"RETFound weights loaded. Missing keys: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")
        feat_dim = 1024  # ViT-Large output dim

    except FileNotFoundError:
        print(
            "\n⚠️  RETFound weights not found at 'retfound_weights/RETFound_cfp_weights.pth'."
            "\n   Download from: https://github.com/rmaphoh/RETFound_MAE"
            "\n   Falling back to ViT-Large with ImageNet weights for now.\n"
        )
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = 1024

    return model, feat_dim


# ──────────────────────────────────────────────────────────────
# FREEZE / UNFREEZE UTILITIES
# ──────────────────────────────────────────────────────────────

def freeze_backbone(backbone: nn.Module):
    """Freeze all backbone parameters — no gradient updates."""
    for param in backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen.")


def unfreeze_top_blocks(backbone: nn.Module, n_blocks: int):
    """
    Unfreeze the top N transformer blocks + the final layer norm.
    Works for ViT backbones (timm naming: backbone.blocks is a list).
    For EfficientNet, unfreezes the last N layers instead.
    """
    # First freeze everything
    freeze_backbone(backbone)

    # Check if it's a ViT (has .blocks attribute)
    if hasattr(backbone, "blocks"):
        total = len(backbone.blocks)
        for block in backbone.blocks[total - n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze the final layer norm
        if hasattr(backbone, "norm"):
            for param in backbone.norm.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Unfrozen top {n_blocks}/{total} transformer blocks. "
              f"Trainable backbone params: {trainable:,}")

    else:
        # EfficientNet: unfreeze last n layers by index
        all_layers = list(backbone.children())
        for layer in all_layers[-n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfrozen top {n_blocks} backbone layers.")


# ──────────────────────────────────────────────────────────────
# ATTENTION MODULE
# ──────────────────────────────────────────────────────────────

class AttentionMIL(nn.Module):
    """
    Attention-based MIL aggregation (Ilse et al., 2018).

    Given N feature vectors (one per image in a bag):
      1. Project each to a scalar score via a small MLP
      2. Softmax across N → attention weights α (sum to 1)
      3. Weighted sum → single patient-level vector

    Input:  (N, D)  — N images, D-dimensional features
    Output: (D,)    — single aggregated patient vector
            (N,)    — attention weights (useful for interpretability)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # The attention network: D → hidden → 1
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        H: (N, D) — bag of N image features
        Returns:
            z:      (D,)  — aggregated patient vector
            alpha:  (N,)  — attention weights per image
        """
        # (N, D) → (N, 1) → (N,)
        scores = self.attention(H).squeeze(-1)        # (N,)
        alpha  = F.softmax(scores, dim=0)             # (N,) sums to 1

        # Weighted sum: (N,) · (N, D) → (D,)
        z = torch.einsum("n,nd->d", alpha, H)         # (D,)

        return z, alpha


# ──────────────────────────────────────────────────────────────
# FULL STAGE 1 MODEL
# ──────────────────────────────────────────────────────────────

class Stage1Model(nn.Module):
    """
    Full Stage 1 pipeline:
      Image → Backbone → Phase Embedding → Attention MIL → Classifier

    Forward input:
        images: (N, 3, H, W)  — N images in the patient's bag
        phases: (N,)           — phase index per image

    Forward output:
        logit:   scalar — raw output before sigmoid
        alpha:   (N,) — attention weight per image (interpretability)
    """

    def __init__(self):
        super().__init__()
        cfg = MODEL

        # 1. Backbone
        self.backbone, feat_dim = load_backbone(cfg["backbone"])

        # If backbone dim != config, override (handles EfficientNet vs ViT)
        actual_combined = feat_dim + cfg["phase_embed_dim"]

        # 2. Phase embedding table: num_phases × phase_embed_dim
        self.phase_embed = nn.Embedding(
            num_embeddings=cfg["num_phases"],
            embedding_dim=cfg["phase_embed_dim"],
        )

        # 3. Attention MIL
        self.attention_mil = AttentionMIL(
            input_dim=actual_combined,
            hidden_dim=cfg["attn_hidden_dim"],
        )

        # 4. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(actual_combined, cfg["classifier_hidden"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["classifier_hidden"], 1),
            # No sigmoid here — we use BCEWithLogitsLoss which is more
            # numerically stable. Apply sigmoid only at inference.
        )

        self._feat_dim = feat_dim

    def forward(
        self,
        images: torch.Tensor,   # (N, 3, H, W)
        phases: torch.Tensor,   # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Per-image encoding ──────────────────────────────
        # Backbone: (N, 3, H, W) → (N, feat_dim)
        img_features = self.backbone(images)

        # Phase embeddings: (N,) → (N, phase_embed_dim)
        phase_features = self.phase_embed(phases)

        # Concatenate: (N, feat_dim + phase_embed_dim)
        H = torch.cat([img_features, phase_features], dim=-1)

        # ── MIL aggregation ─────────────────────────────────
        # (N, combined_dim) → (combined_dim,), (N,)
        z, alpha = self.attention_mil(H)

        # ── Classification ───────────────────────────────────
        # (combined_dim,) → (1,) → scalar
        logit = self.classifier(z).squeeze(-1)

        return logit, alpha


# ──────────────────────────────────────────────────────────────
# WEIGHTED LOSS
# ──────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted BCE.

    For Stage 1 binary classification:
      weight[class] = total_samples / (num_classes * count[class])

    Returns a tensor [weight_for_class_0, weight_for_class_1].
    """
    n_total = len(labels)
    n_pos   = sum(labels)           # RV present
    n_neg   = n_total - n_pos       # Normal

    # Avoid division by zero
    w_pos = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    w_neg = n_total / (2 * n_neg) if n_neg > 0 else 1.0

    print(f"Class weights — Normal: {w_neg:.3f}, RV Present: {w_pos:.3f}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def get_loss_fn(pos_weight: torch.Tensor) -> nn.BCEWithLogitsLoss:
    """
    Binary cross-entropy with logits + positive class upweighting.
    pos_weight > 1 means the model is penalised more for missing RV cases.
    This maximises recall for Stage 1.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig


class ScalarTokenizer(nn.Module):
    """Convert each scalar feature into one learned token."""

    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        tokens = self.value_projection(features.unsqueeze(-1))
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        return self.dropout(self.norm(tokens))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.25, n_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )


class TechnicalMLP(MLPClassifier):
    def forward(self, technical, fundamental):
        return self.network(technical)


class FundamentalMLP(MLPClassifier):
    def forward(self, technical, fundamental):
        return self.network(fundamental)


class ConcatenationMLP(MLPClassifier):
    def forward(self, technical, fundamental):
        return self.network(torch.cat([technical, fundamental], dim=1))


class LateFusionClassifier(nn.Module):
    """Predict per modality, then combine technical and fundamental logits."""

    def __init__(self, n_technical, n_fundamental, hidden_dim=128, dropout=0.25):
        super().__init__()

        def branch(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
            )

        self.technical_branch = branch(n_technical)
        self.fundamental_branch = branch(n_fundamental)
        self.technical_weight_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, technical, fundamental):
        technical_logits = self.technical_branch(technical)
        fundamental_logits = self.fundamental_branch(fundamental)
        weight = torch.sigmoid(self.technical_weight_logit)
        return weight * technical_logits + (1.0 - weight) * fundamental_logits


class DirectCrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention followed by pooled classification."""

    def __init__(self, n_technical, n_fundamental, config: TrainingConfig):
        super().__init__()
        d_model = config.d_model
        self.technical_tokenizer = ScalarTokenizer(
            n_technical, d_model, config.dropout
        )
        self.fundamental_tokenizer = ScalarTokenizer(
            n_fundamental, d_model, config.dropout
        )
        self.technical_queries = nn.MultiheadAttention(
            d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.fundamental_queries = nn.MultiheadAttention(
            d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.technical_norm = nn.LayerNorm(d_model)
        self.fundamental_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 3),
        )

    def forward(self, technical, fundamental):
        technical_tokens = self.technical_tokenizer(technical)
        fundamental_tokens = self.fundamental_tokenizer(fundamental)
        technical_attention, _ = self.technical_queries(
            technical_tokens,
            fundamental_tokens,
            fundamental_tokens,
            need_weights=False,
        )
        fundamental_attention, _ = self.fundamental_queries(
            fundamental_tokens,
            technical_tokens,
            technical_tokens,
            need_weights=False,
        )
        technical_tokens = self.technical_norm(
            technical_tokens + technical_attention
        )
        fundamental_tokens = self.fundamental_norm(
            fundamental_tokens + fundamental_attention
        )
        pooled = torch.cat(
            [
                technical_tokens.mean(dim=1),
                fundamental_tokens.mean(dim=1),
                technical_tokens.max(dim=1).values,
                fundamental_tokens.max(dim=1).values,
            ],
            dim=1,
        )
        return self.head(pooled)


class ConservativeDirectCrossAttentionFusion(nn.Module):
    """Direct cross-attention injected through a small learned residual strength."""

    def __init__(self, n_technical, n_fundamental, config: TrainingConfig):
        super().__init__()
        d_model = config.d_model
        self.technical_tokenizer = ScalarTokenizer(
            n_technical, d_model, config.dropout
        )
        self.fundamental_tokenizer = ScalarTokenizer(
            n_fundamental, d_model, config.dropout
        )
        self.technical_queries = nn.MultiheadAttention(
            d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.fundamental_queries = nn.MultiheadAttention(
            d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.technical_norm = nn.LayerNorm(d_model)
        self.fundamental_norm = nn.LayerNorm(d_model)
        strength = float(np.clip(config.conservative_attn_init, 1e-4, 1 - 1e-4))
        logit = np.log(strength / (1.0 - strength))
        self.attention_strength_logit = nn.Parameter(
            torch.tensor(logit, dtype=torch.float32)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 3),
        )

    def forward(self, technical, fundamental):
        technical_tokens = self.technical_tokenizer(technical)
        fundamental_tokens = self.fundamental_tokenizer(fundamental)
        technical_attention, _ = self.technical_queries(
            technical_tokens,
            fundamental_tokens,
            fundamental_tokens,
            need_weights=False,
        )
        fundamental_attention, _ = self.fundamental_queries(
            fundamental_tokens,
            technical_tokens,
            technical_tokens,
            need_weights=False,
        )
        strength = torch.sigmoid(self.attention_strength_logit)
        technical_tokens = self.technical_norm(
            technical_tokens + strength * technical_attention
        )
        fundamental_tokens = self.fundamental_norm(
            fundamental_tokens + strength * fundamental_attention
        )
        pooled = torch.cat(
            [
                technical_tokens.mean(dim=1),
                fundamental_tokens.mean(dim=1),
                technical_tokens.max(dim=1).values,
                fundamental_tokens.max(dim=1).values,
            ],
            dim=1,
        )
        return self.head(pooled)


class ResidualCrossAttentionTranslator(nn.Module):
    """Translate technical tokens into implied fundamentals and expose the residual."""

    def __init__(self, n_technical, n_fundamental, config: TrainingConfig):
        super().__init__()
        d_model = config.d_model
        self.technical_tokenizer = ScalarTokenizer(
            n_technical, d_model, config.dropout
        )
        self.fundamental_tokenizer = ScalarTokenizer(
            n_fundamental, d_model, config.dropout
        )
        self.translator = nn.MultiheadAttention(
            d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.residual_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 6, 160),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(80, 3),
        )

    def representations(self, technical, fundamental):
        technical_tokens = self.technical_tokenizer(technical)
        fundamental_tokens = self.fundamental_tokenizer(fundamental)
        implied_fundamentals, _ = self.translator(
            fundamental_tokens,
            technical_tokens,
            technical_tokens,
            need_weights=False,
        )
        residual = self.residual_norm(fundamental_tokens - implied_fundamentals)
        return technical_tokens, fundamental_tokens, implied_fundamentals, residual

    def forward(self, technical, fundamental):
        technical_tokens, fundamental_tokens, implied, residual = self.representations(
            technical, fundamental
        )
        pooled = torch.cat(
            [
                technical_tokens.mean(dim=1),
                fundamental_tokens.mean(dim=1),
                implied.mean(dim=1),
                residual.mean(dim=1),
                residual.abs().mean(dim=1),
                residual.max(dim=1).values,
            ],
            dim=1,
        )
        return self.head(pooled)


class GatedResidualCrossAttentionTranslator(ResidualCrossAttentionTranslator):
    """Use the divergence residual to gate the pooled technical representation."""

    def __init__(self, n_technical, n_fundamental, config: TrainingConfig):
        super().__init__(n_technical, n_fundamental, config)
        d_model = config.d_model
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, technical, fundamental):
        technical_tokens, fundamental_tokens, implied, residual = self.representations(
            technical, fundamental
        )
        technical_pool = technical_tokens.mean(dim=1)
        fundamental_pool = fundamental_tokens.mean(dim=1)
        implied_pool = implied.mean(dim=1)
        residual_pool = residual.mean(dim=1)
        absolute_residual_pool = residual.abs().mean(dim=1)
        gate = self.gate(torch.cat([residual_pool, absolute_residual_pool], dim=1))
        pooled = torch.cat(
            [
                gate * technical_pool,
                fundamental_pool,
                implied_pool,
                residual_pool,
                absolute_residual_pool,
                gate,
            ],
            dim=1,
        )
        return self.head(pooled)


MODEL_NAMES = (
    "technical_only_mlp",
    "fundamental_only_mlp",
    "concat_mlp",
    "late_fusion",
    "direct_cross_attention",
    "conservative_direct_cross_attention",
    "residual_cross_attention",
    "gated_residual_cross_attention",
)


def build_model(name, n_technical, n_fundamental, config: TrainingConfig):
    if name == "technical_only_mlp":
        return TechnicalMLP(n_technical, hidden_dim=128, dropout=config.dropout)
    if name == "fundamental_only_mlp":
        return FundamentalMLP(n_fundamental, hidden_dim=128, dropout=config.dropout)
    if name == "concat_mlp":
        return ConcatenationMLP(
            n_technical + n_fundamental,
            hidden_dim=160,
            dropout=config.dropout,
        )
    if name == "late_fusion":
        return LateFusionClassifier(
            n_technical, n_fundamental, dropout=config.dropout
        )
    if name == "direct_cross_attention":
        return DirectCrossAttentionFusion(n_technical, n_fundamental, config)
    if name == "conservative_direct_cross_attention":
        return ConservativeDirectCrossAttentionFusion(
            n_technical, n_fundamental, config
        )
    if name == "residual_cross_attention":
        return ResidualCrossAttentionTranslator(n_technical, n_fundamental, config)
    if name == "gated_residual_cross_attention":
        return GatedResidualCrossAttentionTranslator(
            n_technical, n_fundamental, config
        )
    raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")

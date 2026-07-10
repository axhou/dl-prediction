from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 4096
    learning_rate: float = 7e-4
    patience: int = 3
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.25
    loss: str = "ce"
    focal_gamma: float = 1.0
    class_weight_power: float = 0.3
    bottom_weight_mult: float = 0.95
    middle_weight_mult: float = 1.0
    top_weight_mult: float = 1.08
    selection_metric: str = "balanced_composite"
    conservative_attn_init: float = 0.10
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    random_state: int = 42

    def updated(self, **overrides):
        return replace(self, **overrides)

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "candidate_a": TrainingConfig(),
    "attention_balanced": TrainingConfig(
        dropout=0.20,
        top_weight_mult=1.10,
    ),
    "conservative_final": TrainingConfig(
        learning_rate=5e-4,
        dropout=0.15,
        d_model=32,
        conservative_attn_init=0.15,
    ),
    "top_focus": TrainingConfig(
        bottom_weight_mult=0.90,
        top_weight_mult=1.15,
    ),
}


def get_preset(name: str, **overrides) -> TrainingConfig:
    if name not in PRESETS:
        choices = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}. Available: {choices}")
    return PRESETS[name].updated(**overrides)

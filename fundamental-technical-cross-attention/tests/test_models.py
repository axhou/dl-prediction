import pytest
import torch

from src.config import TrainingConfig
from src.models import MODEL_NAMES, build_model
from src.utils import set_seed


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_output_shape(model_name):
    config = TrainingConfig(d_model=16, n_heads=4, dropout=0.1)
    model = build_model(model_name, 5, 7, config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(8, 5), torch.randn(8, 7))
    assert output.shape == (8, 3)


def test_model_initialization_is_reproducible():
    config = TrainingConfig(d_model=16, n_heads=4)
    set_seed(17)
    first = build_model("direct_cross_attention", 5, 7, config)
    first_state = {key: value.clone() for key, value in first.state_dict().items()}
    set_seed(17)
    second = build_model("direct_cross_attention", 5, 7, config)
    assert all(
        torch.equal(first_state[key], value)
        for key, value in second.state_dict().items()
    )

import pytest
import torch

from neuralrec.nn.transformer import TransformerEncoder


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("d_model", [64, 128])
def test_transformer_encoder_forward_shape(
    batch_size: int, seq_len: int, d_model: int
) -> None:
    model = TransformerEncoder(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        causal=False,
    )
    model.eval()
    src = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out = model(src)
    assert out.shape == (batch_size, seq_len, d_model)

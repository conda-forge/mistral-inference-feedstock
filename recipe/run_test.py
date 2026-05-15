import torch

from mistral_inference.lora import LoraArgs, LoRALinear
from mistral_inference.rope import apply_rotary_emb, precompute_freqs_cis, precompute_freqs_cis_2d
from mistral_inference.transformer_layers import RMSNorm, repeat_kv


def test_rope_round_trip_shapes() -> None:
    freqs = precompute_freqs_cis(dim=4, end=3, theta=10_000.0)
    xq = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    xk = xq + 100

    out_q, out_k = apply_rotary_emb(xq, xk, freqs)

    assert out_q.shape == xq.shape
    assert out_k.shape == xk.shape
    torch.testing.assert_close(out_q[0], xq[0])
    torch.testing.assert_close(out_k[0], xk[0])


def test_rope_2d_shape() -> None:
    freqs = precompute_freqs_cis_2d(dim=8, height=2, width=3, theta=10_000.0)

    assert freqs.shape == (2, 3, 4)
    torch.testing.assert_close(freqs[0, 0].real, torch.ones(4))
    torch.testing.assert_close(freqs[0, 0].imag, torch.zeros(4))


def test_lora_linear_and_args() -> None:
    args = LoraArgs(rank=2, scaling=0.5)
    layer = LoRALinear(3, 2, rank=args.rank, scaling=args.scaling)
    x = torch.ones(4, 3)

    y = layer(x)

    assert y.shape == (4, 2)
    assert layer.bias is False


def test_rms_norm_and_repeat_kv() -> None:
    norm = RMSNorm(dim=3)
    x = torch.tensor([[1.0, 2.0, 3.0]])

    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    keys = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    values = keys + 10
    repeated_keys, repeated_values = repeat_kv(keys, values, repeats=2, dim=1)

    assert repeated_keys.shape == (1, 4, 3)
    assert repeated_values.shape == (1, 4, 3)
    torch.testing.assert_close(repeated_keys[:, 0], repeated_keys[:, 1])
    torch.testing.assert_close(repeated_values[:, 2], repeated_values[:, 3])


if __name__ == "__main__":
    test_rope_round_trip_shapes()
    test_rope_2d_shape()
    test_lora_linear_and_args()
    test_rms_norm_and_repeat_kv()

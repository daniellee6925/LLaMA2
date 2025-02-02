# get all imports
import torch
import torch.nn as nn
import torch.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # num of heads for queries
    n_kv_heads: Optional[int] = None  # num of heads for keys and values
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    norm_eps: float = 1e-5

    # needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RSMNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = pre_compute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the position [start pos, start pos + seq len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # apply all the encoding layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class RSMNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len , 1) -> (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # (B, seq_len, dim)
        return self.weight * self.norm(x.float()).type_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before self attention block
        self.attention_norm = RSMNorm(args.dim, eps=args.norm_eps)
        # normalization before the feed forward block
        self.ffn_norm = RSMNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freq_complex)
        out = h + self.feed_forward(self.ffn_norm(x), start_pos, freq_complex)
        return out

class SelfAttention(nn.Module):
    

def pre_compute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000
):
    """Function to obtain theta parameter
    e^(ix) = cos(x) + i*sin(x)
    """
    assert head_dim % 2 == 0, "Dimension must be an even number"
    # build the theta parameter
    # according to paper, theta_i = 10000^(-2(i-1)/dim) for [1, 2, ... dim/2]
    # shape: (head_dim /2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** ((theta_numerator) / head_dim)).to(device)

    # construct the position (m) parameter
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # multiply each theta by each position using outer product
    # shape: (seq_len) outer.prod (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()

    # compute complex numbers in polar form
    # c = R * exp(i * m * theta) where R = 1
    # (seq_len, head_dim/2)

    # z = r e^{i * theta} = r (cos(theta) + i *sin(theta))
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freq_complex


def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)  # add (B:0, H:2)
    # (B, seq_len, H, head_dim/2) x (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freq_complex
    # (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out

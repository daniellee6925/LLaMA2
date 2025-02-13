# get all imports
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # (vocab_size, dim)
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        # add n_layers of encoder blocks
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # rsm normalization step
        self.norm = RSMNorm(args.dim, eps=args.norm_eps)
        # output step to change dim for (dim -> vocab size)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # step to apply rotary positional embeddings
        self.freqs_complex = pre_compute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"

        # (B, seq_len) -> (B, seq_len, dim), add embeddings for tokens
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the position [start pos, start pos + seq len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # apply all the encoding layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        # apply normalization
        h = self.norm(h)
        # apply final linear layer and set dtype to float
        output = self.output(h).float()
        return output


class RSMNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # eps to prevent dividing by 0:gamma parameter
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len , 1) -> (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # (B, seq_len, dim)
        # set type as float for norm calc and reset type to original
        return self.weight * self.norm(x.float()).type_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        # divide dims by num of heads
        self.head_dim = args.dim // args.n_heads

        # set self attention layer
        self.attention = SelfAttention(args)
        # set feed forward layer
        self.feed_forward = FeedForward(args)

        # normalization before self attention block
        self.attention_norm = RSMNorm(args.dim, eps=args.norm_eps)
        # normalization before the feed forward block
        self.ffn_norm = RSMNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        # each block goes from normalize -> self attention -> add residual layer
        # normalize -> feed forward -> add residual layer
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freq_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # number of heads for key and value matrices
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for query matrix
        self.n_heads_q = args.n_heads
        # number of times the key and value heads should be repeated to match the query heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        # apply linear layer to transform embedding dim for K, Q, V
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # to ensure dimension matches
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # initalize cache for keys and values matrices
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (batch_size, 1, dim)
        # apply wq, wk, and wv to queries, keys, and values
        # (batch_size, 1, dim) ->  (batch_size, 1, num_head_q * head_dim)
        xq = self.wq(x)
        # (batch_size, 1, dim) ->  (batch_size, 1, n_kv_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # expand torch dimension to apply rotary positional embeddings
        # (batch_size, 1, num_head_q * head_dim) -> # (batch_size, 1, num_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (batch_size, 1, n_kv_heads * head_dim) -> (batch_size, 1, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary positional encodings to queries and keys
        xq = apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device=x.device)

        # replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # retrieve all the cache and key values so far
        # (batch_size, seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]

        # repeat the number of heads of K and V to reach the number of Q
        keys = repeat_KV(keys, self.n_rep)
        values = repeat_KV(values, self.n_rep)

        # (batch_size, 1, n_q_heads, head_dim) -> (batch_size, n_q_heads, 1, head_dim)
        xq = xq.transpose(1, 2)
        # (batch_size, seq_len_kv, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (batch_size, n_q_heads, 1, head_dim) @ (batch_size, n_q_heads, head_dim, seq_len_kv)
        # -> (batch_size, n_q_heads, 1, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (batch_size, n_q_heads, 1, head_dim) @ (batch_size, n_q_heads, seq_len_kv, head_dim)
        # -> (batch_size, n_q_heads, 1, head_dim)
        output = torch.matmul(scores, values)

        # (batch_size, n_q_heads, 1, head_dim) -> (batch_size, 1, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (batch_size, 1, dim) -> (batch_size, 1, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden_dim to the nearest multiplier of the multiple of parameter
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


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


def repeat_KV(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]  # add None and expand dim
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # add num heads
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # reshape heads
        )

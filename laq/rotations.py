"""Roles of the block-level Linears of a WAM and the rotation applied at the input of each.

Adaptive layer normalization and per-dimension modulation rule out one global rotation of the residual stream, so
every rotation is applied locally at the input of a Linear and compensated in its own weight (x' = x R, W' = W R).
"""
from __future__ import annotations

from collections import namedtuple
import re
import zlib

import torch

from . import quant as q

Role = namedtuple("Role", "expert block attn kind")
_FW = re.compile(r"^(video_expert|action_expert)\.blocks\.(\d+)\.(?:(self_attn|cross_attn)\.(q|k|v|o)|ffn\.(0|2))$")
_CO = re.compile(r"^net\.blocks\.(\d+)\._checkpoint_wrapped_module\.(?:(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|"
                 r"output_proj)|mlp\.layer(1|2))$")
_KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "output_proj": "o"}


def parse(name):
    """Role of a block-level Linear (Fast-WAM or Cosmos-Policy naming), or None for an interface or
    time-conditioning layer, which stays in BF16."""
    m = _FW.match(name)
    if m:
        expert, block, attn, kind, ffn = m.groups()
        expert = expert.split("_")[0]
        return Role(expert, int(block), attn.split("_")[0], kind) if attn else Role(expert, int(block), None,
                                                                                    "mlp0" if ffn == "0" else "mlp2")
    m = _CO.match(name)
    if m:
        block, attn, kind, mlp = m.groups()
        return Role("net", int(block), attn.split("_")[0], _KINDS[kind]) if attn else Role("net", int(block), None,
                                                                                         "mlp0" if mlp == "1" else "mlp2")
    return None


def stream_of(r):
    return f"{r.expert}.context" if r.attn == "cross" and r.kind in ("k", "v") else f"{r.expert}.hidden"


def attn_key(r):
    return f"{r.expert}.{r.block}.{r.attn}"


def seed_of(base, tag):
    return int(base) + zlib.crc32(tag.encode()) % (1 << 31)


class Had:
    """x -> x diag(signs) H (signs optional)."""

    def __init__(self, signs=None):
        self.s = signs

    def apply(self, x):
        return q.had(x if self.s is None else x * self.s)

    def inv(self, y):
        z = q.had(y, inverse=True)
        return z if self.s is None else z * self.s


class HeadHad:
    """Input of the attention output projection: Hadamard over the head dimension, then across heads."""

    def __init__(self, heads, head_dim):
        self.h, self.d = heads, head_dim

    def apply(self, x):
        y = q.had(x.reshape(*x.shape[:-1], self.h, self.d))
        return q.had(y.transpose(-1, -2)).transpose(-1, -2).reshape(x.shape)

    def inv(self, y):
        z = q.had(y.reshape(*y.shape[:-1], self.h, self.d).transpose(-1, -2), inverse=True).transpose(-1, -2)
        return q.had(z, inverse=True).reshape(y.shape)


def sandwich(xf, h):
    return xf.apply(xf.apply(h).t()).t()


def input_transform(role, in_features, arch, seed, device):
    """The rotation at the input of a block-level Linear. Linears that read the same stream share one rotation."""
    if role.kind == "mlp2":
        # A GELU output has a positive mean, which a plain Hadamard concentrates in one coordinate; random signs remove it.
        return Had(q.random_signs(in_features, seed_of(seed, f"{role.expert}.{role.block}.mlp2"), device))
    if role.kind == "o":
        return HeadHad(arch["heads"], arch["head_dim"])
    return Had(q.random_signs(in_features, seed_of(seed, stream_of(role)), device))


# ------------------------------------------------------------------ weight fitting


def fit(weight, hessian, xf):
    """GPTQ of one Linear in rotated coordinates. weight [rows, in] and the input second moment hessian [in, in] in
    the original coordinates; returns the packed 4-bit weight and its relative proxy error."""
    wc, hc = xf.apply(weight), sandwich(xf, hessian)
    codes, scales = q.gptq_row(wc, hc)
    bank = q.encode_sym(codes, scales[:, None], weight.shape[1])
    return bank, q.proxy_nmse(q.decode_sym(bank, torch.float32), wc, hc)

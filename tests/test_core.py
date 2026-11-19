"""Arithmetic of LAQ on CPU: refinement, selection, the reader metric, costs and the allocation, weight packing."""
from __future__ import annotations

import torch

from laq import checkpoint, fastwam as laq, quant as q, rotations as mt

torch.manual_seed(0)
ARCH = dict(heads=4, head_dim=16)
STEPS, FRAME, VIDEO, ACT, CTX = 2, 2, 6, 3, 3


def test_refine():
    x = torch.randn(2, 5, 64) * torch.linspace(0.1, 3, 5)[None, :, None]
    assert torch.equal(laq.refine(x, None, laq.a_base, laq.a_fine), laq.a_base(x))
    rows = torch.tensor([True, False, True, False, False])
    y = laq.refine(x, rows, laq.a_base, laq.a_fine)
    assert torch.equal(y[:, ~rows], laq.a_base(x)[:, ~rows])
    s1 = x.abs().amax(-1, keepdim=True) / 7
    err = (y - x).abs()
    assert bool((err[:, rows] <= s1[:, rows] / 28 * (1 + 1e-4) + 1e-7).all())
    plain = (laq.a_fine(x) - x).pow(2).sum(-1)
    fine = (laq.two_pass(x, laq.a_fine) - x).pow(2).sum(-1)
    assert bool((fine * 60 < plain).all())
    for base, fine_q in ((laq.k_base, laq.k_fine), (laq.v_base, laq.v_fine)):
        r = torch.randn(1, 5, 4, 16)
        z = laq.refine(r, rows, base, fine_q)
        assert torch.equal(z[:, ~rows], base(r)[:, ~rows])
        assert float((z[:, rows] - r[:, rows]).pow(2).sum()) * 50 < float((fine_q(r)[:, rows] - r[:, rows]).pow(2).sum())
    assert torch.equal(laq.refine(x, torch.zeros(5, dtype=torch.bool), laq.a_base, laq.a_fine), laq.a_base(x))


def test_attention_and_top():
    qr, k = torch.randn(1, 3, 64), torch.randn(1, 9, 64)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[0, 4:] = False
    p = laq.attention(qr, k, 4, mask)
    assert p.shape == (4, 3, 9) and float((p.sum(-1) - 1).abs().max()) < 1e-5 and float(p[:, 0, 4:].abs().max()) == 0
    g = torch.tensor([0.1, 0.9, 0.5, 0.3])
    assert laq.top(g, 2).tolist() == [False, True, True, False] and int(laq.top(g, 0).sum()) == 0 and int(laq.top(g, 9).sum()) == 4
    assert laq.overlap(laq.top(g, 2), laq.top(g, 1)) == 0.5


def test_metric():
    D, r = 48 * 2 * 4 * 4, 3
    E = torch.linalg.qr(torch.randn(D, r))[0].t()
    lam, d = torch.rand(r), torch.rand(D) * 0.01
    tau = float(lam.sum() + d.sum())
    diag = (lam[:, None] * E.square()).sum(0) + d
    saved = dict(E=E, lam=lam, d_res=d, rho=0.1, D=D, tau=tau, diag=diag, shape=[1, 48, 2, 4, 4])
    metric = laq.Metric(saved, torch.device("cpu"))
    dense = 0.9 * D / tau * (E.t() @ torch.diag(lam) @ E + torch.diag(d)) + 0.1 * torch.eye(D)
    v = torch.randn(D)
    assert abs(metric.quad(v) - float(v @ dense @ v)) < 1e-2 * float(v @ dense @ v)
    gaze = laq.token_gaze(saved)
    assert gaze.shape == (2 * 2 * 2,) and abs(float(gaze.sum()) - float(dense.diagonal().sum())) < 1e-2 * D
    block = dense.diagonal().reshape(48, 2, 4, 4)[:, 1, 2:4, 0:2].sum()
    assert abs(float(gaze[4 + 2]) - float(block)) < 1e-3 * float(block)
    target = torch.randn(1, 48, 3, 4, 4)
    future = target + 0.1 * torch.randn_like(target)
    eye = dict(saved, rho=1.0)
    plain = float((future[:, :, 1:] - target[:, :, 1:]).square().sum() / target[:, :, 1:].square().sum())
    assert abs(laq.Metric(eye, torch.device("cpu")).nmse(future, target) - plain) < 1e-5
    action = torch.zeros(32, 7)
    e = laq.errors(metric, action + 0.5, future, dict(action=action - 0.5, future=target))
    assert abs(e["A"] - 1.0) < 1e-6 and e["gripper_flips"] == 10 and abs(e["Z"] - plain) < 1e-6


def names(blocks=2):
    out = []
    for expert, hidden, ffn in (("video_expert", 64, 128), ("action_expert", 32, 64)):
        for b in range(blocks):
            for attn in ("self_attn", "cross_attn"):
                out += [(f"{expert}.blocks.{b}.{attn}.{k}", hidden) for k in ("q", "k", "v")]
                out.append((f"{expert}.blocks.{b}.{attn}.o", 64))
            out += [(f"{expert}.blocks.{b}.ffn.0", hidden), (f"{expert}.blocks.{b}.ffn.2", ffn)]
    return out + [("video_expert.head.head", 64)]


def test_costs_and_solve():
    cost = laq.costs(names(), ARCH, STEPS, FRAME, VIDEO, ACT, CTX)
    assert cost["token_pass"] == dict(video=2 * (64 + 64 + 64 + 64 + 64 + 128), action=2 * (32 + 64 + 32 + 64 + 32 + 64))
    assert cost["context_pass"] == dict(video=2 * 64, action=2 * 32) and cost["kv_token"]["video"] == 2 * 2 * 64
    unit, base = cost["unit"], cost["base"]
    assert unit["action"]["a"] == ACT * STEPS * cost["token_pass"]["action"] and unit["text.video"]["kv"] == CTX * 256
    assert base["a"] == (FRAME + 4 * 3) * cost["token_pass"]["video"] + unit["action"]["a"] + CTX * (128 + 64)
    assert base["kv"] == VIDEO * 256 + 2 * CTX * 256
    benefit = {"text.video": 0.0, "text.action": 0.0, "obsA": 4.0, "obsKV": 1.0, "futKV": 2.0, "fut.0": 0.0, "fut.1": 3.0, "fut.2": 0.5}
    g_fut, g_obs = [0.1, 0.6, 0.2, 0.1], [0.7, 0.3]
    for fraction in (0.2, 0.4, 0.8, 5.0):
        out = laq.solve(benefit, cost, g_fut, g_obs, fraction, STEPS)
        spec, acc = out["spec"], out["accounting"]
        assert spec["action"] and not spec["text_video"] and spec["fut"][0] == 0
        assert acc["fraction"] <= fraction + 1e-9 and acc == laq.accounting(spec, cost)
        assert abs(acc["a_bits"] - 4 * (1 + acc["extra_a"] / base["a"])) < 1e-9
    assert laq.solve(benefit, cost, g_fut, g_obs, 5.0, STEPS)["spec"] == dict(
        text_video=False, text_action=False, action=True, obsA=2, obsKV=2, futKV=4, fut=[0, 4, 4])
    small = laq.solve(benefit, cost, g_fut, g_obs, 0.2, STEPS)
    assert small["spec"]["futKV"] >= 1 and small["spec"]["obsKV"] >= small["spec"]["obsA"]
    try:
        laq.solve(benefit, cost, g_fut, g_obs, 1e-4, STEPS)
        raise AssertionError("a budget below the action rows must fail")
    except ValueError:
        pass


def test_hadamard_is_orthonormal():
    for n in (64, 96, 112):
        x = torch.randn(3, n)
        assert float((q.had(q.had(x), inverse=True) - x).abs().max()) < 1e-4
        assert abs(float(q.had(x).square().sum() / x.square().sum()) - 1) < 1e-4


def test_weight_roundtrip(tmp_path):
    role = mt.parse("video_expert.blocks.0.ffn.0")
    xf = mt.input_transform(role, 64, ARCH, 7, torch.device("cpu"))
    w, x = torch.randn(32, 64), torch.randn(512, 64)
    bank, error = mt.fit(w, x.t() @ x, xf)
    assert bank["packed"].shape == (32, 32) and 0 < error < 0.05
    checkpoint.save(tmp_path, dict(wam="fastwam"), {"video_expert.blocks.0.ffn.0": bank})
    config, weights = checkpoint.load(tmp_path)
    back = weights["video_expert.blocks.0.ffn.0"]
    assert config["wam"] == "fastwam" and torch.equal(q.decode(back), q.decode(bank))
    y, y_hat = x @ w.t(), xf.apply(x) @ q.decode(back, torch.float32).t()
    assert float((y_hat - y).square().sum() / y.square().sum()) < 0.05

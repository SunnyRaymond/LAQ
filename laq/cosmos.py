"""LAQ on Cosmos-Policy: the reference runtime (FP32 arithmetic on rotated operands, decoded 4-bit weights).

Base quantizers:
  weights      GPTQ, one MSE-clipped scale per row, in randomized-Hadamard coordinates
  activations  per-token symmetric INT4, clip 0.9
  keys/values  per-head asymmetric INT4, clip 0.95, Hadamard over the head dimension, at every attention and step

Refinement: a selected token gets two 4-bit passes, x^ = Q(x) + Q(x - Q(x)), each with its own per-token scale and
no clipping, in the activations (every Linear that reads the token) and in the keys and values.

Cosmos-Policy denoises its latent slots jointly with one DiT, so nothing but the text is written once. Token groups
(slots of the latent, `per` tokens each):
  obs   the conditioning slots (taken from the condition mask of the query)
  act   the action slot (4); its activation rows are always refined
  fut   the future-image slots (6, 7), where the reader metric lives
  aux   the other generated slots (future proprio, value)
Units: (group, A | KV, denoiser evaluation s) for the tokens, and the text rows (the context rows at the
cross-attention K/V Linears and the text K/V: the same at every evaluation, so written once and read at each).

Gaze: the attention mass the action tokens put on a token, summed over heads and self-attentions of one denoiser
evaluation. The gaze of one evaluation selects the tokens of the next; the first evaluation of a query uses the gaze of
the last evaluation of the query before it, and the first query of an episode uses the offline gaze.

The reader state: the future-image slots of the solver state just before the cut-th denoiser evaluation, the state on
which the reader metric is defined.
"""
from __future__ import annotations

from collections import Counter
import inspect
import math
import re

import torch
import torch.nn.functional as F

from . import quant as q
from . import rotations as mt

A_CLIP, KV_CLIP = 0.9, 0.95
ACT_SLOT, FUT_SLOTS = 4, (6, 7)
GROUPS = ("obs", "fut", "aux", "act")
UNITS = [(g, k) for g in ("obs", "fut", "aux") for k in ("A", "KV")] + [("act", "KV")]
ATTENTION = re.compile(r"blocks\.(\d+)\._checkpoint_wrapped_module\.(self|cross)_attn$")


def a_base(r):
    return q.a_token(r, sym=True, clip=A_CLIP)


def a_fine(r):
    return q.a_token(r, sym=True, clip=1.0)


def k_base(r):
    return q.a_token(r, sym=False, clip=KV_CLIP)


def k_fine(r):
    return q.a_token(r, sym=False, clip=1.0)


def v_base(r):
    return q.a_group(r, r.shape[-1], clip=KV_CLIP)


def v_fine(r):
    return q.a_group(r, r.shape[-1], clip=1.0)


def two_pass(x, fine):
    first = fine(x)
    return first + fine(x - first)


def refine(x, idx, base, fine):
    """base(x) on every token (axis 1); the tokens `idx` (indices, None for none) get two fine passes."""
    out = base(x)
    if idx is None or idx.numel() == 0:
        return out
    return out.index_copy(1, idx, two_pass(x.index_select(1, idx), fine))


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def spec_none(steps):
    return dict(text=False, action=False, counts={f"{g}.{k}": [0] * steps for g, k in UNITS})


class State:
    def __init__(self, steps, cut, arch, rotation_seed, device):
        self.steps, self.cut = int(steps), int(cut)
        self.arch, self.h, self.d = arch, arch["heads"], arch["head_dim"]
        self.rotation_seed, self.device = rotation_seed, device
        self.quant = False      # False: BF16 weights (pass-through or error injection); True: decoded 4-bit weights
        self.inject = None      # "frozen" | "redrawn": sign-randomized 4-bit rounding error of the text rows (BF16 mode)
        self.seed = 0
        self.gen = torch.Generator(device=device)
        self.signs = {}
        self.spec, self.offline, self.online = None, None, False
        self.watch = False      # collect the attention gaze
        self.layout = None      # dict(slots, per, S, tokens={group: indices}, cond=[slots])
        self.cond = None
        self.n_eval, self.in_query, self.queries = 0, False, 0
        self.idx = {"A": None, "KV": None}
        self.acc = None
        self.step_gaze, self.live, self.last_top = [], None, None
        self.z = None
        self.trace = []
        self.seen, self.counts, self.shapes = set(), Counter(), {}
        self.fast = True        # reuse operands that are provably the same (bit-identical results, fewer GPU calls)
        self.shared = None      # (input tensor, stream, quantized operand): the Linears of one attention that read one input
        self.frozen = {}        # per text site: (input, output) of the last call; the context is the same at every evaluation

    # ---------------- selection
    def use(self, spec, online=False):
        self.spec, self.online = spec, online
        self.reset()

    def reset(self):
        """Start of an episode (or of an independent query): the offline gaze selects the first evaluation."""
        self.n_eval, self.in_query = 0, False
        self.live, self.last_top, self.trace = None, None, []
        self.shared, self.frozen = None, {}
        self.idx = {"A": None, "KV": None}

    def set_layout(self, slots, cond):
        if FUT_SLOTS[1] >= slots or ACT_SLOT in cond or any(s in cond for s in FUT_SLOTS):
            raise RuntimeError(f"Unexpected latent layout: {slots} slots, conditioning slots {cond}")
        if self.cond is None:
            self.cond, self.slots = list(cond), int(slots)
        elif self.cond != list(cond) or self.slots != int(slots):
            raise RuntimeError(f"The latent layout changed: {self.slots} slots / {self.cond} -> {slots} / {cond}")

    def tokens_seen(self, S):
        """Called by every hidden-stream operand: fixes the token layout at the first one and checks it afterwards."""
        if self.layout is not None:
            if S != self.layout["S"]:
                raise RuntimeError(f"A hidden operand holds {S} tokens, expected {self.layout['S']}")
            return
        if self.cond is None or S % self.slots:
            raise RuntimeError(f"{S} tokens do not split into the {getattr(self, 'slots', None)} latent slots")
        per = S // self.slots
        group = {s: "obs" for s in self.cond}
        group[ACT_SLOT] = "act"
        group.update({s: "fut" for s in FUT_SLOTS})
        tokens = {g: torch.cat([torch.arange(s * per, (s + 1) * per) for s in range(self.slots) if group.get(s, "aux") == g]
                               or [torch.zeros(0, dtype=torch.long)]).to(self.device) for g in GROUPS}
        self.layout = dict(slots=self.slots, per=per, S=S, tokens=tokens, cond=self.cond)
        self.shapes.update(tokens=S, per_slot=per, slots=self.slots, conditioning_slots=self.cond,
                           groups={g: int(t.numel()) for g, t in tokens.items()})
        self.acc = torch.zeros(S, device=self.device)
        self.select()

    def select(self):
        """Refined rows of the current evaluation, from the spec and the gaze that is in force."""
        self.idx = {"A": None, "KV": None}
        if self.spec is None or self.layout is None:
            return
        gaze = self.live if (self.online and self.live is not None) else self.offline
        s = self.n_eval
        for kind in ("A", "KV"):
            parts = [self.layout["tokens"]["act"]] if (kind == "A" and self.spec["action"]) else []
            for g in GROUPS:
                n = self.spec["counts"].get(f"{g}.{kind}", [0] * self.steps)[s]
                tok = self.layout["tokens"][g]
                n = min(int(n), tok.numel())
                if n == tok.numel():
                    parts.append(tok)
                elif n > 0:
                    score = gaze[tok] if gaze is not None else torch.zeros(tok.numel(), device=self.device)
                    parts.append(tok[torch.topk(score, n).indices])
            if parts and sum(p.numel() for p in parts):
                self.idx[kind] = torch.cat(parts).sort().values

    def text_idx(self, T):
        return torch.arange(T, device=self.device) if (self.spec is not None and self.spec["text"]) else None

    def same(self, key, x):
        """The cached output of a text site if its input equals the one of the last call, else None."""
        hit = self.frozen.get(key)
        if hit is not None and (hit[0] is x or (hit[0].shape == x.shape and torch.equal(hit[0], x))):
            return hit[1]
        return None

    # ---------------- error injection (BF16 weights, text rows)
    def sign(self, key, shape):
        if self.inject == "frozen":
            if key not in self.signs:
                self.signs[key] = self._draw(shape)
            if self.signs[key].shape != torch.Size(shape):
                raise RuntimeError(f"Frozen sign pattern of {key} changed shape")
            return self.signs[key]
        return self._draw(shape)

    def _draw(self, shape):
        return torch.randint(0, 2, tuple(shape), generator=self.gen, device=self.device).float() * 2 - 1

    # ---------------- operands
    def linear(self, x, module, xf, text, name, stream):
        self.seen.add(name)
        if text:
            self.shapes["context_tokens"] = int(x.shape[1])
        if self.quant:
            if text:
                if self.fast:
                    out = self.same(name, x)
                    if out is not None:
                        return out
                    if self.shared is not None and self.shared[0] is x and self.shared[1] == stream:
                        out = F.linear(self.shared[2], module.weight, module.bias)
                        self.frozen[name] = (x, out)
                        return out
                operand = refine(xf.apply(x.float()), self.text_idx(x.shape[1]), a_base, a_fine).to(x.dtype)
                if stream is not None:
                    self.shared = (x, stream, operand)
                out = F.linear(operand, module.weight, module.bias)
                self.frozen[name] = (x, out)
                return out
            if self.fast and stream is not None and self.shared is not None and self.shared[0] is x and self.shared[1] == stream:
                operand = self.shared[2]
            else:
                shape = x.shape
                flat = x.reshape(shape[0], -1, shape[-1])
                self.tokens_seen(flat.shape[1])
                operand = refine(xf.apply(flat.float()), self.idx["A"], a_base, a_fine).to(x.dtype).reshape(shape)
                if stream is not None:
                    self.shared = (x, stream, operand)
            return F.linear(operand, module.weight, module.bias)
        if not text:
            self.tokens_seen(x.reshape(x.shape[0], -1, x.shape[-1]).shape[1])
        elif self.inject is not None:
            r = xf.apply(x.float())
            x = xf.inv(r + self.sign(f"A.{name}", r.shape) * (a_base(r) - r)).to(x.dtype)
        return F.linear(x, module.weight, module.bias)

    def kv(self, x, kind, site, key):
        """K or V [B, T, heads, head_dim] of one attention (site "self" or "text")."""
        self.counts[site] += 1
        base, fine = (k_base, k_fine) if kind == "K" else (v_base, v_fine)
        text = site == "text"
        if self.quant:
            if text and self.fast:
                out = self.same(key, x)
                if out is not None:
                    return out
            r = q.fht(x.float())
            out = q.fht(refine(r, self.text_idx(x.shape[1]) if text else self.idx["KV"], base, fine)).reshape(x.shape).to(x.dtype)
            if text:
                self.frozen[key] = (x, out)
            return out
        if text and self.inject is not None:
            r = q.fht(x.float())
            return q.fht(r + self.sign(key, r.shape) * (base(r) - r)).reshape(x.shape).to(x.dtype)
        return x

    def observe(self, query, key):
        """Attention mass of the action tokens on every token, added over heads (batch element 0)."""
        if not self.watch or self.layout is None:
            return
        qa = query[0].float().index_select(0, self.layout["tokens"]["act"]).transpose(0, 1)   # [heads, act, d]
        ka = key[0].float().transpose(0, 1)                                                    # [heads, S, d]
        self.acc += torch.softmax(qa @ ka.transpose(1, 2) / math.sqrt(qa.shape[-1]), -1).sum((0, 1))

    # ---------------- query phases
    def start_query(self):
        self.n_eval, self.in_query = 0, True
        self.signs.clear()
        self.gen.manual_seed(int(self.seed))
        self.step_gaze, self.z = [], None
        self.shared = None
        if self.acc is not None:
            self.acc.zero_()

    def begin_eval(self, xt, condition):
        if not self.in_query:
            raise RuntimeError("A denoiser evaluation outside generate_samples_from_batch")
        if self.n_eval >= self.steps:
            raise RuntimeError(f"More than {self.steps} denoiser evaluations in one query")
        mask = condition.condition_video_input_mask_B_C_T_H_W
        cond = [int(i) for i in (mask[0].float().mean((0, 2, 3)) > 0.5).nonzero().flatten().tolist()]
        self.set_layout(int(xt.shape[2]), cond)
        if self.n_eval == self.cut - 1:
            self.z = xt[:, :, FUT_SLOTS[0]:FUT_SLOTS[1] + 1].detach().to(torch.float64).clone()
        self.shared = None
        self.select()

    def end_eval(self):
        if self.watch and self.acc is not None:
            gaze = self.acc.clone()
            self.acc.zero_()
            self.step_gaze.append(gaze)
            if self.online:
                self.live = gaze
        self.n_eval += 1

    def finish_query(self):
        if self.n_eval != self.steps:
            raise RuntimeError(f"The query ran {self.n_eval} denoiser evaluations, expected {self.steps}")
        self.in_query = False
        self.queries += 1
        if self.watch and self.step_gaze and self.layout is not None:
            fut = self.layout["tokens"]["fut"]
            k = max(1, fut.numel() // 4)
            now = torch.topk(self.step_gaze[-1][fut], k).indices.tolist()
            off = None if self.offline is None else torch.topk(self.offline[fut], k).indices.tolist()
            self.trace.append(dict(previous=None if self.last_top is None else overlap(now, self.last_top),
                                   offline=None if off is None else overlap(now, off)))
            self.last_top = now


def install(st, model, modules):
    """Replace the forwards of the block-level Linears, wrap compute_qkv of every attention, and track the denoiser
    evaluations of a query. The model keeps its BF16 behaviour bit for bit until weights are loaded or an error is
    injected."""
    installed = 0
    for name, module in modules.items():
        role = mt.parse(name)
        if role is None:
            continue
        xf = mt.input_transform(role, module.in_features, st.arch, st.rotation_seed, st.device)
        text = role.attn == "cross" and role.kind in ("k", "v")
        stream = f"{mt.stream_of(role)}.{role.block}.{role.attn}" if role.kind in ("q", "k", "v") else None
        module.forward = (lambda x, _m=module, _xf=xf, _t=text, _n=name, _s=stream: st.linear(x, _m, _xf, _t, _n, _s))
        installed += 1

    wrapped = 0
    for name, attn in model.named_modules():
        m = ATTENTION.search(name)
        if m and hasattr(attn, "compute_qkv"):
            if (attn.n_heads, attn.head_dim) != (st.h, st.d):
                raise RuntimeError(f"{name} has {attn.n_heads}x{attn.head_dim} heads, expected {st.h}x{st.d}")
            site = "self" if m.group(2) == "self" else "text"

            def compute(*args, _original=attn.compute_qkv, _site=site, _key=f"{m.group(1)}.{m.group(2)}", **kwargs):
                q_, k, v, *rest = _original(*args, **kwargs)
                k, v = st.kv(k, "K", _site, f"K.{_key}"), st.kv(v, "V", _site, f"V.{_key}")
                if _site == "self":
                    st.observe(q_, k)
                return (q_, k, v, *rest)
            attn.compute_qkv = compute
            wrapped += 1

    generate_original, denoise_original = model.generate_samples_from_batch, model.denoise

    def generate(*args, **kwargs):
        st.start_query()
        out = generate_original(*args, **kwargs)
        st.finish_query()
        return out

    signature = inspect.signature(denoise_original)

    def denoise(*args, **kwargs):
        values = list(signature.bind(*args, **kwargs).arguments.values())  # (xt, sigma, condition)
        st.begin_eval(values[0], values[2])
        out = denoise_original(*args, **kwargs)
        st.end_eval()
        return out
    model.generate_samples_from_batch = generate
    model.denoise = denoise
    return installed, wrapped


def load_weights(st, modules, weights):
    """Decode the packed 4-bit weights ({module name: bank}, see laq.checkpoint) into the Linears; from here on the
    forwards quantize."""
    loaded = 0
    for name, module in modules.items():
        if mt.parse(name) is None:
            continue
        bank = weights[name]
        if tuple(bank["shape"]) != tuple(module.weight.shape):
            raise RuntimeError(f"The checkpoint does not match {name}")
        with torch.no_grad():
            module.weight.copy_(q.decode(bank).to(module.weight.device, module.weight.dtype))
        loaded += 1
    st.quant, st.inject = True, None
    st.shared, st.frozen = None, {}
    return loaded


# ---------------- the reader metric
class Metric:
    """M = (1 - rho) (D / tau) (E^T diag(lam) E + diag(d)) + rho I on the flattened future slots of the cut state."""

    def __init__(self, saved, device):
        self.E = saved["E"].to(device, torch.float32)
        self.lam = saved["lam"].to(device, torch.float32)
        self.dres = saved["d_res"].to(device, torch.float32)
        self.rho, self.D, self.tau = float(saved["rho"]), int(saved["D"]), float(saved["tau"])
        self.c = (1 - self.rho) * self.D / self.tau
        self.device = device

    def quad(self, v):
        v = v.reshape(-1).to(self.device, torch.float32)
        p = self.E @ v
        return float(self.c * ((self.lam * p.square()).sum() + (self.dres * v.square()).sum()) + self.rho * v.square().sum())

    def nmse(self, z, target):
        t = target.to(self.device, torch.float32)
        return self.quad(z.to(self.device, torch.float32) - t) / float(t.square().sum().clamp_min(1e-12))


def token_gaze(saved, patch=2):
    """Offline gaze of the future tokens: the trace of the block of M that belongs to each token, order (slot, row, column)."""
    rho, D, tau = float(saved["rho"]), int(saved["D"]), float(saved["tau"])
    _, c, f, h, w = saved["shape"]
    m = ((1 - rho) * D / tau * saved["diag"].double() + rho).reshape(c, f, h, w).sum(0)
    return m.reshape(f, h // patch, patch, w // patch, patch).sum((2, 4)).reshape(-1).float()


def errors(metric, action, z, teacher):
    a, ta = torch.as_tensor(action).float().cpu(), teacher["action"].float()
    tz = teacher["z"].float()
    zf = z.float().cpu()
    return dict(A=float((a[:, :6] - ta[:, :6]).square().mean()),
                Z=float((zf - tz).square().sum() / tz.square().sum()),
                ZM=metric.nmse(zf, tz), gripper_flips=int((a[:, 6].sign() != ta[:, 6].sign()).sum()))


# ---------------- costs and the allocation (plain Python)
def costs(names_and_widths, arch, steps, groups, context_tokens):
    """Operand values (4-bit numbers) of one query, counted as a deployment computes them: operands shared by several
    Linears once, the text rows and the text K/V once per query. names_and_widths: [(module name, in_features)];
    groups: {group: tokens}."""
    tok, ctx, attentions = 0, 0, 0
    for name, width in names_and_widths:
        role = mt.parse(name)
        if role is None:
            continue
        if role.attn == "cross" and role.kind in ("k", "v"):
            if role.kind == "k":
                ctx += width
        elif not (role.attn == "self" and role.kind in ("k", "v")):
            tok += width
        if role.attn == "self" and role.kind == "q":
            attentions += 1
    kv_tok = attentions * 2 * arch["heads"] * arch["head_dim"]
    S = sum(groups.values())
    unit = {f"{g}.A": dict(a=tok, kv=0) for g in GROUPS}
    unit.update({f"{g}.KV": dict(a=0, kv=kv_tok) for g in GROUPS})
    unit["action"] = dict(a=groups["act"] * steps * tok, kv=0)
    unit["text"] = dict(a=context_tokens * ctx, kv=context_tokens * kv_tok)
    base = dict(a=S * steps * tok + unit["text"]["a"], kv=S * steps * kv_tok + unit["text"]["kv"])
    return dict(unit=unit, base=base, token_pass=tok, context_pass=ctx, kv_token=kv_tok, groups=dict(groups),
                context_tokens=context_tokens)


def solve(benefit, cost, gaze, fraction, steps):
    """Threshold rule: refine a unit iff benefit / cost >= mu, with mu set by the budget (a fraction of the operand
    values of a query). The action rows are always refined and charged first. The benefit of a class (group, kind,
    evaluation) is split over its tokens in proportion to the gaze. gaze: {group: [share per token]}."""
    unit, base = cost["unit"], cost["base"]
    price = {k: v["a"] + v["kv"] for k, v in unit.items()}
    budget = fraction * (base["a"] + base["kv"]) - price["action"]
    if budget < 0:
        raise ValueError("The budget does not cover the action rows")
    share = {}
    for g, values in gaze.items():
        v = [max(float(x), 0.0) for x in values]
        total = sum(v) or 1.0
        share[g] = sorted((x / total for x in v), reverse=True)
    cands = [(benefit["text"] / price["text"], price["text"], "text", 0)]
    for g, k in UNITS:
        for s in range(steps):
            cands += [(benefit[f"{g}.{k}.{s}"] * x / price[f"{g}.{k}"], price[f"{g}.{k}"], f"{g}.{k}", s) for x in share[g]]
    cands.sort(key=lambda t: -t[0])
    spec = spec_none(steps)
    spec["action"] = True
    spent, mu = 0.0, None
    for ratio, c, cls, s in cands:
        if ratio <= 0:
            break
        if spent + c <= budget:
            if cls == "text":
                spec["text"] = True
            else:
                spec["counts"][cls][s] += 1
            spent += c
            mu = ratio
    return dict(spec=spec, mu=mu, accounting=accounting(spec, cost), budget_values=budget + price["action"])


def accounting(spec, cost):
    unit, base = cost["unit"], cost["base"]
    extra = dict(a=0.0, kv=0.0)
    items = [("text", int(spec["text"])), ("action", int(spec["action"]))] + [(k, sum(v)) for k, v in spec["counts"].items()]
    for cls, n in items:
        extra["a"] += n * unit[cls]["a"]
        extra["kv"] += n * unit[cls]["kv"]
    return dict(extra_a=extra["a"], extra_kv=extra["kv"], base_a=base["a"], base_kv=base["kv"],
                a_bits=4 * (1 + extra["a"] / base["a"]), kv_bits=4 * (1 + extra["kv"] / base["kv"]),
                fraction=(extra["a"] + extra["kv"]) / (base["a"] + base["kv"]))


# ---------------- loading a checkpoint
def target_linears(model):
    """The block-level Linears of the DiT, by name."""
    return {name: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and mt.parse(name) is not None}


def new_state(model, config, device=None):
    """Hook a Cosmos-Policy model (BF16 behaviour until weights are loaded). Returns (state, block-level Linears)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    st = State(config["steps"], config["cut"], config["arch"], config["rotation_seed"], device)
    modules = target_linears(model)
    install(st, model, modules)
    return st, modules


def attach(model, checkpoint, device=None, online=True):
    """Turn a loaded Cosmos-Policy model into its W4A4KV4 LAQ version in place.

    checkpoint: the name of a released model, a checkpoint directory, or a (config, weights) pair. The returned State
    carries the selection; call state.reset() at the start of every episode."""
    from . import checkpoint as ckpt
    config, weights = ckpt.resolve(checkpoint)
    if config["wam"] != "cosmos-policy":
        raise ValueError(f"A {config['wam']} checkpoint cannot be attached to Cosmos-Policy")
    st, modules = new_state(model, config, device)
    if set(modules) != set(weights):
        raise RuntimeError(f"The model has {len(modules)} block-level Linears, the checkpoint {len(weights)}")
    load_weights(st, modules, weights)
    st.offline = torch.tensor(config["offline_gaze"], device=st.device)
    st.watch = True
    st.use(config["spec"], online=online)
    return st


# ---------------- calibration
EVERY = 10 ** 9


def class_spec(steps, name):
    """The base (the action rows refined) plus one class refined entirely; name None is the base itself."""
    spec = spec_none(steps)
    spec["action"] = True
    if name == "text":
        spec["text"] = True
    elif name is not None:
        group, kind, s = name.split(".")
        spec["counts"][f"{group}.{kind}"][int(s)] = EVERY
    return spec


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def calibrate(st, modules, weights, metric_saved, contexts, query, fraction=0.15, seed=0, log=print):
    """The read count of the text rows, the damage each class removes, and the closed-form allocation.

    st, modules: from new_state on the BF16 model; weights: the packed 4-bit weights; metric_saved: the reader metric;
    contexts: calibration contexts; query(context) runs one policy query and returns the action chunk [steps, dims].
    The objective of a run is A + lambda_M Z_M against the BF16 teacher of the same context. Returns the entries a
    checkpoint config needs (spec, offline_gaze) with the measured quantities."""
    metric = Metric(metric_saved, st.device)
    pi = token_gaze(metric_saved)

    def run(index, context):
        st.seed = seed + index
        st.reset()
        action = torch.as_tensor(query(context)).detach().float().cpu()
        return action, st.z.cpu()

    st.use(None)
    st.watch = True
    teachers = []
    for i, context in enumerate(contexts):
        action, z = run(i, context)
        teachers.append(dict(action=action, z=z, gaze=torch.stack(st.step_gaze).cpu()))
    st.watch = False
    tokens = {g: t.cpu() for g, t in st.layout["tokens"].items()}
    fut = tokens["fut"]
    if pi.numel() != fut.numel():
        raise RuntimeError(f"The metric covers {pi.numel()} tokens, the future slots hold {fut.numel()}")

    def measure(name):
        units = [errors(metric, *run(i, context), teachers[i]) for i, context in enumerate(contexts)]
        out = dict(A=_mean(u["A"] for u in units), ZM=_mean(u["ZM"] for u in units))
        log(f"calibrate {name}: A={out['A']:.3e} Z_M={out['ZM']:.3e}")
        return out

    pair = {}
    for mode in ("frozen", "redrawn"):
        st.inject = mode
        pair[mode] = measure(f"read count text ({mode})")
    st.inject = None
    read_count = dict(text=pair["frozen"]["ZM"] / max(pair["redrawn"]["ZM"], 1e-30))

    load_weights(st, modules, weights)
    st.use(spec_none(st.steps))
    plain = measure("no refinement")
    lam = plain["A"] / plain["ZM"]

    def loss(unit):
        return unit["A"] + lam * unit["ZM"]
    st.use(class_spec(st.steps, None))
    base = loss(measure("action rows"))
    benefit = {}
    for name in ["text", *(f"{g}.{k}.{s}" for g, k in UNITS for s in range(st.steps))]:
        st.use(class_spec(st.steps, name))
        benefit[name] = max(base - loss(measure(name)), 0.0)

    att = torch.stack([t["gaze"] / t["gaze"].sum(1, keepdim=True) for t in teachers])  # [contexts, steps, tokens]
    mean_att = att.mean((0, 1))
    pin = pi / pi.sum()
    offline = mean_att.clone()
    offline[fut] = pin * mean_att[fut].sum()
    shares = {g: (offline[t] / offline[t].sum().clamp_min(1e-30)).tolist() for g, t in tokens.items()}
    names = [(n, m.in_features) for n, m in modules.items()]
    cost = costs(names, st.arch, st.steps, {g: int(t.numel()) for g, t in tokens.items()}, st.shapes["context_tokens"])
    solved = solve(benefit, cost, shares, fraction, st.steps)
    return dict(spec=solved["spec"], offline_gaze=offline.tolist(), mu=solved["mu"], accounting=solved["accounting"],
                lambda_M=lam, benefit=benefit, read_count=read_count)

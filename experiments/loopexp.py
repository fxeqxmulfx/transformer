"""Looped vs flat pre-norm GPT on FineWeb sp1024: quantization-error propagation.

A scaled-down parameter-golf block (resid_mix with x0 injection, per-channel
attn/mlp scales, zero-init output maps, q/k RMS norm, relu^2 MLP, tied
embeddings, logit softcap, Muon), no U-Net skips.  After training it measures,
per virtual layer, the stream norm, the error a quantized block injects at the
clean state, the propagated error, the gain along the actual error direction,
and the top singular value of the layer's Jacobian.
"""
import argparse, glob, json, math, os, time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DATA = Path(os.environ.get("PG_DATA", Path.home() / ".cache/param-golf-data"))
HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--virtual", required=True, help="physical index per virtual layer, e.g. 0,1,2,3,2,3,2,3,4,5")
p.add_argument("--pre_loop", default="", help="virtual layout before --loop_start (default: same)")
p.add_argument("--loop_start", type=float, default=0.0)
p.add_argument("--dim", type=int, default=384)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--kv_heads", type=int, default=3)
p.add_argument("--mlp_mult", type=int, default=3)
p.add_argument("--seq_len", type=int, default=1024)
p.add_argument("--micro_bs", type=int, default=8)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--warmdown", type=float, default=0.4)
p.add_argument("--matrix_lr", type=float, default=0.04)
p.add_argument("--scalar_lr", type=float, default=0.04)
p.add_argument("--embed_lr", type=float, default=0.05)
p.add_argument("--val_tokens", type=int, default=2_000_000)
p.add_argument("--noisy_qat", type=int, default=0, help="uniform weight noise of int-N step on looped blocks")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--out", default=str(HERE / "runs"))
p.add_argument("--diag_only", type=int, default=0, help="load model.pt, skip training")
p.add_argument("--compile", type=int, default=1)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
VIRTUAL = [int(v) for v in args.virtual.split(",")]
PRE = [int(v) for v in args.pre_loop.split(",")] if args.pre_loop else VIRTUAL
NPHYS = max(VIRTUAL + PRE) + 1
counts = {i: VIRTUAL.count(i) for i in range(NPHYS)}
LOOPED = {i for i, c in counts.items() if c > 1}
outdir = Path(args.out) / args.name
outdir.mkdir(parents=True, exist_ok=True)
logf = open(outdir / ("diag.txt" if args.diag_only else "log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


# ---------------- data ----------------
def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    assert h[0] == 20240520 and h[1] == 1
    return np.fromfile(f, dtype="<u2", count=int(h[2]), offset=1024)


train_tokens = torch.from_numpy(load_shard(DATA / "datasets/fineweb10B_sp1024/fineweb_train_000000.bin").astype(np.int32))
val_all = load_shard(DATA / "datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
nval = (min(args.val_tokens, len(val_all) - 1) // args.seq_len) * args.seq_len
val_tokens = torch.from_numpy(val_all[: nval + 1].astype(np.int32))

sp = spm.SentencePieceProcessor(model_file=str(DATA / "tokenizers/fineweb_1024_bpe.model"))
V = sp.vocab_size()
base_bytes = np.zeros(V, np.int16)
lead_space = np.zeros(V, np.bool_)
boundary = np.ones(V, np.bool_)
for t in range(V):
    if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t):
        continue
    boundary[t] = False
    if sp.is_byte(t):
        base_bytes[t] = 1
        continue
    piece = sp.id_to_piece(t)
    if piece.startswith("▁"):
        lead_space[t] = True
        piece = piece[1:]
    base_bytes[t] = len(piece.encode("utf-8"))
base_bytes = torch.tensor(base_bytes, device=dev)
lead_space = torch.tensor(lead_space, device=dev)
boundary = torch.tensor(boundary, device=dev)

pos = 0


def next_batch():
    global pos
    n = args.micro_bs * args.seq_len
    if pos + n + 1 > len(train_tokens):
        pos = 0
    chunk = train_tokens[pos : pos + n + 1].to(dev, non_blocking=True).long()
    pos += n
    return chunk[:-1].view(args.micro_bs, -1), chunk[1:].view(args.micro_bs, -1)


# ---------------- model ----------------
QUANT: dict = {"bits": 0, "noise_bits": 0}  # forward-time weight quantization / noise for looped blocks


def quantize_rows(w: Tensor, bits: int) -> Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


class Lin(nn.Linear):
    def __init__(self, i, o, zero=False):
        super().__init__(i, o, bias=False)
        self.zero = zero
        self.looped = False

    def forward(self, x):
        w = self.weight
        if QUANT["bits"] and (QUANT.get("all") or self.looped):
            w = quantize_rows(w.float(), QUANT["bits"])
        elif self.training and self.looped and QUANT["noise_bits"]:
            with torch.no_grad():
                step = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / (2 ** (QUANT["noise_bits"] - 1) - 1)
            w = w + (torch.rand_like(w) - 0.5) * step
        return F.linear(x, w.to(x.dtype))


def rope(T, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(torch.arange(T, device=dev).float(), inv)
    return f.cos()[None, None], f.sin()[None, None]


def apply_rope(x, cos, sin):
    h = x.size(-1) // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat((a * cos + b * sin, -a * sin + b * cos), -1)


class Block(nn.Module):
    def __init__(self, d, H, Hk, mult):
        super().__init__()
        self.H, self.Hk, self.hd = H, Hk, d // H
        self.q, self.k, self.v = Lin(d, d), Lin(d, Hk * self.hd), Lin(d, Hk * self.hd)
        self.o = Lin(d, d, zero=True)
        self.fc, self.down = Lin(d, mult * d), Lin(mult * d, d, zero=True)
        self.q_gain = nn.Parameter(torch.full((H,), 1.5))
        self.attn_scale = nn.Parameter(torch.ones(d))
        self.mlp_scale = nn.Parameter(torch.ones(d))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(d), torch.zeros(d))))

    def attn(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hk, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hk, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.Hk != self.H)
        return self.o(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, x0, cos, sin):
        m = self.resid_mix.to(x.dtype)
        x = m[0] * x + m[1] * x0
        x = x + self.attn_scale.to(x.dtype) * self.attn(F.rms_norm(x, (x.size(-1),)), cos, sin)
        h = F.relu(self.fc(F.rms_norm(x, (x.size(-1),))))
        return x + self.mlp_scale.to(x.dtype) * self.down(h.square())


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, args.dim)
        nn.init.normal_(self.emb.weight, std=0.005)
        self.blocks = nn.ModuleList(Block(args.dim, args.heads, args.kv_heads, args.mlp_mult) for _ in range(NPHYS))
        for i, b in enumerate(self.blocks):
            for m in b.modules():
                if isinstance(m, Lin):
                    m.looped = i in LOOPED
                    if m.zero:
                        nn.init.zeros_(m.weight)
                    else:
                        nn.init.orthogonal_(m.weight)
        self.layout = PRE

    def embed(self, idx):
        x = F.rms_norm(self.emb(idx), (args.dim,))
        return x, rope(idx.size(1), args.dim // args.heads)

    def logits(self, x):
        x = F.rms_norm(x, (args.dim,))
        return 30.0 * torch.tanh(F.linear(x, self.emb.weight.to(x.dtype)) / 30.0)

    def forward(self, idx, tgt):
        x, (cos, sin) = self.embed(idx)
        x0 = x
        for i in self.layout:
            x = self.blocks[i](x, x0, cos, sin)
        return F.cross_entropy(self.logits(x).float().view(-1, V), tgt.reshape(-1))


# ---------------- optimizer ----------------
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    tr = G.size(0) > G.size(1)
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                u *= max(1, u.size(0) / u.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-g["lr"])


model = GPT().to(dev)
mats = [p for n, p in model.blocks.named_parameters() if p.ndim == 2 and "resid_mix" not in n]
scalars = [p for n, p in model.blocks.named_parameters() if p.ndim < 2 or "resid_mix" in n]
opts = [
    torch.optim.Adam([model.emb.weight], lr=args.embed_lr, betas=(0.9, 0.95), fused=True),
    Muon(mats, lr=args.matrix_lr),
    torch.optim.Adam(scalars, lr=args.scalar_lr, betas=(0.9, 0.95), fused=True),
]
for o in opts:
    for g in o.param_groups:
        g["base_lr"] = g["lr"]
QUANT["noise_bits"] = args.noisy_qat
fwd = torch.compile(model, dynamic=False) if args.compile else model
nparams = sum(p.numel() for p in model.parameters())
log(f"run {args.name} virtual={VIRTUAL} pre={PRE} looped={sorted(LOOPED)} params={nparams}")


# ---------------- eval ----------------
@torch.no_grad()
def evaluate(bits=0, all_layers=False):
    QUANT["bits"], QUANT["all"] = bits, all_layers
    model.eval()
    tot_loss = tot_tok = tot_bytes = 0.0
    bs = 8
    nseq = nval // args.seq_len
    for s in range(0, nseq, bs):
        e = min(s + bs, nseq)
        chunk = val_tokens[s * args.seq_len : e * args.seq_len + 1].to(dev).long()
        x, y = chunk[:-1].view(e - s, -1), chunk[1:].view(e - s, -1)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, y)
        tot_loss += loss.item() * y.numel()
        tot_tok += y.numel()
        tb = base_bytes[y].to(torch.int32) + (lead_space[y] & ~boundary[x]).to(torch.int32)
        tot_bytes += tb.sum().item()
    QUANT["bits"], QUANT["all"] = 0, False
    model.train()
    l = tot_loss / tot_tok
    return l, l / math.log(2) * tot_tok / tot_bytes


# ---------------- train ----------------
t0 = time.time()
t_last, s_last = t0, 0
tokens_per_step = args.micro_bs * args.seq_len * args.accum
for step in range(0 if args.diag_only else args.steps + 1):
    frac = step / args.steps
    if args.loop_start and frac >= args.loop_start and model.layout is not VIRTUAL:
        model.layout = VIRTUAL
        if args.compile:
            fwd = torch.compile(model, dynamic=False)
        log(f"step {step}: looping on, layout {VIRTUAL}")
    if step == args.steps:
        break
    lrm = min(1.0, (1 - frac) / args.warmdown) if frac > 1 - args.warmdown else 1.0
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * lrm
    opts[1].param_groups[0]["momentum"] = 0.85 + 0.10 * min(1.0, step / 300)
    for _ in range(args.accum):
        x, y = next_batch()
        with torch.autocast("cuda", torch.bfloat16):
            loss = fwd(x, y)
        (loss / args.accum).backward()
    for o in opts:
        o.step()
        o.zero_grad(set_to_none=True)
    if step % 100 == 0 or step == args.steps - 1:
        lv = loss.item()
        now = time.time()
        log(f"step {step} loss {lv:.4f} lr_mul {lrm:.3f} t {now - t0:.0f}s "
            f"tok/s {tokens_per_step * (step + 1 - s_last) / (now - t_last):.0f}")
        t_last, s_last = now, step + 1
train_time = time.time() - t0
model.layout = VIRTUAL
if args.diag_only:
    model.load_state_dict(torch.load(outdir / "model.pt"))
else:
    torch.save(model.state_dict(), outdir / "model.pt")

res = {"args": vars(args), "virtual": VIRTUAL, "looped": sorted(LOOPED), "params": nparams, "train_time": train_time}
res["eval"] = {}
for bits in ([] if args.diag_only else [0, 8, 6, 5, 4]):
    l, b = evaluate(bits, all_layers=True)
    res["eval"][f"all_int{bits}" if bits else "fp"] = (l, b)
    log(f"eval {'fp' if not bits else f'int{bits} all blocks'}: loss {l:.4f} bpb {b:.4f}")


# ---------------- diagnostics ----------------
# Per virtual layer k, with F_k the block and F~_k its quantized copy, over a
# batch of validation sequences (Frobenius norms / sqrt(tokens), fp32, no TF32):
#   norm   = |x_k|                     stream norm entering layer k
#   eta    = |F~_k(x_k) - F_k(x_k)|    error the quantized block injects at the clean state
#   eta_p  = |F~_k(y_k) - F_k(y_k)|    the same at the quantized trajectory y
#   err    = |y_{k+1} - x_{k+1}|       propagated error after layer k
#   gain   = |F_k(y_k) - F_k(x_k)| / |y_k - x_k|   gain along the actual error
#   sigma  = top singular value of the Jacobian of F_k at x_k (power iteration)
#   Pg     = eta_p + gain * Pg         Gronwall with the actual gains, >= err
#   Ps     = eta + sigma * Ps          Gronwall with the worst-case gains
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model.eval()
model.requires_grad_(False)
N = len(VIRTUAL)
dseqs = 4
didx = val_tokens[: dseqs * args.seq_len].to(dev).long().view(dseqs, -1)
ntok = didx.numel()


def nrm(t):
    return (t.float().norm() / math.sqrt(ntok)).item()


def call(k, X, x0, cs, bits=0):
    QUANT["bits"], QUANT["all"] = bits, True
    try:
        return model.blocks[VIRTUAL[k]](X, x0, *cs)
    finally:
        QUANT["bits"] = 0


@torch.no_grad()
def trajectory(bits):
    x, cs = model.embed(didx)
    x0, xs = x, [x]
    for k in range(N):
        x = call(k, x, x0, cs, bits)
        xs.append(x)
    return xs, x0, cs


def top_sigma(k, X, x0, cs, iters=12):
    f = lambda Y: model.blocks[VIRTUAL[k]](Y, x0, *cs)
    v = torch.randn_like(X)
    v /= v.norm()
    sig = float("nan")
    with sdpa_kernel(SDPBackend.MATH):
        _, vjp_fn = torch.func.vjp(f, X)
        for _ in range(iters):
            _, jv = torch.func.jvp(f, (X,), (v,))
            (w,) = vjp_fn(jv)
            sig = jv.norm().item()
            v = (w / w.norm()).detach()
    return sig


xs, x0, cs = trajectory(0)
sigmas = [top_sigma(k, xs[k], x0, cs) for k in range(N)]
res["diag"] = {"sigma": sigmas, "norm": [nrm(x) for x in xs]}
for bits in [8, 6, 4]:
    ys, _, _ = trajectory(bits)
    rows, Pg, Ps = [], 0.0, 0.0
    with torch.no_grad():
        for k in range(N):
            eta = nrm(call(k, xs[k], x0, cs, bits) - xs[k + 1])
            Fy = call(k, ys[k], x0, cs, 0)
            eta_p = nrm(ys[k + 1] - Fy)
            e_in = nrm(ys[k] - xs[k])
            gain = nrm(Fy - xs[k + 1]) / e_in if e_in > 0 else float("nan")
            Pg = eta_p + (gain if e_in > 0 else 0.0) * Pg
            Ps = eta + sigmas[k] * Ps
            rows.append(dict(k=k, phys=VIRTUAL[k], norm=nrm(xs[k]), eta=eta, eta_p=eta_p,
                             err=nrm(ys[k + 1] - xs[k + 1]), gain=gain, sigma=sigmas[k], Pg=Pg, Ps=Ps))
    res["diag"][f"int{bits}"] = rows
    log(f"--- int{bits}, all blocks quantized ---")
    log(" k phys    norm       eta     err/norm_out    gain  sigma   err/Pg  err/sum(eta)")
    s_eta = 0.0
    for r in rows:
        s_eta += r["eta"]
        nout = res["diag"]["norm"][r["k"] + 1]
        log(f"{r['k']:2d} {r['phys']:3d} {r['norm']:8.2f} {r['eta']:9.2e} {r['err'] / nout:12.2e} "
            f"{r['gain']:8.3f} {r['sigma']:6.2f} {r['err'] / r['Pg']:7.3f} {r['err'] / s_eta:9.3f}")

stats = []
for i, b in enumerate(model.blocks):
    a, c = b.resid_mix[0].detach().float(), b.resid_mix[1].detach().float()
    st = dict(phys=i, uses=counts.get(i, 0), a_min=a.min().item(), a_max=a.max().item(), a_mean=a.mean().item(),
              a_absmax=a.abs().max().item(), a_absmin=a.abs().min().item(), a_neg=(a < 0).float().mean().item(),
              b_absmean=c.abs().mean().item(), attn_scale=b.attn_scale.abs().mean().item(),
              mlp_scale=b.mlp_scale.abs().mean().item())
    stats.append(st)
    log(f"phys {i} uses {st['uses']}: mix0 in [{st['a_min']:.3f}, {st['a_max']:.3f}] mean {st['a_mean']:.3f} "
        f"neg {st['a_neg']:.2%} |mix1| {st['b_absmean']:.3f} attn_s {st['attn_scale']:.3f} mlp_s {st['mlp_scale']:.3f}")
res["mix"] = stats
json.dump(res, open(outdir / ("diag.json" if args.diag_only else "result.json"), "w"), indent=1)
log("done", f"{time.time() - t0:.0f}s", f"peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

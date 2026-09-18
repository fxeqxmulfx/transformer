"""Does a network learn to *write* to an addressed memory, and does the write
survive being overwritten?

Everything measured in `vm-rs/alm-margin` is about retrieval.  The write side
is untested, and this is the test.  The task is state tracking: a stream of
`SET a v` and `GET a` events over `A = E * F` addresses, where the answer to a
`GET` is the *latest* prior `SET` to that address.

The ALM model trains as a soft overwrite, which is the write-side analogue of
the read-side mixture and is likewise an exact expectation, not an estimate:

    M_t(a) = (1 - p_t(a)) * M_{t-1}(a) + p_t(a) * v_t     at a SET
    read_t = sum_a q_t(a) * M_t(a)

At one-hot `p` the recursion is exactly "the last write to `a`", so the
inference-time recency term is its hard limit.  No temperature, no
straight-through, no Gumbel, no annealing.

Two baselines read the same stream by content instead: ordinary causal
attention over the event positions, and a GRU carrying the table in its state.

The axes are the number of addresses `A`, the distance from write to read, and
`R`, how many times the probed address was overwritten in between.  The
arithmetic in `alm-margin` predicts the third one collapses at `levels(f, a)`,
which in float32 at 4096 addresses is below one.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SET, GET = 0, 1


def episodes(B, T, A, V, dev, g, active=32, probe_r=None, probe_gap=None):
    """A batch of event streams.  Each episode touches `active` of the `A`
    addresses, so a GET is usually answerable while the softmax still has to
    be right over all of them -- a game has many slots and few live ones.

    With `probe_r` the last position is a GET to one address written exactly
    `probe_r + 1` times, and no other event touches it, so R is exact."""
    pool = torch.stack([torch.randperm(A, generator=g, device=dev)[:active] for _ in range(B)])
    lo = 1 if probe_r is not None else 0  # index 0 is reserved for the probe
    idx = torch.randint(lo, active, (B, T), generator=g, device=dev)
    addr = torch.gather(pool, 1, idx)
    val = torch.randint(0, V, (B, T), generator=g, device=dev)
    typ = torch.where(torch.rand(B, T, generator=g, device=dev) < 0.2, GET, SET)
    typ[:, 0] = SET
    if probe_r is None:
        return typ, addr, val, None
    gap = probe_gap if probe_gap is not None else T // 2
    last = T - 1 - gap
    slots = torch.stack([torch.randperm(last, generator=g, device=dev)[: probe_r + 1] for _ in range(B)])
    slots, _ = slots.sort(dim=1)
    rows = torch.arange(B, device=dev)[:, None]
    typ[rows, slots] = SET
    addr[rows, slots] = pool[:, :1]
    typ[:, T - 1] = GET
    addr[:, T - 1] = pool[:, 0]
    return typ, addr, val, val[rows, slots[:, -1:]].squeeze(1)


class Trunk(nn.Module):
    def __init__(self, A, V, d, L, T):
        super().__init__()
        self.et = nn.Embedding(2, d)
        self.ea = nn.Embedding(A, d)
        self.ev = nn.Embedding(V + 1, d)
        self.ep = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)

    def forward(self, typ, addr, val):
        B, T = typ.shape
        v = torch.where(typ == SET, val, torch.full_like(val, self.ev.num_embeddings - 1))
        x = self.et(typ) + self.ea(addr) + self.ev(v) + self.ep(torch.arange(T, device=typ.device))
        m = nn.Transformer.generate_square_subsequent_mask(T, device=typ.device)
        return self.enc(x, mask=m, is_causal=True)


class Alm(nn.Module):
    """Addressed memory: a softmax over addresses, written by soft overwrite."""

    def __init__(self, A, V, d, L, T):
        super().__init__()
        self.trunk, self.A, self.V = Trunk(A, V, d, L, T), A, V
        self.wa = nn.Linear(d, A)

    def forward(self, typ, addr, val):
        B, T = typ.shape
        h = self.trunk(typ, addr, val)
        p = self.wa(h).softmax(-1)
        vh = F.one_hot(val, self.V).to(p.dtype)
        w = (typ == SET).to(p.dtype)
        M = torch.zeros(B, self.A, self.V, device=p.dtype and p.device, dtype=p.dtype)
        out = []
        for t in range(T):
            out.append(torch.einsum("ba,bav->bv", p[:, t], M))
            pt = (w[:, t, None] * p[:, t])[:, :, None]
            M = M * (1 - pt) + pt * vh[:, t][:, None, :]
        return torch.stack(out, 1).clamp_min(1e-9).log()


class Soft(nn.Module):
    """The same stream read by content: causal attention over the positions."""

    def __init__(self, A, V, d, L, T):
        super().__init__()
        self.trunk, self.V = Trunk(A, V, d, L, T), V
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.d = d

    def forward(self, typ, addr, val):
        B, T = typ.shape
        h = self.trunk(typ, addr, val)
        s = self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(self.d)
        m = torch.full((T, T), float("-inf"), device=h.device).triu(0)
        s = s + m + torch.where(typ == SET, 0.0, float("-inf"))[:, None, :]
        a = s.softmax(-1).nan_to_num()
        return (a @ F.one_hot(val, self.V).to(a.dtype)).clamp_min(1e-9).log()


class Gru(nn.Module):
    def __init__(self, A, V, d, L, T):
        super().__init__()
        self.trunk = Trunk(A, V, d, 1, T)
        self.rnn = nn.GRU(d, d, L, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, typ, addr, val):
        h = self.trunk.et(typ) + self.trunk.ea(addr) + self.trunk.ev(
            torch.where(typ == SET, val, torch.full_like(val, self.trunk.ev.num_embeddings - 1))
        )
        return self.out(self.rnn(h)[0]).log_softmax(-1)


MODELS = {"alm": Alm, "soft": Soft, "gru": Gru}


def latest(typ, addr, val, A):
    """The true answer at every position: the last SET to this address before it."""
    B, T = typ.shape
    tab = torch.full((B, A), -1, dtype=torch.long, device=typ.device)
    y = torch.full((B, T), -1, dtype=torch.long, device=typ.device)
    rows = torch.arange(B, device=typ.device)
    for t in range(T):
        y[:, t] = tab[rows, addr[:, t]]
        s = typ[:, t] == SET
        tab[rows[s], addr[s, t]] = val[s, t]
    return y


def train(name, A, V, T, d, L, B, steps, dev, seed, active):
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    net = MODELS[name](A, V, d, L, T).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4 if name == "alm" else 1e-3)
    for i in range(steps):
        typ, addr, val, _ = episodes(B, T, A, V, dev, g, active)
        y = latest(typ, addr, val, A)
        m = (typ == GET) & (y >= 0)
        if m.sum() == 0:
            continue
        loss = F.nll_loss(net(typ, addr, val)[m], y[m])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


@torch.no_grad()
def probe(net, A, V, T, B, dev, seed, rs, active, gap=None):
    g = torch.Generator(device=dev).manual_seed(seed + 7919)
    acc = {}
    for r in rs:
        hit = n = 0
        for _ in range(8):
            typ, addr, val, tgt = episodes(B, T, A, V, dev, g, active, probe_r=r, probe_gap=gap)
            pred = net(typ, addr, val)[:, -1].argmax(-1)
            hit += (pred == tgt).sum().item()
            n += B
        acc[r] = hit / n
    return acc


ALPHA = 0.3  # RECENCY_ALPHA, vm-rs/alm-margin/src/rewrite.rs


def recency_term(pos, horizon):
    """The two features of `Recency`, in float64: the shipped `inv_log_pos`,
    which saturates, and the linear one of the same span."""
    p = np.asarray(pos, dtype=np.float64)
    inv = 1.0 / math.log(2.0) - 1.0 / np.log(p + 2.0)
    lin = np.minimum(p / max(horizon, 1), 1.0) / math.log(2.0)
    return ALPHA * inv, ALPHA * lin


def resolves(dtype, addrs, poss, term, q):
    """The ALM head, in one format: key `a` lifts to `(2a, -a^2 + recency)`,
    query `a` to `(a, 1)`, and the cell read is the argmax.  True when that is
    the latest write to `q` -- which it is over the reals, by
    `sScore q k = q^2 - (k - q)^2` and a recency span below the unit gap."""
    a = np.asarray(addrs, dtype=dtype)
    t = np.asarray(term, dtype=dtype)
    s = dtype(2) * a * dtype(q) - a * a + t
    live = np.flatnonzero(np.asarray(addrs) == q)
    return int(np.argmax(s)) == int(live[np.argmax(np.asarray(poss)[live])])


def precision(As, rs, active, trials, seed):
    """No training: the arithmetic alone.  `levels(f, a) = span / ulp(a^2)` is
    how many rewrites of address `a` the format can still order, and at 4096 it
    is 0.22 in float32 and 1.2e8 in float64."""
    rng = np.random.default_rng(seed)
    print("head precision, no training: fraction of GETs that read the latest write\n")
    print(f"{'addresses':>10} {'R':>4} {'f32 invlog':>12} {'f32 linear':>12} {'f64 invlog':>12} {'f64 linear':>12}")
    for A in As:
        for r in rs:
            hit = {k: 0 for k in ("f32i", "f32l", "f64i", "f64l")}
            for _ in range(trials):
                pool = rng.choice(A, size=active, replace=False)
                addrs = np.repeat(pool, r + 1)
                rng.shuffle(addrs)
                poss = np.arange(len(addrs))
                inv, lin = recency_term(poss, len(addrs))
                q = int(rng.choice(pool))
                hit["f32i"] += resolves(np.float32, addrs, poss, inv, q)
                hit["f32l"] += resolves(np.float32, addrs, poss, lin, q)
                hit["f64i"] += resolves(np.float64, addrs, poss, inv, q)
                hit["f64l"] += resolves(np.float64, addrs, poss, lin, q)
            row = "".join(f"{hit[k] / trials:>13.3f}" for k in ("f32i", "f32l", "f64i", "f64l"))
            print(f"{A:>10} {r:>4}{row}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addresses", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--rewrites", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--values", type=int, default=16)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["train", "precision"], default="train")
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    if a.mode == "precision":
        precision(a.addresses, a.rewrites, a.active, a.trials, a.seed)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}, values {a.values} (chance {1/a.values:.3f}), length {a.length}\n")
    for A in a.addresses:
        print(f"addresses {A}")
        print("  " + f"{'model':>6}" + "".join(f"{('R=' + str(r)):>8}" for r in a.rewrites) + f"{'train s':>10}")
        for name in MODELS:
            t0 = time.time()
            net = train(name, A, a.values, a.length, a.dmodel, a.layers, a.batch, a.steps, dev, a.seed, a.active)
            acc = probe(net, A, a.values, a.length, a.batch, dev, a.seed, a.rewrites, a.active)
            print("  " + f"{name:>6}" + "".join(f"{acc[r]:>8.3f}" for r in a.rewrites) + f"{time.time()-t0:>10.0f}")
        print()


if __name__ == "__main__":
    main()

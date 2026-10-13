"""Ordinary-transformer orbit probes without training interference.

Source: arXiv:2301.05217v1, section 5.1, adapted to prime-field division.
Projection averages over (u*x, u*y) for nonzero u; all invariant functions
are retained, replacing final-checkpoint-selected addition frequencies.
Main energy removes row shifts, global class bias and zero quotient.
Restricted/excluded logits retain global class bias. This is a functional
projection, not a claim to isolate specific attention/FFN circuits.
"""

import torch
from torch.nn import functional as F

from ...domain.benchmarks import is_prime
from ..benchmarks.modular import complete_rows, fingerprint
from .grokking_internal import InternalProbe


def statistics(logits, train, heldout, targets, energy_floor=1e-12):
    """Current complete-orbit logits, shaped quotient x denominator x vocab.

    Quotient zero is first. Constant input-independent logits supply no
    structural evidence. Losses follow projection without training it.
    """
    prime, orbit, vocab = logits.shape
    if prime < 3 or not is_prime(prime) or orbit != prime - 1:
        raise ValueError("A division probe needs full nontrivial prime-field orbits")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Grokking logits must be finite")
    x = logits.detach().double()
    restricted = x.mean(1, keepdim=True).expand_as(x)
    excluded = x - restricted + x.mean((0, 1), keepdim=True)
    nonzero = x[1:]
    z = nonzero - nonzero.mean(-1, keepdim=True)
    z = z - z.mean((0, 1), keepdim=True)
    invariant = z.mean(1, keepdim=True).expand_as(z)
    total, structured = float(z.square().mean()), float(invariant.square().mean())
    residual = float((z - invariant).square().mean())
    fraction = min(1., max(0., structured / total)) if total > energy_floor else None
    raw, kept, removed = (a.reshape(-1, vocab) for a in (x, restricted, excluded))
    mask = torch.zeros((prime, orbit), dtype=torch.bool, device=x.device)
    mask.reshape(-1)[heldout] = True
    counts = mask.sum(1)
    heldout_means = (x * mask[..., None]).sum(1) / counts.clamp_min(1)[:, None]
    heldout_kept = heldout_means[:, None, :].expand_as(x).reshape(-1, vocab)
    valid = mask[1:]
    h = nonzero - nonzero.mean(-1, keepdim=True)
    h = h - h[valid].mean(0)
    hmean = (h * valid[..., None]).sum(1) / counts[1:].clamp_min(1)[:, None]
    hprojected = hmean[:, None, :].expand_as(h)
    htotal = float(h[valid].square().mean())
    hstructured = float(hprojected[valid].square().mean())
    hresidual = float((h - hprojected)[valid].square().mean())
    coverage = float((counts[1:] >= 2).double().mean())
    hfraction = min(1., max(0., hstructured / htotal)) if htotal > energy_floor and coverage == 1 else None

    def loss(a, indices):
        return float(F.cross_entropy(a[indices], targets[indices]))

    return {"invariant_energy_fraction": fraction, "total_energy": total,
            "invariant_energy": structured, "residual_energy": residual,
            "energy_decomposition_error": abs(total - structured - residual),
            "raw_train_loss": loss(raw, train), "raw_heldout_loss": loss(raw, heldout),
            "restricted_train_loss": loss(kept, train), "restricted_heldout_loss": loss(kept, heldout),
            "heldout_invariant_energy_fraction": hfraction, "heldout_total_energy": htotal,
            "heldout_invariant_energy": hstructured, "heldout_residual_energy": hresidual,
            "heldout_energy_decomposition_error": abs(htotal - hstructured - hresidual),
            "heldout_orbit_coverage": coverage, "heldout_min_orbit_size": int(counts[1:].min()),
            "heldout_restricted_loss": loss(heldout_kept, heldout),
            "heldout_restricted_accuracy": float((heldout_kept[heldout].argmax(-1) == targets[heldout]).double().mean()),
            "excluded_train_loss": loss(removed, train), "excluded_heldout_loss": loss(removed, heldout),
            "raw_heldout_accuracy": float((raw[heldout].argmax(-1) == targets[heldout]).double().mean()),
            "restricted_heldout_accuracy": float((kept[heldout].argmax(-1) == targets[heldout]).double().mean()),
            "excluded_train_accuracy": float((removed[train].argmax(-1) == targets[train]).double().mean()),
            "nonzero_quotients": prime - 1, "orbit_size": orbit, "constant_logits": fraction is None}


class GrokkingObserver:
    """Fixed full-domain observation; optimizer, sampler and RNG are retained."""

    def __init__(self, spec, task):
        self.spec = spec
        corpus = task.corpus
        self.prime, self.vocab = corpus.prime, len(corpus.tokens)
        rows = torch.tensor(complete_rows(self.prime), dtype=torch.long)
        self.inputs, self.targets = rows[:, :5], rows[:, 5]
        self.fingerprint = fingerprint(rows.numpy())
        offset = corpus.tokens.index("0")

        def indices(data):
            rows = torch.tensor(data, dtype=torch.long)
            quotient, denominator = rows[:, 5] - offset, rows[:, 3] - offset
            return quotient * (self.prime - 1) + denominator - 1

        self.train, self.heldout = indices(corpus.train), indices(corpus.heldout)
        for selected, original in ((self.train, corpus.train), (self.heldout, corpus.heldout)):
            if not torch.equal(rows[selected], torch.tensor(original)):
                raise ValueError("Division probe indices must reproduce both raw splits")

    @torch.no_grad()
    def observe(self, model, step):
        device = next(model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), InternalProbe(
                    model, self.heldout, self.prime, self.spec.energy_floor) as internal:
                logits = torch.cat([model(part.to(device))[:, 4].detach().cpu()
                                    for part in self.inputs.split(self.spec.batch)])
        finally:
            for module, training in modes.items():
                module.training = training
        result = statistics(logits.reshape(self.prime, self.prime - 1, self.vocab),
                            self.train, self.heldout, self.targets, self.spec.energy_floor)
        return {"step": step, "rows_sha256": self.fingerprint, "examples": len(self.inputs),
                "internal": internal.result(),
                "forward": "uncompiled_raw_answer_prefix_without_answer_token",
                "projection": "all_common_scaling_orbits; zero_removed_from_main_energy", **result}

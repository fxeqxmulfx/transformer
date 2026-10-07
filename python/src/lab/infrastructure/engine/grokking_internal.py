"""Inside ordinary transformer blocks during the current division probe.

Source: Nanda et al., arXiv:2301.05217v1, section 5.1. Deviations: measure
held-out scaling-orbit coherence of actual attention/FFN/residual features
and attention entropy, rather than selecting addition Fourier circuits.
Hidden-feature energy has no logit row-shift correction. These observations
are associations, not causal importance or a guarantee of correct answers.
"""

import torch


def hidden_energy(features, heldout, prime, floor):
    """Weighted orthogonal projection on held-out nonzero-quotient orbits."""
    x = features.detach().double().reshape(prime, prime - 1, -1)[1:]
    mask = torch.zeros((prime, prime - 1), dtype=torch.bool)
    mask.reshape(-1)[heldout] = True
    mask = mask[1:]
    counts = mask.sum(1)
    centered = x - x[mask].mean(0)
    average = (centered * mask[..., None]).sum(1) / counts.clamp_min(1)[:, None]
    projected = average[:, None, :].expand_as(x)
    total = float(centered[mask].square().mean())
    signal = float(projected[mask].square().mean())
    residual = float((centered - projected)[mask].square().mean())
    fraction = min(1., max(0., signal / total)) if total > floor and bool((counts >= 2).all()) else None
    return {"invariant_energy_fraction": fraction, "total_energy": total,
            "invariant_energy": signal, "residual_energy": residual,
            "energy_decomposition_error": abs(total - signal - residual),
            "activation_rms": float(x[mask].square().mean().sqrt())}


class InternalProbe:
    """Read existing forward outputs with temporary hooks; remove every hook."""

    def __init__(self, model, heldout, prime, floor):
        self.model, self.heldout, self.prime, self.floor = model, heldout, prime, floor
        self.features, self.weights, self.handles = {}, {}, []

    def __enter__(self):
        def capture(name):
            def hook(module, inputs, output):
                self.features.setdefault(name, []).append(output[:, 4].detach().cpu())
            return hook

        def routes(name, attention):
            def hook(module, inputs, scores):
                group = inputs[-1]
                key = name if group is None else f"{name}.head{group}"
                probabilities = attention.weights(scores.detach())[..., 4, :]
                if group is not None:
                    probabilities = probabilities[:, None, :]
                self.weights.setdefault(key, []).append(probabilities.cpu())
            return hook

        for name, module in self.model.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
                if len(parts) == 2 or (len(parts) == 3 and parts[2] in ("attention", "ffn")):
                    self.handles.append(module.register_forward_hook(capture(name)))
                if len(parts) == 3 and parts[2] == "attention" and not module.fused:
                    self.handles.append(module.scores.register_forward_hook(routes(name, module)))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def result(self):
        features = {name: hidden_energy(torch.cat(chunks), self.heldout, self.prime, self.floor)
                    for name, chunks in self.features.items()}
        attention = {}
        for name, chunks in self.weights.items():
            p = torch.cat(chunks)[self.heldout].double()
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
            attention[name] = {"entropy_by_head": entropy.mean(0).tolist(),
                               "mean_weights_by_head_and_prompt_token": p.mean(0).tolist()}
        return {"features": features, "attention": attention,
                "position": "answer_query_at_equals_token",
                "prompt_positions": ["EOS", "numerator", "operator", "denominator", "equals"],
                "scope": "current_heldout_nonzero_orbit_feature_energy; heldout_route_statistics; no_ablation"}

"""Semantic checks before expensive training, including RoPE causality."""

import json

import numpy as np
import torch

from .certify import encode, make_example, simplex_projection, train_metric
from .convex import ConvexRecall
from .data import validate_batch
from .rope import RopeTransformer


def check(device):
    torch.set_num_threads(6)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    w, _ = train_metric(6, rng)
    rows = [make_example(rng, 16, 4, 64, 0.1) for _ in range(8)]
    tokens, positions, labels = (torch.tensor(np.stack(items), device=device)
                               for items in zip(*rows))
    validate_batch(tokens, positions, labels, 64)
    convex = ConvexRecall(64, w).to(device)
    assert torch.equal(convex(tokens, positions), labels)
    first_keys = torch.arange(0, 8, 2, device=device).expand(len(tokens), -1)
    assert torch.all(convex(tokens, first_keys) == -1)
    z = rng.normal(size=(32, 9))
    expected = simplex_projection(z)
    actual = convex.project_simplex(torch.tensor(z, device=device)).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    model = RopeTransformer(64, 32).to(device).eval()
    with torch.no_grad():
        full_logits = model(tokens)
        selected_logits = model(tokens, positions)
        gathered = full_logits.gather(1, positions[..., None].expand(-1, -1, 64))
        torch.testing.assert_close(selected_logits, gathered)
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 7) % 64
        torch.testing.assert_close(model(changed)[:, :8], full_logits[:, :8])
        attention = model.blocks[0].attention
        x = torch.randn(2, 1, 16, 32, device=device)
        rotated = attention.rotate(x)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    model.train()
    torch.nn.functional.cross_entropy(model(tokens, positions).flatten(0, 1),
                                       labels.flatten()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    print(json.dumps({"semantic_checks": "passed", "device": device,
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None}), flush=True)

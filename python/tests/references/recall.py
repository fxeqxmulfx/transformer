"""Measure the original convex MQAR models at 3bae214, including tied gradients."""

import torch
import torch.nn.functional as F

from common import hashes, sha


def replay(golden):
    from convex_mqar.rope import RopeTransformer
    from convex_mqar.sparse_attention import SparsemaxTransformer

    for name, expected in golden["models"].items():
        config = golden["runs"][name]["config"]
        factory = SparsemaxTransformer if golden["runs"][name]["attention"] == "sparsemax" else RopeTransformer
        torch.manual_seed(expected["seed"])
        model = factory(expected["vocab"], expected["width"], config["layers"], expected["heads"],
                        config["mlp_ratio"], config["rope_base"])
        expected["parameters"] = hashes(dict(model.named_parameters()))
        generator = torch.Generator().manual_seed(7)
        tokens = torch.randint(expected["vocab"], (8, expected["length"]), generator=generator)
        positions = torch.randint(expected["length"], (8, expected["length"] // 4), generator=generator)
        labels = torch.randint(expected["vocab"], positions.shape, generator=generator)
        output = model(tokens, positions)
        loss = F.cross_entropy(output.flatten(0, 1), labels.flatten())
        loss.backward()
        expected.update(logits=sha(output), loss=loss.item().hex(),
                        gradients=hashes({key: value.grad for key, value in model.named_parameters()}))

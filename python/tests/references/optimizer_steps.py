"""Frozen zoo updates on shared gradients, without feedback from roundoff.

The algorithms come from the zoo revision in reference_fixtures.REVISIONS.
Only the inputs are prescribed here; no optimizer equation is reimplemented.
"""

from types import SimpleNamespace

import torch


def replay(golden):
    from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.optimizer import ExtensionOptimizer
    from gpt_mini.infrastructure.benchmark.full_compile_benchmark.optimizer import initialize, propose
    from gpt_mini.infrastructure.benchmark.full_compile_benchmark.step import mask_plan
    from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import make_optimizer

    result = {}
    names = ("embed.weight", "blocks.0.attn.q.weight", "temperature")
    shapes = ((2, 3), (3, 3), (3,))
    for name in ("adam", "amsgrad", "adamw", "magma_adam", "magma_adamw", "amsgradw"):
        config = golden["runs"][name]["config"]
        result[name] = {}
        for dtype in (torch.float32, torch.float64):
            parameters = {key: torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
                          for key, shape in zip(names, shapes, strict=True)}
            model = SimpleNamespace(named_parameters=lambda: parameters.items(), magma_seed=config["seed"])
            counter = torch.zeros((), dtype=torch.float64)
            if name == "amsgradw":
                optimizer = ExtensionOptimizer(parameters, name, config["rate"])
            else:
                optimizer = make_optimizer(name, model, config["rate"], config["seed"])
                if hasattr(optimizer, "masked_parameters"):
                    optimizer.masked_parameters = dict.fromkeys(optimizer.masked_parameters)
                initialize(optimizer, counter)
            masks = mask_plan(40, 1 if name.startswith("magma_") else 0, config["seed"])
            generator = torch.Generator().manual_seed(731)
            rows = []
            with torch.no_grad():
                for step in range(40):
                    points, gradients = {}, {}
                    for key, parameter in parameters.items():
                        point = torch.randn(parameter.shape, dtype=dtype, generator=generator)
                        gradient = torch.randn(parameter.shape, dtype=dtype, generator=generator)
                        # Include zero gradients and small gradients where epsilon matters.
                        gradient *= (0., 1e-6, 1., 1e3)[step % 4]
                        parameter.copy_(point)
                        parameter.grad = gradient
                        points[key], gradients[key] = point.tolist(), gradient.tolist()
                    if name == "amsgradw":
                        optimizer.step({key: parameter.grad for key, parameter in parameters.items()})
                    else:
                        counter.add_(1)
                        directions = propose(optimizer, list(parameters.values()), {}, masks[step])
                        for parameter, direction in zip(parameters.values(), directions, strict=True):
                            parameter.add_(direction, alpha=-config["rate"])
                    rows.append({"point": points, "gradient": gradients,
                                 "next": {key: parameter.tolist() for key, parameter in parameters.items()}})
            result[name][str(dtype).removeprefix("torch.")] = rows
    golden["optimizer_steps"] = result

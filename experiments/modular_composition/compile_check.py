"""CUDA max-autotune numerical check before the declared learning screen.

One no-update forward/backward comparison with the independent source
equations. This diagnostic does not observe a learning result or tune a run.
"""

import importlib.util
import json
from pathlib import Path
import platform

import torch

from lab.dsl import SourceTransformer
from lab.infrastructure.nn import build_model

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("source_equations", ROOT / "review/sequence_model_source_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = build_model(SourceTransformer(16, 2, 2, 9), 17, 11).cuda()
    tokens = torch.arange(18, device="cuda").reshape(2, 9) % 17
    targets = (tokens + 1) % 17
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    result = module.compare(model, tokens, targets, lambda x: (compiled(x), None))
    result.update(python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  compilation="max-autotune/fullgraph/static", width=16, depth=2, heads=2, context=9)
    (Path(__file__).parent / "compile_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

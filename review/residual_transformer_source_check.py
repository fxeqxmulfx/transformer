"""Explicitly rebuild the task145 oracle from the pinned external GPT source.

Run with Python3.15/Torch2.15, passing the GPT checkout as the only argument.
Tests consume the bundled fixture; they never regenerate it from the lab port.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

SOURCE = "ad76812"
ROOT = Path(__file__).resolve().parents[1]


def digest(model):
    values = torch.cat([p.detach().flatten() for p in model.parameters()])
    return hashlib.sha256(values.numpy().astype("<f8").tobytes()).hexdigest()


def main():
    source = Path(sys.argv[1]).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if not head.startswith(SOURCE):
        raise ValueError("Oracle rebuild requires the pinned GPT commit")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src"], text=True):
        raise ValueError("Oracle source files must be clean")
    sys.path.insert(0, str(source / "src"))
    from diff_gpt.domain.model.residual_gpt import MODES, ResidualGPT
    torch.set_num_threads(1)
    cases = []
    for mode in MODES:
        for clipped in (False, True):
            torch.manual_seed(145)
            model = ResidualGPT(11, 8, 8, 2, 2, mixing=mode, use_checkpoint=False,
                                pad_vocab_size_to=1).double()
            if clipped:
                with torch.no_grad():
                    model.blocks[0].sa.c_attn.weight.mul_(1e-14)
            tokens = torch.tensor([[1, 2, 3, 4], [2, 2, 2, 5], [3, 3, 3, 3]])
            logits = model(tokens)[0]
            logits.square().mean().backward()
            cases.append({"mode": mode, "clipped": clipped, "parameter_sha256": digest(model),
                          "tokens": tokens.tolist(), "logits": logits.detach().tolist(),
                          "gradients": {name: p.grad.tolist() for name, p in model.named_parameters()}})
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((source / "src/diff_gpt/domain/model").glob("*.py"))}
    result = {"source_commit": head, "source_files": files, "torch": torch.__version__,
              "python": sys.version, "loss": "mean(square(all logits))", "cases": cases}
    (ROOT / "python/tests/fixtures/residual_transformer.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

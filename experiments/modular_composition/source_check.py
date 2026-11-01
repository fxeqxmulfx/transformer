"""Compare the lab port to the pinned external source and freeze CPU fixtures.

Run explicitly with uv run --locked inside python/. This uses the supplied
diff_gpt checkout, never the implementation under test as the fixture oracle.
The normal test suite consumes the saved fixture without that checkout.
"""

import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
from torch.nn import functional as F

from lab.dsl import SourceTransformer
from lab.infrastructure.nn import build_model

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    pinned = json.loads((ROOT / "review/sequence_model_source.json").read_text())
    source = Path(pinned["source_root"])
    for name, digest in pinned["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned source changed: {name}")
    sys.path.insert(0, str(source.parents[2]))
    from diff_gpt.domain.model.gpt import GPT

    records = []
    for width, depth, heads, length in ((8, 2, 2, 5), (16, 3, 2, 7), (8, 1, 1, 1)):
        seed, prime = 41, 7
        torch.manual_seed(seed)
        original = GPT(prime, width, length, heads, depth, use_checkpoint=False, pad_vocab_size_to=1)
        port = build_model(SourceTransformer(width, depth, heads, length), prime, seed)
        assert original.state_dict().keys() == port.state_dict().keys()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(port.state_dict()[name], value, rtol=0, atol=0)
        tokens = torch.arange(2 * length).reshape(2, length) % prime
        expected = original(tokens)[0]
        actual = port(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        labels = (tokens[:, -1] + 1) % prime
        ga = torch.autograd.grad(F.cross_entropy(actual[:, -1], labels), tuple(port.parameters()))
        ge = torch.autograd.grad(F.cross_entropy(expected[:, -1], labels), tuple(original.parameters()))
        for a, e in zip(ga, ge, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        flat = torch.cat([p.detach().flatten() for p in original.parameters()])
        records.append({"width": width, "depth": depth, "heads": heads, "length": length,
                        "seed": seed, "prime": prime, "tokens": tokens.tolist(),
                        "initial_sha256": hashlib.sha256(flat.numpy().astype("<f4").tobytes()).hexdigest(),
                        "logits": expected.detach().tolist(), "max_logit_error": 0., "max_gradient_error": 0.})
    result = {"source_commit": pinned["repository_commit"], "source_hashes": pinned["files"],
              "python": platform.python_version(), "torch": torch.__version__, "cases": records}
    (ROOT / "python/tests/fixtures/source_transformer.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(records), "source": pinned["repository_commit"],
                      "initialization_logits_and_gradients": "bitwise_equal", "python": platform.python_version(),
                      "torch": torch.__version__}))


if __name__ == "__main__":
    main()

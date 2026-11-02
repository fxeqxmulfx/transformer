"""Zero-update fullgraph preparation of the actual historical hard-recall model."""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import Clock
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import provenance

root = Path(sys.argv[1]).resolve()
folder = root / "experiments/basis_control"
sys.path.insert(0, str(folder))
from protocol import frozen, sha

declaration = frozen()
spec = load(folder).experiments["gptmini-hard-recall-s0"]
torch.set_num_threads(1)
torch.cuda.reset_peak_memory_stats()
task = build_task(spec.benchmark, spec.seeds.data, torch.device("cuda"))
model = build_model(spec.model, task.vocab, spec.seeds.model).cuda()
stepper = ResidualCompiledStepper(spec, task, model, Clock(torch.device("cuda")))
sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
before = copy.deepcopy((model.state_dict(), stepper.state_dict(), sampler.state()))
assert stepper.conditioner is None
started = time.monotonic()
stepper.prepare(sampler.sizes())
torch.cuda.synchronize()
elapsed = time.monotonic() - started
torch.testing.assert_close(before, (model.state_dict(), stepper.state_dict(), sampler.state()), rtol=0, atol=0)
assert task.fingerprints == declaration["cases"][0]["fingerprints"]
result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(Path(__file__)),
          "protocol_sha256": sha(folder / "protocol.json"), "provenance": provenance("cuda"),
          "label": "gptmini-hard-recall-s0", "training_updates": 0, "prepare_seconds": elapsed,
          "sampler_sizes": sampler.sizes(), "complete_frame_exact": True, "final_test_opened": False,
          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
          "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
out = folder / "runs/preflight.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k:v for k,v in result.items() if k!='provenance'}, indent=2))

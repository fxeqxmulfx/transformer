# Current GPT and conditional residual gains on Basis

Task160 asks whether the conditional gain candidate also improves the
repository's original Basis tasks. Modular division measures one kind of
generalization; these runs add depth, associative recall, and parity. No
Basis learning result is claimed until its source-pinned run is measured.

| Setting | Small/easy | Large/hard |
| --- | --- | --- |
| Width / layers / heads | 64 / 2 / 4 | 128 / 6 / 4 |
| Depth | E2, lengths32–64; final test at128 | E4, lengths32–64; validation at128 |
| Recall | 256 symbols, 8 bindings, 8 queries | 256 symbols, 16 writes, 8 overwrites, 8 queries |
| Parity | 1–16 bits, generate answer and EOS | Same task, larger model |
| Model seeds | 0 and1 | 0 and1 |

Each setting compares the current `diff_gpt` GPT at `df1e9f7` with its
conditional post-RMS gain candidate. The candidate adds one zero-initialized
bias-free row per branch; its multiplier is
`2*sigmoid(W_gate*RMS(branch_input))`. The common initial weights, logits,
gradients and RNG state are identical. See
[`adaptive_residual`](../adaptive_residual) for exact source correspondence
and [`context_gains`](../context_gains) for the finite-solution gradient test.

The task generators, tokens, labels, length distributions and full split
fingerprints must equal `lab.domain.basis.BASIS`. Both variants use data
seed1, 20,000 training examples, 512 validation and 512 test examples. The
batch seed is model seed plus10,000, as in Basis. Rates, training batches,
50-update warmup and observations every3,200 examples are unchanged.
The original recipe used AdamW and GPTMini on CPU. This comparison uses
only Schedule-Free AdamW, betas(.9,.98), decay.1, the actual normalized
source GPT, Python3.15/Torch2.15, FP32 CUDA fullgraph max-autotune and math
SDPA. Evaluation batch64 fits the local4GiB GPU; historical Basis used256.

The stopping score is **sequence accuracy**, at the first observation
reaching99%. Hard depth uses `validation/length-128`; the other settings
use ordinary validation. Parity is scored through free generation,
including EOS. The current complete Schedule-Free endpoint is retained;
no best checkpoint is selected. Test remains sealed until learning closes.
The initial100,000-update chunk is a continuation point, not an impossibility
result. Promising uncrossed cases need longer training before final test.
The second seed reverses model order in each task/size pair.

After a complete chunk, `./run.py basis-continue <decisions.json>` records
an explicit validation-only decision for every unfinished case. Each record
names its label, exact checkpoint hash and update count, an `extend` or
`censor` action, and its reason; extensions also name a larger `updates`
budget. The document pins `protocol_sha256`. This separate runner preserves
the original frozen sources, retains each previous checkpoint, changes only
the budget and stops at the first99% crossing. A censored case remains an
unresolved result. Every decision and runner hash travels with the phase;
closed learning and disclosed final tests cannot be reopened by this command.

After a process interruption, `./run.py basis-recover <decision.json>` resumes
the one unfinished extension from its exact saved state. The recovery names
the phase, checkpoint and original continuation hashes and cannot change the
declared update horizon. It preserves the interrupted phase, manifest, raw
logs and checkpoint before the engine discards any records beyond that state.
This is an interrupted segment, not a completed budget chunk. Recovery1 starts
at102,750 updates, within the existing200,000-update extension; its saved
validation is178/512 exact sequences and3,579/4,096 correct tokens.

Original Basis samples can repeat inputs across splits. We keep those
splits intact, disclose their overlap, and separately report test examples
whose inputs never occurred in training. For parity, the input key is its
prompt, not the teacher-forced answer. Length transfer, sampled test and
novel-input results are distinct. CPU endpoint replay is identified as such;
the canonical validation crossing comes from the recorded CUDA history.

The mathematical depth hierarchy has specific precision, head-count and
RoPE assumptions. It does not directly prove a depth lower bound for these
four-head, exact-angle, finite-context runs. Likewise, poor finite training
is not a proof that a model cannot represent the task.

Run `./run.py check experiments/basis_residual`, then
`./run.py basis-residual freeze`, `run`, and `report`. The frozen protocol
pins all lab/optimizer code, experiment scripts, runtime and full data
fingerprints. Complete-state checkpoints preserve model, optimizer,
sampler, epoch remainders and gradient traces. Report every pair and every
unresolved case; retain the current production model if the candidate
regresses on a Basis task. Final test never chooses the architecture.

Reporting first runs the unchanged frozen reporter. A separate storage
reduction then retains its complete output under ignored `runs/`, records
that file's hash, and removes only repeated per-class history tables from
the tracked summary. Every history aggregate and all endpoint class tables
remain unchanged. This does not change evaluation or select an endpoint.

## Historical solved baseline

The original GPTMini already solved the standard hard mode in all three
seeds. At width128/depth6 its depth crossings were800/1,400/5,800,
hard recall3,350/2,850/2,600, and parity5,000/8,400/8,800. These are the
recorded99% validation crossings from bd63e509, documented at18888e2a in
[`basis`](../basis/README.md). The later b7d6e673 softmax controls solved
hard recall at2,950/2,150/2,750, also in all three seeds.

[The historical evidence](analysis/historical_baseline.json) pins those
committed records and their source hashes. The original ignored
`experiments/basis/runs/` directory is absent in this checkout; this is
recorded history, not a new replay of its saved weights.

The historical model and optimizer differ materially from this comparison:

| Component | Successful historical GPTMini | Current source GPT |
| --- | --- | --- |
| Optimizer | AdamW | Schedule-Free AdamW |
| Q/K normalization | Unit vectors with learned per-head temperature | Parameter-free RMSNorm and default SDPA scale |
| Residual branches | PreNorm; unnormalized branch outputs | PeriRMS and fixed1/sqrt(2L) branch scale |
| Feed-forward | ReLU squared, hidden width4d | SwiGLU, hidden width about8d/3 |
| Embedding / epsilon / initialization | Historical GPTMini definitions | Source embedding RMS, dtype epsilon, depth-scaled projections |

Both use RoPE, XSA and tied readout. The successful repeat's learned
attention temperatures ranged from about5.01 to7.63 across heads and seeds.
This is a concrete difference to test, not evidence that it causes the gap.
Task162 prioritizes the original architecture under Schedule-Free AdamW
before attributing current misses to normalization or residual connections.

## Initial validation findings and continuation

All24 initial chunks have finished:20 reached99% sequence accuracy and the
four hard-recall cases remain unresolved at100,000 updates. These are
validation results; every final-test split is still sealed.

| Task / model | Seed | Canonical updates | Conditional-gain updates |
| --- | ---: | ---: | ---: |
| Easy depth / small | 0 | 200 | 200 |
| Easy depth / small | 1 | 200 | 200 |
| Hard depth / large | 0 | 600 | 2,000 |
| Hard depth / large | 1 | 1,600 | 7,800 |
| Easy recall / small | 0 | 25,100 | 38,350 |
| Easy recall / small | 1 | 11,600 | 71,100 |
| Hard recall / large | 0 | Unresolved at100,000 | Unresolved at100,000 |
| Hard recall / large | 1 | Unresolved at100,000 | Unresolved at100,000 |
| Parity / small | 0 | 4,600 | 5,000 |
| Parity / small | 1 | 5,400 | 6,200 |
| Parity / large | 0 | 14,000 | 8,400 |
| Parity / large | 1 | 14,000 | 10,200 |

Easy-depth ties have200-update observation resolution. Both hard-depth
models fit short sequences, but the conditional gains transfer to length128
later. Both easy-recall and small-parity pairs also take more updates with
gains. Large parity improves in both seeds:40% and27.14% fewer updates.
The effect depends on the task and size; it is not a general improvement.

At100,000 updates, canonical hard recall has29.61% and82.52% token accuracy;
the gain candidate has9.77% and9.40%. Only canonical seed1 has any exact
validation sequences (20.70%). Its token accuracy rose from51.27% to82.52%
over the last25,000 updates, so [continuation1](analysis/continuation-1.json)
extends it to at most200,000 with the same first99% stopping rule. The other
three states are retained and explicitly censored as unresolved. Their
different horizons must remain visible; an unobserved crossing is not an
infinite time-to-target or a proof of impossibility. These decisions use
validation only and prioritize the matched historically successful control.

An exploratory checkpoint probe of canonical large hard recall at26,100
updates distinguishes memorization from a Schedule-Free weight-state
mismatch. It evaluates the same first512 training rows and all512 validation
rows at both working and averaged weights, on CPU:

| Weights | Training subset sequence | Validation sequence | Validation token |
| --- | ---: | ---: | ---: |
| Working | 98.0469% | 0% | 17.5293% |
| Averaged | 100% | 0% | 17.3828% |

The gap occurs at both weight states. It does not establish a cause, measure
the entire training set, or rule out later grokking. The copied checkpoint
and complete per-class measurements are retained in ignored `runs/probes/`;
[the compact evidence](analysis/canonical-large-recall-s0-step26100.json)
records their hashes. The replay program reproduces all28 aggregate
measurements exactly and never reads final test or changes the active run:

```sh
./run.py basis-state-probe canonical-large-recall-s0 \
  --snapshot experiments/basis_residual/runs/probes/canonical-large-recall-s0-step26100.pt
```

The same control reached its first100,000-update boundary with0/512 exact
validation sequences and1,213/4,096 correct validation tokens (29.6143%).
Its averaged weights still solve all512 examples of the fixed training
subset. Working weights solve493/512 training sequences,0/512 validation
sequences and1,193/4,096 validation tokens (29.12598%). The same gap is
therefore visible at both weight states. Validation token accuracy and loss
have improved since26,100, so the boundary remains an unresolved continuation
checkpoint, not a claim that learning is impossible. The paired candidate
is running next, under the frozen order. Final test remains sealed.

[The boundary probe](analysis/canonical-large-recall-s0-step100000.json)
pins the copied complete state and source. Its six validation counts and
accuracies agree exactly with the canonical CUDA observation; CPU loss is
reported as a CPU replay and differs slightly in rounding.

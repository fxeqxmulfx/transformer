# Convex recall versus a conventional transformer

Does the convex answer-CE matching model proved at `0fcc0d1b` learn held-out
MQAR dictionaries, and can its copying mechanism also learn `a / b mod p`?
This is a native Rust/Kernite comparison. The protocol below was committed
before training at `8564ccaf`. All 24 runs completed their fixed budgets.

## Findings

Both models solve this MQAR setting in all three seeds. The convex matcher
first shows 100% test query and sequence accuracy at step 5, and exactly
equals the identity matrix at step 10, in every seed. The transformer first
shows 100% query accuracy at steps 1,000 / 2,000 / 1,000 for seeds 0 / 1 / 2.
These are scheduled observations, not exact first-passage times. Both finish
at 100% after 6,000 updates. The matching model uses 64 parameters versus
26,368, with the substantial advantage of its supplied pairing/copying prior.

Endpoint accuracies below are percentages, averaged over all three seeds;
brackets give their range. CE is mean held-out ordinary cross-entropy.

| Task | Method | Train accuracy | Test accuracy [range] | Test CE |
| --- | --- | ---: | ---: | ---: |
| MQAR | Proved convex matching | 100 | **100 [100, 100]** | 2.247202 |
| MQAR | Transformer | 100 | **100 [100, 100]** | 0.000000119 |
| Division mod 17 | Copy adaptation | 15.44 | 9.07 [6.62, 10.29] | 2.917312 |
| Division mod 17 | Convex table control | 100 | 5.15 [2.94, 7.35] | 2.944439 |
| Division mod 17 | Transformer | 100 | 11.76 [8.82, 14.71] | 4.487515 |
| Division mod 97 | Copy adaptation | 2.60 | 1.62 [1.52, 1.76] | 4.590720 |
| Division mod 97 | Convex table control | 100 | 1.13 [1.10, 1.16] | 4.595120 |
| Division mod 97 | Transformer | 19.37 | 16.26 [7.56, 20.70] | 2.793301 |

The matcher reaches its analytically known CE minimum; its larger raw CE
than the transformer's reflects the fixed unit logit scale, not recall errors.
For division, the copy adaptation cannot represent most outputs. The table
control fits every training equation but leaves all held-out logits exactly
uniform, as predicted by its zero held-out gradient. With the specified
first-maximum tie rule it always predicts zero there; its accuracy is the
held-out fraction of zero labels, not a sampled uniform-class accuracy.

The transformer memorizes mod 17 without learning the whole division rule
in this budget. On mod 97 it has not even fit the training split, and its
held-out accuracy varies substantially by seed. It exceeds the copy-only
whole-table ceiling, showing a capability the copying family lacks, but
does not solve division here. These small-width, 20,000-update runs do not
establish the transformer's eventual performance or a general grokking limit.
No run was extended, replaced, or selected by held-out accuracy.

The useful conclusion is that ordinary convex answer training suffices for
this structured recall architecture. Its fixed value-copying prior does not
supply a general arithmetic model. Adding an independent convex answer table
supplies capacity to memorize arithmetic, without a mechanism that identifies
unseen pairs. The Lean flat-spectrum result is consistent with all of these
observations.

The three seed streams ran concurrently, each transformer with four CPU
workers. Mean recorded times are 0.78 s for matching and 27.81 s for the
transformer on MQAR; 30.23 s and 37.94 s for the transformer on mod 17 and
mod 97. They include preceding evaluations/checkpoint I/O within a run,
but exclude initial evaluation and the current final evaluation. CPU
contention, different precisions and different architectural work prevent
interpreting these as controlled speedup measurements.

The native report validates all **24 endpoints**, matched input/split hashes,
finite losses, the copy-support ceilings, unchanged unseen table logits,
and exact matching identity recovery. Every run records trainer source SHA256
`cb81c4018e20260d58fad7f06fea0407606824b9de6a5411fdbb60a14d60e51d`.
Raw metrics, metadata, resumable checkpoints and the generated summary remain
in ignored `runs/`; only the source and authored interpretation are committed.

## Fixed protocol

Seeds are **0, 1, 2**, with identical data and minibatch streams across
methods within each seed. No validation-based selection, early stopping,
learning-rate search, or favorable-checkpoint selection is used. Report the
endpoint, all checkpoint evaluations, and any numerical failures.

| Task | Input and split | Updates | Batch |
| --- | --- | ---: | ---: |
| MQAR | 8 keys, 16 values; 512 unique training dictionaries and 512 disjoint test dictionaries; independently shuffled binding/query orders; all 8 queries | 6,000 | 32 dictionaries |
| Division mod 17 | `[a, /, b, =]`, `a=0..16`, `b=1..16`; random half of all 272 pairs for training | 20,000 | 64 equations |
| Division mod 97 | Same encoding; random half of all 9,312 pairs for training | 20,000 | 64 equations |

Each MQAR input contains eight adjacent key/value pairs followed by eight
query keys, for 24 input tokens. Only the eight query positions are scored.
No answers are appended to the queries. A test dictionary's assignment of
values never appears in training, regardless of order. The checked training
separation condition uses only input dictionary values.

All methods minimize the **ordinary answer softmax cross-entropy over the
same whole token vocabulary**, including the key/operator tokens. Evaluation
reports mean CE, query accuracy, and the fraction of sequences with every
query correct. Division has one query per sequence. Sampling is uniformly
with replacement, using a separate fixed MT19937 stream.

| Method | Trainable model | Training |
| --- | --- | --- |
| `matching` (MQAR) | The proved `8×8` stochastic key-matching matrix, fixed categorical value copying, unit logit scale | Uniform initialization; projected gradient, learning rate 1; no decay |
| `transformer` (both tasks) | Two causal softmax-attention blocks; width 32, four heads, ReLU FFN width 128, post-LayerNorm, residuals, sinusoidal positions, learned embeddings and untied readout | Source initialization and AdamW, betas `(0.9,0.98)`, epsilon `1e-8`, 10-step warmup; MQAR LR `0.003`, decay `0.1`; division LR `0.001`, decay `1` |
| `copy` (division) | Explicit adaptation: `z(a,b)=α_a e_a + β_a e_b`, `(α_a,β_a)` on the two-point simplex | Uniform initialization; projected gradient, learning rate 1; no decay |
| `table` (division) | Separate convex control: independent logits for each input pair and output class | Zero initialization; gradient descent, learning rate 1; no decay |

The transformer is the conventional decoder of `grok_division` at
`891e4525`, using its validated Kernite x86 GEMM, parameter layout, optimizer,
and RNG. The model source is copied with only its two attention scratch arrays
changed from length 16 to the actual context length. The double-precision
reference tests its longer-context logits/gradients. This smaller model and
budget are **not a replication of the paper's width-128 million-update run**.

## Architectural scope

`matching` implements `FlatSpectrum.ConvexRecall` exactly, with the input
key/value pairing supplied. The transformer must learn the local pairing
from positions and tokens. This is an architectural-prior comparison, not a
parameter-matched ablation or a drop-in embedding/attention replacement.
The matching model has 64 parameters; the transformer has 26,368 on MQAR,
26,048 on mod 17, and 31,168 on mod 97.

The proved model needs a dictionary input. Its division adaptation is stated
explicitly and is **not covered by the dictionary-label learning theorem**.
It can only output one of the numeric input tokens. Exhaustive native checks
find `3p−4` copyable equations among `p(p−1)`, giving whole-table ceilings of
**47/272 = 17.2794%** for mod 17 and **287/9312 = 3.0820%** for mod 97. The
held-out split has its own recorded ceiling. No division table or modular
inverse is supplied as a forward feature; arithmetic is used only to build
labels and check results.

The `table` control can memorize training pairs but has exactly zero gradient
on held-out rows; zero initialization therefore keeps their logits uniform.
It is a different convex architecture and does not inherit the recall theorem.
These controls distinguish representational limitations from unseen-pair
identification, rather than attributing every failure to optimization.

Unit-scale matching bounds confidence even at exact recall. Its optimal CE
is `log(e + V − 1) − 1`, with `V=24`; compare accuracy and the gap to this
known minimum as well as raw CE. A conventional transformer can increase
logit scale. No auxiliary route labels, calibration objective, or answer
features are used by any method.

## Reproduction and checks

Kernite is clean at `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`, under
`/home/misha/rust_projects/kgen`. The current host reports an Intel Xeon
E5-2680 v4 and, when queried outside the sandbox, a GTX 1050 with 2 GiB.
This comparison uses the existing verified **Kernite CPU** transformer engine
with four workers, not a new GPU port. Timings are local execution measurements,
not GPU throughput or an exclusive-device benchmark. Transformer arithmetic is
f32; small convex models use f64; evaluation CE is f64 for both.

```sh
cd experiments/convex_recall_compare
cargo fmt --check
cargo test --offline --release
cargo build --offline --release
for seed in 0 1 2; do
  for method in matching transformer; do
    target/release/convex-recall-compare run mqar "$method" "$seed" 6000 "runs/mqar-$method-$seed"
  done
  for task in div17 div97; do
    for method in copy table transformer; do
      target/release/convex-recall-compare run "$task" "$method" "$seed" 20000 "runs/$task-$method-$seed"
    done
  done
done
target/release/report runs > runs/summary.json
```

Repeating a command resumes its checkpoint, including the minibatch RNG and
optimizer moments. Metadata rejects changed source/configuration/splits.
`run.json` records source and data hashes, shapes, separation and copy-support
ceilings. `metrics.jsonl` retains evaluations at steps 0, 1, 2, 5, 10, 20,
50, 100, 200, 500, every 1,000 updates, and the endpoint. All generated
artifacts are ignored in `runs/` and preserved outside temporary worktrees.

Tests cover arithmetic labels, train/test disjointness, input separation,
finite-difference convex gradients, sparse/full gradient agreement, the
projected optimizer's identity solution, longer-context transformer agreement
with an independent double reference, causal masking, and exact checkpoint
continuation. The existing Lean proof remains the mathematical guarantee;
finite training observations are not an optimizer convergence proof.

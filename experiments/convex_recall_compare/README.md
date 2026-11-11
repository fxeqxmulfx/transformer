# Convex recall versus a conventional transformer

Does the convex answer-CE matching model proved at `0fcc0d1b` learn held-out
MQAR dictionaries, and can its copying mechanism also learn `a / b mod p`?
This is a native Rust/Kernite comparison. The protocol below was written
before training. Results are pending.

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

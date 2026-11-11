# Division mod 97 under the author's Adam and AdamW

Does the transformer of arXiv:2201.02177 (Power et al.), trained on half of the division table
mod 97 exactly as the author's code trains it, come to answer every held-out equation: under
Adam without weight decay within the `10⁶` updates of the paper's Figure 1, and under AdamW
with weight decay 1 within the `10⁵` updates of its default setting?

[PROTOCOL.md](PROTOCOL.md) pins the task, the model, the loss, the optimizer and the sampler
to the paper and to openai/grok at `3d64b1d8`, lists where the two disagree, and fixes the
recorded steps, the terminal window and the outcome classes. It was frozen before training.

## Runs

| Cohort | Optimizer | Weight decay | Updates | Seeds | Supervised tokens |
| --- | --- | ---: | ---: | --- | --- |
| `adam` | Adam | 0 | 1 000 000 | 0, 1, 2 | answer and closing `<\|eos\|>` (the code's loss) |
| `adamw` | AdamW | 1 | 100 000 | 0, 1, 2 | answer and closing `<\|eos\|>` |
| `adamw --answer-only` | AdamW | 1 | 100 000 | 0, 1, 2 | answer alone (the paper's wording) |
| `adam`, loss in doubles | Adam | 0 | 1 000 000 | 0, 1, 2 | answer and closing `<\|eos\|>`; the trainer of commit `b8ff967f` |

Equal seeds have equal splits, initial parameters and batches in every cohort. Seed 0 has the
split of the author's code.

## Commands

The trainer is native Rust; its matrix products run Kernite's blocked x86 kernel (AVX2 and
FMA). Kernite is expected at `/home/misha/rust_projects/kgen`, clean at
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`; `check` refuses any other state.

```sh
cd experiments/grok_division
cargo test --offline --release           # 31 tests
cargo build --offline --release
target/release/grok-division check       # pins, split fingerprints, f32 model against the model in doubles
target/release/grok-division bench 8     # the time of an update
for seed in 0 1 2; do
  target/release/grok-division run adamw $seed runs/adamw-seed$seed --threads 8
  target/release/grok-division run adam $seed runs/adam-seed$seed --threads 8
done
target/release/grok-division report runs/*-seed*
target/release/diagnose runs/adamw-seed1 1 10000 50000   # what checkpoints do at the answer
```

`diagnose RUN SEED STEP...` reads checkpoints with the model in doubles: accuracy on 512
training equations, the hidden units that fire, the entropy of the attention, what the
normalizations divide by, and the norms of the parameter groups. It is a second binary and
does not change the trainer or its source fingerprint.

A run resumes from its last checkpoint when the command is repeated, on any number of
threads, and is then bit for bit the run that was not stopped. `runs/` is ignored by git.

## Findings

The runs are in progress; this section is written when they end.

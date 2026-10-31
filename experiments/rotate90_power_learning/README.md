# Learning powered modular division in the current Rotate90 model

Can training learn `(a/b)^K mod P`, beyond the proved arithmetic uniqueness
and the one-block representability result for K=1?

Requested on 2026-10-31. This study uses the current `grok-simple.py`
Peri-RMSNorm SwiGLU forward, with tied E and no biases or learned norm gains.
The previous `rotate90_analytic` experiment used a different forward and
initializer. Every run here uses Python 3.15 and PyTorch 2.15, CUDA,
`torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)` for
forward/loss/backward and inference, and fused native AdamW updates.
The installed runtime is Python 3.15.2 / PyTorch 2.15.0+cu132 on an RTX 3050
Laptop GPU with 4 GiB. TorchInductor reports that this GPU has too few SMs
for its GEMM autotuning search; max-autotune compilation still runs. CPU
source-fixture weights and logits match exactly. Compiled GPU losses and
gradients match eager evaluation within float32 tolerances. Continuation
checks compare every saved state with rtol 2e-6 / atol 2e-7, with exact
sampler state and accuracies; bitwise CUDA trajectories are not assumed.

| Factor | Values |
| --- | --- |
| Prime P | 17, 97, 193 |
| Exponent K | 1, 2, 3 |
| Width/depth | 16/1, 64/1, 128/1, 64/2, 64/4 |
| AdamW learning rate | .001, .003, .01 |
| Weight decay | 0, .1, 1 |
| Betas | (.9,.98), (.9,.999), (.5,.98) |
| Base | width 64, depth 1, lr .003, decay 1, betas (.9,.98) |
| Split and budget | 50% train, 5000 updates, batch 512, 10-step warmup |
| Evaluation | complete train/heldout tables every 250 updates |

The 45 architecture runs cross five shapes with nine tasks. The 54 optimizer
runs change one optimizer factor from the base on the same nine tasks.
Three full-table controls at K=3 measure fitting without held-out data.
The `timing` label is a 100-update compilation and performance pilot.
Twelve follow-up runs were added after the initial width-16 results: nine
learning-rate/decay combinations at P=193, K=1, width 16, depth 1 (one
repeats the base setting), and K=5 on all three primes with the base model.
K=5 preserves every output class at these primes. The narrow K=1 follow-up
tests optimization where the existing Lean theorem guarantees expressivity.
Seeds initially use 11 for initialization, split and sampler. Operand pairs
and batch orders match across K and all model/optimizer variants for each P.
K=2 and K=3 can have fewer output classes; result metadata records class
counts and majority-class accuracy. Zero numerators are included and zero
denominators excluded. Every label passes `b^K*c = a^K mod P` independently.

Runs are budgeted equally in optimizer updates, not FLOPs or epochs across P.
Compilation/preparation time appears in the initial wall time; training
seconds exclude compilation and evaluation. A failure at 5000 updates is a
failure within this budget. Held-out results are exploratory sweep data.

Status: source parity, arithmetic labels, gradients and compiled continuation
are validated. The sweep is running. Portable results and complete final
checkpoint audits will be recorded here when the runs finish.

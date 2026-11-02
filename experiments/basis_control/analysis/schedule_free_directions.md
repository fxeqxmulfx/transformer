# Schedule-Free directions before another training run

Task166 asks whether averaging opposes the fresh gradient in saved slow Basis
states. This is a cheap falsification check, not an optimizer sweep. The
prediction below precedes the measurements. Use all six closed hard-recall
endpoints: original GPTMini, canonical GPT and the temperature candidate,
each with seeds0/1. Read only each saved sampler's next batch. Do not advance
the optimizer, evaluate final tests, select a new checkpoint or change a model.

The source is the local arXiv:2405.15682 manuscript, §2 and §4.4, together with
`ResidualScheduleFreeAdamW` atccf91e8e. The local implementation uses corrected
second moments and squared maximum-rate averaging weights; the paper's
printed algorithm instead absorbs bias correction into its rate. Derive from
the actual implementation, rather than treating those early-step conventions
as identical. Its relevant state after a completed update is the working
weight `y`, base weight `z`, second moment `v` and averaging total `S`.

For the next update, let `eta` be the rate, `c = M²/(S+M²)` with
`M = max(previous_maximum, eta)`, and let `p` be the freshly bias-corrected
Adam direction. The exact real-arithmetic working-weight increment is

```
delta_y = a + d + w
a = c * (z - y)
d = -eta * (1 - beta1 + beta1*c) * p
w = -eta * (1 - beta1 + beta1*c) * weight_decay * y
```

The coefficient of `p` alone is insufficient to infer learning speed. In a
constant-direction thought experiment with no warmup or decay, `c=1/t`,
`z_t=y_0-t*eta*p` and `x_t=y_0-(t+1)*eta*p/2` for `t>=1`, because `x_t`
averages `z_1,...,z_t`. Therefore

```
y_t = (1-beta1)*z_t + beta1*x_t
    = y_0 - ((1-beta1/2)*t + beta1/2)*eta*p.
```

The first update has coefficient1; subsequent increments have coefficient
`1-beta1/2`, which is0.55 when beta1=0.9. A blanket tenfold rate increase does
not follow. Conversely, a sudden reversal of `p` makes `a` retain the old
direction. It can dominate the fresh update and move uphill on that batch.
These are algebraic witnesses, not assumptions about a transformer's gradient
sequence. The manuscript's convex convergence bounds do not prove grokking.

Before the probe, define the proposed signature of strong local opposition:
`dot(g,a) >= 0.5 * -dot(g,d)`, where `g` is the current batch gradient.
Report `dot(g,w)` separately: weight decay can raise training loss and still
help generalization. Also report vector cancellation
`1 - norm(a+d+w)/(norm(a)+norm(d+w))`, and the separate norms and inner products
for every parameter, rather than relying only on a global norm.

Reject the claim that this is a common endpoint mechanism if the signature
is absent in any of the three zero-exact-validation endpoints. Presence would
only motivate further diagnostics: endpoints have unequal ages and learning
progress, one batch is noisy, and local training descent is not generalization.
Do not launch a rate or momentum sweep from this diagnostic alone.

The CPU forward/backward uses the saved FP32 weights and the math attention
backend. Direction algebra and reductions use FP64 after that gradient is
computed. This approximates the real recurrence at that point; it is not an
exact prediction of a compiled CUDA update. Frozen model/optimizer/sampler
sources, task fingerprints, recipes, checkpoint hashes and next-batch indices
are checked and recorded. No saved-state bytes may change. Replay with
`./run.py basis-direction-probe`.

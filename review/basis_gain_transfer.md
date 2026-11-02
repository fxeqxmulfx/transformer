# Conditional residual gains: same-assistant diagnostic review

Task161 examines why the task160 gate candidate reaches length128 validation
later. This is a post-hoc diagnostic of closed models, not a new architecture
comparison. No model is promoted. The current diff_gpt remains df1e9f7.

The probe checks the frozen implementations of the actual canonical/gated
models, RMS backward, centered XSA and Schedule-Free optimizer. It checks all
original data fingerprints and checkpoint hashes, then reads the first512
training rows and validation in working and averaged modes. Easy depth did
not have length128 validation during training; its new deterministic validation
stream is separately fingerprinted. No final-test predictions enter this probe.
All original validation prediction counts agree with the closed report.

| Hard-depth case | First99 short validation | First99 length128 |
| --- | ---: | ---: |
| Canonical seed0 | 200 | 600 |
| Conditional seed0 | 200 | 2,000 |
| Canonical seed1 | 400 | 1,600 |
| Conditional seed1 | 600 | 7,800 |

For a gain g(z)=2sigmoid(z), the derivative divided by its maximum1/2 is
`g*(2-g)`. The diagnostic calls a gate saturated when this fraction is below
0.1. Over nonpadding validation tokens and all12 gates, the averaged hard-depth
seed0 endpoint has no saturated activations. Seed1 has10.15% on short sequences
and10.76% at length128. Its final FFN gain averages1.9589 and1.9763 respectively.
Small-model endpoints have none. Saturation is present in one slow endpoint;
it is not a common explanation established for both slow seeds.

Inference ablations act on copied averaged models without changing any saved
weights. Setting every gate to one reduces length128 accuracy to55.66% and
20.31% for hard-depth seeds0/1. Replacing each gate with its mean over the
training subset's tokens gives93.95% and36.52%, versus100% and99.02% with the
learned conditional gates. The final function depends on both amplitude and
conditioning. This does not show that conditioning caused the earlier delay:
other weights have co-adapted, and the endpoints have unequal update counts.
Easy-depth length128 validation is mixed: canonical/gated seed0 scores67.77%/
100%; seed1 scores99.61%/88.67%. This is additional validation, not a fresh test.

The original diagnostics do retain per-parameter gradient and update norms,
though not gradient vectors or intermediate weights. All588 sampled hard-depth
gate gradients are finite and nonzero; the smallest are about1e-10 on nearly
fitted training batches. Global gradient traces cover every update. The real
RMS formula gives `abs(w dot RMS(x)) <= sqrt(width)*norm(w)`. Saturation below
10% of maximum derivative needs `abs(z)>2acosh(sqrt(10))`, about3.63689. The
recorded norm bound remains below that threshold through update400 in seed0
and600 in seed1. It first permits saturation at600/800, but an upper bound
above the threshold does not establish it. These bounds use recorded FP32
norms with the real formula, not a formal floating-point certificate. The
fixed-input update bound also omits changes to earlier features.

The independent width oracle has two opposite RMS-one feature vectors and
target gains1.5/0.5. A finite exact solution is `w_i=log(3)/width`. At zero
weights every gradient coordinate is-1/4. The actual Schedule-Free first step
matches `z=width*rate*(1/4)/(1/4+eps)` in FP64 for widths16–1024 and two rates.
At rate0.001 and widths64/128, the logits are0.064/0.128, far from saturation.
At width1024 and rate0.01, the first-step derivative falls to0.0143% of its
maximum despite the finite solution. This isolates a possible scaling issue
for aligned features; it does not diagnose real transformer features. It uses
an isolated constant-rate first update. Actual Basis has a50-update warmup,
and its recorded first learning rate is zero.

The evidence supports neither deleting gates nor changing their scaling as a
production repair. Task165 will retain intermediate full states in declared
matched depth repeats so actual activations can be aligned with learning.
Any resulting intervention must be declared before training and pass task164's
powered-division and Basis confirmation. No new architecture flag is added.

`./run.py basis-gain-probe` replays the endpoint measurements, recorded norms
and finite-solution optimizer oracle; all outputs are required to match their
saved JSON exactly. Raw diagnostic hashes and source hashes are retained in
`analysis/`. All review is performed by the same assistant.

Verification passes 513 lab tests in 910.052 seconds with13 CUDA skips and
104 optimizer tests in13.817 seconds with14 skips. The endpoint replay's
complete hash is `a53c0a87468dfffa1c79f1089c531be2a717060a7d51008f840aabb906a65328`;
the trajectory-norm report hash is
`1b72a511e3b4e42300b73c2a3be793ea37a40365446b52528e2bac5fe6819944`.
All three numerical reports reproduce exactly through the registered command.

# Grokking: how it works and when it fails

Started 2026-10-08 at the user's request. Status: **active**. Work solo and
follow AGENTS.md. This plan replaces three earlier plans:

- the progress-measurement plan this file held, whose runs are written up in
  [grokking_progress](experiments/grokking_progress/README.md) and
  [grokking_internals](experiments/grokking_internals/README.md);
- the RASP realizability plan of the same day, which is absorbed here;
- the route list of [grokking_lean_plan.md](grokking_lean_plan.md), which
  stays as the ledger of its results.

The [Basis/convex cycle](plan.md) stays paused, and ANSR stays stopped.

## Question

Grokking is delayed generalization: a model fits its training split long
before it answers held-out inputs from the same distribution.

1. **Mechanism.** What changes inside the model between the fit and the
   generalization, and why does it take so long?
2. **Conditions.** For which task, model size, data, decay, numerics and
   budget does it happen? When it does not, which condition was missing?

Tasks with a known algorithm sharpen both questions. The algorithm's RASP
program names the circuit the model has to form. The program's size says
which models can hold that circuit at all.

## Outcome of a run

These definitions are pinned in lab code, with tests, before any run.

- **Four phases.** We use Liu et al.'s definitions (arXiv:2205.10343v2,
  Appendix A, Table "Definitions of the four phases of learning") with our
  budget.
  - Comprehension: train and held-out accuracy both pass 90%, and held-out
    passes less than 10³ updates after train.
  - Grokking: both pass 90%, but held-out passes later than that.
  - Memorization: only train passes.
  - Confusion: neither passes.

  The 99% times of Power et al. are reported beside them.
- **Semi-grokking and ungrokking** (Varma et al., arXiv:2309.02390v1,
  §4.2, §5.2–5.3).
  - Semi-grokking: held-out accuracy rises well after the fit but settles at
    a middling value. We count a final value between chance and 90%.
  - Ungrokking: held-out accuracy falls back after a sustained success,
    once training continues on a smaller split.
- **Censoring.** A memorization run means "no transition by update T". It
  never means "no transition".
- **Held-out** means inputs that no training row contains, at the training
  lengths. **Length transfer** is the accuracy at longer lengths. It is
  reported on its own and never merged into held-out accuracy.
- **Recording.** Every run records full curves at a fixed interval, with no
  early stopping and no checkpoint selection, and keeps checkpoints for
  stage 6.

## What the sources establish

Each claim below is a hypothesis to test here, not a premise.

| Source | Claim | Setting and caveat |
| --- | --- | --- |
| Power et al., arXiv:2201.02177v1, §3.1–3.3 | Updates to 99% held-out grow steeply as the training fraction falls: on S₅ near 25–30%, 1% less data costs 40–50% more updates (§3.1.1). x³+xy²+y mod 97 did not generalize within the budget at any fraction up to 95%: the models memorized (§3.2). Weight decay is the strongest intervention, and the learning rate works only within about one order of magnitude (§3.3). | Two layers, width 128, four heads, AdamW at decay 1, 10⁵ updates; 5·10⁵ for the learning-time curves (App. A.1.2) |
| Liu et al., arXiv:2205.10343v2, §3–4 | Generalization comes with structured representations. The time to generalize diverges as the training set shrinks toward a critical size, which their effective theory predicts (§3.2). Over the decoder's learning rate and weight decay the outcomes form four phases: a fast, capable decoder memorizes, and a fast, simple one is confused (§4.1). | Effective theory of a toy model. The transformer diagram (§4.2) applies decay to the decoder only |
| Nanda et al., arXiv:2301.05217v1, §5 and appendix "Hypothesis: Phase Transitions are inherent to composition" | Training passes through memorization, circuit formation and cleanup. The circuit forms well before held-out loss falls, and the fall comes during cleanup. A part of a composed circuit lowers the loss only together with the other parts, so its gradient stays weak until they form. Then the gradients grow together, which would make the transition abrupt. | One-layer transformer, modular addition. The composition point is a hypothesis |
| Varma et al., arXiv:2309.02390v1, §3–5 | Grokking needs a generalizing circuit (Gen) that is more efficient than memorization (Mem), giving larger logits per parameter norm, but is learned more slowly. Mem's efficiency falls as the data grow, so a critical size D_crit exists. Below it there is no grokking, near it there is semi-grokking, and shrinking the data below it causes ungrokking. D_crit does not depend on the decay strength. | One-layer transformer, modular addition mod 113, AdamW. Efficiency is measured on Mem-only runs with random labels and Gen-only runs at large sizes (§5.1) |
| Prieto et al., arXiv:2501.04697v1, §3–5 | Without regularization, after the fit the gradient aligns with "naive loss minimization", which only scales the logits (§4). Float softmax eventually collapses and learning stops. The collapse comes earlier in float32 than in float64, and with little enough data it comes before any held-out progress (§3). StableMax prevents the collapse and brings grokking without regularization (§3.3). ⊥Grad prevents the scaling, and the model generalizes without first overfitting (§5.1). | MLPs with two hidden layers of width 200 and a one-layer transformer with four heads, full batch, no decay unless stated |
| Golechha, arXiv:2405.12755v1 | Grokking occurs outside the expected range of weight norms. Norm correlates with it but does not cause it. | Classifiers on MNIST and IMDb |
| Morris et al., arXiv:2505.24832, abstract and §3 | GPT-style models store about 3.6 bits per parameter, and 3.83 when trained in float32. On growing datasets they memorize until that capacity fills; then generalization begins and unintended memorization falls. | Language models of 500K to 1.5B parameters; capacity measured on uniform random data |
| Weiss et al., arXiv:2106.06981v2, §5, Tables 1–2 | A RASP program compiles to L layers and H heads, an upper bound. Below it accuracy drops. Reverse at (2,1) falls from 99.9% to 23.1% with one layer, and to 41.2% with one layer of twice the heads and width. Double histogram at (2,2) falls from 99.0% to 40.5%. Sort at (2,1) keeps 99.0% with one layer. The authors read this as a bucket sort: over a bounded alphabet one full-attention head counts every symbol. | Four runs per size at width 256, on 50,000 sequences of length 0–100, for 100 epochs. Adam, with no weight decay reported. Alphabets of 100 symbols for reverse and sort, and 26 for the rest. The test set comes from the same distribution, and train accuracy is not reported. So this measures capacity, not grokking |
| Yang et al., arXiv:2506.16055v3, §4–5 | Proved: a transformer that rounds to fixed precision outside attention needs depth exactly k to recognize L_k (k alternating blocks a⁺b⁺…) at every length. This holds without positions. It also holds for E_k (L_k with a neutral letter) under sinusoidal or RoPE encodings at rational angles, or ALiBi. Trained NoPE models follow the bound at the training lengths. On L₆ at lengths 201–250, depths 1–3 reach 36–74% and depth 4 reaches 100%. | The bound is uniform in length. At bounded lengths the language is finite, and the theorem says nothing there. The experiment uses the prediction problem, over prefixes of strings in L_k only, which needs depth k−2. It trains on 800 strings of length 201–250, with Adam, for at most 25 epochs |
| Zhou et al., arXiv:2310.16028v1, §2–5 | Conjecture: a transformer trained to completion is likely to generalize to longer lengths under three conditions. One transformer must solve the task at every length; a short RASP-L program must express that solution; and the training data must rule out every shorter program that agrees only in distribution (§2). | A phenomenological conjecture. Training is online: every batch is drawn fresh, with no finite training split. Model sizes are chosen per task to fit (App. A). The model class admits weights in ℝ ∪ {±∞}, so softmax can saturate at every length, which no finite-weight model does (§2) |

## The frame to test

Three boundaries, each measured on its own, should predict the phase of a
run:

1. **Expressivity.** Does some model of this architecture and size compute
   the rule on the evaluated inputs? If not, there is nothing to grok.
2. **Memorization capacity D_mem.** The largest training split the model
   fits with random labels. Above it, the split cannot be memorized.
3. **Critical size D_crit.** Below it, memorizing the split costs less
   parameter norm than the rule.

| Rule expressible | Training split size D | Predicted phase |
| --- | --- | --- |
| yes | D_crit < D < D_mem | grokking |
| yes | D ≥ D_mem | comprehension |
| yes | D ≤ D_crit | memorization; semi-grokking near D_crit |
| no | D < D_mem | memorization, or a shortcut valid only at the training lengths |
| no | D ≥ D_mem | confusion |

When D_mem ≤ D_crit the window is empty, and the model never groks. Even
inside the window, decay, learning rate or numerics can still prevent
grokking. D counts training rows. Across tasks, compare the number of label
bits instead.

## Hypotheses

- **H1, the expressivity gate** (Weiss §5; Yang et al. §4–5; Liu §4).
  Below the size that expresses the rule, no amount of data, decay or
  budget produces it.
  1. On L_k without positions, no model shallower than k is right at every
     length. For the rounded class this is a theorem; for the float32 lab
     model it is a prediction. Runs can only show failure at the lengths
     they evaluate, so the test is whether a shallower model already fails
     at the longer evaluated lengths. If it stays right there, the bound
     does not bite at those lengths. Then extend the lengths before drawing
     any conclusion.
  2. At the training lengths the uniform bound says nothing, because there
     the language is finite. A shallow model may still answer held-out
     inputs through a shortcut tied to the training lengths. Depth 1 is an
     exception. Without positions, one attention layer sees only the current
     token and the counts of each token in the prefix. So it answers alike
     on two words with the same counts and the same last letter, such as
     aabb ∈ L₂ and abab ∉ L₂. Such pairs exist for every k ≥ 2, so depth 1
     fails at every length. The paper's commutativity lemma for depth-1
     formulas (`CRASP.commutativeOnMiddle_of_mem_TLCP_one`) is the logic
     version of this argument.
  3. For the Weiss tasks, removing a layer below the causal program's depth
     lowers held-out accuracy in the grokking regime too. The exceptions are
     a width large enough for a lookup and an alphabet small enough for a
     bucket sort. On a bounded domain a depth bound is really a depth–width
     trade-off: one layer of width |Σ|·n_max can store the whole input.
- **H2, the data window** (Varma §4–5; Morris; Power §3.1.1; Liu §3.2).
  The phases follow the table above, and the plateau t_gen − t_fit diverges
  as D falls toward D_crit. D_crit does not depend on decay strength, and
  D_mem grows with the parameter count.
- **H3, composition delay** (Nanda, appendix on composition). Suppose a rule
  needs m attention steps composed in sequence. Each step lowers the loss
  only together with the others, so its gradient stays weak while they are
  rough. So at fixed D/D_crit the plateau should grow with m, which is k for
  L_k and the causal layer count for the Weiss tasks. Whether extra depth
  beyond the minimum shortens the plateau is open.
- **H4, decay and numerics** (Prieto §3–5; Power §3.3; Liu §4).
  - At decay 0 the update aligns with logit scaling after the fit.
  - Softmax then collapses in float32 and learning stops. With few enough
    data, it stops before held-out accuracy rises.
  - Float64 postpones the collapse, and StableMax prevents it.
  - Heavy decay at a high learning rate gives confusion (Liu), and the
    learning rate works only within about one order of magnitude (Power).
- **H5, the mechanism** (Nanda §5; Varma §3, §5.1). During the plateau the
  rule's circuit grows while memorization is cleaned up. Parameter norm
  moves from Mem to Gen because Gen gives more margin per unit norm. Expect
  four observations:
  - heads align with the program's selectors before t_gen;
  - before cleanup, ablating those heads removes held-out accuracy but
    leaves train accuracy;
  - Gen's efficiency does not depend on D, while Mem's falls as D grows;
  - the two efficiencies cross at the D_crit of H2.

  The learned circuit need not be the compiled one: Weiss's reverse model
  found another way to compute `length`.

## Tasks

Every task has an explicit rule. Model size means depth, heads and width,
each reported, with the parameter count. Alphabet size and the number of
classes describe the task, not the model.

| Task (lab word) | Rule and its size | Proved in Lean | Role |
| --- | --- | --- | --- |
| L_k, k = 1…6: `AlternatingBlocks(blocks=k, neutral_fraction=0.0)` with `NoPositions()` | Every prefix is labeled by membership in L_k. The lab samples words of 1 to 2k+3 runs, starting with either letter. That is recognition, which needs depth exactly k, not the paper's prediction problem, which needs k−2 | `CRASP.rtfr_depth_hierarchy`, both directions, for the rounded class | Core: a proved gate, and composition depth m = k |
| E_k: `AlternatingBlocks(blocks=k)` with `RoPE()` | Depth k at rational RoPE angles | `CRASP.not_recognizes_altPlusNeutral_rope` covers rational angles only, so it does not cover the lab's RoPE at base 10000 | Control: whether the bound survives real positional encodings |
| `Histogram`, `Reverse`, `DoubleHistogram`, `Sort`, `MostFrequent` | Weiss's encoder sizes are (1,1), (2,1), (2,2), (2,1), (3,2). The lab writes each answer after the input, so stage 0 derives the causal depth | Semantics of reverse (`RASP.reverse_apply`) and sort (`RASP.sortProg_apply`) in encoder RASP. No lower bound beyond zero heads (`RASP.Expr.not_reverse_of_heads_eq_zero`) | Weiss's size reduction, repeated in the grokking regime |
| `Parity()`, no scratchpad | No rule works at every length; at bounded length it is a lookup | `CRASP.not_recognizes_parity`, at any depth | Control: a rule that exists only at bounded lengths |
| `ModularDivision(97, …)` | A finite table | `Grokking.DivisionOrbits` | Calibration against the archived grokking runs |

## Stages

**Rules for every stage**

- **Commit.** Each stage ends in a commit containing:
  - an experiment folder in the lab's language;
  - its README: the question, a table of how the runs differ, and what
    they found;
  - its row in `experiments/README.md`;
  - its status in this file.
- **Fixed in advance.** Grids, budgets and seeds go into `experiment.py`
  before any run.
- **Seeds.** At least three model seeds per cell, with the data seed also
  varied at the boundaries.
- **Comparing sizes.** Runs of different sizes are compared at equal
  training tokens and at equal FLOPs.
- **Training.**
  - Unchanged AdamW from ordinary initialization, on answer cross-entropy.
  - A compiled program shows what a solution can look like. It is never a
    training target.
- **Default recipe.** The reference transformer of
  [mod97_grokking](experiments/mod97_grokking/README.md), which grokked in
  all three confirmation runs:
  - post-norm with LayerNorm, ReLU and an untied readout;
  - AdamW at rate 1e-3 with betas (0.9, 0.98) and a warmup of 10 updates;
  - for L_k, no positional encoding.

  GPTMini follows once this recipe has results.
- **Budget.** 10⁵ updates, or 5·10⁵ near a boundary, as Power et al. used
  for their learning-time curves (App. A.1.2).

### 0. Protocol and expressivity ledger

- Pin the outcome definitions in the lab's `phases`, with tests.
- For each task, write its rule as a causal program for the lab's exact
  serialization. Weiss's programs are noncausal, and the lab writes their
  outputs after the input, so his (L, H) do not carry over.
- Record each program's depth and heads, and what is proved about it:
  - a construction, which gives an upper bound;
  - an impossibility for a named model class, which gives a lower bound;
  - nothing, so the bound is only empirical.

  RASP-L allows infinite weights. Every construction here uses finite
  weights.
- Where a construction is explicit, set a lab model's weights to it and
  check it on every evaluated length. This checks the step from the rounded
  class to the float32 lab model directly.
- Test the lab's L_k labels against an independent implementation of
  Lean's `altPlus`, on every word up to length 12.
- Measure milliseconds per update on one core for each model size. Size the
  grids of stages 1–5 to finish overnight on the 28-core farm.

### 1. Memorization capacity D_mem

- **Runs.** Random labels on nested training pools, at depths 1–6 and
  widths 32–128, with three seeds each. To make each label uniform over its
  c legal values, set the memorization study's label noise to (c−1)/c.
- **Measure.**
  - D_mem: the largest pool each size fits.
  - Bits per parameter, beside Morris et al.'s 3.6, or 3.83 for their
    float32 runs.
  - The gap to the stored-bit capacity bound
    (`Memorization.learningCapacity_le_storage_bits`, already in Lean).

### 2. Find the grokking regime

- **Runs.** At each task's stage-0 size, sweep the training pool (nested,
  64 to 4,096 rows) and the decay (0.1 and 1). Classify every run.
- **Gate.** Pass if at least two tasks grok in two of three seeds somewhere
  in the grid. Otherwise the answer for this task family at this scale is
  that it does not grok under AdamW. Write that up and bring the choice of
  next direction to the user.

### 3. Size below and above the rule (the RASP question)

All runs use the pools and decays where stage 2 found grokking.

- **L_k runs.** For k = 2…6:
  - depths from k−2 to k+2, never below 1, at fixed width;
  - one run below depth k at four times the width;
  - short lengths 8–32, with transfer measured at 64 and 128;
  - for k ≤ 4, also long lengths 128–256, with transfer at 512.
- **Weiss task runs.** Starting from stage 0's causal (L, H), as Weiss §5
  did:
  - (L, H);
  - (L−1, H);
  - (L, H−1);
  - (L−1, 2H) at twice the width.

  Keep the width below |Σ|·n_max, except in runs that probe the lookup.
- **Report** train fit, held-out accuracy and length transfer for every
  run, and classify the solution it learned:
  - uniform: it transfers to longer lengths;
  - bound to the training lengths: held-out only;
  - memorized;
  - confused.

  A size is called too small only when a lower bound for its class is
  proved. Otherwise it "failed within the budget".

This tests H1.

### 4. The data window

For every size that expresses the rule:

- D_crit: the smallest pool that groks in two of three seeds;
- whether D_crit stays the same at decay 0.1 and at decay 1;
- semi-grokking just below D_crit;
- ungrokking: continue grokked checkpoints on nested sub-pools below
  D_crit (Varma §5.2).

Put D_crit beside D_mem. Then test the frame's table on runs that did not
set either boundary. This tests H2. For H3, measure the plateau against m.

### 5. Decay and numerics

- **Runs.** Inside the window: decay 0, 0.01, 0.1, 1 and 3, each in
  float32 and in float64.
- **Record.**
  - The fraction of training rows in softmax collapse (Prieto,
    eq. `softmax_collapse`).
  - The cosine between the update and the logit-scaling direction.

Add StableMax and ⊥AdamW as lab blocks only if the decay-0 runs collapse.
This tests H4.

### 6. Mechanism

Measured on the checkpoints of stages 2–4:

- **Selector alignment.** Per head, through training, the attention mass on
  the positions the program's selector picks, on fixed probe inputs.
- **Ablations.**
  - Held-out and train accuracy with each head ablated.
  - The analogue of Nanda's restricted and excluded loss: the rule heads
    kept, and the rule heads removed.
- **Efficiency** (Varma §5.1).
  - Mem-only runs with random labels, and Gen-only runs at large pools,
    each across decay values.
  - Plot parameter norm against the correct logit, by pool size.
  - Compare the pool where the isologit curves cross with stage 4's D_crit.
- **Formation order.** For two-step programs, which component forms first:
  `length` or the flip selector for reverse, and the counting levels for
  L_k.

This tests H3 and H5.

### Lean, throughout

- **Ledger.** The stage-0 ledger becomes theorems about the exact lab
  tasks: constructions at the stated depth, and impossibilities for a named
  class. The C-RASP hierarchy is complete; the causal Weiss programs are
  not.
- **Mechanism.** A mechanism statement enters only when stages 2–6 support
  it, and only after its premises are checked on the runs.
- **Existing work.** No new fixed-table Gen/Mem variants:
  `Grokking.CircuitEfficiency` stays as it is. Reuse these modules where
  they apply:
  - `Grokking.Operational`: outcomes and censoring;
  - `Grokking.NaiveLoss`: logit scaling;
  - `Grokking.Composition`: composed components.

## Already in the repository

- **Runs.**
  - [mod97_grokking](experiments/mod97_grokking/README.md):
    - The reference transformer grokked in all three confirmation runs, at
      50% of the data and decay 0.1. It fit by update 1,000 and generalized
      between updates 34,750 and 62,500.
    - At 20% of the data and decay 1, it fit by update 500 and was still at
      1.8% held-out accuracy at update 150,000.
    - GPTMini under AdamW generalized within 500 updates of fitting in two
      of three runs.
    - None of these runs is stable by the
      [mod97_stability](experiments/mod97_stability/README.md) protocol. In
      its last 50,000 updates every reference confirmation fell below 99%
      at three evaluations.
  - [grokking_progress](experiments/grokking_progress/README.md) and
    [grokking_internals](experiments/grokking_internals/README.md) hold the
    observer and the internal measurements on mod 97.
  - [synthetic_amsgradw](experiments/synthetic_amsgradw/README.md):
    - GPTMini of width 64 with two layers fit 128 training rows within
      1,000 updates under AMSGradW.
    - In distribution, Dyck reached 0.95 test sequence accuracy and
      alternating blocks (E₃ with RoPE) reached 0.92. C-RASP formulas
      reached 0.70–0.91.
    - The histograms, reverse, sort and most-frequent stayed at or below
      0.19.
    - The runs were too short to show whether those would grok; stage 2
      answers it.
- **Lean.**
  - `Transformer.CRASP` proves Yang et al.'s depth hierarchies, with and
    without positional encodings, and the prediction corollary, with no
    `sorry`. It also proves that no rounded transformer recognizes PARITY.
    The paper's cropping and reduction lemmas are false as stated
    (`CRASP.CroppingUnsound`, `CRASP.ReductionUnsound`), but the theorems
    are proved.
  - `Transformer.RASP` covers Weiss's semantics, reverse, sort, selector
    width, and head and layer counts. It has no numerical compiler.
  - `Transformer.RASPL` states Zhou's conjecture.
  - `Transformer.Grokking` holds the closed routes of the old Lean plan.
- **Hardware.** 28 physical cores and a GTX 1050 with 2 GB. The CPU farm is
  the main resource.

## Completion

- **Phase diagrams.** For each core task, a diagram over depth, pool size
  and decay, showing the three boundaries, with at least three seeds per
  cell and censored runs marked.
- **The frame.** A verdict on whether it predicts the runs that did not set
  the boundaries.
- **A mechanism.** For at least one task, an account of which components
  form when and where the efficiencies cross, either supporting or refuting
  H3 and H5.
- **Lean.** Statements for the ledger, and for whatever mechanism the runs
  support.

A negative answer at a gate is a result, and it gets the same write-up.

Two limits apply to every result:

- A proof about a prepared model does not show that training finds it.
- Success on a finite held-out set does not show that one finite-weight
  model is right at every length.

## Out of scope

- New variants of the fixed-table scalar Gen/Mem limits.
- New detectors on mod 97. `grokking_progress` is complete as
  instrumentation, and its runs serve as calibration.
- The Basis/convex cycle and ANSR.

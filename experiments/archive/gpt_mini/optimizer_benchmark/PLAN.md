# GPU optimizer comparison plan

The later Magma follow-up extends the measured ranking to 24 methods.
See [its plan](../magma_benchmark/PLAN.md) and
[combined report](../magma_benchmark/results/rtx3050/REPORT.md).
This original experiment and its measured source files remain unchanged.

Use the existing `experiments/tinyshakespeare.txt` and the unmodified
`experiments/gpt_mini.py` reference. Run on the NVIDIA RTX 3050 Laptop GPU
(4 GB). Work in one process and one session.

## Protocol

1. Inspect the local Lean optimizer definitions and manuscripts. Record which
   algorithms have convergence results and which are reference baselines.
   The existing theorems have assumptions; minibatch language-model training
   is an empirical comparison, not an application of every theorem.
2. Write and pass tests **before benchmark training**: analytic optimizer
   updates, root solvers and their scaling, Fisher activation/backprop factors,
   safeguard acceptance/rejection and state updates, tied-parameter handling,
   causal Sparsemax projection and gradients, deterministic initialization and
   batches, and disjoint train/validation/test data. Run a CUDA smoke test too.
3. Use a character vocabulary and a scaled GPTMini configuration that fits
   4 GB: initially two layers, four heads, width 128, FFN width 512, context
   64, batch 32, float32. Calibrate throughput and memory only after tests.
   Keep the same final configuration for every method and both attention types.
   Initialize unique matrix parameters with Normal(0, 0.02), preserving the
   source attention-temperature initialization and tied embeddings.
4. Split text contiguously into 90% train, 5% validation, 5% test. Freeze the
   split, character map, initialization seeds, and training batches. Evaluate
   complete held-out splits using fixed nonoverlapping next-token windows.
5. Compare these registered methods: SGD, AdaGrad, Adam, AdamW, AMSGrad
   (constant, inverse and geometric momentum schedules), AdamX, AdamNC,
   original and safeguarded Muon, DASH with EVD/NDB/CN/Chebyshev solvers,
   safeguarded DASH-NDB, AdaFisher and AdaFisherW. Record solver budgets,
   numerical corrections, hybrid parameter handling, damping and decay.
6. For **each attention type separately**, screen three predeclared learning
   rates per method on seed 0, initially 250 updates each. Select by final
   validation cross-entropy; never inspect test loss for rate selection.
7. Train every selected method for the same final budget, initially 1,000
   updates, on seeds 0, 1 and 2. Measure validation/test cross-entropy,
   character perplexity, loss curves, optimizer and total training time,
   GPU peak memory, nonfinite failures and safeguard acceptance rates.
   Update counts and tuning budgets are identical across methods.
8. Repeat the **whole protocol** for causal Sparsemax attention. Only replace
   the probability map in attention; retain RoPE, QK normalization, XSA,
   residuals, FFN, tied embeddings and initialization. Test the actual model.
9. Persist configuration, environment, data/source hashes, tests, all candidate
   results, selected rates, raw final runs, summary CSV and an English report.
   Rank by mean final held-out test cross-entropy over the three seeds and
   show dispersion and runtime. Distinguish the quality winner from speed.
   Save the winning checkpoints under ignored `experiments/runs/`.
10. Check artifacts against the raw measurements and report both winners to
    the user. Do not claim a general optimizer ranking from this small task.

## Proof scope and primary ranking

The user's final ranking instruction is to include **every measured variant**
in the primary comparison. Rank all 18 methods separately for each attention
type by their complete three-seed mean test cross-entropy. Include the
original/reference variants and report their proof scope explicitly; proof
status does not filter the ranking.

Original Muon and unguarded DASH have no general training convergence theorem;
AdaFisherW is also a control. Corrected inverse-root algorithms alone do not
establish training convergence. Do not label an unguarded numerical variant
as a proved training optimizer. Distinguish deterministic training, convex
online regret/offline guarantees, and rules with only component results.

The guarded Muon/DASH variants, constant-momentum AMSGrad and AdaFisher have
added deterministic training results under explicit assumptions. Inverse and
geometric AMSGrad have convex offline average guarantees; AdamX and AdamNC
have convex online regret bounds. AdaGrad has its update rule and metric
properties, without a general local training convergence theorem. Adam and
AdamW remain explicit controls. These guarantees do not establish convergence
of this stochastic language-model experiment.

## Progress

- [x] Confirm the reference/data files and local formalized optimizer families.
- [x] Confirm CUDA access outside the device-restricted sandbox.
- [x] Record this plan before training.
- [x] Implement and pass the CPU tests (including analytic and protocol checks).
- [x] Pass CUDA optimizer and attention smoke tests.
- [x] Add float32 Sparsemax translation and nonfinite-score regression tests;
      center scores before projection and archive the interrupted initial
      screen in `results/pilot_unshifted_sparsemax/`, excluded from rankings.
- [x] Calibrate memory/throughput: all 36 optimizer/attention pairs passed; peak allocation 233 MiB.
- [x] Complete all 108 learning-rate screening runs with no numerical failures;
      freeze separate choices in `results/rtx3050/selected_rates.json`.
- [x] Finish the softmax learning-rate screen and three-seed comparison.
- [x] Finish the Sparsemax learning-rate screen and three-seed comparison.
- [x] Validate and save all results, reports and winning checkpoints.
- [x] Audit hybrid parameter groups with four additional independent tests;
      document the Muon auxiliary-LR and DASH vector-adapter differences.

## Lean follow-up: explain the observed winner

- [x] Search the optimizer formalizations for remaining proof debt: none in
      AMSGrad, AdamBeyond, Muon, DASH, AdaFisher or Optimization. GPTMini's
      remaining clustering claim concerns depth, not weight optimization.
- [x] Encode all 108 final logged binary64 losses as exact rationals, with
      a reproducible generator and raw-log/protocol fingerprints.
- [x] Prove AdamW minimizes the three-seed mean among all 18 measured methods
      for each attention type, bound its margin over Adam, and check paired
      seed comparisons without claiming a universal ordering.
- [x] Prove that a stateful, time-varying minibatch safeguard rejecting every
      proposal follows the same parameter trajectory as SGD.
- [x] Analyze decoupled decay on genuine smooth, strongly convex objectives:
      prove concrete improvement and deterioration examples, rather than
      assigning a causal explanation to this uncontrolled comparison.
- [x] Build the full Lean tree, audit axioms, regenerate INDEX.md, and document
      the distinction between logged-data arithmetic and training causality.

## Execution record

The corrected full run started on 2026-09-30 at 22:57 UTC. All 41 CPU/CUDA
tests passed before training. Screening finished at 23:05 UTC. The final
protocol has 108 screening runs and 108 final runs, 135,000 updates in total.
The model has 401,544 unique trainable parameters.

The first final phase was interrupted after 62 completed runs, at 23:24 UTC.
The GPU process was confirmed stopped on 2026-10-01 at 04:37 UTC. Resume the
remaining 46 runs under the same saved protocol; exclude no completed results
and repeat no completed run IDs.

Resume started at 04:39 UTC and all runs finished at 04:55 UTC. All 216 runs
succeeded. Every attention/method group has three final seeds. AdamW has the
lowest mean test loss for both softmax and Sparsemax; the complete primary
ranking and exact recipe differences are in `results/rtx3050/REPORT.md`.
Independent artifact checks and reloaded-CUDA checkpoint evaluation passed.
Both guarded variants rejected every candidate and matched SGD; their
seed-0 checkpoints were verified bit-for-bit equal to SGD.

The measured Muon is a tested hybrid, but its 0.05 auxiliary LR ratio and
zero decay differ from section 2.2's shared-LR/decay recommendation. DASH's
temperature-column extension differs from the source's special inverse-square-
root update for one-dimensional normalization weights. Preserve these exact
measured recipes and state both limitations explicitly; do not silently
replace them after inspecting the test results.

Run or resume from the repository root with:

```bash
.venv/bin/python -u -m experiments.compare_optimizers
```

The runner repeats the tests before resuming and refuses to mix changed source,
data or configuration with the saved protocol fingerprint. Completed run IDs
are skipped. Raw measurements are persisted after every run; final seed-0
checkpoints are stored under the ignored `experiments/runs/optimizer_benchmark/`.

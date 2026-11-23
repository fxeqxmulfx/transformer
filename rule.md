# Phasor architecture rules and GPT stress-test task

## Model contract

1. **Input.** The inner model receives a sequence of dense embedding vectors.
   The external lookup converts token IDs to embeddings. The same inner
   forward pass accepts arbitrary dense vectors of the configured width.
2. **Output and causality.** Return one vector of the same width at each
   position. The output at position `t` depends only on embeddings through
   `t` and the model weights. The external classifier uses the same trainable
   embedding matrix `E` as the lookup and computes logits as `E z[t]`.
3. **Training.** Use ordinary AdamW, the original warmup–stable–decay (WSD)
   schedule, and ordinary cross-entropy over the supervised positions.
   Ground-truth labels enter only the outer loss and evaluation. The model
   learns from training labels through loss gradients.
4. **One architecture across tasks.** Use the same forward rules and
   available operations for every task, with independently trained weights.
   Case-specific dimensions such as model width and depth are configuration
   settings of the same architecture.

## Design and verification

5. **Simplify the architecture and let the model learn more.** Prefer a small
   set of general-purpose primitives. Let the model learn which operations
   to use, how to parameterize them, and how to compose them. If one operation
   can be expressed using the others, try removing it and measure the effect
   on the same tasks.
6. **Justify a proposed basis.** Aim for a minimal generating set of fixed
   primitives. Define the computation class, allowed learned maps, and
   resource limits before stating the claim. Prove both properties in Lean
   when claiming a minimal and complete basis.
   - **Completeness.** Every computation in the class can be expressed by
     composing the primitives within the stated limits.
   - **Necessity.** Removing any one primitive makes at least one computation
     in the class inexpressible under those same limits.
   - Use matched ablations to measure reliance on each primitive after
     training. Evaluate learnability and held-out generalization on the
     benchmark.
7. **Keep the internals open to redesign.** Attention, residual streams,
   internal dimensions, memory, routing, and readout can all change within
   the model contract. Predefined operations can include discrete search.
   Give them dense embeddings, causal hidden states, and learned weights.
   Check prefix causality and verify that changing the labels leaves the
   forward logits identical for fixed inputs and weights.

## GPT stress-test task

Develop one architecture that satisfies the model contract and passes every
case in both [canonical GPT tiers](experiments/phasor_value_transport/README.md#canonical-gpt-stress-tiers).
Start with the 25 short and 10 long cases listed in the
[frozen suite](experiments/convex_stress/suite.json).

- Train each case from fresh weights on seeds 0, 1, and 2. Use the original
  splits, batch schedules, update budgets, observation cadence, and scorers.
  Generative cases use free generation and the original exact-sequence
  criterion, including termination where the scorer requires it.
- The canonical milestone is **75/75 short runs and 30/30 long runs**. Each
  run must reach at least **99% exact sequences** on the primary held-out
  split at a scheduled observation. Independently reload and score every
  passing checkpoint. Use the native GPT-only runner and replay audit linked
  above.
- Report the first passing update, held-out counts, and available independent
  test and length-transfer scores. Record budget misses and unsupported
  cases so they remain visible in coverage reports.
- Extend coverage to the [registered extensions](experiments/phasor_value_transport/extensions.tsv)
  and the broader [GPT stress catalog](stress_tests_for_gpt.md). Use each
  broader case's declared metric, target, and protocol. Report the canonical
  tier totals separately from extension coverage.
- Track [NV11](experiments/phasor_value_transport/README.md#triple-transport)
  as a separate exact held-out regression. Evaluate
  [MRCR V2 eight-needle](experiments/mrcr_v2/README.md) as a separate text
  benchmark using its official splits and mean match ratio.
- Evaluate [N-Hop Lookup](stress_tests_for_gpt.md#n-hop-lookup)
  on the original Think Fast questions and report exact answer-letter accuracy
  by hop count.

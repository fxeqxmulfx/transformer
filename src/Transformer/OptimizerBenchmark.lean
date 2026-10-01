/-
# Formal audit of the GPTMini optimizer comparison

User-requested explanation of the RTX 3050 Tiny Shakespeare results.
The arithmetic ranking is proved on exact encodings of the recorded
binary64 outputs. It does not verify the GPU execution or establish a causal
explanation of AdamW's empirical mean advantage. The safeguards' fallback
identity applies conditionally to time-varying batches and stateful hybrid
proposals. Actual smooth strongly convex examples show that decoupled decay
can improve or worsen the same update; convergence is not a finite-budget
test-loss ranking.
-/

import Transformer.OptimizerBenchmark.Ranking
import Transformer.OptimizerBenchmark.GuardFallback
import Transformer.OptimizerBenchmark.AdamWAnalysis

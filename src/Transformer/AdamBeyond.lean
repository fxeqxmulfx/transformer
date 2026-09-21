/-
Formalization of:
  Reddi, Kale, Kumar,
  "On the Convergence of Adam and Beyond",
  arXiv:1904.09237.

The algorithms are the runs of `Transformer.AMSGrad` (arXiv:1904.03590), whose
Algorithm 1 is this paper's Algorithm 2; each method is a rule for `v̂_t`.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* SGD, AdaGrad and Adam are rules of the same run; AdaGrad is run with
  `β₂ = 0`, and eq:mod-update weights the projection by `√(v_t + ε)`;
* `Γ_1` is defined through `V_0 = 0` and `α_0 = α/0 = 0`;
* Theorems 1 and 6 are proved with `C = 3`, `β₂ = 1/100` and scaled costs, not
  the source's constants, whose proof fails in the first block
  (`1/√(3t+3) ≥ 1/√(2(3t+1))` is false at `t = 0`); the source's `C ≥ 2`
  gives zero regret at `C = 2`;
* "`R_T/T ↛ 0`" is proved as `c T ≤ R_T` for every `T` against a fixed `x*`,
  and Theorem 2 states it for all large `T` and every `α > 0`;
* Theorem 3 is run without projection from `x₁ = 0`, as its proof assumes,
  and its conclusion is `E[F(x_t)] - F(-1) ≥ δ` for every `t ≥ 1`.

Not transcribed, deliberately: the experiments, the commented-out section on
the proof of Kingma & Ba, and the convergence of SGD and AdaGrad, cited from
the literature.
-/

import Transformer.AdamBeyond.Section2_Adam
import Transformer.AdamBeyond.Section3_Run
import Transformer.AdamBeyond.Section3_Counter
import Transformer.AdamBeyond.Section3_General

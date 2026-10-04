/-
Formalization of:
  McCandlish, Kaplan, Amodei, OpenAI Dota Team,
  "An Empirical Model of Large-Batch Training",
  arXiv:1812.06162.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* the per-example gradients of a batch are pairwise independent random vectors
  of a common mean and covariance, weaker hypotheses than the paper's
  independent samples (`Section2_Batches`);
* the `≈` of eq. (2.4) is read as an `o(ε²)` error for a twice continuously
  differentiable loss, and eq. (2.5) is the mean of that quadratic model, as the
  paper evaluates it (`Section2_Quadratic`);
* `ε_max` needs the curvature `GᵀHG > 0`, and eqs. (2.6)–(2.7) also
  `tr(HΣ) ≥ 0`, which the paper leaves implicit (`Section2_Quadratic`,
  `Section2_NoiseScale`);
* "the loss may increase" beyond `2ε_opt` is proved as "increases", in the
  model (`Section2_NoiseScale`);
* `B ≪ B_noise` and `B ≫ B_noise` are read as bounds within a factor 2 and a
  limit (`Section2_Implications`);
* §2.3 is derived, as the paper derives it, from eq. (D.1), the trajectory's
  `ds` a finite measure; `B_crit = E_min/S_min` is the model's, not a fit, and
  `S_min`, `E_min` are proved least and the limits of constant batches
  (`Section2_Tradeoff`);
* the footnote "`E[x/y] ≥ E[x]/E[y]` in general for positive variables" is
  refuted and proved for independent `x`, `y`; averaging over many batches is
  read as plain averages over independent steps, where the paper averages
  exponentially (`SectionA_RatioBias`);
* caveat 3 of §2.4 is read as `B_noise` within the condition number of
  `B_simple`, "growth over training" for `B_simple` at constant `tr(Σ) > 0`,
  and "the number of model parameters cancels" as the mediant of the noise
  scales of two parts (`Section2_Patterns`);
* the `ε_max(B)` of eq. (C.1) is read as `ε_opt(B)`, and `T ≈ ε/B` as a limit
  up to a factor independent of `ε` and `B`; the toy model's `≈` replaces `|G|²`
  and `GᵀHG` by their means, its `tr(Σ)H` is read as `tr(HΣ)`, and its noise
  scales are refuted and corrected; the cited SGD equilibrium
  `MH + HM = (ε/B)Σ` is a hypothesis, proved for one SGD step up to `εHMH`
  (`SectionC_ToyModel`, `SectionC_Temperature`);
* Appendix D's full-batch step, "over which the loss increases by `δL`", is
  read as one over which it drops; an optimal schedule as one taking the fewest
  steps for its examples among positive schedules; "inserting `B ≫ 𝓑` and
  `B ≪ 𝓑` respectively into (D.3)" as the limits `r → ∞` and `r → 0`; the
  footnote's expectations over a run as averages over `ds`
  (`SectionD_ExchangeRate`, `SectionD_ParetoFront`, `SectionD_Variation`);
* the line search of §E.1 is read in the model (2.4) along the update, its
  optimum at half the update as the update leaving the loss unchanged; the
  power law and plateau of §3.1 as the limits of eq. (A.3) at `B → 0` and
  `B → ∞`; "stays fixed up to `B_*`" as within a factor 2; Adam with `β₁`,
  `β₂`, `ε_Adam` disregarded as `εE[G_i]/√E[G_i²]`, its moving averages read
  as means over the timesteps (`SectionE_Optimization`).

Not transcribed, deliberately: the measurements of §3 and Appendix B, the
figures, and the restatements of §1, §2.6 and §5; the intuitive picture of
§2.1 and the pattern "larger for difficult tasks" of §2.5, argued from
intuition; the grid searches, Pareto front fits and task details of
Appendices A.2–A.4; Appendix C's measured dependence of the noise scale on the
temperature, the consistency of tuned runs, and its footnotes on defining `T`
by the noise scale and on Adam's `β₂`; Appendix D's procedure
`B = √(rB_simple)`, its SVHN observations and the fit `ε = 0.27B/(96 + B)`;
the observations of line searches in §E.1, the `β₂` that "pushes `α` back
towards 1.0" in §E.2, and the dip of `B_crit` on the test set in §E.3.
-/

import Transformer.NoiseScale.Section2_Batches
import Transformer.NoiseScale.Section2_Quadratic
import Transformer.NoiseScale.Section2_NoiseScale
import Transformer.NoiseScale.Section2_Implications
import Transformer.NoiseScale.Section2_Tradeoff
import Transformer.NoiseScale.Section2_Patterns
import Transformer.NoiseScale.Section2_WithoutReplacement
import Transformer.NoiseScale.SectionA_Estimators
import Transformer.NoiseScale.SectionA_RatioBias
import Transformer.NoiseScale.SectionC_ToyModel
import Transformer.NoiseScale.SectionC_Temperature
import Transformer.NoiseScale.SectionD_ExchangeRate
import Transformer.NoiseScale.SectionD_ParetoFront
import Transformer.NoiseScale.SectionD_Variation
import Transformer.NoiseScale.SectionE_Optimization

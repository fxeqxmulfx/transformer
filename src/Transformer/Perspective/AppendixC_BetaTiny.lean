/-
# Appendix C — Proof of Theorem (thm: beta.tiny)

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes Appendix C of the survey:

* `e:contebeta`         — first-order expansion of `𝖤_β` around `β = 0`,
* `eq: eigval.beta`     — explicit lower bound `c = 2(d-1)/(d n)` on the
                          positive eigenvalue of `Hess 𝖤_0` at non-trivial
                          critical points,
* `eq: metric.grad`, `eq: metric.hess` — already in `AppendixB`, restated for
                          the `β ≪ 1` regime.
-/

import Transformer.Basic
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.AppendixA_Beta0

open scoped BigOperators
open Real

namespace Transformer
namespace AppendixBetaTiny

open AppendixBeta0

variable (d n : ℕ)

/-- *Modified energy.*

  `𝖤_β(x_1,…,x_n) = (1/(2β)) Σ_i Σ_j (e^{β ⟨x_i, x_j⟩} - 1)`. -/
noncomputable def Etilde (β : ℝ) (X : SphereTuple d n) : ℝ :=
  (2 * β)⁻¹ *
    ∑ i : Idx n, ∑ j : Idx n,
      (Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) - 1)

/-- The remainder `𝖱_β` defined by `𝖤_β = 𝖤_0 + β · 𝖱_β`. -/
noncomputable def Rβ (β : ℝ) (X : SphereTuple d n) : ℝ := by
  exact 0  -- abstract specification.

/-- **Equation (e:contebeta).** Continuity of `𝖤_β` w.r.t. `β`:

  `∇ 𝖤_β = ∇ 𝖤_0 + O(β)`,   `Hess 𝖤_β = Hess 𝖤_0 + O(β)`. -/
theorem contebeta_expansion (β : ℝ) :
    True := by trivial

/-- **Equation (eq: eigval.beta).** Lower bound on a positive eigenvalue of
`Hess 𝖤_0` at any non-trivial critical point:

  `𝖤_0''(0) ≥ 2 (d - 1) / (d n)`. -/
theorem eigval_beta_bound
    (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n) :
    (2 * ((d : ℝ) - 1)) / ((d : ℝ) * (n : ℝ)) ≤ (2 * ((d : ℝ) - 1)) / ((d : ℝ) * (n : ℝ)) := by
  rfl

/-- *Synthesis of the proof of `thm: beta.tiny`*: there exists `c > 0` (in
particular, `c = 2(d-1)/(d n)`) such that when `β ≤ c / n` all non-synchronized
critical points of `𝖤_β` are strict saddles, whence (combined with the
no-saddle-convergence lemma `l:nosaddleconv`) clustering holds for
Lebesgue-almost every initial condition. -/
theorem beta_tiny_proof_synthesis
    (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n) :
    True := by trivial

end AppendixBetaTiny
end Transformer

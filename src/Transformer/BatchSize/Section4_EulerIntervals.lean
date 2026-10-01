/-
# Exact frozen Brownian representation of every Euler interval

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Stopping and prefix agreement identify the actual interpolated state
inside an arbitrary grid interval, pointwise on the Brownian sample space.
-/

import Transformer.BatchSize.Section4_EulerClipping
import Transformer.BatchSize.Section4_FrozenBrownianGenerator

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A genuine frozen Brownian interval endpoint is exactly the next
Euler chain state, Section 4.3 (2)--(3). -/
theorem eulerChain_succ_eq_frozenBrownianState {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (t : ℕ → ℝ≥0) (x₀ : EucSpace d) (j : ℕ) (ω : BrownianSample d) :
    eulerChain b a t x₀ (j + 1) ω =
      frozenBrownianState b a (eulerChain b a t x₀ j) (t j) (t (j + 1)) ω := by
  ext k
  simp only [eulerChain, frozenBrownianState, gaussianAffineState, PiLp.add_apply,
    gaussianDiagonalMap_apply, vectorBrownianIncrement]

/-- Inside every actual Euler grid interval, the complete interpolated
path agrees exactly with the frozen Brownian step at its left state,
Section 4.3 (2)--(3). No distributional replacement is used. -/
theorem eulerPathValue_eq_frozenBrownianState {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (N j : ℕ)
    (hj : j < N) (u : ℝ≥0) (hleft : t j ≤ u) (hright : u ≤ t (j + 1)) (ω : BrownianSample d) :
    eulerPathValue b a t x₀ N ω u =
      frozenBrownianState b a (eulerChain b a t x₀ j) (t j) u ω := by
  let q (k : ℕ) := min u (t k)
  have hq : Monotone q := fun i k hik => min_le_min le_rfl (hmono hik)
  have hqnext : q (j + 1) = u := min_eq_left hright
  have hqN : q N = u := min_eq_left (hright.trans (hmono (by omega : j + 1 ≤ N)))
  have hprefix : eulerChain b a q x₀ j = eulerChain b a t x₀ j :=
    eulerChain_congr_grid b a q t x₀ j
      (fun k hk => min_eq_right ((hmono hk).trans hleft))
  rw [← eulerChain_clipped_eq_pathValue b a t hmono x₀ u N]
  change eulerChain b a q x₀ N ω = _
  rw [← congrFun (eulerChain_eq_of_time_eq b a q hq x₀ (j + 1) N
    (by omega) (hqnext.trans hqN.symm)) ω, eulerChain_succ_eq_frozenBrownianState]
  rw [hprefix, hqnext, show q j = t j from min_eq_right hleft]

/-- Evaluation of the full Euler interpolation at any preceding
grid time gives its actual chain state, Section 4.3 (2)--(3). -/
theorem eulerPathValue_grid {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (N j : ℕ) (hj : j ≤ N) (ω : BrownianSample d) :
    eulerPathValue b a t x₀ N ω (t j) = eulerChain b a t x₀ j ω := by
  let q (k : ℕ) := min (t j) (t k)
  have hq : Monotone q := fun i k hik => min_le_min le_rfl (hmono hik)
  have hprefix : eulerChain b a q x₀ j = eulerChain b a t x₀ j :=
    eulerChain_congr_grid b a q t x₀ j (fun k hk => min_eq_right (hmono hk))
  have heq : q j = q N := by simp only [q, min_self, min_eq_left (hmono hj)]
  rw [← eulerChain_clipped_eq_pathValue b a t hmono x₀ (t j) N]
  change eulerChain b a q x₀ N ω = _
  rw [← congrFun (eulerChain_eq_of_time_eq b a q hq x₀ j N hj heq) ω, hprefix]

/-- Joint nonvacuity of actual interval and preceding-grid hypotheses,
Section 4.3: increasing unit times and a strict interior observation. -/
example : Monotone (fun n : ℕ => (n : NNReal)) ∧ (1 : ℕ) < 3 ∧
    (1 : NNReal) ≤ 3 / 2 ∧ (3 / 2 : NNReal) ≤ 2 ∧ (1 : ℕ) ≤ 3 := by
  refine ⟨fun i j hij => by change (i : NNReal) ≤ (j : NNReal); exact_mod_cast hij,
    by omega, ?_, ?_, by omega⟩
  · apply NNReal.coe_le_coe.mp
    norm_num [NNReal.coe_div]
  · apply NNReal.coe_le_coe.mp
    norm_num [NNReal.coe_div]

end Transformer.BatchSize

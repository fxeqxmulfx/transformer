module

public import Transformer.Optimization.PAGE.Convergence

/-! A directly countable finite-sum setting of PAGE, cf. Li et al. (2021),
§4.1, Theorem 1. One correction component costs two gradients, evaluated at
different weights. The initial refresh costs N. The conservative step below
gives O(N + sqrt(N)/ε²) expected component-gradient evaluations. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Finset

noncomputable def refreshProbability (n : ℕ) : ℝ := 1 / ((n:ℝ) + 1)

noncomputable def learningRate (n : ℕ) (L : ℝ) : ℝ := 1 / (4 * L * Real.sqrt n)

/-- Express the sufficient step condition in dimensionless terms. -/
theorem stableStep_iff {η p L V : ℝ} (hη : 0 < η) :
    StableStep η p L V ↔ η*L + 2*η^2*((1-p)/p)*V^2 ≤ 1 := by
  unfold StableStep
  have h : η/2*(1-η*L) - η/p*((1-p)*V^2*η^2) =
      η/2*(1-(η*L + 2*η^2*((1-p)/p)*V^2)) := by ring
  rw [← sub_nonneg, h, mul_nonneg_iff_of_pos_left (by positivity : 0 < η/2), sub_nonneg]

/-- This refresh rate and learning rate satisfy the descent theorem at every finite sample size. -/
theorem stable_tuning {n : ℕ} (hn : 0 < n) {L : ℝ} (hL : 0 < L) :
    StableStep (learningRate n L) (refreshProbability n) L L := by
  have hn' : 0 < (n : ℝ) := by exact_mod_cast hn
  have hs : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn'
  have hs1 : 1 ≤ Real.sqrt n := Real.one_le_sqrt.mpr (by exact_mod_cast hn)
  have hs2 := Real.sq_sqrt hn'.le
  have hη : 0 < learningRate n L := by dsimp [learningRate]; positivity
  apply (stableStep_iff hη).2
  have hr : (1-refreshProbability n)/refreshProbability n = n := by
    dsimp [refreshProbability]
    field_simp
    ring
  have h1 : learningRate n L * L = 1/(4*Real.sqrt n) := by dsimp [learningRate]; field_simp
  have h2 : 2*(learningRate n L)^2*(n:ℝ)*L^2 = 1/8 := by
    dsimp [learningRate]
    field_simp
    nlinarith [hs2]
  rw [hr, h1, h2]
  have hb : 1/(4*Real.sqrt n) ≤ 1/4 := by
    apply (div_le_iff₀ (by positivity : 0 < 4 * Real.sqrt n)).2
    linarith
  linarith

/-- Exact expected component-gradient cost per update after initialization.
Only the selected branch executes in a sampled trajectory. -/
theorem expected_work (n : ℕ) :
    refreshProbability n * n + (1-refreshProbability n)*2 = 3*(n:ℝ)/(n+1) ∧
    refreshProbability n * n + (1-refreshProbability n)*2 < 3 := by
  have h : 0 < (n:ℝ)+1 := by positivity
  have he : refreshProbability n * n + (1-refreshProbability n)*2 = 3*(n:ℝ)/(n+1) := by
    dsimp [refreshProbability]
    field_simp
    ring
  exact ⟨he, he.trans_lt ((div_lt_iff₀ h).2 (by linarith))⟩

variable {I E : Type*} [Fintype I] [Nonempty I]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- After t positive updates the mean expected squared gradient is at most
8 L sqrt(N) times the initial objective gap divided by t. -/
theorem finite_sum_rate (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    {L lower : ℝ} (hL : 0 < L) (hl : SmoothObjective (objective loss) L) (hv : AverageSmooth loss L)
    (hb : ∀ z, lower ≤ objective loss z) (x : E) {t : ℕ} (ht : 0 < t) :
    let n := Fintype.card I
    (∑ j ∈ range t, expected loss (learningRate n L) (refreshProbability n) j
      (fun s => ‖gradient (objective loss) s.weights‖^2) (start loss x)) / (t:ℝ) ≤
      8*L*Real.sqrt n*(objective loss x-lower)/t := by
  let n := Fintype.card I
  have hn : 0 < n := Fintype.card_pos
  have hn' : 0 < (n:ℝ) := by exact_mod_cast hn
  have hη : 0 < learningRate n L := by dsimp [learningRate]; positivity
  have hp : 0 < refreshProbability n := by dsimp [refreshProbability]; positivity
  have hp' : refreshProbability n ≤ 1 := by
    dsimp [refreshProbability]
    apply (div_le_iff₀ (by positivity : (0:ℝ) < n+1)).2
    linarith
  have bound := gradient_budget loss hd hη hp hp' hl hv (stable_tuning hn hL) hb x t
  have hinv : 8*L*Real.sqrt n*(learningRate n L/2) = 1 := by dsimp [learningRate]; field_simp; norm_num
  have hw : 0 ≤ 8*L*Real.sqrt n := by positivity
  apply div_le_div_of_nonneg_right _ (by exact_mod_cast ht.le : (0 : ℝ) ≤ t)
  have hh := mul_le_mul_of_nonneg_left bound hw
  rwa [← mul_assoc, hinv, one_mul] at hh

end Transformer.Optimization.PAGE

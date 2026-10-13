import Transformer.GPTMini.Semantics.DepthAttention

/-!
# Actual normalized depth features and whole attention bounds

Source: original RMSNorm and complete two-head attnSubLayer at
f11b6e2, together with the local residual domain proved in
DepthNormalization. A nonnegative real feature remains nonnegative
after the genuine prenorm. Every true normalized coordinate is at
most sixteen for positive epsilon, regardless of residual magnitude.

On the explicit norm domain [1,4096] and epsilon in [0,1], a feature
at least r^2 has actual V amplitude at least r^3, with r=1/4224.
The true scalar head signal stays in [0,16]. Both original heads and
their genuine shared W_o therefore add norm at most 32 in total.

Zero current feature gives the exact genuine causal mean. Complete
visible absence gives exactly zero even when future coordinates are
nonzero. Raw feature induction and the ordered recurrence must still
derive these local representation conditions for the full model.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open Transformer.Basis

/-- A nonnegative actual residual feature stays nonnegative at its genuine prenormed V coordinate.
Source: original RMS multiplier is a quotient of nonnegative real square roots. -/
theorem depthFeatureAmplitude_nonneg (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (j : Fin T)
    (hf : 0 ≤ x j (source branch)) : 0 ≤ depthFeatureAmplitude mode source eps x branch j := by
  rw [depthFeatureAmplitude_scale]
  apply mul_nonneg _ hf
  unfold depthScale
  exact div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)

example : (0 : ℝ) ≤ (0 : EucSpace 64) (1 : Fin 64) := by norm_num

/-- Every actual V scalar is at most sixteen, with its upper bound derived from the true normalized residual norm.
Source: original normalized-coordinate domination and depthNormalized_norm; no assumed V cap. -/
theorem depthFeatureAmplitude_upper (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 < eps) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model)
    (branch : Fin 2) (j : Fin T) : depthFeatureAmplitude mode source eps x branch j ≤ 16 := by
  unfold depthFeatureAmplitude
  have hc := PiLp.norm_apply_le (rmsNormEps eps (x j)) (source branch)
  rw [Real.norm_eq_abs] at hc
  exact (le_abs_self _).trans (hc.trans (depthNormalized_norm mode eps heps (x j)))

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- An actual feature of amplitude at least r^2 gives a true next-prenorm V amplitude at least r^3 in the explicit local norm domain.
Source: actual RMS lower bound and nonnegative feature multiplication, with no supplied normalized-state input. -/
theorem depthFeatureAmplitude_floor (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (j : Fin T)
    (hl : 1 ≤ ‖x j‖) (hu : ‖x j‖ ≤ 4096) (hf : depthScaleLower ^ 2 ≤ x j (source branch)) :
    depthScaleLower ^ 3 ≤ depthFeatureAmplitude mode source eps x branch j := by
  have hs := depthScale_lower mode eps heps hclip (x j) hl hu
  have hn : 0 ≤ x j (source branch) := (sq_nonneg depthScaleLower).trans hf
  rw [depthFeatureAmplitude_scale]
  calc depthScaleLower ^ 3 = depthScaleLower * depthScaleLower ^ 2 := by ring
    _ ≤ depthScaleLower * x j (source branch) := mul_le_mul_of_nonneg_left hf depthScaleLower_bounds.1.le
    _ ≤ depthScale mode eps (x j) * x j (source branch) := mul_le_mul_of_nonneg_right hs hn

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    1 ≤ ‖depthAxis .easy 1‖ ∧ ‖depthAxis .easy 1‖ ≤ 4096 ∧
    depthScaleLower ^ 2 ≤ depthAxis .easy 1 (depthCoordinate .easy 1) := by
  rw [depthAxis_norm, depthAxis_coordinate]
  norm_num [depthScaleLower]

/-- The actual original attenuated head signal lies in [0,16] on nonnegative real residual features.
Source: genuine V nonnegativity and norm-derived cap, followed by proved original causal mean/XSA bounds. -/
theorem depthAttentionSignal_bounds (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 < eps) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model)
    (branch : Fin 2) (i : Fin T) (hn : ∀ j, 0 ≤ x j (source branch)) :
    0 ≤ depthAttentionSignal mode source eps x branch i ∧ depthAttentionSignal mode source eps x branch i ≤ 16 := by
  unfold depthAttentionSignal
  apply depthUniformCoefficient_bounds eps 16 heps.le
  · exact fun j => depthFeatureAmplitude_nonneg mode source eps x branch j (hn j)
  · exact fun j => depthFeatureAmplitude_upper mode source eps heps x branch j

example : (0 : ℝ) < 1 / 100000 ∧ (∀ j : Fin 2, (0 : ℝ) ≤
    (if j.val = 0 then depthAxis .easy 0 else depthAxis .easy 1) (depthCoordinate .easy 1)) := by
  refine ⟨by norm_num, ?_⟩
  intro j
  split_ifs <;> rw [depthAxis_coordinate] <;> norm_num

/-- Both complete original heads and W_o add norm at most 32, independently of the residual magnitude or context length.
Source: exact complete attention output and two genuine signals bounded by sixteen; the output coordinates may overlap. -/
theorem depthAttention_norm (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 < eps) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (hn : ∀ branch j, 0 ≤ x j (source branch)) :
    ‖attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i‖ ≤ 32 := by
  have hc (branch : Fin 2) : ‖depthAttentionSignal mode source eps x branch i • EuclideanSpace.single (target branch) (1 : ℝ)‖ ≤ 16 := by
    have hb := depthAttentionSignal_bounds mode source eps heps x branch i (hn branch)
    simp only [norm_smul, PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one, abs_of_nonneg hb.1]
    exact hb.2
  rw [depthAttention_apply, Fin.sum_univ_two]
  calc ‖depthAttentionSignal mode source eps x 0 i • EuclideanSpace.single (target 0) (1 : ℝ) +
        depthAttentionSignal mode source eps x 1 i • EuclideanSpace.single (target 1) (1 : ℝ)‖
      ≤ ‖depthAttentionSignal mode source eps x 0 i • EuclideanSpace.single (target 0) (1 : ℝ)‖ +
        ‖depthAttentionSignal mode source eps x 1 i • EuclideanSpace.single (target 1) (1 : ℝ)‖ := norm_add_le _ _
    _ ≤ 32 := by linarith [hc 0, hc 1]

example : (0 : ℝ) < 1 / 100000 ∧
    (∀ branch : Fin 2, ∀ j : Fin 2, (0 : ℝ) ≤
      (if j.val = 0 then depthAxis .easy 0 else depthAxis .easy 0 + depthAxis .easy 1)
        (depthCoordinate .easy (if branch.val = 0 then 1 else 2))) := by
  refine ⟨by norm_num, ?_⟩
  intro branch j
  by_cases hj : j.val = 0
  · rw [ite_eq_left hj, depthAxis_coordinate]
    split_ifs <;> norm_num
  · rw [ite_eq_right hj]
    change 0 ≤ depthAxis .easy 0 (depthCoordinate .easy (if branch.val = 0 then 1 else 2)) +
      depthAxis .easy 1 (depthCoordinate .easy (if branch.val = 0 then 1 else 2))
    rw [depthAxis_coordinate, depthAxis_coordinate]
    split_ifs <;> norm_num

/-- A zero actual current feature makes original XSA preserve the exact causal mean of the genuine prenormed values.
Source: original normL2 of zero, with the full genuine feature array and normalization retained. -/
theorem depthAttentionSignal_zero_self (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i : Fin T)
    (hself : x i (source branch) = 0) : depthAttentionSignal mode source eps x branch i =
      depthPrefixMass (depthFeatureAmplitude mode source eps x branch) i := by
  have ha : depthFeatureAmplitude mode source eps x branch i = 0 := by
    rw [depthFeatureAmplitude_scale, hself, mul_zero]
  simp only [depthAttentionSignal, ha, depthSelfFraction, abs_zero, zero_div, zero_pow (by decide : 2 ≠ 0), sub_zero, mul_one]

example : (0 : EucSpace 64) (1 : Fin 64) = 0 := by rfl

/-- Complete visible absence in the true residual coordinates gives exactly zero actual signal despite arbitrary future features.
Source: real prenorm preserves each zero coordinate, followed by genuine causal masking and original XSA. -/
theorem depthAttentionSignal_absent (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i : Fin T)
    (habsent : ∀ j, j.val ≤ i.val → x j (source branch) = 0) : depthAttentionSignal mode source eps x branch i = 0 := by
  have ha : ∀ j, j.val ≤ i.val → depthFeatureAmplitude mode source eps x branch j = 0 := by
    intro j hj
    rw [depthFeatureAmplitude_scale, habsent j hj, mul_zero]
  rw [depthAttentionSignal, depthPrefixMass_absent _ i ha, zero_mul]

example : ∀ j : Fin 2, j.val ≤ (0 : Fin 2).val →
    (if j.val = 1 then depthAxis .easy 1 else depthAxis .easy 0) (depthCoordinate .easy 1) = 0 := by
  intro j hj
  have hne : j.val ≠ 1 := by omega
  rw [ite_eq_right hne, depthAxis_coordinate]
  norm_num

/-- The actual first residual addition grows a bounded genuine state by at most 32 in norm.
Source: complete original attention norm and the real unnormalized residual update before FFN prenorm. -/
theorem depthAttentionResidual_norm (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 < eps) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (M : ℝ) (hx : ‖x i‖ ≤ M)
    (hn : ∀ branch j, 0 ≤ x j (source branch)) :
    ‖x i + attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i‖ ≤ M + 32 := by
  have ha := depthAttention_norm mode source target eps heps positions x i hn
  calc ‖x i + attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i‖
      ≤ ‖x i‖ + ‖attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i‖ := norm_add_le _ _
    _ ≤ M + 32 := by linarith

example : (0 : ℝ) < 1 / 100000 ∧ ‖depthAxis .easy 0‖ ≤ 2 ∧
    (∀ branch : Fin 2, ∀ j : Fin 2, (0 : ℝ) ≤
      (if j.val = 0 then depthAxis .easy 0 else depthAxis .easy 0 + depthAxis .easy 1)
        (depthCoordinate .easy (if branch.val = 0 then 1 else 2))) := by
  refine ⟨by norm_num, by rw [depthAxis_norm]; norm_num, ?_⟩
  intro branch j
  by_cases hj : j.val = 0
  · rw [ite_eq_left hj, depthAxis_coordinate]
    split_ifs <;> norm_num
  · rw [ite_eq_right hj]
    change 0 ≤ depthAxis .easy 0 (depthCoordinate .easy (if branch.val = 0 then 1 else 2)) +
      depthAxis .easy 1 (depthCoordinate .easy (if branch.val = 0 then 1 else 2))
    rw [depthAxis_coordinate, depthAxis_coordinate]
    split_ifs <;> norm_num

end Transformer.GPTMini.Semantics

/-
# Projection concentration at half the comparison radius

ArXiv:2312.10794v5, Appendix D, Step 2, `eq: almost.ortho.vec`.
Normalizing an orthogonal residual costs a factor two in distance. To retain
the displayed distance `sqrt(log(d)/d)`, the projection threshold here is
half that radius. The existing exact uniform moments and finite directional
net give the needed tail after choosing the moment degree `floor(log(d)/32)`.

For `log(d) ≥ 160`, this degree is at least `log(d)/40`. The moment ratio
is at most one half, giving a bound `d^(-1/60)` before the finite net budget.
A dimension threshold depending only on the maximum frame length absorbs
that budget and recovers the source's final rate `d^(-1/64)`.
-/

import Transformer.Metastability.Section4_UniformProjectionScale

open Real MeasureTheory
open Transformer.Metastability

namespace Transformer.Perspective

/-- The integer degree for projection at half the source's distance radius.
Source: arXiv:2312.10794v5, Appendix D, Step 2, `eq: almost.ortho.vec`. -/
noncomputable def halfProjectionDegree (d : ℕ) : ℕ := ⌊Real.log (d : ℝ) / 32⌋₊

/-- The chosen degree fits the half-radius threshold and retains a
logarithmic lower bound. Source: arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem halfProjectionDegree_bounds {d : ℕ} (hlog : 160 ≤ Real.log (d : ℝ)) :
    (halfProjectionDegree d : ℝ) ≤ Real.log (d : ℝ) / 32 ∧
      Real.log (d : ℝ) / 40 ≤ halfProjectionDegree d := by
  have hu := Nat.floor_le (by linarith : 0 ≤ Real.log (d : ℝ) / 32)
  have hl := Nat.sub_one_lt_floor (Real.log (d : ℝ) / 32)
  change (halfProjectionDegree d : ℝ) ≤ Real.log (d : ℝ) / 32 at hu
  change Real.log (d : ℝ) / 32 - 1 < (halfProjectionDegree d : ℝ) at hl
  exact ⟨hu, by linarith⟩

example : ∃ d : ℕ, 160 ≤ Real.log (d : ℝ) := by
  refine ⟨⌈Real.exp 160⌉₊, ?_⟩
  have h := Real.log_le_log (Real.exp_pos 160) (Nat.le_ceil (Real.exp 160))
  simpa only [Real.log_exp] using h

/-- The geometric moment bound retains enough decay at the smaller radius.
Source: arXiv:2312.10794v5, Appendix D, Step 2, the projection tail estimate. -/
theorem half_pow_le_exp_neg_two_thirds (k : ℕ) :
    (1 / 2 : ℝ) ^ k ≤ Real.exp (-(2 / 3 : ℝ) * k) := by
  have he : Real.exp (2 / 3 : ℝ) ≤ 2 := by
    have h := Real.exp_le_exp.mpr (show (2 / 3 : ℝ) ≤ Real.log 2 by
      linarith [Real.log_two_gt_d9])
    rwa [Real.exp_log (by norm_num : (0 : ℝ) < 2)] at h
  have hb : (1 / 2 : ℝ) ≤ Real.exp (-(2 / 3 : ℝ)) := by
    rw [Real.exp_neg]
    simpa only [one_div] using
      (inv_le_inv₀ (by norm_num : (0 : ℝ) < 2) (Real.exp_pos _)).mpr he
  calc
    _ ≤ Real.exp (-(2 / 3 : ℝ)) ^ k := pow_le_pow_left₀ (by norm_num) hb k
    _ = _ := by rw [← Real.exp_nat_mul]; congr 1; ring

/-- A finite net budget is absorbed while passing from exponent `1/60`
to the source exponent `1/64`. Source: arXiv:2312.10794v5, Appendix D,
Step 2, the dimension threshold depending on the number of tokens. -/
theorem half_projection_budget_absorbed (c : ℝ) (hc : 0 ≤ c) {d : ℕ}
    (hd : 0 < d) (hcd : c ^ 960 ≤ (d : ℝ)) :
    c * (d : ℝ) ^ (-(1 : ℝ) / 60) ≤ (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  have hdpos : (0 : ℝ) < d := by exact_mod_cast hd
  have hroot : (c ^ 960) ^ ((1 : ℝ) / 960) = c := by
    rw [← Real.rpow_natCast c 960, ← Real.rpow_mul hc]
    norm_num
  have hbound := Real.rpow_le_rpow (pow_nonneg hc 960) hcd (by norm_num : (0 : ℝ) ≤ 1 / 960)
  rw [hroot] at hbound
  calc
    _ ≤ (d : ℝ) ^ ((1 : ℝ) / 960) * (d : ℝ) ^ (-(1 : ℝ) / 60) :=
      mul_le_mul_of_nonneg_right hbound (Real.rpow_nonneg hdpos.le _)
    _ = _ := by rw [← Real.rpow_add hdpos]; congr 1; norm_num

example : (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 ∧ (1 : ℝ) ^ 960 ≤ ((1 : ℕ) : ℝ) := by
  norm_num

/-- An explicit large-dimension threshold for the source's distance radius.
Source: arXiv:2312.10794v5, Appendix D, Step 2, under the standing
threshold `d ≥ d⋆(n,β)` of `thm: phase.transition.curve`. -/
noncomputable def halfProjectionThreshold (n : ℕ) : ℕ :=
  max (n + 3) (max ⌈Real.exp 160⌉₊ (sphericalNetBudget n ^ 960 + 1))

/-- The actual threshold exceeds the sample size and provides both the
moment-degree scale and the finite-budget bound. Source:
arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem halfProjectionThreshold_spec (n : ℕ) :
    n < halfProjectionThreshold n ∧ ∀ d : ℕ, halfProjectionThreshold n ≤ d →
      1 < d ∧ 160 ≤ Real.log (d : ℝ) ∧ (sphericalNetBudget n : ℝ) ^ 960 ≤ (d : ℝ) := by
  have hn := le_max_left (n + 3) (max ⌈Real.exp 160⌉₊ (sphericalNetBudget n ^ 960 + 1))
  refine ⟨by unfold halfProjectionThreshold; omega, fun d hd => ?_⟩
  have hnd : n + 3 ≤ d := hn.trans hd
  have hceil : ⌈Real.exp 160⌉₊ ≤ d :=
    (le_max_left _ _).trans ((le_max_right _ _).trans hd)
  have hbudget : sphericalNetBudget n ^ 960 + 1 ≤ d :=
    (le_max_right _ _).trans ((le_max_right _ _).trans hd)
  have hexp : Real.exp 160 ≤ (d : ℝ) := (Nat.le_ceil _).trans (by exact_mod_cast hceil)
  have hlog := Real.log_le_log (Real.exp_pos 160) hexp
  rw [Real.log_exp] at hlog
  exact ⟨by omega, hlog, by exact_mod_cast (show sphericalNetBudget n ^ 960 ≤ d by omega)⟩

/-- Uniform projection onto every previous actual orthonormal frame has
failure probability at most `d^(-1/64)` at half the comparison radius.
Source: arXiv:2312.10794v5, Appendix D, Step 2, `eq: almost.ortho.vec`.
The tail follows from the proved exact moments and finite directional net. -/
theorem uniform_frameProjection_tail_half_scale {d m n : ℕ}
    (hd : halfProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v) :
    ν.real {x : SSphere d | uniformProjectionRadius d / 2 ≤
      ‖frameProjection v (x : EucSpace d)‖} ≤ (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  obtain ⟨hdone, hlog, hbudget⟩ := (halfProjectionThreshold_spec n).2 d hd
  have hdpos : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hr : 0 < uniformProjectionRadius d / 2 := by
    exact div_pos (uniformProjectionRadius_pos hdone) (by norm_num)
  obtain ⟨hdegree, hdegreeLog⟩ := halfProjectionDegree_bounds hlog
  have hsq : (d : ℝ) * ((uniformProjectionRadius d / 2) / 2) ^ 2 = Real.log (d : ℝ) / 16 := by
    rw [div_pow, div_pow, uniformProjectionRadius_sq hdone.le]
    field_simp
    ring
  have hbase : (halfProjectionDegree d : ℝ) /
      ((d : ℝ) * ((uniformProjectionRadius d / 2) / 2) ^ 2) ≤ 1 / 2 := by
    apply (div_le_iff₀ (by positivity)).mpr
    rw [hsq]
    linarith
  calc
    _ ≤ ((sphericalHalfNet m).card : ℝ) *
        ((halfProjectionDegree d : ℝ) /
          ((d : ℝ) * ((uniformProjectionRadius d / 2) / 2) ^ 2)) ^ halfProjectionDegree d :=
      uniform_frameProjection_tail_le_pow (by omega) ν hν v hv _ hr _
    _ ≤ (sphericalNetBudget n : ℝ) * (1 / 2 : ℝ) ^ halfProjectionDegree d :=
      mul_le_mul (by exact_mod_cast sphericalHalfNet_card_le_budget hmn)
        (pow_le_pow_left₀ (by positivity) hbase _) (by positivity) (by positivity)
    _ ≤ (sphericalNetBudget n : ℝ) * Real.exp (-(2 / 3 : ℝ) * halfProjectionDegree d) :=
      mul_le_mul_of_nonneg_left (half_pow_le_exp_neg_two_thirds _) (by positivity)
    _ ≤ (sphericalNetBudget n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 60) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      rw [Real.rpow_def_of_pos hdpos]
      apply Real.exp_le_exp.mpr
      linarith
    _ ≤ _ := half_projection_budget_absorbed _ (by positivity) (by omega) hbudget

/-- A genuine uniform law and one actual unit direction satisfy every
half-scale concentration hypothesis at the stated dimension threshold. -/
example : ∃ d : ℕ, ∃ v : Fin 1 → EucSpace d,
    halfProjectionThreshold 2 ≤ d ∧ (1 : ℕ) ≤ 2 ∧
      IsUniformOn d (MeanField.uniformLaw d) ∧ Orthonormal ℝ v := by
  let d := halfProjectionThreshold 2
  have hd := ((halfProjectionThreshold_spec 2).2 d le_rfl).1
  let v : Fin 1 → EucSpace d := fun _ => EuclideanSpace.single ⟨0, by omega⟩ 1
  refine ⟨d, v, le_rfl, by decide, isUniformOn_uniformLaw (by omega), ?_⟩
  rw [orthonormal_iff_ite]
  intro i j
  simp [v, Subsingleton.elim i j]

end Transformer.Perspective

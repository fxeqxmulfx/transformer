/-
# Bounded distinct configurations spread under the original discrete head sum

ArXiv:2305.05465v6, §7, the first paragraph of the proof of `l:onlyone`,
and §12, `sec:conclusion`.
The source uses compactness and strict convexity to exclude a uniformly
bounded configuration of separated tokens. Here the quantitative softmax
monotonicity gives an explicit positive cubic bound for each original
head. All other positive heads contribute nonnegative gap increments.

For an actual discrete trajectory, a fixed positive initial gap persists.
If all tokens stayed in a fixed interval, this gap would consequently
increase by a fixed positive amount at every layer. The resulting linear
lower bound contradicts that interval. The conclusion concerns the
original separate head matrices and the original finite-step equation.
-/

import Transformer.Clusters.Section12_MultiHeadOrder
import Transformer.Clusters.Section7_OnlyOneSpread

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {m H : ℕ}

/-- An original positive scalar head strictly separates two distinct bounded tokens.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, the strict convexity step,
with its own original query-key coefficient as required by §12. -/
theorem scalarHeadAverage_gap_of_bounded (c : ℝ) (hc : 0 < c)
    (y : Idx (m + 1) → ℝ) {R : ℝ} (hR : ∀ j, |y j| ≤ R)
    (L M : Idx (m + 1)) (hLM : y L < y M) :
    (Real.exp (-(2 * (c * R ^ 2))) / ((m : ℝ) + 1) / 2 * c) * (y M - y L) ^ 3 ≤
      scalarHeadAverage c y (y M) - scalarHeadAverage c y (y L) := by
  have hR0 : 0 ≤ R := (abs_nonneg _).trans (hR L)
  have hsc : ∀ k j, |c * y k * y j| ≤ c * R ^ 2 := by
    intro k j
    calc
      |c * y k * y j| = c * |y k * y j| := by rw [mul_assoc, abs_mul, abs_of_pos hc]
      _ ≤ c * (R * R) := mul_le_mul_of_nonneg_left
        (by rw [abs_mul]; exact mul_le_mul (hR k) (hR j) (abs_nonneg _) hR0) hc.le
      _ = c * R ^ 2 := by ring
  have hab : M ≠ L := fun he => by rw [he] at hLM; exact lt_irrefl _ hLM
  have h := sq_le_softmax_monotone (fun j => c * y M * y j) (fun j => c * y L * y j)
    (hsc M) (hsc L) hab
  have he : (∑ j, (Perspective.softmaxWeight (fun l => c * y M * y l) j -
      Perspective.softmaxWeight (fun l => c * y L * y l) j) * (c * y M * y j - c * y L * y j)) =
      (c * (y M - y L)) * (scalarHeadAverage c y (y M) - scalarHeadAverage c y (y L)) := by
    simp only [scalarHeadAverage, ← Finset.sum_sub_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [he] at h
  apply le_of_mul_le_mul_left _ (mul_pos hc (sub_pos.mpr hLM))
  have hid : c * (y M - y L) *
      ((Real.exp (-(2 * (c * R ^ 2))) / ((m : ℝ) + 1) / 2 * c) * (y M - y L) ^ 3) =
      Real.exp (-(2 * (c * R ^ 2))) / ((m : ℝ) + 1) / 2 *
        ((c * y M * y M - c * y L * y M) - (c * y M * y L - c * y L * y L)) ^ 2 := by ring
  rwa [hid]

/-- Any one positive original head gives a strict lower bound for the whole drift gap.
Source: arXiv:2305.05465v6, §12's head sum and §7, proof of `l:onlyone`. -/
theorem multiHeadScalarDrift_gap_of_bounded (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 < v h) (y : Idx (m + 1) → ℝ) {R : ℝ} (hR : ∀ j, |y j| ≤ R)
    (L M : Idx (m + 1)) (hLM : y L < y M) (h : Idx H) :
    v h * (Real.exp (-(2 * (c h * R ^ 2))) / ((m : ℝ) + 1) / 2 * c h) * (y M - y L) ^ 3 ≤
      multiHeadScalarDrift c v y (y M) - multiHeadScalarDrift c v y (y L) := by
  have hnn : ∀ a, 0 ≤ v a * (scalarHeadAverage (c a) y (y M) - scalarHeadAverage (c a) y (y L)) :=
    fun a => mul_nonneg (hv a).le (sub_nonneg.mpr (scalarHeadAverage_monotone (c a) (hc a).le y hLM.le))
  calc
    _ = v h * ((Real.exp (-(2 * (c h * R ^ 2))) / ((m : ℝ) + 1) / 2 * c h) *
        (y M - y L) ^ 3) := by ring
    _ ≤ v h * (scalarHeadAverage (c h) y (y M) - scalarHeadAverage (c h) y (y L)) :=
      mul_le_mul_of_nonneg_left (scalarHeadAverage_gap_of_bounded (c h) (hc h) y hR L M hLM) (hv h).le
    _ ≤ ∑ a, v a * (scalarHeadAverage (c a) y (y M) - scalarHeadAverage (c a) y (y L)) :=
      Finset.single_le_sum (fun a _ => hnn a) (Finset.mem_univ h)
    _ = _ := by simp only [multiHeadScalarDrift, mul_sub, Finset.sum_sub_distrib]

/-- A genuinely distinct original discrete multi-head configuration cannot stay uniformly bounded.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, first paragraph, for §12's original iteration. -/
theorem multiHeadTransformer_not_all_bounded {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (L M : Idx (m + 1))
    (hLM : X 0 L 0 < X 0 M 0) (h : Idx H) :
    ¬ ∃ R : ℝ, ∀ k j, |X k j 0| ≤ R := by
  rintro ⟨R, hR⟩
  let c := fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1)
  let v := fun a => (V a unit1) 0
  have hc : ∀ a, 0 < c a := fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).1
  have hv : ∀ a, 0 < v a := fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).2
  let g := X 0 M 0 - X 0 L 0
  have hg : 0 < g := sub_pos.mpr hLM
  let C := v h * (Real.exp (-(2 * (c h * R ^ 2))) / ((m : ℝ) + 1) / 2 * c h)
  have hC : 0 < C := by
    have hch := hc h
    have hvh := hv h
    dsimp [C]
    positivity
  let η := Δt * C * g ^ 3
  have hη : 0 < η := by dsimp [η]; positivity
  have hstep : ∀ k, X k M 0 - X k L 0 + η ≤ X (k + 1) M 0 - X (k + 1) L 0 := by
    intro k
    have hgap := multiHeadTransformer_initial_gap hΔt.le hV hQK hX L M hLM.le k
    have hord := multiHeadTransformer_strict_order hΔt.le hV hQK hX L M hLM k
    have hd := multiHeadScalarDrift_gap_of_bounded c v hc hv (fun j => X k j 0) (hR k) L M hord h
    have hp := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hg.le hgap 3) hC.le
    have hs := mul_le_mul_of_nonneg_left (hp.trans hd) hΔt.le
    rw [multiHeadTransformer_coord hX k M, multiHeadTransformer_coord hX k L]
    dsimp [η, C, c, v] at hs ⊢
    nlinarith
  have hlinear : ∀ k : ℕ, g + (k : ℝ) * η ≤ X k M 0 - X k L 0 := by
    intro k
    induction k with
    | zero => simp only [Nat.cast_zero, zero_mul, add_zero]; exact le_rfl
    | succ k ih =>
        have hs := hstep k
        push_cast
        nlinarith
  obtain ⟨k, hk⟩ := exists_nat_gt ((2 * R - g) / η)
  have hbig : 2 * R - g < (k : ℝ) * η := (div_lt_iff₀ hη).mp hk
  have hL := (abs_le.mp (hR k L)).1
  have hM := (abs_le.mp (hR k M)).2
  have hlin := hlinear k
  linarith

/-- Every proposed global bound is exceeded along an actual distinct original trajectory.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, first paragraph, extended to §12. -/
theorem multiHeadTransformer_exceeds_every_bound {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (L M : Idx (m + 1))
    (hLM : X 0 L 0 < X 0 M 0) (h : Idx H) :
    ∀ R : ℝ, ∃ k j, R < |X k j 0| := by
  have hb := multiHeadTransformer_not_all_bounded hΔt hV hQK hX L M hLM h
  push Not at hb
  exact hb

/-- Distinct bounded data realize the static strict-head estimate with two different heads. -/
example : ∃ (c v : Idx 2 → ℝ) (y : Idx 2 → ℝ),
    (∀ h, 0 < c h) ∧ (∀ h, 0 < v h) ∧ (∀ j, |y j| ≤ 1) ∧ y 0 < y 1 := by
  refine ⟨![1, 2], ![1, 3], ![-1, 1], ?_, ?_, ?_, by norm_num⟩
  · intro h; fin_cases h <;> norm_num
  · intro h; fin_cases h <;> norm_num
  · intro j; fin_cases j <;> norm_num

/-- Two genuine positive heads and distinct nonzero initial tokens jointly realize
all original trajectory premises of the impossibility of a uniform bound. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ X 0 0 0 < X 0 1 0 := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  refine ⟨Q, Q, Q, multiHeadTrajectory 1 Q Q Q Y, one_pos,
    fun _ => isPosDefOp_id 1, fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1),
    multiHeadTrajectory_solution _ _ _ _ _, ?_⟩
  change Y 0 0 < Y 1 0
  norm_num [Y, unit1]

end Transformer.Clusters

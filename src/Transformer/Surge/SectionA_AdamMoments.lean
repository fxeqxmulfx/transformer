/-
# The moments of Adam

arXiv:2405.14578, Appendix A.  Adam keeps the moments `m_t = β₁m_{t-1} + (1 - β₁)G_t` of the
gradients and `v_t = β₂v_{t-1} + (1 - β₂)G_t²` of their squares, from `m_0 = v_0 = 0`,
eqs. (25) and (26) (`adamMoment`, `adamMoment_eq`), and updates by `V = m̂_t/(√v̂_t + ε_Adam)`,
with `m̂_t = m_t/(1 - β₁ᵗ)` and `v̂_t = v_t/(1 - β₂ᵗ)`, eq. (27) (`adamUpdate`, `adamUpdate_eq`).

Without `ε_Adam`, as `β₁, β₂ → 1` the update tends to `E_t[G]/√E_t[G²]`, the first step of
eq. (28), the means taken over the iterations `1, …, t` (`iterMean`, `tendsto_adamUpdate_one`).
The limit is joint in `(β₁, β₂)` and two-sided: it excludes only `β = 1`, where the bias
correction `(1 - β)/(1 - βᵗ)` is `0/0`.  The variance `var_t(G) = E_t[G²] - E_t[G]²` of eq. (28)
is `∑_{i=1}^t (G_i - E_t[G])²/t`, never negative (`iterVar`, `mul_iterVar`, `iterVar_nonneg`),
and the limit is `sign(E_t[G])/√(1 + var_t(G)/E_t[G]²)`, the rest of eq. (28)
(`adamLimit_eq_sign`).  At `β = 0` the moment is the last gradient (`adamMoment_zero`).
-/

import Mathlib.Algebra.Ring.GeomSum
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Basic.Real.Sign
import Mathlib.Tactic.LinearCombination

open Filter Topology

namespace Transformer.Surge

/-- A moment of Adam, eqs. (25) and (26): `m_0 = 0` and `m_t = βm_{t-1} + (1 - β)G_t`, for the
gradients `G`, or their squares. -/
noncomputable def adamMoment (β : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => β * adamMoment β g t + (1 - β) * g (t + 1)

/-- Eq. (27): `m_t = (1 - β) ∑_{i=1}^t β^{t-i} G_i`. -/
theorem adamMoment_eq (β : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    adamMoment β g t = (1 - β) * ∑ i ∈ Finset.Icc 1 t, β ^ (t - i) * g i := by
  induction t with
  | zero => simp [adamMoment]
  | succ t ih =>
    have h : ∑ i ∈ Finset.Icc 1 t, β ^ (t + 1 - i) * g i =
        β * ∑ i ∈ Finset.Icc 1 t, β ^ (t - i) * g i := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [show t + 1 - i = t - i + 1 by have := (Finset.mem_Icc.1 hi).2; omega, pow_succ]
      ring
    rw [adamMoment, ih, Finset.sum_Icc_succ_top (by omega), h, Nat.sub_self, pow_zero, one_mul]
    ring

/-- The update of Adam, eq. (27): `V = m̂_t/(√v̂_t + ε_Adam)`, with the bias-corrected moments
`m̂_t = m_t/(1 - β₁ᵗ)` of the gradients `G` and `v̂_t = v_t/(1 - β₂ᵗ)` of their squares. -/
noncomputable def adamUpdate (β₁ β₂ εA : ℝ) (g : ℕ → ℝ) (t : ℕ) : ℝ :=
  adamMoment β₁ g t / (1 - β₁ ^ t) / (√(adamMoment β₂ (fun i => g i ^ 2) t / (1 - β₂ ^ t)) + εA)

/-- **Eq. (27)**: `V = (1 - β₁)/(1 - β₁ᵗ) ∑ β₁^{t-i}G_i/(√((1 - β₂)/(1 - β₂ᵗ) ∑ β₂^{t-i}G_i²)
+ ε_Adam)`. -/
theorem adamUpdate_eq (β₁ β₂ εA : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    adamUpdate β₁ β₂ εA g t =
      (1 - β₁) / (1 - β₁ ^ t) * (∑ i ∈ Finset.Icc 1 t, β₁ ^ (t - i) * g i) /
        (√((1 - β₂) / (1 - β₂ ^ t) * ∑ i ∈ Finset.Icc 1 t, β₂ ^ (t - i) * g i ^ 2) + εA) := by
  simp only [adamUpdate, adamMoment_eq]
  rw [mul_div_right_comm, mul_div_right_comm (1 - β₂)]

/-- The mean `E_t[G] = (G_1 + ⋯ + G_t)/t` over the iterations, eq. (28). -/
noncomputable def iterMean (g : ℕ → ℝ) (t : ℕ) : ℝ := (∑ i ∈ Finset.Icc 1 t, g i) / t

/-- For `β ≠ 1` the bias correction `(1 - β)/(1 - βᵗ)` is `1/(1 + β + ⋯ + β^{t-1})`. -/
theorem adamMoment_div_eq {β : ℝ} (hβ : β ≠ 1) (g : ℕ → ℝ) (t : ℕ) :
    adamMoment β g t / (1 - β ^ t) =
      (∑ i ∈ Finset.Icc 1 t, β ^ (t - i) * g i) / ∑ k ∈ Finset.range t, β ^ k := by
  rw [adamMoment_eq, show 1 - β ^ t = (1 - β) * ∑ k ∈ Finset.range t, β ^ k by
    linear_combination geom_sum_mul β t, mul_div_mul_left _ _ (sub_ne_zero.2 hβ.symm)]

/-- The hypothesis of `adamMoment_div_eq` is satisfiable: `β = 0`. -/
example (g : ℕ → ℝ) (t : ℕ) := adamMoment_div_eq zero_ne_one g t

/-- As `β → 1`, the bias-corrected moment `m_t/(1 - βᵗ)` tends to the mean `E_t[G]`. -/
theorem tendsto_adamMoment_div_one (g : ℕ → ℝ) (t : ℕ) :
    Tendsto (fun β => adamMoment β g t / (1 - β ^ t)) (𝓝[≠] 1) (𝓝 (iterMean g t)) := by
  rcases eq_or_ne t 0 with rfl | ht
  · simp [adamMoment, iterMean]
  have h1 : Tendsto (fun β : ℝ => ∑ i ∈ Finset.Icc 1 t, β ^ (t - i) * g i) (𝓝 1)
      (𝓝 (∑ i ∈ Finset.Icc 1 t, g i)) := by
    have : Continuous fun β : ℝ => ∑ i ∈ Finset.Icc 1 t, β ^ (t - i) * g i := by fun_prop
    simpa using this.tendsto 1
  have h2 : Tendsto (fun β : ℝ => ∑ k ∈ Finset.range t, β ^ k) (𝓝 1) (𝓝 (t : ℝ)) := by
    have : Continuous fun β : ℝ => ∑ k ∈ Finset.range t, β ^ k := by fun_prop
    simpa using this.tendsto 1
  refine ((h1.div h2 (Nat.cast_ne_zero.2 ht)).mono_left nhdsWithin_le_nhds).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with β hβ
  exact (adamMoment_div_eq hβ g t).symm

/-- **Eq. (28)**, first step: without `ε_Adam`, as `β₁, β₂ → 1` the update of Adam tends to
`E_t[G]/√E_t[G²]`. -/
theorem tendsto_adamUpdate_one (g : ℕ → ℝ) (t : ℕ) :
    Tendsto (fun β : ℝ × ℝ => adamUpdate β.1 β.2 0 g t) (𝓝[≠] 1 ×ˢ 𝓝[≠] 1)
      (𝓝 (iterMean g t / √(iterMean (fun i => g i ^ 2) t))) := by
  simp only [adamUpdate, add_zero]
  by_cases h : iterMean (fun i => g i ^ 2) t = 0
  · have hg : ∀ i ∈ Finset.Icc 1 t, g i = 0 := by
      intro i hi
      unfold iterMean at h
      rcases div_eq_zero_iff.1 h with h | h
      · exact pow_eq_zero_iff two_ne_zero |>.1 <|
          (Finset.sum_eq_zero_iff_of_nonneg fun i _ => sq_nonneg (g i)).1 h i hi
      · rw [Nat.cast_eq_zero] at h
        subst h
        simp at hi
    have h0 (β : ℝ) : adamMoment β g t = 0 := by
      rw [adamMoment_eq, Finset.sum_eq_zero fun i hi => by rw [hg i hi, mul_zero], mul_zero]
    simp [h, h0]
  · have hpos : 0 < iterMean (fun i => g i ^ 2) t := lt_of_le_of_ne
      (div_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg (g i)) (Nat.cast_nonneg t)) (Ne.symm h)
    exact ((tendsto_adamMoment_div_one g t).comp tendsto_fst).div
      ((tendsto_adamMoment_div_one _ t).comp tendsto_snd).sqrt (Real.sqrt_ne_zero'.2 hpos)

/-- At `β = 0` the moment is the last gradient: `m_t = G_t`, `t ≥ 1`. -/
theorem adamMoment_zero (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0) : adamMoment 0 g t = g t := by
  obtain ⟨s, rfl⟩ := Nat.exists_eq_succ_of_ne_zero ht
  simp [adamMoment]

/-- The hypothesis of `adamMoment_zero` is satisfiable: `t = 1`. -/
example (g : ℕ → ℝ) := adamMoment_zero g one_ne_zero

/-- The variance `var_t(G) = E_t[G²] - E_t[G]²` over the iterations, eq. (28). -/
noncomputable def iterVar (g : ℕ → ℝ) (t : ℕ) : ℝ :=
  iterMean (fun i => g i ^ 2) t - iterMean g t ^ 2

/-- `t var_t(G) = ∑_{i=1}^t (G_i - E_t[G])²`. -/
theorem mul_iterVar (g : ℕ → ℝ) (t : ℕ) :
    t * iterVar g t = ∑ i ∈ Finset.Icc 1 t, (g i - iterMean g t) ^ 2 := by
  rcases eq_or_ne t 0 with rfl | ht
  · simp
  have ht' : (t : ℝ) ≠ 0 := Nat.cast_ne_zero.2 ht
  simp only [iterVar, sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul,
    ← Finset.mul_sum, Finset.sum_const, Nat.card_Icc, Nat.add_sub_cancel, nsmul_eq_mul, iterMean]
  field_simp
  ring

/-- `var_t(G) ≥ 0`. -/
theorem iterVar_nonneg (g : ℕ → ℝ) (t : ℕ) : 0 ≤ iterVar g t := by
  rcases eq_or_ne t 0 with rfl | ht
  · simp [iterVar, iterMean]
  have h : 0 ≤ (t : ℝ) * iterVar g t :=
    mul_iterVar g t ▸ Finset.sum_nonneg fun i _ => sq_nonneg _
  exact (mul_nonneg_iff_of_pos_left (Nat.cast_pos.2 (Nat.pos_of_ne_zero ht))).1 h

/-- `m/√s = sign(m)/√(1 + (s - m²)/m²)`. -/
theorem div_sqrt_eq_sign (m s : ℝ) : m / √s = Real.sign m / √(1 + (s - m ^ 2) / m ^ 2) := by
  rcases eq_or_ne m 0 with rfl | hm
  · simp
  rw [one_add_div (pow_ne_zero 2 hm), add_sub_cancel, Real.sqrt_div' s (sq_nonneg m),
    Real.sqrt_sq_eq_abs, div_div_eq_mul_div]
  rcases hm.lt_or_gt with hm | hm
  · rw [Real.sign_of_neg hm, abs_of_neg hm]
    ring
  · rw [Real.sign_of_pos hm, abs_of_pos hm]
    ring

/-- **Eq. (28)**: `E_t[G]/√E_t[G²] = sign(E_t[G])/√(1 + var_t(G)/E_t[G]²)`. -/
theorem adamLimit_eq_sign (g : ℕ → ℝ) (t : ℕ) :
    iterMean g t / √(iterMean (fun i => g i ^ 2) t) =
      Real.sign (iterMean g t) / √(1 + iterVar g t / iterMean g t ^ 2) :=
  div_sqrt_eq_sign _ _

end Transformer.Surge

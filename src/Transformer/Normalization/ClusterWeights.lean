/-
# Attention weights inside a local cluster

Appendix C, proof of Theorem 4.3 of arXiv:2510.22026v2: a small
score diameter makes attention close to the uniform probability vector.
Absolute inverse temperature also covers negative beta.
-/

import Transformer.Normalization.Basic
import Transformer.Perspective.Softmax
import Mathlib.Analysis.Complex.Exponential

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- An attention row, with the scores of arXiv:2510.22026v2, equation (NA). -/
noncomputable def attentionWeight (β : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j k : Idx n) : ℝ :=
  Perspective.softmaxWeight (fun l => β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))) k

/-- Attention with identity values is the weighted token mean. Source:
arXiv:2510.22026v2, equation (NA) and Appendix C, proof of Theorem 4.3. -/
theorem attentionVec_id_eq_sum (β : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j : Idx n) :
    attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j =
      ∑ k, attentionWeight β Q K Θ j k • Θ k := by
  rw [attentionVec, Finset.smul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [attentionWeight, Perspective.softmaxWeight,
    ContinuousLinearMap.id_apply, div_eq_mul_inv, smul_smul]
  congr 1
  ring

/-- Attention weights are positive. Source: arXiv:2510.22026v2, (NA). -/
theorem attentionWeight_pos (β : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j k : Idx n) :
    0 < attentionWeight β Q K Θ j k :=
  div_pos (Real.exp_pos _) (Perspective.softmaxPartition_pos (Fin.pos j) _)

/-- An attention row has total mass one. Source: arXiv:2510.22026v2, (NA). -/
theorem sum_attentionWeight (β : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j : Idx n) :
    ∑ k, attentionWeight β Q K Θ j k = 1 :=
  Perspective.sum_softmaxWeight (Fin.pos j) _

/-- A score diameter of `r` bounds each weight between `exp(-r)/n` and
`exp(r)/n`. Source: arXiv:2510.22026v2, Appendix C, the bounds on `w_kj`. -/
theorem softmaxWeight_bounds_of_diameter (hn : 0 < n) (u : Idx n → ℝ) (r : ℝ)
    (hu : ∀ j k, |u j - u k| ≤ r) (j : Idx n) :
    Real.exp (-r) / n ≤ Perspective.softmaxWeight u j ∧
      Perspective.softmaxWeight u j ≤ Real.exp r / n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hZ := Perspective.softmaxPartition_pos hn u
  have hlo : (n : ℝ) * Real.exp (u j - r) ≤ ∑ k, Real.exp (u k) := by
    have h := Finset.sum_le_sum (s := Finset.univ) fun k _ =>
      Real.exp_le_exp.mpr (show u j - r ≤ u k by linarith [le_of_abs_le (hu j k)])
    simpa only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using h
  have hhi : (∑ k, Real.exp (u k)) ≤ (n : ℝ) * Real.exp (u j + r) := by
    have h := Finset.sum_le_sum (s := Finset.univ) fun k _ =>
      Real.exp_le_exp.mpr (show u k ≤ u j + r by linarith [neg_le_of_abs_le (hu j k)])
    simpa only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using h
  constructor
  · rw [Perspective.softmaxWeight, div_le_div_iff₀ hn' hZ]
    calc Real.exp (-r) * (∑ k, Real.exp (u k))
        ≤ Real.exp (-r) * ((n : ℝ) * Real.exp (u j + r)) :=
          mul_le_mul_of_nonneg_left hhi (Real.exp_pos _).le
      _ = Real.exp (u j) * n := by
        rw [mul_left_comm, ← Real.exp_add]
        rw [show -r + (u j + r) = u j by ring, mul_comm]
  · rw [Perspective.softmaxWeight, div_le_div_iff₀ hZ hn']
    calc Real.exp (u j) * n
        = Real.exp r * ((n : ℝ) * Real.exp (u j - r)) := by
          rw [mul_left_comm, ← Real.exp_add]
          rw [show r + (u j - r) = u j by ring, mul_comm]
      _ ≤ Real.exp r * (∑ k, Real.exp (u k)) :=
          mul_le_mul_of_nonneg_left hlo (Real.exp_pos _).le

/-- Equal scores witness the diameter hypothesis with `r = 0`. -/
example : 0 < 1 ∧ ∀ j k : Idx 1,
    |(fun _ : Idx 1 => (0 : ℝ)) j - (fun _ : Idx 1 => (0 : ℝ)) k| ≤ 0 :=
  ⟨Nat.one_pos, fun _ _ => by simp⟩

/-- The weight deviation from `1/n` is at most `(exp(r)-1)/n`.
Source: arXiv:2510.22026v2, Appendix C, proof of Theorem 4.3. -/
theorem abs_softmaxWeight_sub_uniform_le (hn : 0 < n) (u : Idx n → ℝ) (r : ℝ)
    (hu : ∀ j k, |u j - u k| ≤ r) (j : Idx n) :
    |Perspective.softmaxWeight u j - 1 / n| ≤ (Real.exp r - 1) / n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  obtain ⟨hlo, hhi⟩ := softmaxWeight_bounds_of_diameter hn u r hu j
  have hexp : 2 ≤ Real.exp r + Real.exp (-r) := by
    linarith [Real.add_one_le_exp r, Real.add_one_le_exp (-r)]
  have hlo' := (div_le_iff₀ hn').mp hlo
  have hhi' := (le_div_iff₀ hn').mp hhi
  have hunit : (1 / (n : ℝ)) * n = 1 := one_div_mul_cancel hn'.ne'
  rw [abs_le, ← neg_div, div_le_iff₀ hn', le_div_iff₀ hn']
  constructor <;> nlinarith

/-- Equal scores give zero deviation from uniform weights. -/
example : 0 < 1 ∧ ∀ j k : Idx 1,
    |(fun _ : Idx 1 => (0 : ℝ)) j - (fun _ : Idx 1 => (0 : ℝ)) k| ≤ 0 :=
  ⟨Nat.one_pos, fun _ _ => by simp⟩

/-- With identity values, a lower bound on all pairwise inner products is
also a lower bound on the radial attention component. Source:
arXiv:2510.22026v2, Appendix C, estimates on the speed factors. -/
theorem inner_attentionVec_id_ge (β m : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d)
    (hΘ : ∀ j k, m ≤ inner (𝕜 := ℝ) (Θ j) (Θ k)) (j : Idx n) :
    m ≤ inner (𝕜 := ℝ) (Θ j)
      (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j) := by
  rw [attentionVec_id_eq_sum, inner_sum]
  have h := Finset.sum_le_sum (s := Finset.univ) fun k _ =>
    mul_le_mul_of_nonneg_left (hΘ j k) (attentionWeight_pos β Q K Θ j k).le
  simpa only [← Finset.sum_mul, sum_attentionWeight, one_mul, real_inner_smul_right] using h

/-- A unit singleton realizes the lower bound `m = 1`. -/
example : ∀ j k : Idx 1, (1 : ℝ) ≤
    inner (𝕜 := ℝ) ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) j)
      ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) := fun _ _ => by simp

/-- For at least two unit tokens with pairwise height `1-4δ`, the theorem's
temperature-dependent smallness condition keeps attention within `1/(3n)`
of uniform. Source: arXiv:2510.22026v2, Appendix C, the bound on `w_kj` in
the proof of Theorem 4.3. -/
theorem attentionWeight_close_to_uniform (hn : 2 ≤ n) (β δ : ℝ) (hδ0 : 0 ≤ δ)
    (hδ : δ < 1 / (100 * (n : ℝ) ^ 2 * β ^ 2))
    (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d)
    (hQK : ∀ x y, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖)
    (hunit : ∀ j, ‖Θ j‖ = 1)
    (hpair : ∀ j k, 1 - 4 * δ ≤ inner (𝕜 := ℝ) (Θ j) (Θ k)) :
    ∀ j k, |attentionWeight β Q K Θ j k - (n : ℝ)⁻¹| ≤ (1 / 3 : ℝ) / n := by
  have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn0 : 0 < n := by omega
  have hβ : β ≠ 0 := by
    intro hzero
    simp [hzero] at hδ
    linarith
  have hden : 0 < 100 * (n : ℝ) ^ 2 * β ^ 2 := by positivity
  have hprod := (lt_div_iff₀ hden).mp hδ
  have hsmall : δ * β ^ 2 < 1 / 400 := by
    nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) ^ 2 - 4 by nlinarith)
      (mul_nonneg hδ0 (sq_nonneg β))]
  intro j k
  let u : Idx n → ℝ := fun l => β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l))
  have hdiam : ∀ i l, |u i - u l| ≤ (1 / 6 : ℝ) := by
    intro i l
    have hdist : ‖Θ i - Θ l‖ ^ 2 ≤ 8 * δ := by
      rw [norm_sub_sq_real, hunit i, hunit l]
      nlinarith [hpair i l]
    have hnorm := hQK (Θ j) (Θ i - Θ l)
    rw [hunit j, one_mul] at hnorm
    have heq : u i - u l = β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ i - Θ l)) := by
      simp only [u, map_sub, inner_sub_right]
      ring
    have habs : |u i - u l| ≤ |β| * ‖Θ i - Θ l‖ := by
      rw [heq, abs_mul]
      exact mul_le_mul_of_nonneg_left hnorm (abs_nonneg β)
    have hsq := (sq_le_sq₀ (abs_nonneg _) (mul_nonneg (abs_nonneg β) (norm_nonneg _))).mpr habs
    simp only [mul_pow, sq_abs] at hsq
    have hp := mul_le_mul_of_nonneg_left hdist (sq_nonneg β)
    nlinarith [abs_nonneg (u i - u l), sq_abs (u i - u l)]
  have hw := abs_softmaxWeight_sub_uniform_le hn0 u (1 / 6) hdiam k
  have hexp : Real.exp (1 / 6) - 1 ≤ (1 / 3 : ℝ) := by
    have h := Real.abs_exp_sub_one_le (show |(1 / 6 : ℝ)| ≤ 1 by norm_num)
    norm_num at h
    exact le_of_abs_le h
  simpa only [one_div, attentionWeight, u] using hw.trans
    (div_le_div_of_nonneg_right hexp (Nat.cast_nonneg n))

/-- Two identical unit directions, identity parameters and `δ=0` satisfy
the temperature and cone hypotheses. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / (100 * ((2 : ℕ) : ℝ) ^ 2 * (1 : ℝ) ^ 2) ∧
      (∀ j : Idx 2, ‖(fun _ : Idx 2 => u) j‖ = 1) ∧
      (∀ j k : Idx 2, 1 - 4 * (0 : ℝ) ≤ inner (𝕜 := ℝ) ((fun _ : Idx 2 => u) j)
        ((fun _ : Idx 2 => u) k)) ∧
      ∀ x y : EucSpace 1, |inner (𝕜 := ℝ) (idParams 1 0 x) (idParams 1 0 y)| ≤ ‖x‖ * ‖y‖ := by
  refine ⟨le_rfl, by norm_num, fun _ => by simp, fun _ _ => by simp,
    fun x y => abs_real_inner_le_norm x y⟩

end Transformer.Normalization

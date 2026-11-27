module

public import Transformer.Grokking.FiniteRecovery.ArgandNative
public import Transformer.Grokking.FiniteRecovery.ArgandStrictRecovery
public import Transformer.Grokking.FlatSpectrum.WeightedPhases

/-! Bounds on the nuclear norm of the finite-weight Argand phase table.
The centered-code bound uses only the learned embedding norms and the
final RMS. The row-factor bound uses the model's unlabeled probabilities. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Finset Matrix FlatSpectrum
open scoped BigOperators

variable {A U V : Type*} [Fintype A] {T H d layers : ℕ}

/-- If the distribution contains at least `b` units of mass from the
uniform law, at most `1-b` of the unit phase survives centering. -/
private theorem norm_phaseExpectation_of_floor [Nonempty A] (code : A → ℂ) (p : A → ℝ)
    (hunit : ∀ a, ‖code a‖ = 1) (hzero : ∑ a, code a = 0) (hp : ∑ a, p a = 1)
    (b : ℝ) (hfloor : ∀ a, b / Fintype.card A ≤ p a) : ‖phaseExpectation code p‖ ≤ 1 - b := by
  let s := b / Fintype.card A
  have hc : (Fintype.card A : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_pos.ne'
  have heq : phaseExpectation code p = ∑ a, ((p a - s : ℝ) : ℂ) * code a := by
    simp [phaseExpectation, sub_mul, sum_sub_distrib, ← mul_sum, hzero]
  rw [heq]
  calc
    _ ≤ ∑ a, ‖((p a - s : ℝ) : ℂ) * code a‖ := norm_sum_le _ _
    _ = ∑ a, (p a - s) := by
      apply sum_congr rfl
      intro a _
      rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (sub_nonneg.mpr (hfloor a)), hunit, mul_one]
    _ = 1 - b := by simp [s, sum_sub_distrib, hp, nsmul_eq_mul, mul_div_cancel₀ _ hc]

/-- A diameter bound on the complete logit vector yields an entrywise
bound on the centered phase, for every word and every token alphabet. -/
theorem Model.phaseEntry_le_of_embeddingBound [Nonempty A] (model : Model A H d layers)
    (heps : 0 < model.eps) {B : ℝ} (hB : ∀ a, ‖model.embedding a‖ ≤ B)
    (repeats : ℕ) (word : Word A (T + 1)) (code : A → ℂ)
    (hunit : ∀ a, ‖code a‖ = 1) (hzero : ∑ a, code a = 0) :
    ‖phaseExpectation code (model.probability repeats word (Fin.last T))‖ ≤
      1 - Real.exp (-(B * Real.sqrt (H * d : ℕ))) := by
  let i := Fin.last T
  let z := model.logits repeats word i
  let D := B * Real.sqrt (H * d : ℕ)
  have hz (a : A) : |z a| ≤ D / 2 := (model.abs_logits_le heps repeats word i a).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hB a) (Real.sqrt_nonneg _)) (by norm_num))
  apply norm_phaseExpectation_of_floor code _ hunit hzero (model.sum_probability repeats word i)
  intro a
  have hc : 0 < (Fintype.card A : ℝ) := Nat.cast_pos.mpr Fintype.card_pos
  have hs : 0 < ∑ t, Real.exp (z t) := sum_pos' (fun _ _ => (Real.exp_pos _).le)
    ⟨a, mem_univ _, Real.exp_pos _⟩
  change Real.exp (-D) / Fintype.card A ≤ Real.exp (z a) / ∑ t, Real.exp (z t)
  apply (div_le_div_iff₀ hc hs).mpr
  rw [Finset.mul_sum]
  calc
    _ ≤ ∑ _ : A, Real.exp (z a) := sum_le_sum fun t _ => by
      rw [← Real.exp_add]
      exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hz t)).2, (abs_le.mp (hz a)).1])
    _ = _ := by simp [nsmul_eq_mul, mul_comm]

variable [Fintype U] [Fintype V] [DecidableEq V]

/-- Factor converting uniform entry bounds into nuclear norm bounds. -/
noncomputable def phaseTableSizeFactor (m n : ℕ) : ℝ :=
  Real.sqrt ((min m n : ℕ) * (m : ℝ) * n)

theorem nuclearNorm_le_of_entryBound (X : Matrix U V ℂ) {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ u v, ‖X u v‖ ≤ δ) :
    nuclearNorm X ≤ phaseTableSizeFactor (Fintype.card U) (Fintype.card V) * δ := by
  let m := Fintype.card U
  let n := Fintype.card V
  have hf : frobeniusSq X ≤ (m : ℝ) * n * δ ^ 2 := by
    calc
      _ ≤ ∑ _ : U, ∑ _ : V, δ ^ 2 := sum_le_sum fun i _ => sum_le_sum fun j _ =>
        pow_le_pow_left₀ (norm_nonneg _) (h i j) _
      _ = _ := by simp [m, n, nsmul_eq_mul, mul_assoc]
  have hr : (X.rank : ℝ) ≤ (min m n : ℕ) :=
    Nat.cast_le.mpr (le_min X.rank_le_card_height X.rank_le_card_width)
  have hs := (nuclearNorm_sq_le_rank_mul_frobeniusSq X).trans
    (mul_le_mul hr hf (frobeniusSq_nonneg _) (Nat.cast_nonneg _))
  have he := Real.sq_sqrt (by positivity : 0 ≤ ((min m n : ℕ) : ℝ) * m * n)
  change nuclearNorm X ≤ √(((min m n : ℕ) : ℝ) * m * n) * δ
  apply le_of_sq_le_sq _ (mul_nonneg (Real.sqrt_nonneg _) hδ)
  rw [mul_pow, he]
  linarith only [hs]

/-- A rank-one column profile with unit left phases need not use hidden
labels. The residual is computed from the learned table on input words.
Its Frobenius norm controls the excess over the profile's exact norm. -/
theorem Model.nuclearNorm_le_rowFactor (model : Model A H d layers) (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (a : U → ℂ) (b : V → ℂ)
    (ha : ∀ u, ‖a u‖ = 1) :
    let K := model.phaseKernel repeats cut code
    nuclearNorm K ≤ √(Fintype.card U : ℝ) * √(∑ v, ‖b v‖ ^ 2) +
      √((min (Fintype.card U) (Fintype.card V) : ℕ) * frobeniusSq (K - vecMulVec a b)) := by
  dsimp only
  let K := model.phaseKernel repeats cut code
  let Q := vecMulVec a b
  have h := nuclearNorm_add_le Q (K - Q)
  rw [add_sub_cancel _ _] at h
  have hQ : nuclearNorm Q = √(Fintype.card U : ℝ) * √(∑ v, ‖b v‖ ^ 2) := by
    simp [Q, nuclearNorm_vecMulVec, ha]
  rw [hQ] at h
  refine h.trans (add_le_add le_rfl (Real.le_sqrt_of_sq_le ?_))
  exact (nuclearNorm_sq_le_rank_mul_frobeniusSq _).trans
    (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr (le_min (K - Q).rank_le_card_height
      (K - Q).rank_le_card_width)) (frobeniusSq_nonneg _))

example (model : Model (ZMod 2) 4 16 1) :=
  model.nuclearNorm_le_rowFactor (T := 7) 1 (enumeratedCut (wordEnumeration 2 4)) ZMod.stdAddChar
    (fun u => ZMod.stdAddChar (∑ j, wordEnumeration 2 4 u j)) (fun _ => 0) (by simp)

end Transformer.Grokking.FiniteRecovery.Argand

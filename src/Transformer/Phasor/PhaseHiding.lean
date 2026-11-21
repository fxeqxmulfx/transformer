module

public import Transformer.Phasor.StreamSum

/-!
# Phases can hide equal streams

T3d of `review/residual_connections.md`, *What the extra streams can hold*. Read a stream of even
width `d` as `d/2` complex coordinates, as `headPhase` does. Then the phase-conjugated skip
`D H D*` sends four equal streams `h` to `a_s h`, with one complex number per stream,
`a_s = ∑_t H_st e^{i(φ_s − φ_t)} = C_s + i S_s` (`phaseCos`, `phaseSin`): on every channel pair,
stream `s` is `h` turned and scaled by `a_s` (`phaseMix_const_fst`, `phaseMix_const_snd`).

* `sum_phaseSin_eq_zero`: for a symmetric `H` the imaginary parts sum to zero, `sin` being odd.
* `sum_phaseMix_const_eq_zero`: if moreover the real parts sum to zero, the streams after the skip
  sum to zero, so a read with equal gates (`weightedSum_const_weights`) no longer sees `h`;
* `phaseMix_const_ne_zero`: yet stream `s` still holds `a_s h ≠ 0` whenever `C_s ≠ 0`;
* `weightedSum_eq_sum_add`: gates equal off stream `0` read the sum plus `g₀ − γ` times stream `0`,
  so unequal gates see it again.

Without phases this cannot happen (`PositiveReads`). `HideSkip` and `HideRouter` give a skip map
and a router of main that do it.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual
  Transformer.SequenceModel

variable {ι : Type*} [Fintype ι] {d : ℕ}

/-- The first coordinate of pair `k` after the phase `a`. -/
theorem headPhase_fst (a : ℝ) (y : EucSpace d) (k : Fin (d / 2)) :
    headPhase d a y (GPTMini.ropeFst d k) =
      y (GPTMini.ropeFst d k) * Real.cos a - y (GPTMini.ropeSnd d k) * Real.sin a := by
  simp [headPhase, rotateHead_apply_fst, rotate, splitPairs]

/-- The second coordinate of pair `k` after the phase `a`. -/
theorem headPhase_snd (a : ℝ) (y : EucSpace d) (k : Fin (d / 2)) :
    headPhase d a y (GPTMini.ropeSnd d k) =
      y (GPTMini.ropeFst d k) * Real.sin a + y (GPTMini.ropeSnd d k) * Real.cos a := by
  simp [headPhase, rotateHead_apply_snd, rotate, splitPairs]

/-- The real part `C_s = ∑_t H_st cos(φ_s − φ_t)` of the number `a_s` of stream `s`. -/
noncomputable def phaseCos (H : Matrix ι ι ℝ) (φ : ι → ℝ) (s : ι) : ℝ :=
  ∑ t, H s t * Real.cos (φ s - φ t)

/-- The imaginary part `S_s = ∑_t H_st sin(φ_s − φ_t)` of the number `a_s` of stream `s`. -/
noncomputable def phaseSin (H : Matrix ι ι ℝ) (φ : ι → ℝ) (s : ι) : ℝ :=
  ∑ t, H s t * Real.sin (φ s - φ t)

/-- **The skip of equal streams, first coordinate of a pair**: `h_f C_s − h_s S_s`. -/
theorem phaseMix_const_fst (H : Matrix ι ι ℝ) (φ : ι → ℝ) (h : EucSpace d) (s : ι)
    (k : Fin (d / 2)) :
    phaseMix (headPhaseAction d) H φ (fun _ => h) s (GPTMini.ropeFst d k) =
      h (GPTMini.ropeFst d k) * phaseCos H φ s - h (GPTMini.ropeSnd d k) * phaseSin H φ s := by
  change headPhase d (φ s) (weightedSum (H s) fun t => headPhase d (-φ t) h) _ = _
  simp only [headPhase_fst, headPhase_snd, weightedSum, WithLp.ofLp_sum, Finset.sum_apply,
    PiLp.smul_apply, smul_eq_mul, phaseCos, phaseSin, Finset.mul_sum, Finset.sum_mul,
    ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Real.cos_sub, Real.sin_sub, Real.cos_neg, Real.sin_neg]
  ring

/-- **The skip of equal streams, second coordinate of a pair**: `h_f S_s + h_s C_s`. -/
theorem phaseMix_const_snd (H : Matrix ι ι ℝ) (φ : ι → ℝ) (h : EucSpace d) (s : ι)
    (k : Fin (d / 2)) :
    phaseMix (headPhaseAction d) H φ (fun _ => h) s (GPTMini.ropeSnd d k) =
      h (GPTMini.ropeFst d k) * phaseSin H φ s + h (GPTMini.ropeSnd d k) * phaseCos H φ s := by
  change headPhase d (φ s) (weightedSum (H s) fun t => headPhase d (-φ t) h) _ = _
  simp only [headPhase_fst, headPhase_snd, weightedSum, WithLp.ofLp_sum, Finset.sum_apply,
    PiLp.smul_apply, smul_eq_mul, phaseCos, phaseSin, Finset.mul_sum, Finset.sum_mul,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Real.cos_sub, Real.sin_sub, Real.cos_neg, Real.sin_neg]
  ring

/-- For a symmetric `H` the imaginary parts sum to zero: `sin` is odd. -/
theorem sum_phaseSin_eq_zero {H : Matrix ι ι ℝ} (hH : ∀ s t, H s t = H t s) (φ : ι → ℝ) :
    ∑ s, phaseSin H φ s = 0 := by
  have h : ∑ s, phaseSin H φ s = -∑ s, phaseSin H φ s := by
    simp only [phaseSin]
    conv_lhs => rw [Finset.sum_comm]
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [hH s t, Real.sin_sub, Real.sin_sub]
    ring
  linarith

/-- **A symmetric skip whose real parts sum to zero hides equal streams**: the streams after it sum
to zero, at every even width. -/
theorem sum_phaseMix_const_eq_zero (hd : Even d) {H : Matrix ι ι ℝ} (hH : ∀ s t, H s t = H t s)
    {φ : ι → ℝ} (hC : ∑ s, phaseCos H φ s = 0) (h : EucSpace d) :
    ∑ s, phaseMix (headPhaseAction d) H φ (fun _ => h) s = 0 := by
  ext i
  have hi := (GPTMini.ropeSplit d).symm_apply_apply i
  rcases hs : GPTMini.ropeSplit d i with (k | k) | j <;> rw [hs] at hi <;> subst hi
  · rw [GPTMini.ropeSplit_symm_inl_inl]
    simp only [WithLp.ofLp_sum, Finset.sum_apply, phaseMix_const_fst, Finset.sum_sub_distrib,
      ← Finset.mul_sum, hC, sum_phaseSin_eq_zero hH, mul_zero, sub_zero, PiLp.zero_apply]
  · rw [GPTMini.ropeSplit_symm_inl_inr]
    simp only [WithLp.ofLp_sum, Finset.sum_apply, phaseMix_const_snd, Finset.sum_add_distrib,
      ← Finset.mul_sum, hC, sum_phaseSin_eq_zero hH, mul_zero, add_zero, PiLp.zero_apply]
  · exact absurd j.isLt (by have := Nat.two_mul_div_two_of_even hd; omega)

/-- **The hidden streams are kept**: stream `s` after the skip is not zero when `h ≠ 0` and
`C_s ≠ 0`, at every even width; on every pair it is `h` scaled by `|a_s| ≥ |C_s|`. -/
theorem phaseMix_const_ne_zero (hd : Even d) {H : Matrix ι ι ℝ} {φ : ι → ℝ} {s : ι}
    (hC : phaseCos H φ s ≠ 0) {h : EucSpace d} (hh : h ≠ 0) :
    phaseMix (headPhaseAction d) H φ (fun _ => h) s ≠ 0 := by
  intro h0
  have hpos : phaseCos H φ s ^ 2 + phaseSin H φ s ^ 2 ≠ 0 := by positivity
  have hpair : ∀ k, h (GPTMini.ropeFst d k) = 0 ∧ h (GPTMini.ropeSnd d k) = 0 := fun k => by
    have e1 := phaseMix_const_fst H φ h s k
    have e2 := phaseMix_const_snd H φ h s k
    rw [h0, PiLp.zero_apply] at e1 e2
    refine ⟨(mul_eq_zero.mp ?_).resolve_left hpos, (mul_eq_zero.mp ?_).resolve_left hpos⟩
    · linear_combination (-phaseCos H φ s) * e1 - phaseSin H φ s * e2
    · linear_combination phaseSin H φ s * e1 - phaseCos H φ s * e2
  apply hh
  ext i
  have hi := (GPTMini.ropeSplit d).symm_apply_apply i
  rcases hs : GPTMini.ropeSplit d i with (k | k) | j <;> rw [hs] at hi <;> subst hi
  · rw [GPTMini.ropeSplit_symm_inl_inl, (hpair k).1, PiLp.zero_apply]
  · rw [GPTMini.ropeSplit_symm_inl_inr, (hpair k).2, PiLp.zero_apply]
  · exact absurd j.isLt (by have := Nat.two_mul_div_two_of_even hd; omega)

/-- **Unequal gates see what equal gates miss**: gates equal to `γ` off stream `0` read `γ` times
the sum of the streams plus `g₀ − γ` times stream `0`. -/
theorem weightedSum_eq_sum_add {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (g : Fin 4 → ℝ) {γ : ℝ} (hg : ∀ s, s ≠ 0 → g s = γ) (y : Fin 4 → E) :
    weightedSum g y = γ • ∑ s, y s + (g 0 - γ) • y 0 := by
  simp only [weightedSum, Fin.sum_univ_four, hg 1 (by decide), hg 2 (by decide),
    hg 3 (by decide), smul_add, sub_smul]
  abel

/-- At main's width: gates `1/2` on stream `0` and `1/4` elsewhere. -/
example (y : Fin 4 → EucSpace 128) :
    weightedSum ![1 / 2, 1 / 4, 1 / 4, 1 / 4] y =
      (1 / 4 : ℝ) • ∑ s, y s + (1 / 2 - 1 / 4 : ℝ) • y 0 :=
  weightedSum_eq_sum_add _ (fun s hs => by fin_cases s <;> simp_all) y

end Transformer.Phasor

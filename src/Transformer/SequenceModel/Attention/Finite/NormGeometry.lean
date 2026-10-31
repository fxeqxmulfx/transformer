module

public import Transformer.SequenceModel.Attention.Finite.Compiler

/-!
# Positive-epsilon normalization retains finite context information

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F3.
Unlike zero-epsilon normalization, the source RMSNorm is injective. A fixed
positive output scale can make every context in a finite table small enough
for the next normalized residual branch to reach its answer code.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped BigOperators

/-- F3: the exact norm, including positive epsilon, determines the radius. -/
theorem rms_norm_sq (d : ℕ) (hd : 0 < d) {eps : ℝ} (heps : 0 < eps) (x : EucSpace d) :
    ‖rmsNormEps eps x‖ ^ 2 = (d : ℝ) * ‖x‖ ^ 2 / (‖x‖ ^ 2 + d * eps) := by
  have hn : 0 ≤ ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  simp only [rmsNormEps, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, div_pow,
    Real.sq_sqrt hn, Real.sq_sqrt (Nat.cast_nonneg d)]
  ring

/-- F3: the actual positive-epsilon RMSNorm loses no exact vector information.
This statement would be false for the ideal zero-epsilon normalization. -/
theorem rms_injective (d : ℕ) (hd : 0 < d) {eps : ℝ} (heps : 0 < eps) :
    Function.Injective (rmsNormEps eps : EucSpace d → EucSpace d) := by
  intro x y h
  have hd' : (0 : ℝ) < d := Nat.cast_pos.mpr hd
  have hx : 0 < ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  have hy : 0 < ‖y‖ ^ 2 + (d : ℝ) * eps := by positivity
  have hn := congrArg (fun v : EucSpace d => ‖v‖ ^ 2) h
  rw [rms_norm_sq d hd heps x, rms_norm_sq d hd heps y] at hn
  have hcross := (div_eq_div_iff hx.ne' hy.ne').mp hn
  have hnorm : ‖x‖ ^ 2 = ‖y‖ ^ 2 := by
    have hz : (d : ℝ) ^ 2 * eps * (‖x‖ ^ 2 - ‖y‖ ^ 2) = 0 := by
      linear_combination hcross
    exact sub_eq_zero.mp ((mul_eq_zero.mp hz).resolve_left
      (mul_ne_zero (pow_ne_zero 2 hd'.ne') heps.ne'))
  rw [rms_of_normSq d eps x _ rfl, rms_of_normSq d eps y _ rfl, ← hnorm] at h
  have hs := normScale_pos d hd eps heps (‖x‖ ^ 2) (sq_nonneg _)
  have hc := congrArg (fun v : EucSpace d => (normScale d eps (‖x‖ ^ 2))⁻¹ • v) h
  simpa only [smul_smul, inv_mul_cancel₀ hs.ne', one_smul] using hc

/-- F3: one strictly positive projection scale puts every normalized
context of a finite table inside the unit ball. No input-dependent scale
is installed as a trainable parameter. -/
theorem exists_small_normalized_context {A : Type*} [Fintype A]
    (d : ℕ) (hd : 0 < d) {eps : ℝ} (heps : 0 < eps) (v : A → EucSpace d) :
    ∃ τ : ℝ, 0 < τ ∧ ∀ a, ‖rmsNormEps eps (τ • v a)‖ < 1 := by
  let M := (∑ a, ‖v a‖) + 1
  have hm : 0 < M := by
    dsimp only [M]
    positivity
  have hv (a : A) : ‖v a‖ < M := by
    have h := Finset.single_le_sum (fun b (_ : b ∈ Finset.univ) => norm_nonneg (v b))
      (Finset.mem_univ a)
    dsimp only [M]
    linarith
  have he : 0 < Real.sqrt eps := Real.sqrt_pos.mpr heps
  let τ := Real.sqrt eps / (2 * M)
  have hτ : 0 < τ := div_pos he (by positivity)
  refine ⟨τ, hτ, ?_⟩
  intro a
  have h := rmsNormEps_lipschitz eps heps hd (τ • v a) 0
  have hz : rmsNormEps eps (0 : EucSpace d) = 0 := by simp [rmsNormEps]
  rw [hz, sub_zero, sub_zero, norm_smul, Real.norm_eq_abs, abs_of_pos hτ] at h
  have hc : 2 / Real.sqrt eps * (τ * ‖v a‖) = ‖v a‖ / M := by
    dsimp only [τ]
    field_simp
  rw [hc] at h
  exact lt_of_le_of_lt h ((div_lt_one hm).mpr (hv a))

/-- F3: unit input/output codes and a context of norm below one give a
displacement of norm below three, within every source width above nine. -/
theorem reachable_unit_answer {d : ℕ} (hw : 9 < d)
    (base answer context : EucSpace d) (hb : ‖base‖ = 1) (ha : ‖answer‖ = 1)
    (hc : ‖context‖ < 1) : ‖answer - (base + context)‖ ^ 2 < (d : ℝ) := by
  have h := norm_sub_le answer (base + context)
  have h' := norm_add_le base context
  rw [hb] at h'
  rw [ha] at h
  have hn : ‖answer - (base + context)‖ < 3 := by linarith
  have hw' : (9 : ℝ) < d := by exact_mod_cast hw
  nlinarith [norm_nonneg (answer - (base + context))]

example := rms_norm_sq 4 (by decide) (eps := 1 / 8388608) (by norm_num)
  (WithLp.toLp 2 ![1, -2, 3, 4])
example := rms_injective 4 (by decide) (eps := 1 / 8388608) (by norm_num)
example := exists_small_normalized_context 2 (by decide) (eps := 1 / 8388608) (by norm_num)
  (fun a : Fin 7 => WithLp.toLp 2 ![(a.val : ℝ), 100 - a.val])
example := reachable_unit_answer (d := 10) (by decide)
  (EuclideanSpace.single 0 1) (EuclideanSpace.single 1 1) (0 : EucSpace 10)
  (by simp) (by simp) (by simp)

end Transformer.SequenceModel.Attention.Finite

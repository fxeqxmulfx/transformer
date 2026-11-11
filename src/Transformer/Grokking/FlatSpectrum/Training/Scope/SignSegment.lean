module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSegmentCore
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSet

/-!
# The `2 × 2` kernel of signs: the trained products are a segment

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE16. For the
kernel of signs `K = [[1, 1], [−1, 1]]` with the code `±1`, every cell seen once and a weight
decay `0 < λ ≤ √2`, `SignSet` describes the trained products by their real parts and their
nuclear norm. Here the description is solved. With `J = [[1, −1], [1, 1]]` (`signRot`) and
`t = t(λ)`,

    X is a trained product   ⟺   X = t K + i a J  for a real `a` with `|a| ≤ t`

(`isNuclearMin_signKernel_iff`): a segment with the midpoint `t K`, whose ends `t (K ∓ i J)`
have rank one (`signTwin`, `signCotwin`).

"⟸": `t K + i a J` is a convex combination of the two ends, each of at most the nuclear norm
of `t K` (`nuclearNorm_add_smul_signRot_le`). "⟹" is `exists_eq_add_smul_signRot` of
`SignSegmentCore`.

The run of XE3 from stacked factors of full row rank ends at a trained fit, so the product of
its limit is a point of the segment (`exists_tendsto_decayRun_signKernel`): the limit is
determined up to the one real number `a`. From real factors `a = 0` (XE11). Which point a
start that is not real reaches is not determined here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

/-- The matrix `K + i J = [[1 + i, 1 − i], [−(1 − i), 1 + i]]`, the conjugate of `signTwin`. -/
def signCotwin : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1 + Complex.I, 1 - Complex.I; -(1 - Complex.I), 1 + Complex.I]

/-- The conjugate twin has at most the nuclear norm of the kernel of signs: it has rank one,
`(1, i)ᵀ (1 + i, 1 − i)`, and the squared norm `8`. -/
lemma nuclearNorm_signCotwin_le : nuclearNorm signCotwin ≤ nuclearNorm signKernel := by
  have h1 := nuclearNorm_sq_le_rank_mul_frobeniusSq signCotwin
  have h2 : (signCotwin.rank : ℝ) ≤ 1 := by
    have e : signCotwin = vecMulVec ![1, Complex.I] ![1 + Complex.I, 1 - Complex.I] := by
      ext i j
      fin_cases i <;> fin_cases j <;> simp [signCotwin, vecMulVec_apply]
      · linear_combination -Complex.I_mul_I
      · linear_combination Complex.I_mul_I
    rw [e]
    exact_mod_cast rank_vecMulVec_le _ _
  have h3 : frobeniusSq signCotwin = 8 := by
    simp [frobeniusSq, Fin.sum_univ_two, signCotwin, Complex.sq_norm, Complex.normSq_apply]
    norm_num
  rw [h3] at h1
  rw [nuclearNorm_signKernel]
  refine le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_
  rw [mul_pow, Real.sq_sqrt zero_le_two]
  nlinarith

/-- The points of the segment have at most the nuclear norm of `t K`. -/
lemma nuclearNorm_add_smul_signRot_le {t a : ℝ} (ha : |a| ≤ t) :
    nuclearNorm (t • signKernel + ((a : ℂ) * Complex.I) • signRot) ≤ t * (2 * √2) := by
  have h1 : 0 ≤ (t + a) / 2 := by linarith [neg_abs_le a]
  have h2 : 0 ≤ (t - a) / 2 := by linarith [le_abs_self a]
  have e : t • signKernel + ((a : ℂ) * Complex.I) • signRot =
      ((t + a) / 2) • signCotwin + ((t - a) / 2) • signTwin := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [signKernel, signRot, signCotwin, signTwin, Complex.ext_iff] <;>
      constructor <;> ring
  rw [e]
  calc nuclearNorm (((t + a) / 2) • signCotwin + ((t - a) / 2) • signTwin)
      ≤ nuclearNorm (((t + a) / 2) • signCotwin) + nuclearNorm (((t - a) / 2) • signTwin) :=
        nuclearNorm_add_le _ _
    _ = (t + a) / 2 * nuclearNorm signCotwin + (t - a) / 2 * nuclearNorm signTwin := by
        rw [nuclearNorm_real_smul h1, nuclearNorm_real_smul h2]
    _ ≤ (t + a) / 2 * (2 * √2) + (t - a) / 2 * (2 * √2) :=
        add_le_add
          (mul_le_mul_of_nonneg_left (nuclearNorm_signCotwin_le.trans_eq nuclearNorm_signKernel)
            h1)
          (mul_le_mul_of_nonneg_left (nuclearNorm_signTwin_le.trans_eq nuclearNorm_signKernel)
            h2)
    _ = t * (2 * √2) := by ring

variable {lam : ℝ} {X : Matrix (Fin 2) (Fin 2) ℂ}

/-- **The trained products of the `2 × 2` kernel of signs are a segment**
(`review/flat_spectrum.md`, item XE16): with the code `±1`, every cell seen once and
`0 < λ ≤ √2`, a matrix is a trained product exactly when it is `t(λ) K + i a J` for a real
`a` with `|a| ≤ t(λ)`. -/
theorem isNuclearMin_signKernel_iff (hlam : 0 < lam) (hle : lam ≤ √2) :
    IsNuclearMin (ceLoss (codeProbe signCode id) signLabel) lam X ↔
      ∃ a : ℝ, |a| ≤ signScale 2 lam ∧
        X = signScale 2 lam • signKernel + ((a : ℂ) * Complex.I) • signRot := by
  have ht := signScale_nonneg hlam hle
  rw [isNuclearMin_signCode_iff Function.bijective_id signKernel_apply two_pos isFlat_signKernel
    hlam hle, nuclearNorm_signKernel]
  refine ⟨fun h => ?_, ?_⟩
  · have r00 := h.1 0 0
    have r01 := h.1 0 1
    have r10 := h.1 1 0
    have r11 := h.1 1 1
    simp only [signKernel, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_fin_one, Complex.one_re, Complex.neg_re, mul_one,
      mul_neg] at r00 r01 r10 r11
    exact exists_eq_add_smul_signRot ht r00 r01 r10 r11 h.2
  · rintro ⟨a, ha, rfl⟩
    refine ⟨fun i j => ?_, nuclearNorm_add_smul_signRot_le ha⟩
    fin_cases i <;> fin_cases j <;> simp [signKernel, signRot]

/-- **The run ends on the segment** (`review/flat_spectrum.md`, items XE3 and XE16): for
`0 < λ ≤ √2`, under the bounds of XE3 with `a = 1`, `N = 4` and `|C| = 2`, the run from a start
of full row rank converges, and the product of its limit is `t(λ) K + i a J` for a real `a`
with `|a| ≤ t(λ)`. No convergence is assumed. -/
theorem exists_tendsto_decayRun_signKernel {k : Type*} [Fintype k] {η ρ : ℝ} (hη : 0 < η)
    (hlam : 0 < lam) (hle : lam ≤ √2) (hρ : 16 * Real.log 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + 64) ≤ lam) (hdesc : η * (lam + 8 + 8 * ρ) ≤ 1)
    {U₀ V₀ : Matrix (Fin 2) k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U V : Matrix (Fin 2) k ℂ) (a : ℝ),
      Tendsto (decayRun (ceGrad (codeProbe signCode id) signLabel) η lam U₀ V₀) atTop
        (𝓝 (U, V)) ∧ |a| ≤ signScale 2 lam ∧
        U * Vᴴ = signScale 2 lam • signKernel + ((a : ℂ) * Complex.I) • signRot := by
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin
    (A := codeProbe signCode id) (y := signLabel) (a := 1) (ρ := ρ) hη hlam zero_le_one
    (fun s c => by rw [frobeniusSq_codeProbe, norm_signCode])
    (by rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_bool]; push_cast; linarith)
    (by rw [Fintype.card_prod, Fintype.card_fin]; push_cast; linarith)
    (by rw [Fintype.card_prod, Fintype.card_fin]; push_cast; linarith) hw hfull
  obtain ⟨a, ha, hX⟩ := (isNuclearMin_signKernel_iff hlam hle).1 (hmin.isNuclearMin hlam.le)
  exact ⟨U, V, a, hlim, ha, hX⟩

/-! The witnesses: the weight decay `1 ≤ √2`, the midpoint `a = 0` and the end `a = t(1)`; the
run from `U₀ = [1 0]`, `V₀ = [0 1]` of width four with the step `1/200` and `ρ = 16`. -/

example := nuclearNorm_add_smul_signRot_le (t := 1) (a := 1) (by simp)

example : IsNuclearMin (ceLoss (codeProbe signCode id) signLabel) 1
    (signScale 2 1 • signKernel + ((signScale 2 1 : ℂ) * Complex.I) • signRot) :=
  (isNuclearMin_signKernel_iff one_pos one_le_sqrt_two).2
    ⟨signScale 2 1, (abs_of_nonneg (signScale_nonneg one_pos one_le_sqrt_two)).le, rfl⟩

example : ∃ a : ℝ, |a| ≤ signScale 2 1 ∧ signScale 2 1 • signKernel =
    signScale 2 1 • signKernel + ((a : ℂ) * Complex.I) • signRot :=
  (isNuclearMin_signKernel_iff one_pos one_le_sqrt_two).1
    (isNuclearMin_signScale_smul Function.bijective_id signKernel_apply two_pos isFlat_signKernel
      one_pos one_le_sqrt_two)

example := exists_tendsto_decayRun_signKernel (lam := 1) (η := 1 / 200) (ρ := 16) (by norm_num)
  one_pos one_le_sqrt_two (by nlinarith [Real.log_two_lt_d9]) (by norm_num) (by norm_num)
  (U₀ := fromCols (1 : Matrix (Fin 2) (Fin 2) ℂ) 0) (V₀ := fromCols 0 1)
  (by
    rw [← frobeniusSq_stack, stack_fromCols_one]
    norm_num [frobeniusSq, Matrix.one_apply])
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope

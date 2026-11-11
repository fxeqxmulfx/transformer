module

public import Transformer.Grokking.FlatSpectrum.Training.SignDirection
public import Transformer.Grokking.FlatSpectrum.Training.SignReal

/-!
# The code `±1`: all trained products for one weight decay

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"The direction when several products of the unit ball have the largest margin": "For factors
that are not real only the real parts of the trained product are determined", and item XE16.

Let a flat kernel `K`, `Kᴴ K = c · 1` or `K Kᴴ = c · 1`, show the labels of the code `±1`,
with every cell seen once, and let `0 < λ ≤ √c`. XE11 gives one trained product, `t(λ) K`
with `t(λ) = ½ log(2 √c / λ − 1)`, and the real parts of all of them. This file gives all of
them (`isNuclearMin_signCode_iff`):

    X is a trained product   ⟺   Re X = t(λ) K  and  ‖X‖_S1 ≤ t(λ) ‖K‖_S1.

The loss reads only the real parts, so a matrix with the real parts of `t(λ) K` has its loss,
and it is a trained product exactly when it has at most its nuclear norm. Every trained
product then has exactly that norm (`nuclearNorm_eq_signScale_mul_of_isNuclearMin`). The set is
convex, it contains `t(λ) K`, and by the first statement the imaginary parts of its members
are what the weight decay leaves free.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]
  {cell : S → m × n} {y : S → Bool} {K X : Matrix m n ℂ} {c₀ lam : ℝ}

/-- **All trained products of the code `±1` for one weight decay** (`review/flat_spectrum.md`,
item XE16): for a flat kernel that shows the labels, with every cell seen once and
`0 < λ ≤ √c`, a matrix is a trained product exactly when it has the real parts of `t(λ) K`
and at most its nuclear norm. -/
theorem isNuclearMin_signCode_iff (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hlam : 0 < lam) (hle : lam ≤ √c₀) :
    IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam X ↔
      (∀ i j, (X i j).re = signScale c₀ lam * (K i j).re) ∧
        nuclearNorm X ≤ signScale c₀ lam * nuclearNorm K := by
  have ht := signScale_nonneg hlam hle
  have hmin := isNuclearMin_signScale_smul (y := y) hcell hK hc hflat hlam hle
  have hre : (∀ i j, (X i j).re = signScale c₀ lam * (K i j).re) →
      ∀ i j, (X i j).re = ((signScale c₀ lam • K) i j).re := fun h i j => by
    rw [h, Matrix.smul_apply, Complex.smul_re, smul_eq_mul]
  refine ⟨fun hX => ?_, fun h => ?_⟩
  · have h1 := fun i j => hX.re_eq_signScale_mul hcell hK hc hflat hlam hle i j
    have h2 := hX (signScale c₀ lam • K)
    rw [ceLoss_signCode_congr (hre h1), nuclearNorm_real_smul ht] at h2
    exact ⟨h1, le_of_mul_le_mul_left (by linarith) hlam⟩
  · exact hmin.of_le hlam.le (ceLoss_signCode_congr (hre h.1))
      (by rw [nuclearNorm_real_smul ht]; exact h.2)

/-- **Every trained product has the nuclear norm of `t(λ) K`** (`review/flat_spectrum.md`,
item XE16). -/
theorem nuclearNorm_eq_signScale_mul_of_isNuclearMin (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hlam : 0 < lam) (hle : lam ≤ √c₀)
    (hX : IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam X) :
    nuclearNorm X = signScale c₀ lam * nuclearNorm K := by
  have ht := signScale_nonneg hlam hle
  obtain ⟨h1, h2⟩ := (isNuclearMin_signCode_iff hcell hK hc hflat hlam hle).1 hX
  refine le_antisymm h2 ?_
  have h3 := isNuclearMin_signScale_smul (y := y) hcell hK hc hflat hlam hle X
  rw [ceLoss_signCode_congr (X' := X) (X := signScale c₀ lam • K) (fun i j => by
    rw [h1, Matrix.smul_apply, Complex.smul_re, smul_eq_mul]), nuclearNorm_real_smul ht] at h3
  exact le_of_mul_le_mul_left (by linarith) hlam

/-! The witnesses: the `2 × 2` kernel of signs with all four cells seen once, `c = 2`, at the
weight decay `1 ≤ √2`. The matrix `t(1) signTwin` has the real parts of `t(1) K` and at most
its nuclear norm, so it is a trained product, and it is not real. -/

example : IsNuclearMin (ceLoss (codeProbe signCode id) signLabel) 1
    (signScale 2 1 • signTwin) :=
  (isNuclearMin_signCode_iff Function.bijective_id signKernel_apply two_pos isFlat_signKernel
    one_pos one_le_sqrt_two).2
    ⟨fun i j => by rw [Matrix.smul_apply, Complex.smul_re, smul_eq_mul, signTwin_re],
      by
        rw [nuclearNorm_real_smul (signScale_nonneg one_pos one_le_sqrt_two)]
        exact mul_le_mul_of_nonneg_left nuclearNorm_signTwin_le
          (signScale_nonneg one_pos one_le_sqrt_two)⟩

example := nuclearNorm_eq_signScale_mul_of_isNuclearMin Function.bijective_id signKernel_apply
  two_pos isFlat_signKernel one_pos one_le_sqrt_two
  (isNuclearMin_signScale_smul (y := signLabel) Function.bijective_id signKernel_apply two_pos
    isFlat_signKernel one_pos one_le_sqrt_two)

end Transformer.Grokking.FlatSpectrum.Training.Scope

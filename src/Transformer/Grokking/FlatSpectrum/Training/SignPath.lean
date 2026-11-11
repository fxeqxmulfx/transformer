module

public import Transformer.Grokking.FlatSpectrum.Training.MirroredUnique
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin

/-!
# The code `±1`: the multiples of a flat kernel of signs are trained products

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE11 and XE13.
The code `±1` is mirrored at the gap `g = 2`, the value that statement 4 of item XE11 leaves
out. Its logits read only the real parts of the product (`pairing_codeProbe_signCode`), so the
loss and the margin do too (`ceLoss_signCode_congr`, `margin_signCode_congr`).

Let every cell be seen once and let the flat kernel `K`, of squared singular value `c`, show
the labels. At `t K` the softmax weights of a sample differ by `1 − β`, `β = 2 / (1 + e^{2t})`
(`probability_signCode_sub`), so `⟨G(t K), H⟩ = −β ⟨K, H⟩` (`pairing_ceGrad_smul_signCode`).
As `⟨K, Y⟩ ≤ √c ‖Y‖_S1` with equality at `K`, the first-order inequality gives, for `t ≥ 0`,

    t K is a trained product for the weight decay λ = 2 √c / (1 + e^{2t})

(`isNuclearMin_smul_signCode`, by the certificate `isNuclearMin_of_pairing`). So is `t K'` for
every `K'` with the real parts of `K` and at most its nuclear norm
(`isNuclearMin_smul_of_re_eq`, by `IsNuclearMin.of_le`). The direction of `t K` is that of `K`
(`inv_nuclearNorm_smul_smul`).

The witness is the `1 × 1` kernel `1` on its cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Geometry (probability)
open scoped ComplexConjugate

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]
  {cell : S → m × n} {y : S → Bool} {K K' X X' : Matrix m n ℂ} {c₀ t lam : ℝ}
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}

omit [DecidableEq m] [Fintype S] in
/-- **A certificate of a trained product** (`review/flat_spectrum.md`, items WD2 and XE11): for
a convex loss, `X` minimizes `L + λ ‖·‖_S1` when its gradient pairs with every `Y` to at least
`−λ ‖Y‖_S1`, with equality at `X`. -/
theorem isNuclearMin_of_pairing (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hY : ∀ Y, -(lam * nuclearNorm Y) ≤ pairing (G X) Y)
    (hX : pairing (G X) X = -(lam * nuclearNorm X)) : IsNuclearMin L lam X := fun Y => by
  have h := hconv X Y
  rw [pairing_sub_right, hX] at h
  linarith [hY Y]

omit [DecidableEq m] [Fintype S] in
/-- A product with the loss of a trained product and at most its nuclear norm is a trained
product. -/
theorem IsNuclearMin.of_le (hlam : 0 ≤ lam) (h : IsNuclearMin L lam X) (hL : L X' = L X)
    (hn : nuclearNorm X' ≤ nuclearNorm X) : IsNuclearMin L lam X' := fun Y =>
  le_trans (by rw [hL]; gcongr) (h Y)

omit [DecidableEq m] in
/-- The direction of a positive multiple is the direction of the product. -/
lemma inv_nuclearNorm_smul_smul (ht : 0 < t) (X : Matrix m n ℂ) :
    (nuclearNorm (t • X))⁻¹ • t • X = (nuclearNorm X)⁻¹ • X := by
  rw [nuclearNorm_real_smul ht.le, smul_smul, mul_inv, mul_comm t⁻¹, mul_assoc,
    inv_mul_cancel₀ ht.ne', mul_one]

omit [Fintype S] in
/-- The logits of the code `±1` read the real part of the entry. -/
lemma pairing_codeProbe_signCode (s : S) (c : Bool) (X : Matrix m n ℂ) :
    pairing (codeProbe signCode cell s c) X
      = (signCode c).re * (X (cell s).1 (cell s).2).re := by
  rw [pairing_codeProbe]
  cases c <;> simp [signCode]

/-- The cross-entropy of the code `±1` reads only the real parts of the product. -/
lemma ceLoss_signCode_congr (h : ∀ i j, (X' i j).re = (X i j).re) :
    ceLoss (codeProbe signCode cell) y X' = ceLoss (codeProbe signCode cell) y X := by
  simp only [ceLoss, pairing_codeProbe_signCode, h]

omit [Fintype S] in
/-- The margin of the code `±1` reads only the real parts of the product. -/
lemma margin_signCode_congr (h : ∀ i j, (X' i j).re = (X i j).re) :
    margin (codeProbe signCode cell) y X' = margin (codeProbe signCode cell) y X := by
  simp only [margin, gap, pairing_codeProbe_signCode, h]

/-- The two softmax weights at the logits `±t` of the code `±1`, the label `b` having the
logit `t`, differ by `±(1 − 2 / (1 + e^{2t}))`. -/
lemma probability_signCode_sub (b : Bool) (t : ℝ) :
    probability (fun c : Bool => (signCode c).re * ((signCode b).re * t)) true
        - probability (fun c : Bool => (signCode c).re * ((signCode b).re * t)) false
      = (signCode b).re * (1 - 2 / (1 + Real.exp (2 * t))) := by
  have h := Real.exp_pos t
  cases b <;> simp only [probability, Fintype.sum_bool, signCode, ite_true, Bool.false_eq_true,
    ite_false, Complex.one_re, Complex.neg_re, one_mul, neg_mul, neg_neg] <;>
  · rw [Real.exp_neg, show 2 * t = t + t by ring, Real.exp_add]
    field_simp
    ring

/-- **The gradient at a multiple of a kernel of signs** (`review/flat_spectrum.md`, item
XE11): for the code `±1` and a kernel that shows the labels, `⟨G(t K), H⟩ = −β Σ_s
Re(conj K_s H_s)` with `β = 2 / (1 + e^{2t})`. -/
theorem pairing_ceGrad_smul_signCode (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s))
    (t : ℝ) (H : Matrix m n ℂ) :
    pairing (ceGrad (codeProbe signCode cell) y (t • K)) H
      = -(2 / (1 + Real.exp (2 * t))) *
        ∑ s, (conj (K (cell s).1 (cell s).2) * H (cell s).1 (cell s).2).re := by
  rw [pairing_ceGrad, Finset.mul_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  have hz : (fun c => pairing (codeProbe signCode cell s c) (t • K))
      = fun c => (signCode c).re * ((signCode (y s)).re * t) := by
    funext c
    rw [pairing_codeProbe_signCode, Matrix.smul_apply, hK s, Complex.smul_re, smul_eq_mul,
      mul_comm t]
  have hp := probability_signCode_sub (y s) t
  rw [hz, Fintype.sum_bool, hK s, ← pairing_codeProbe signCode cell s (y s) H]
  simp only [pairing_codeProbe_signCode]
  have h1 : (signCode true).re = 1 := by simp [signCode]
  have h0 : (signCode false).re = -1 := by simp [signCode]
  rw [h1, h0]
  linear_combination (H (cell s).1 (cell s).2).re * hp

/-- **For the code `±1` the multiples of a flat kernel of signs are trained products**
(`review/flat_spectrum.md`, item XE11): with every cell seen once and `t ≥ 0`, `t K` minimizes
`L + λ ‖·‖_S1` for `λ = 2 √c / (1 + e^{2t})`. -/
theorem isNuclearMin_smul_signCode (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (ht : 0 ≤ t) :
    IsNuclearMin (ceLoss (codeProbe signCode cell) y) (2 * √c₀ / (1 + Real.exp (2 * t)))
      (t • K) := by
  have hβ : 0 < 2 / (1 + Real.exp (2 * t)) := by positivity
  have hlam : 2 * √c₀ / (1 + Real.exp (2 * t)) = 2 / (1 + Real.exp (2 * t)) * √c₀ := by ring
  have hfn : frobeniusSq K = √c₀ * nuclearNorm K := by
    rw [hflat.frobeniusSq_eq hc.ne', hflat.nuclearNorm_eq hc.ne', mul_left_comm,
      Real.mul_self_sqrt hc.le]
  have hpair : ∀ H, pairing (ceGrad (codeProbe signCode cell) y (t • K)) H
      = -(2 / (1 + Real.exp (2 * t))) * pairing K H := fun H => by
    rw [pairing_ceGrad_smul_signCode hK, pairing_eq_sum_re, ← Fintype.sum_prod_type']
    exact congrArg _ (hcell.sum_comp fun p : m × n => (conj (K p.1 p.2) * H p.1 p.2).re)
  refine isNuclearMin_of_pairing (ceLoss_first_order _ _) (fun Y => ?_) ?_
  · rw [hpair, hlam]
    nlinarith [hflat.pairing_le hc Y]
  · rw [hpair, hlam, pairing_smul_right, pairing_self, nuclearNorm_real_smul ht, hfn]
    ring

/-- **So are the multiples of a matrix with the real parts of the kernel and at most its
nuclear norm** (`review/flat_spectrum.md`, item XE11): the loss of the code `±1` does not read
the imaginary parts. -/
theorem isNuclearMin_smul_of_re_eq (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hre : ∀ i j, (K' i j).re = (K i j).re) (hle : nuclearNorm K' ≤ nuclearNorm K)
    (ht : 0 ≤ t) :
    IsNuclearMin (ceLoss (codeProbe signCode cell) y) (2 * √c₀ / (1 + Real.exp (2 * t)))
      (t • K') :=
  (isNuclearMin_smul_signCode hcell hK hc hflat ht).of_le (by positivity)
    (ceLoss_signCode_congr fun i j => by
      rw [Matrix.smul_apply, Matrix.smul_apply, Complex.smul_re, Complex.smul_re, hre])
    (by rw [nuclearNorm_real_smul ht, nuclearNorm_real_smul ht]; gcongr)

/-! The witness: the `1 × 1` kernel `1` on its cell, with the label of the codeword `1`. Its
multiple `1 • 1` is a trained product for the weight decay `2 √1 / (1 + e²)`. -/

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))

example := inv_nuclearNorm_smul_smul one_pos (1 : Matrix (Fin 1) (Fin 1) ℂ)

/-- The certificate at the witness, with the gradient of the cross-entropy. -/
example : IsNuclearMin (ceLoss (codeProbe signCode cell₀) fun _ => true)
    (2 / (1 + Real.exp (2 * 1))) ((1 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) := by
  have hβ : (0 : ℝ) < 2 / (1 + Real.exp (2 * 1)) := by positivity
  have hp := pairing_ceGrad_smul_signCode (cell := cell₀) (y := fun _ => true) (K := 1)
    (fun _ => by simp [signCode]) 1
  simp only [Finset.univ_unique, Finset.sum_singleton, Matrix.one_apply_eq, map_one,
    one_mul] at hp
  refine isNuclearMin_of_pairing (ceLoss_first_order _ _) (fun Y => ?_) ?_
  · rw [hp]
    nlinarith [(Complex.re_le_norm (Y 0 0)).trans (norm_apply_le_nuclearNorm Y 0 0)]
  · rw [hp, one_smul, nuclearNorm_one_fin_one, Matrix.one_apply_eq, Complex.one_re]
    ring

/-- The same from the statements above, and through a matrix with the real parts of `1`. -/
example := (isNuclearMin_smul_signCode (cell := cell₀) (y := fun _ => true) (K := 1)
  ⟨fun _ _ _ => Subsingleton.elim _ _, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩
  (fun _ => by simp [signCode]) one_pos isFlat_one zero_le_one).of_le (by positivity) rfl le_rfl

example := isNuclearMin_smul_of_re_eq (cell := cell₀) (y := fun _ => true) (K := 1) (K' := 1)
  ⟨fun _ _ _ => Subsingleton.elim _ _, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩
  (fun _ => by simp [signCode]) one_pos isFlat_one (fun _ _ => rfl) le_rfl zero_le_one

end Transformer.Grokking.FlatSpectrum.Training

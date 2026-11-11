module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Training.Basic

/-!
# The nuclear norm of a `2 × 2` matrix from below

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE15, its tool.
A `2 × 2` matrix with singular values `σ₁`, `σ₂` has `(σ₁ + σ₂)² = σ₁² + σ₂² + 2 σ₁ σ₂`, that
is `‖Z‖_S1² = ‖Z‖_S2² + 2 |det Z|`. This file proves the inequality that XE15 uses,

    ‖Z‖_S2² + 2 |det Z| ≤ ‖Z‖_S1²

(`frobeniusSq_add_two_mul_norm_det_le`), by a certificate. Let `ω` be a unit number with
`conj ω · det Z = |det Z|` (`exists_unit_conj_mul`) and

    C = Z + ω adj(Z)ᴴ = [[a + ω conj d, b − ω conj c], [c − ω conj b, d + ω conj a]].

Then `Cᴴ C = (‖Z‖_S2² + 2 |det Z|) · 1` (`isFlat_detCertificate`): `C` is flat, and it pairs
with `Z` to the same number (`pairing_detCertificate`). The bound `⟨C, Z⟩ ≤ √c ‖Z‖_S1` of a
flat matrix (R3, `IsFlat.pairing_le`) gives the claim. With `|b| |c| − |a| |d| ≤ |det Z|`
(`norm_mul_sub_le_norm_det`) it yields the form that XE15 reads the margin from,

    (|a| − |d|)² + (|b| + |c|)² ≤ ‖Z‖_S1²

(`sq_sub_add_sq_add_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix
open scoped ComplexConjugate

/-- A unit number that turns `z` to its modulus: `conj ω · ω = 1` and `conj ω · z = |z|`. -/
lemma exists_unit_conj_mul (z : ℂ) : ∃ ω : ℂ, conj ω * ω = 1 ∧ conj ω * z = (‖z‖ : ℂ) := by
  by_cases hz : z = 0
  · exact ⟨1, by simp, by simp [hz]⟩
  · have hn : (‖z‖ : ℂ) ≠ 0 := by exact_mod_cast (norm_pos_iff.2 hz).ne'
    refine ⟨z / ‖z‖, ?_, ?_⟩
    · rw [map_div₀, Complex.conj_ofReal, div_mul_div_comm, Complex.conj_mul']
      field_simp
    · rw [map_div₀, Complex.conj_ofReal, div_mul_eq_mul_div, Complex.conj_mul']
      field_simp

/-- The certificate `Z + ω adj(Z)ᴴ` of a `2 × 2` matrix with the entries `a`, `b`, `c`, `d`. -/
def detCertificate (ω a b c d : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![a + ω * conj d, b - ω * conj c; c - ω * conj b, d + ω * conj a]

/-- The squared norm of `[[a, b], [c, d]]`. -/
lemma frobeniusSq_fin_two (a b c d : ℂ) :
    frobeniusSq !![a, b; c, d] = ‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 + ‖d‖ ^ 2 := by
  simp [frobeniusSq, Fin.sum_univ_two]
  ring

/-- **The certificate is flat**, with the squared singular value `‖Z‖_S2² + 2 |det Z|`. -/
lemma isFlat_detCertificate {ω a b c d : ℂ} (h1 : conj ω * ω = 1)
    (h2 : conj ω * (a * d - b * c) = (‖a * d - b * c‖ : ℂ)) :
    IsFlat (detCertificate ω a b c d)
      (‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 + ‖d‖ ^ 2 + 2 * ‖a * d - b * c‖) := by
  have h3 : ω * conj (a * d - b * c) = (‖a * d - b * c‖ : ℂ) := by
    have h := congrArg conj h2
    rwa [map_mul, Complex.conj_conj, Complex.conj_ofReal] at h
  have ha := Complex.conj_mul' a
  have hb := Complex.conj_mul' b
  have hc := Complex.conj_mul' c
  have hd := Complex.conj_mul' d
  simp only [map_sub, map_mul] at h3
  refine Or.inl ?_
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp only [detCertificate, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply,
      Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one, Fin.zero_eta, Fin.mk_one, Matrix.smul_apply,
      Matrix.one_apply_eq, Matrix.one_apply_ne, ne_eq, zero_ne_one, one_ne_zero,
      not_false_eq_true, smul_eq_mul, mul_one, mul_zero, Complex.star_def, map_add, map_sub,
      map_mul, Complex.conj_conj, Complex.ofReal_add, Complex.ofReal_mul, Complex.ofReal_pow,
      Complex.ofReal_ofNat]
  · linear_combination (d * conj d + b * conj b) * h1 + h3 + h2 + ha + hb + hc + hd
  · linear_combination (-(conj a * b + conj c * d)) * h1
  · linear_combination (-(conj b * a + conj d * c)) * h1
  · linear_combination (c * conj c + a * conj a) * h1 + h3 + h2 + ha + hb + hc + hd

/-- **The certificate pairs with the matrix to `‖Z‖_S2² + 2 |det Z|`.** -/
lemma pairing_detCertificate {ω a b c d : ℂ}
    (h2 : conj ω * (a * d - b * c) = (‖a * d - b * c‖ : ℂ)) :
    pairing (detCertificate ω a b c d) !![a, b; c, d] =
      ‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 + ‖d‖ ^ 2 + 2 * ‖a * d - b * c‖ := by
  have ha := Complex.conj_mul' a
  have hb := Complex.conj_mul' b
  have hc := Complex.conj_mul' c
  have hd := Complex.conj_mul' d
  have key : ((detCertificate ω a b c d)ᴴ * !![a, b; c, d]).trace =
      ((‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 + ‖d‖ ^ 2 + 2 * ‖a * d - b * c‖ : ℝ) : ℂ) := by
    simp only [detCertificate, Matrix.trace, Matrix.diag, Matrix.mul_apply, Fin.sum_univ_two,
      conjTranspose_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_fin_one, Complex.star_def, map_add, map_sub, map_mul,
      Complex.conj_conj, Complex.ofReal_add, Complex.ofReal_mul, Complex.ofReal_pow,
      Complex.ofReal_ofNat]
    linear_combination 2 * h2 + ha + hb + hc + hd
  rw [pairing, key, Complex.ofReal_re]

/-- **The nuclear norm of a `2 × 2` matrix from below**
(`review/flat_spectrum.md`, item XE15): `‖Z‖_S2² + 2 |det Z| ≤ ‖Z‖_S1²`. -/
theorem frobeniusSq_add_two_mul_norm_det_le (Z : Matrix (Fin 2) (Fin 2) ℂ) :
    frobeniusSq Z + 2 * ‖Z.det‖ ≤ nuclearNorm Z ^ 2 := by
  obtain ⟨ω, h1, h2⟩ := exists_unit_conj_mul (Z 0 0 * Z 1 1 - Z 0 1 * Z 1 0)
  have hZ : Z = !![Z 0 0, Z 0 1; Z 1 0, Z 1 1] := Matrix.eta_fin_two Z
  have hF : frobeniusSq Z = ‖Z 0 0‖ ^ 2 + ‖Z 0 1‖ ^ 2 + ‖Z 1 0‖ ^ 2 + ‖Z 1 1‖ ^ 2 := by
    conv_lhs => rw [hZ]
    exact frobeniusSq_fin_two _ _ _ _
  rw [Matrix.det_fin_two, hF]
  set s := ‖Z 0 0‖ ^ 2 + ‖Z 0 1‖ ^ 2 + ‖Z 1 0‖ ^ 2 + ‖Z 1 1‖ ^ 2 +
    2 * ‖Z 0 0 * Z 1 1 - Z 0 1 * Z 1 0‖ with hs
  have hs0 : 0 ≤ s := by positivity
  rcases hs0.eq_or_lt with h0 | hpos
  · rw [← h0]
    positivity
  have h := (isFlat_detCertificate h1 h2).pairing_le hpos Z
  have hp : pairing (detCertificate ω (Z 0 0) (Z 0 1) (Z 1 0) (Z 1 1)) Z = s := by
    conv_lhs => rw [hZ]
    exact pairing_detCertificate h2
  rw [hp] at h
  have hroot : √s ≤ nuclearNorm Z := by
    have hr := Real.sqrt_pos.2 hpos
    have hsq := Real.sq_sqrt hs0
    by_contra hlt
    nlinarith [not_le.1 hlt]
  calc s = √s ^ 2 := (Real.sq_sqrt hs0).symm
    _ ≤ nuclearNorm Z ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg s) hroot 2

/-- The entries of a `2 × 2` matrix against its determinant: `|b| |c| − |a| |d| ≤ |det|`. -/
lemma norm_mul_sub_le_norm_det (Z : Matrix (Fin 2) (Fin 2) ℂ) :
    ‖Z 0 1‖ * ‖Z 1 0‖ - ‖Z 0 0‖ * ‖Z 1 1‖ ≤ ‖Z.det‖ := by
  rw [Matrix.det_fin_two, ← norm_mul, ← norm_mul, norm_sub_rev]
  exact norm_sub_norm_le _ _

/-- `(|a| − |d|)² + (|b| + |c|)² ≤ ‖Z‖_S1²` for a `2 × 2` matrix. -/
lemma sq_sub_add_sq_add_le (Z : Matrix (Fin 2) (Fin 2) ℂ) :
    (‖Z 0 0‖ - ‖Z 1 1‖) ^ 2 + (‖Z 0 1‖ + ‖Z 1 0‖) ^ 2 ≤ nuclearNorm Z ^ 2 := by
  have h := frobeniusSq_add_two_mul_norm_det_le Z
  have hdet := norm_mul_sub_le_norm_det Z
  have hF : frobeniusSq Z = ‖Z 0 0‖ ^ 2 + ‖Z 0 1‖ ^ 2 + ‖Z 1 0‖ ^ 2 + ‖Z 1 1‖ ^ 2 := by
    conv_lhs => rw [Matrix.eta_fin_two Z]
    exact frobeniusSq_fin_two _ _ _ _
  nlinarith

/-- A complex number of modulus at most `r` whose real part is at least `r` is `r`. -/
lemma eq_ofReal_of_norm_le_of_le_re {z : ℂ} {r : ℝ} (h1 : ‖z‖ ≤ r) (h2 : r ≤ z.re) :
    z = (r : ℂ) := by
  have h3 := Complex.re_le_norm z
  have h4 := Complex.sq_norm z
  rw [Complex.normSq_apply] at h4
  have hre : z.re = r := le_antisymm (h3.trans h1) h2
  have hn : ‖z‖ = r := le_antisymm h1 (h2.trans h3)
  have him : z.im = 0 := by
    rw [hn, hre] at h4
    nlinarith
  exact Complex.ext (by simpa using hre) (by simpa using him)

/-! The witnesses: the identity, with `‖1‖_S2² + 2 |det 1| = 4`. -/

example := frobeniusSq_add_two_mul_norm_det_le (1 : Matrix (Fin 2) (Fin 2) ℂ)

example := isFlat_detCertificate (ω := 1) (a := 1) (b := 0) (c := 0) (d := 1) (by simp)
  (by simp)

example := pairing_detCertificate (ω := 1) (a := 1) (b := 0) (c := 0) (d := 1) (by simp)

example := eq_ofReal_of_norm_le_of_le_re (z := 1) (r := 1) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum.Training.Scope

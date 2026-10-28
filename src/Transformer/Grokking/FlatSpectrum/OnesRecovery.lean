module

public import Transformer.Grokking.FlatSpectrum.OnesCertificate
public import Transformer.Grokking.FlatSpectrum.Interpolation

/-!
# Exact recovery of the all-ones matrix

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let `Y`
vanish off `Ω`, have unit row and column sums, and let `c (Y − Q)` be a contraction. Pairing
`J + H` with the contraction `Q + Π Z₀ Π`, where `Z₀` attains `‖Π H Π‖_S1`, gives
`tr = N + tr (Q H) + ‖Π H Π‖_S1`. For `H` vanishing on `Ω`, `tr (Yᴴ H) = 0` and
`Π (Y − Q)ᴴ Π = (Y − Q)ᴴ`, so `tr (Q H) = − tr ((Y − Q)ᴴ Π H Π)`, of modulus at most
`‖Π H Π‖_S1 / c`. Hence `N + (1 − 1/c) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1`
(`le_nuclearNorm_onesMatrix_add`). For `c > 1` a minimum has `Π H Π = 0`, so
`H = Q H + Π H Q` has entries `r_i + s_j`, and a rigid `Ω` forces `H = 0`
(`isMinOn_onesMatrix_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **The certificate bound**, `review/flat_spectrum.md`, *Exact recovery of a rank-one
kernel*: if `Y` vanishes off `Ω`, has unit row and column sums and `c (Y − Q)` is a
contraction, `c > 0`, then `N + (1 − 1/c) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1` for every `H` vanishing
on `Ω`, where `Q = J / N` and `Π = 1 − Q`. -/
theorem le_nuclearNorm_onesMatrix_add [Nonempty n] {Ω : Set (n × n)} {Y : Matrix n n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0) (hrow : ∀ i, ∑ j, Y i j = 1)
    (hcol : ∀ j, ∑ i, Y i j = 1) {c : ℝ} (hc : 0 < c)
    (hW : IsContraction ((c : ℂ) • (Y - meanProj n))) {H : Matrix n n ℂ}
    (hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0) :
    Fintype.card n + (1 - c⁻¹) * nuclearNorm ((1 - meanProj n) * H * (1 - meanProj n)) ≤
      nuclearNorm (onesMatrix n + H) := by
  have hQ : (meanProj n)ᴴ = meanProj n := conjTranspose_meanProj
  have hQQ : meanProj n * meanProj n = meanProj n := meanProj_mul_meanProj
  have hQJ : meanProj n * onesMatrix n = onesMatrix n := meanProj_mul_onesMatrix
  set Q := meanProj n with hQdef
  set A := (1 - Q) * H * (1 - Q) with hAdef
  obtain ⟨Z₀, hZ₀, htr⟩ := exists_isContraction_trace_eq A
  have hP : (1 - Q)ᴴ = 1 - Q := by rw [conjTranspose_sub, conjTranspose_one, hQ]
  have hPJ : (1 - Q) * onesMatrix n = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hQJ, sub_self]
  have htrJ : (onesMatrix n).trace = Fintype.card n := by simp [onesMatrix, trace]
  -- the pairing of `J + H` with the contraction `Q + Π Z₀ Π`
  have h1 : ((Q + (1 - Q) * Z₀ * (1 - Q))ᴴ * (onesMatrix n + H)).trace =
      Fintype.card n + (Q * H).trace + nuclearNorm A := by
    have h0 : ((1 - Q) * (Z₀ᴴ * (1 - Q)) * onesMatrix n).trace = 0 := by
      rw [Matrix.mul_assoc, Matrix.mul_assoc, hPJ, Matrix.mul_zero, Matrix.mul_zero, trace_zero]
    have hA : ((1 - Q) * (Z₀ᴴ * (1 - Q)) * H).trace = (Z₀ᴴ * A).trace := by
      rw [Matrix.mul_assoc, Matrix.trace_mul_comm, hAdef]
      simp only [Matrix.mul_assoc]
    rw [conjTranspose_add, conjTranspose_mul, conjTranspose_mul, hQ, hP, Matrix.add_mul,
      Matrix.mul_add, Matrix.mul_add, trace_add, trace_add, trace_add, hQJ, htrJ, h0, hA, htr]
    ring
  -- `tr (Q H) = − tr ((Y − Q)ᴴ A)`
  have hQYh : Q * Yᴴ = Q := by
    have h := congrArg conjTranspose (mul_meanProj hrow)
    rwa [conjTranspose_mul, ← hQdef, hQ] at h
  have hYhQ : Yᴴ * Q = Q := by
    have h := congrArg conjTranspose (meanProj_mul hcol)
    rwa [conjTranspose_mul, ← hQdef, hQ] at h
  have hPW : (1 - Q) * ((Y - Q)ᴴ * (1 - Q)) = (Y - Q)ᴴ := by
    rw [conjTranspose_sub, hQ]
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_one, hQYh, hYhQ, hQQ]
    abel
  have hYH : (Yᴴ * H).trace = 0 := by
    rw [trace_conjTranspose_mul]
    refine sum_eq_zero fun i _ => sum_eq_zero fun j _ => ?_
    by_cases hij : (i, j) ∈ Ω
    · rw [hH (i, j) hij, mul_zero]
    · rw [hYΩ i j hij, star_zero, zero_mul]
  have h2 : (Q * H).trace = -((Y - Q)ᴴ * A).trace := by
    have e1 : (Y - Q)ᴴ * A = (Y - Q)ᴴ * ((1 - Q) * H) * (1 - Q) := by
      simp only [hAdef, Matrix.mul_assoc]
    have e2 : (1 - Q) * ((Y - Q)ᴴ * ((1 - Q) * H)) = (Y - Q)ᴴ * H := by
      calc (1 - Q) * ((Y - Q)ᴴ * ((1 - Q) * H)) = (1 - Q) * ((Y - Q)ᴴ * (1 - Q)) * H := by
            simp only [Matrix.mul_assoc]
        _ = (Y - Q)ᴴ * H := by rw [hPW]
    rw [e1, Matrix.trace_mul_comm ((Y - Q)ᴴ * ((1 - Q) * H)), e2, conjTranspose_sub, hQ,
      Matrix.sub_mul, trace_sub, hYH, zero_sub, neg_neg]
  -- `|tr ((Y − Q)ᴴ A)| ≤ ‖A‖_S1 / c`
  have h3 : ‖((Y - Q)ᴴ * A).trace‖ ≤ c⁻¹ * nuclearNorm A := by
    have h := norm_trace_le_nuclearNorm hW A
    rw [conjTranspose_smul, Complex.star_def, Complex.conj_ofReal, Matrix.smul_mul, trace_smul,
      smul_eq_mul, norm_mul, Complex.norm_real, Real.norm_of_nonneg hc.le] at h
    rwa [le_inv_mul_iff₀ hc]
  have h4 := norm_trace_le_nuclearNorm (hZ₀.proj_add hQ hQQ) (onesMatrix n + H)
  rw [h1, h2] at h4
  have h5 := Complex.re_le_norm
    ((Fintype.card n : ℂ) + -((Y - Q)ᴴ * A).trace + (nuclearNorm A : ℂ))
  have h6 := Complex.re_le_norm ((Y - Q)ᴴ * A).trace
  simp only [Complex.add_re, Complex.neg_re, Complex.natCast_re, Complex.ofReal_re] at h5
  nlinarith

example := le_nuclearNorm_onesMatrix_add (n := Fin 2) (Ω := Set.univ) (Y := meanProj (Fin 2))
  (fun _ _ h => absurd (Set.mem_univ _) h) sum_meanProj
  (fun j => by rw [← sum_meanProj j]; exact sum_congr rfl fun _ _ => rfl) one_pos
  (by rw [sub_self, smul_zero]; exact isContraction_zero) (H := 0) fun _ _ => rfl

/-- **Exact recovery of `J`**, `review/flat_spectrum.md`, *Exact recovery of a rank-one
kernel*: with a certificate `Y` as in `le_nuclearNorm_onesMatrix_add` for some `c > 1`, and
`Ω` rigid — `r_i + s_j = 0` on `Ω` only if everywhere — `J` is the unique minimum of the
nuclear norm among the matrices equal to `J` on `Ω`. -/
theorem isMinOn_onesMatrix_iff [Nonempty n] {Ω : Set (n × n)} {Y : Matrix n n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → Y i j = 0) (hrow : ∀ i, ∑ j, Y i j = 1)
    (hcol : ∀ j, ∑ i, Y i j = 1) {c : ℝ} (hc : 1 < c)
    (hW : IsContraction ((c : ℂ) • (Y - meanProj n)))
    (hΩ : ∀ r s : n → ℂ, (∀ ij ∈ Ω, r ij.1 + s ij.2 = 0) → ∀ i j, r i + s j = 0)
    {X : Matrix n n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2} X ↔
      X = onesMatrix n := by
  have hlow (X' : Matrix n n ℂ) (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
      Fintype.card n + (1 - c⁻¹) * nuclearNorm ((1 - meanProj n) * (X' - onesMatrix n) *
        (1 - meanProj n)) ≤ nuclearNorm X' := by
    have h := le_nuclearNorm_onesMatrix_add hYΩ hrow hcol (zero_lt_one.trans hc) hW
      (H := X' - onesMatrix n) fun ij hij => by rw [Matrix.sub_apply, hX' ij hij, sub_self]
    rwa [add_sub_cancel] at h
  have hc1 : 0 < 1 - c⁻¹ := sub_pos.mpr (inv_lt_one_of_one_lt₀ hc)
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h1 := (hlow X hX).trans ((hmin _ fun _ _ => rfl).trans nuclearNorm_onesMatrix.le)
    set Q := meanProj n
    set H := X - onesMatrix n with hHdef
    have h0 := nuclearNorm_nonneg ((1 - Q) * H * (1 - Q))
    have hA := eq_zero_of_nuclearNorm_eq_zero (le_antisymm (by nlinarith) h0)
    have hdec : H = Q * H + (1 - Q) * H * Q + (1 - Q) * H * (1 - Q) := by
      simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one]
      abel
    rw [hA, add_zero] at hdec
    obtain ⟨i₀⟩ := ‹Nonempty n›
    have hrs (i j : n) : H i j = ((1 - Q) * H * Q) i i₀ + (Q * H) i₀ j := by
      conv_lhs => rw [hdec]
      rw [Matrix.add_apply, add_comm, mul_meanProj_apply _ i j i₀, meanProj_mul_apply _ i i₀ j]
    have hzero := hΩ (fun i => ((1 - Q) * H * Q) i i₀) (fun j => (Q * H) i₀ j)
      fun ij hij => by rw [← hrs, hHdef, Matrix.sub_apply, hX ij hij, sub_self]
    ext i j
    have h := hrs i j
    rwa [hzero i j, hHdef, Matrix.sub_apply, sub_eq_zero] at h
  · intro hXJ X' hX'
    rw [hXJ, nuclearNorm_onesMatrix]
    have := nuclearNorm_nonneg ((1 - meanProj n) * (X' - onesMatrix n) * (1 - meanProj n))
    nlinarith [hlow X' hX']

example := (isMinOn_onesMatrix_iff (n := Fin 2) (Ω := Set.univ) (Y := meanProj (Fin 2))
  (fun _ _ h => absurd (Set.mem_univ _) h) sum_meanProj
  (fun j => by rw [← sum_meanProj j]; exact sum_congr rfl fun _ _ => rfl) one_lt_two
  (by rw [sub_self, smul_zero]; exact isContraction_zero)
  (fun _ _ h i j => h (i, j) (Set.mem_univ _)) (X := onesMatrix (Fin 2)) fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum

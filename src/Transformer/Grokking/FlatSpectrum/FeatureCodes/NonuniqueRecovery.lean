module

public import Transformer.Grokking.FlatSpectrum.RankCertificate

/-!
# Correct classification without unique completion

Source: `review/flat_spectrum.md`, *Classification without unique recovery*, items 1–2.
A strict dual certificate places every optimal completion in the target's tangent
space. Unique recovery asks the observations to distinguish every tangent direction.
Here only a real linear readout, or every correct-versus-incorrect score contrast,
must annihilate the tangent directions invisible to the observations. Thus all
global minima classify correctly even when their matrices and scores differ.

The certificate, global optimality and readout condition remain hypotheses. This
does not assert that a transformer or its optimizer satisfies them. Concrete
nonunique witnesses for both theorems are in `NonuniqueWitness`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r]
  [DecidableEq m] [DecidableEq n] [DecidableEq r]

/-- **Recovery of a linear readout**, source item 1. The readout need only vanish on
unobserved tangent perturbations; the perturbations themselves may be nonzero. -/
theorem linear_readout_eq_of_certificate {K : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hY : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (L : Matrix m n ℂ →ₗ[ℝ] ℝ)
    (hL : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → L (U * Aᴴ + B * Vᴴ) = 0)
    {X : Matrix m n ℂ} (hfit : ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = K ij.1 ij.2} X) :
    L X = L K := by
  let H := X - K
  have hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0 := by
    intro ij hij
    simp only [H, Matrix.sub_apply, hfit ij hij, sub_self]
  have hbound := le_nuclearNorm_add_of_certificate hU hV hKU hK hY hWU hWV
    (zero_lt_one.trans hc) hW hH
  have hbest := (isMinOn_iff.mp hmin) K (fun _ _ => rfl)
  have hpositive : 0 < 1 - c⁻¹ := sub_pos.mpr (inv_lt_one_of_one_lt₀ hc)
  have hnonneg := nuclearNorm_nonneg ((1 - U * Uᴴ) * H * (1 - V * Vᴴ))
  have hKH : K + H = X := by dsimp [H]; abel
  rw [hKH] at hbound
  have hnormal : (1 - U * Uᴴ) * H * (1 - V * Vᴴ) = 0 := by
    apply eq_zero_of_nuclearNorm_eq_zero
    nlinarith
  have hdecomp : H = U * (Hᴴ * U)ᴴ + ((1 - U * Uᴴ) * H * V) * Vᴴ := by
    have hsplit : H = U * Uᴴ * H + (1 - U * Uᴴ) * H * (V * Vᴴ) +
        (1 - U * Uᴴ) * H * (1 - V * Vᴴ) := by
      simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one]
      abel
    rw [hnormal, add_zero] at hsplit
    rw [conjTranspose_mul, conjTranspose_conjTranspose]
    simpa only [Matrix.mul_assoc] using hsplit
  have hzero := hL (Hᴴ * U) ((1 - U * Uᴴ) * H * V)
    (fun ij hij => by rw [← hdecomp]; exact hH ij hij)
  rw [← hdecomp] at hzero
  change L (X - K) = 0 at hzero
  rw [L.map_sub] at hzero
  exact sub_eq_zero.mp hzero

/-- **Every global minimum predicts the correct label**, source item 2. The known target
has a strict correct-class gap. Unobserved tangent directions must preserve only those
gaps, permitting a common change in all class scores and nonunique feature matrices. -/
theorem classification_of_minimum_without_unique_recovery {K : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hY : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    {Q A : Type*} (score : Q → A → Matrix m n ℂ →ₗ[ℝ] ℝ) (answer : Q → A)
    (hgap : ∀ q a, a ≠ answer q → score q a K < score q (answer q) K)
    (hcontrast : ∀ (P : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Pᴴ + B * Vᴴ) ij.1 ij.2 = 0) →
      ∀ q a, (score q (answer q) - score q a) (U * Pᴴ + B * Vᴴ) = 0)
    {X : Matrix m n ℂ} (hfit : ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = K ij.1 ij.2} X) :
    ∀ q a, a ≠ answer q → score q a X < score q (answer q) X := by
  intro q a ha
  have heq := linear_readout_eq_of_certificate hU hV hKU hK hY hWU hWV hc hW
    (score q (answer q) - score q a) (fun P B h => hcontrast P B h q a) hfit hmin
  simp only [LinearMap.sub_apply] at heq
  have hpositive := hgap q a ha
  linarith

end Transformer.Grokking.FlatSpectrum.FeatureCodes

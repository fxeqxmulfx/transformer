module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Conditioning
public import Transformer.Grokking.FlatSpectrum.RankCertificate

/-!
# Approximate recovery of readouts without a unique matrix optimum

Source: `review/flat_spectrum.md`, *Approximate nonunique recovery*, QR2–QR3.
Let `E` correct the observed fitting error, with `‖E‖_* ≤ ρ`, and let
`‖X‖_* ≤ ‖K‖_* + ε`. A strict certificate with slack `s = 1 - 1/c > 0`
bounds the normal part of `X - E - K` by `(ε + ρ)/s`. A readout conditioned
by `κ` on hidden perturbations and globally bounded by `λ` therefore has
error at most `κ(ε + ρ)/s + λρ`. A finite `κ` follows from annihilating hidden
tangent directions (`exists_readout_conditioning`); full recovery is unnecessary.
The objective gap is a measured or proved premise, not an optimizer guarantee.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r]
  [DecidableEq m] [DecidableEq n] [DecidableEq r]

/-- **Normal error from noisy fitting and a finite objective gap**, QR2.
`E` is any correction that makes the observations exact; choosing the masked
residual supplies the usual noisy-observation interpretation. -/
theorem normal_error_le_of_approximate_minimum {K : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hsupport : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    {X E : Matrix m n ℂ} {ε ρ : ℝ}
    (hfit : ∀ ij ∈ Ω, (X - E) ij.1 ij.2 = K ij.1 ij.2)
    (hobjective : nuclearNorm X ≤ nuclearNorm K + ε) (hnoise : nuclearNorm E ≤ ρ) :
    nuclearNorm ((1 - U * Uᴴ) * (X - E - K) * (1 - V * Vᴴ)) ≤
      (ε + ρ) / (1 - c⁻¹) := by
  have hhidden : ∀ ij ∈ Ω, (X - E - K) ij.1 ij.2 = 0 := by
    intro ij hij
    rw [Matrix.sub_apply, hfit ij hij, sub_self]
  have hcert := le_nuclearNorm_add_of_certificate hU hV hKU hK hsupport hWU hWV
    (zero_lt_one.trans hc) hW hhidden
  have hneg : nuclearNorm (-E) = nuclearNorm E := by
    simpa using nuclearNorm_smul (-1 : ℂ) E
  have htriangle := nuclearNorm_add_le X (-E)
  rw [hneg, ← sub_eq_add_neg] at htriangle
  rw [show K + (X - E - K) = X - E by abel] at hcert
  have hslack : 0 < 1 - c⁻¹ := sub_pos.mpr (inv_lt_one_of_one_lt₀ hc)
  apply (le_div_iff₀ hslack).mpr
  nlinarith

/-- **Readout error with explicit conditioning**, QR3. Only hidden tangent
directions need be harmless; the whole feature matrix may remain nonunique.
`κ` and `λ` can be tighter than the existential and explicit bounds of QR1. -/
theorem linear_readout_error_le_of_approximate_minimum {K : Matrix m n ℂ}
    {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K)
    {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hsupport : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0)
    (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (L : Matrix m n ℂ →ₗ[ℝ] ℝ) {κ ℓ : ℝ} (hκ : 0 ≤ κ) (hℓ : 0 ≤ ℓ)
    (hcondition : ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      |L H| ≤ κ * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)))
    (hbound : ∀ H, |L H| ≤ ℓ * nuclearNorm H)
    {X E : Matrix m n ℂ} {ε ρ : ℝ}
    (hfit : ∀ ij ∈ Ω, (X - E) ij.1 ij.2 = K ij.1 ij.2)
    (hobjective : nuclearNorm X ≤ nuclearNorm K + ε) (hnoise : nuclearNorm E ≤ ρ) :
    |L X - L K| ≤ κ * ((ε + ρ) / (1 - c⁻¹)) + ℓ * ρ := by
  have hn := normal_error_le_of_approximate_minimum hU hV hKU hK hsupport
    hWU hWV hc hW hfit hobjective hnoise
  have hh : ∀ ij ∈ Ω, (X - E - K) ij.1 ij.2 = 0 := by
    intro ij hij
    rw [Matrix.sub_apply, hfit ij hij, sub_self]
  have hhidden := (hcondition _ hh).trans (mul_le_mul_of_nonneg_left hn hκ)
  have he := (hbound E).trans (mul_le_mul_of_nonneg_left hnoise hℓ)
  have hsplit : L X - L K = L (X - E - K) + L E := by rw [map_sub, map_sub]; ring
  rw [hsplit]
  exact (abs_add_le _ _).trans (add_le_add hhidden he)

/-- **Each contrast's recovery and alignment errors add**, QR3. This lemma uses
contrast errors directly, so no extra factor two is introduced for alignment. -/
theorem contrast_error_le_of_alignment {actual recovered target α β : ℝ}
    (halign : |actual - recovered| ≤ α) (hrecover : |recovered - target| ≤ β) :
    |actual - target| ≤ α + β := by
  calc
    |actual - target| = |(actual - recovered) + (recovered - target)| := by ring_nf
    _ ≤ |actual - recovered| + |recovered - target| := abs_add_le _ _
    _ ≤ α + β := add_le_add halign hrecover

example := contrast_error_le_of_alignment (actual := 3) (recovered := 2) (target := 1)
  (α := 1) (β := 1) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.FeatureCodes

/-
# Homogenized Transformers — the drift of an overlap

Formalization of `lem:Ito_formula` of arXiv:2604.01978v1, *Homogenized
Transformers*, §5: the Itô decomposition of `R_ij(t) = ⟨x_i(t), x_j(t)⟩` along
`eq:Diffusive_gaussian_case`, and its explicit drift `eq:Dij_explicit_clean`.

**What the source says and what is written here.**  `eq:Rij_decomp_clean` is
`dR_ij = dM_ij + 𝒟_ij dt`.  Mathlib has no stochastic integral and
`Generator.IsItoSolution` is a solution concept in the one-dimensional
marginals, so neither `dR_ij` nor `dM_ij` can be written down.  What can be,
and what is the whole content of the lemma, is:

* the integrand of the martingale part `eq:martingale_Mij_clean` is the
  derivative of the observable along the diffusion kernel,
  `Dφ(X)[G(·,θ)] = ⟨G_i(θ), x_j⟩ + ⟨x_i, G_j(θ)⟩` — this is `fderiv_overlap`
  of `Transformer.Homogenized.OverlapObservable`, and it is proved;
* the drift `𝒟_ij` is the generator of `eq:Diffusive_gaussian_case` applied to
  `φ(X) = ⟨x_i, x_j⟩` — this is `sphGenerator_overlap_diffusive`
  (`eq:drift_Dij_clean`, proved) and `ito_formula_overlap`
  (`eq:Dij_explicit_clean`).

The scalars `s_μ(x)` and `s_μ(x,x')` of `eq:s_mu_defs_clean` are
`Barycenter.baryCorr`; `s_μ(x) = s_μ(x,x)` is `baryCorr_self`.
-/

import Transformer.Homogenized.OverlapObservable
import Transformer.Homogenized.GaussianKernel

open scoped BigOperators NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The drift along the diffusive equation -/

/-- **Equation (eq:drift_Dij_clean).**  The drift of `R_ij` along
`eq:Diffusive_gaussian_case` is

  `𝒟_ij = ∫_Θ ⟨G_i,G_j⟩ ρ*(dθ) - (R_ij/2) ∫_Θ (‖G_i‖² + ‖G_j‖²) ρ*(dθ)`.

The vanishing of the drift `b_{ρ*}` — which under (G) is `bField_gaussian`,
still unproved — is taken as the hypothesis `hb`, so that this identity is
proved outright and its dependence is legible in its signature.

Source: arXiv:2604.01978v1, `eq:drift_Dij_clean`. -/
theorem sphGenerator_overlap_diffusive {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (i j : Idx n) (hb : ∀ k : Idx n, bField β ρ X k = 0)
    (h1 : Integrable (fun θ => inner (𝕜 := ℝ) (Gfield β ρ X θ i) (Gfield β ρ X θ j)) ρ)
    (h2 : Integrable (fun θ => ‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2) ρ) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
        (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X
      = (∫ θ, inner (𝕜 := ℝ) (Gfield β ρ X θ i) (Gfield β ρ X θ j) ∂ρ)
          - inner (𝕜 := ℝ) (X i) (X j) / 2
            * ∫ θ, (‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2) ∂ρ := by
  rw [sphGenerator_overlap, hb i, hb j]
  have hsplit : ∫ θ, (2 * inner (𝕜 := ℝ) (Gfield β ρ X θ i) (Gfield β ρ X θ j)
        - inner (𝕜 := ℝ) (X i) (X j) * (‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2)) ∂ρ
      = 2 * ∫ θ, inner (𝕜 := ℝ) (Gfield β ρ X θ i) (Gfield β ρ X θ j) ∂ρ
        - inner (𝕜 := ℝ) (X i) (X j)
          * ∫ θ, (‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2) ∂ρ := by
    rw [integral_sub (h1.const_mul 2) (h2.const_mul _), integral_const_mul, integral_const_mul]
  simp only [noiseField_one_one, inner_zero_left, inner_zero_right, add_zero, zero_add]
  rw [hsplit]
  ring

/-- The hypotheses of `sphGenerator_overlap_diffusive` are satisfiable: at
`ρ* = δ_0` the drift and the diffusion kernel both vanish, and a constant
function is integrable against a Dirac mass. -/
example {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) :
    (∀ k : Idx n, bField β (Measure.dirac (0 : HeadParam d)) X k = 0) ∧
      Integrable (fun θ => inner (𝕜 := ℝ)
        (Gfield β (Measure.dirac (0 : HeadParam d)) X θ i)
        (Gfield β (Measure.dirac (0 : HeadParam d)) X θ j))
          (Measure.dirac (0 : HeadParam d)) ∧
      Integrable (fun θ => ‖Gfield β (Measure.dirac (0 : HeadParam d)) X θ i‖ ^ 2
          + ‖Gfield β (Measure.dirac (0 : HeadParam d)) X θ j‖ ^ 2)
          (Measure.dirac (0 : HeadParam d)) :=
  ⟨fun _ => by simp, integrable_dirac enorm_lt_top, integrable_dirac enorm_lt_top⟩

/-! ### The explicit drift under Gaussian initialization -/

/-- **Equation (eq:Dij_explicit_clean).**  The drift `𝒟_ij(X)` of the overlap
at Gaussian initialization:

  `𝒟_ij = (1/d)(d - 2 + R_ij²) s_μ(x_i,x_j) - ((d-1)/2d) R_ij (s_μ(x_i) + s_μ(x_j))`,

with `μ = μ_X` the empirical measure and `s_μ` the correlation `baryCorr` of
the softmax barycenters.  The two coefficients are the traces
`Tr(Proj_{x_i}Proj_{x_j}) = d - 2 + R_ij²` and `Tr(Proj_{x_i}) = d - 1` of
`eq:trace_proj_proj_clean`, divided by the `d` of the Gaussian identity
`E[VᵀMV] = (1/d)Tr(M) I_d`.

Source: arXiv:2604.01978v1, `eq:Dij_explicit_clean`. -/
noncomputable def overlapDrift {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (i j : Idx n) : ℝ :=
  (1 / (d : ℝ)) * ((d : ℝ) - 2 + inner (𝕜 := ℝ) (X i) (X j) ^ 2)
      * baryCorr β ρ (empMeasure X) (X i) (X j)
    - ((d : ℝ) - 1) / (2 * (d : ℝ)) * inner (𝕜 := ℝ) (X i) (X j)
      * (baryCorr β ρ (empMeasure X) (X i) (X i) + baryCorr β ρ (empMeasure X) (X j) (X j))

/-- The drift of `⟨x_i, x_i⟩` vanishes: `‖x_i(t)‖ ≡ 1`, which is what the
sphere-preserving correction of `eq:ambient_form_clean` is chosen for.  This is
the formula of `eq:Dij_explicit_clean` at `i = j`, where
`Tr(Proj_{x_i}Proj_{x_i}) = d - 2 + 1 = d - 1 = Tr(Proj_{x_i})`. -/
theorem overlapDrift_self {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (i : Idx n) (hX : ‖X i‖ = 1) :
    overlapDrift β ρ X i i = 0 := by
  have h : inner (𝕜 := ℝ) (X i) (X i) = 1 := by
    rw [real_inner_self_eq_norm_sq, hX, one_pow]
  rw [overlapDrift, h]
  ring

/-- The hypothesis of `overlapDrift_self` is satisfiable. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

/-- **Lemma (lem:Ito_formula).**  Along `eq:Diffusive_gaussian_case` the drift
of `R_ij(t) = ⟨x_i(t), x_j(t)⟩` is `eq:Dij_explicit_clean`: the generator of
the equation, applied to the observable `φ(X) = ⟨x_i, x_j⟩`, is
`overlapDrift`.

**What the source says and what is written here.**

* `eq:Rij_decomp_clean` splits `dR_ij` into a martingale and a drift.  Only the
  drift is a statement about the generator; the martingale part is identified
  separately, through its integrand, by `fderiv_overlap`.  See the module
  docstring.
* `σ_V² = 1/d` is an explicit hypothesis, as in `gaussian_drift_and_kernel`:
  the Gaussian identity `E[VᵀMV] = (1/d)Tr(M) I_d` the proof turns on is the
  one of `lem:lemma_app`, and it is that normalization of `σ_V` and no other.
* The source fixes `i, j ∈ [n]²`; the statement here is for all `i, j`,
  including `i = j`, where it degenerates correctly — `overlapDrift_self`.
* The tokens are unit vectors, which is where `Tr(Proj_{x_i}) = d - 1` and
  `eq:trace_proj_proj_clean` come from.

Not proved here.

Source: arXiv:2604.01978v1, `lem:Ito_formula`. -/
theorem ito_formula_overlap {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : ((σV : ℝ)) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k : Idx n, ‖X k‖ = 1) (i j : Idx n) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
        (fun y : Idx n → EucSpace d => inner (𝕜 := ℝ) (y i) (y j)) X
      = overlapDrift β ρ X i j := by
  sorry

/-- The hypotheses of `ito_formula_overlap` are satisfiable: the Gaussian
ensemble at the normalization `σ_V² = 1/d` of `lem:lemma_app`, on a
configuration frozen at the north pole of `𝕊^d`. -/
example (d n : ℕ) (σA : ℝ≥0) :
    IsGaussianHeadLaw (d + 1) (stdSigmaV (d + 1)) σA
        (gaussHeadLaw (d + 1) (stdSigmaV (d + 1)) σA) ∧
      ((stdSigmaV (d + 1) : ℝ)) ^ 2 = 1 / ((d + 1 : ℕ) : ℝ) ∧
      ∀ _k : Idx n, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  ⟨isGaussianHeadLaw_gaussHeadLaw _ _ _, stdSigmaV_sq _, fun _ => by
    simp [basePoint, PiLp.norm_single]⟩

end Homogenized
end Transformer

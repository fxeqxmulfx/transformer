/-
# Homogenized Transformers — the variance proxy at Gaussian initialization

Consequence (i) of assumption (G) of arXiv:2604.01978v1: the variance proxy
`ς`, the upper bound

  `ς² = max_X max_j E‖Proj_{x_j} ξ_θ[μ_X](x_j)‖²`

of `eq: defining.alpha`, is explicit, `ς² = σ_V²(d-1)`
(`varianceProxy_gaussian`), so that `eq:Alpha_Sec2` gives
`α = (η/H) σ_V²(d-1)` (`alphaOf_gaussian`).

The computation has three steps.

* The drift vanishes under (G) (`meanField_gaussian`), so `ξ_θ = B_θ`, and at a
  token the field is the value matrix applied to the attention average,
  `B_θ[μ_X](x_j) = V m_j(A)` with `m_j(A) = Σ_k π^A_{j→k} x_k`
  (`attnField_self_eq_valueMap`, in `Homogenized.AttnAverage`).  The attention
  average is the softmax barycenter `m_{β,A}[μ_X](x_j)` of `eq: mbetaA`
  (`softBary_empMeasure`).
* `V` is independent of `A`, with uncorrelated entries of variance `σ_V²`
  (`Homogenized.GaussianEntries`), so
  `E‖G(X,θ)_j‖² = σ_V²(d-1) E‖m_j(A)‖²` (`integral_norm_Gfield_sq_gaussian`,
  from `integral_norm_proj_toEuclideanLin_sq`).
* `m_j(A)` is a convex combination of unit vectors, so `‖m_j(A)‖ ≤ 1`, with
  equality when all tokens coincide: the maximum over configurations is
  `σ_V²(d-1)`, attained at a collapsed configuration.

Source: arXiv:2604.01978v1, §2.3.3, item (i); `eq:G_def`,
`eq: defining.alpha`, `eq:Alpha_Sec2`, and the proof of `lem:Ito_formula`.
-/

import Transformer.Homogenized.AttnAverage
import Transformer.Homogenized.GaussianEntries
import Transformer.Homogenized.GaussianMoments
import Transformer.Homogenized.GaussianDrift
import Transformer.Homogenized.SimplexWellDefined

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-! ### The second moment of `G` -/

/-- **`E‖G(X,θ)_j‖² = σ_V²(d-1) E‖m_j(A)‖²`.**  Under (G) the drift vanishes, so
`G(X,θ)_j = Proj_{x_j} V m_j(A)`; `V` is independent of `A`, with uncorrelated
entries of variance `σ_V²`, and the projection removes one of `d` directions.

Source: arXiv:2604.01978v1, §2.3.3, item (i) (the computation behind it), with
`G` of `eq:G_def`. -/
theorem integral_norm_Gfield_sq_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ i, ‖x i‖ = 1) (j : Idx n) :
    ∫ θ, ‖Gfield β ρ x θ j‖ ^ 2 ∂ρ =
      (σV : ℝ) ^ 2 * ((d : ℝ) - 1) * ∫ θ, ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2 ∂ρ := by
  have hG : ∀ θ, Gfield β ρ x θ j =
      proj d (x j) (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 x j k • x k)) := fun θ => by
    rw [Gfield, fluct, meanField_gaussian β hρ, sub_zero, attnField_self_eq_valueMap]
    rfl
  simp_rw [hG]
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', hind, hmV, -, -, rfl⟩ := hρ
  have hY := measurable_sum_attnProb_smul β x j
  have hA := measurable_qk hW hW'
  have hVA : Measurable fun ω => ((Vr ω, Wr ω * (Wr' ω).transpose) : HeadParam d) :=
    hV.prodMk hA
  have hn : Measurable fun θ : HeadParam d => ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2 :=
    (hY.comp measurable_snd).norm.pow_const 2
  have h1 : ∫ θ, ‖proj d (x j) (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 x j k • x k))‖ ^ 2
      ∂(P.map fun ω => ((Vr ω, Wr ω * (Wr' ω).transpose) : HeadParam d)) =
      ∫ ω, ‖proj d (x j) (Matrix.toEuclideanLin (Vr ω)
        (∑ k, attnProb β (Wr ω * (Wr' ω).transpose) x j k • x k))‖ ^ 2 ∂P :=
    integral_map hVA.aemeasurable (measurable_norm_proj_sq (x j) hY).aestronglyMeasurable
  have h2 : ∫ θ, ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2
      ∂(P.map fun ω => ((Vr ω, Wr ω * (Wr' ω).transpose) : HeadParam d)) =
      ∫ ω, ‖∑ k, attnProb β (Wr ω * (Wr' ω).transpose) x j k • x k‖ ^ 2 ∂P :=
    integral_map hVA.aemeasurable hn.aestronglyMeasurable
  have hYA : Measurable fun ω => ∑ k, attnProb β (Wr ω * (Wr' ω).transpose) x j k • x k :=
    hY.comp hA
  have hind' : IndepFun Vr (fun ω => ∑ k, attnProb β (Wr ω * (Wr' ω).transpose) x j k • x k) P :=
    (indepFun_value_qk hV hW hW' hind).comp measurable_id hY
  rw [h1, h2]
  exact integral_norm_proj_toEuclideanLin_sq hV hYA hind'
    (integral_value_mul_value hV hind hmV) (integrable_value_mul_value hV hmV)
    (fun ω => norm_sum_attnProb_smul_le β _ hx j) (hx j)

/-- The hypotheses of `integral_norm_Gfield_sq_gaussian` are satisfiable: the
degenerate law `ρ* = δ_0` and one token at `basePoint d`. -/
example (d : ℕ) :
    IsGaussianHeadLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      ∀ _i : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  ⟨isGaussianHeadLaw_dirac_zero _, fun _ => by simp [basePoint, PiLp.norm_single]⟩

/-! ### Consequence (i) -/

/-- **Consequence (i) of (G).**  Under `eq: tformers.at.initialization` the
variance proxy `ς` — the upper bound of `eq: defining.alpha`, which
`eq:Alpha_Sec2` turns into `α = ης²/H` — is explicit: `ς² = σ_V²(d-1)`.

The maximum over configurations is attained where all tokens coincide, since
there `m_j(A) = x_j` for every `A`.

The source states this consequence as `α = (η/H)σ_V²(d-1)`; dividing by the
definition of `α` is what isolates `ς`, and `alphaOf_gaussian` puts it back.

Source: arXiv:2604.01978v1, §2.3.3, item (i). -/
theorem varianceProxy_gaussian {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (s : ℝ) (hs : IsVarianceProxy d n β ρ s) :
    s ^ 2 = (σV : ℝ) ^ 2 * ((d : ℝ) - 1) := by
  obtain ⟨-, ⟨x, hx, j, hv⟩, hmax⟩ := hs
  have := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hd : (1 : ℝ) ≤ d := by
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · have h := hx j
      simp [EuclideanSpace.norm_eq] at h
    · exact_mod_cast hd
  have hc : 0 ≤ (σV : ℝ) ^ 2 * ((d : ℝ) - 1) := mul_nonneg (sq_nonneg _) (by linarith)
  have hle : ∫ θ, ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2 ∂ρ ≤ 1 := by
    have h1 : ∀ θ : HeadParam d, ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2 ≤ 1 := fun θ =>
      pow_le_one₀ (norm_nonneg _) (norm_sum_attnProb_smul_le β θ.2 hx j)
    have h2 : ∫ θ, ‖∑ k, attnProb β θ.2 x j k • x k‖ ^ 2 ∂ρ ≤ ∫ _θ, (1 : ℝ) ∂ρ :=
      integral_mono_of_nonneg (Filter.Eventually.of_forall fun θ => sq_nonneg _)
        (integrable_const 1) (Filter.Eventually.of_forall h1)
    simpa using h2
  refine le_antisymm ?_ ?_
  · rw [hv, integral_norm_Gfield_sq_gaussian β hρ hx j]
    calc _ ≤ (σV : ℝ) ^ 2 * ((d : ℝ) - 1) * 1 := mul_le_mul_of_nonneg_left hle hc
      _ = _ := mul_one _
  · obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by have := Fin.pos j; omega⟩
    have h : ∫ θ, ‖Gfield β ρ (fun _ => x j) θ j‖ ^ 2 ∂ρ ≤ s ^ 2 :=
      hmax ⟨fun _ => x j, fun _ => hx j, j, rfl⟩
    rw [integral_norm_Gfield_sq_gaussian β hρ (fun _ => hx j) j] at h
    simpa [sum_attnProb_smul_const, hx j] using h

/-- **Consequence (i) of (G), the scaling parameter.**  Under
`eq: tformers.at.initialization` the scaling parameter of `eq:Alpha_Sec2` is

  `α = (η/H) σ_V²(d-1)`,

which is what suggests the standard choice `σ_V² = 1/d`.

Source: arXiv:2604.01978v1, §2.3.3, item (i). -/
theorem alphaOf_gaussian {d n H : ℕ} (η β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (s : ℝ) (hs : IsVarianceProxy d n β ρ s) :
    alphaOf η s H = η * ((σV : ℝ) ^ 2 * ((d : ℝ) - 1)) / (H : ℝ) := by
  rw [alphaOf, varianceProxy_gaussian β σV σA ρ hρ s hs]

/-- The hypotheses of `varianceProxy_gaussian` and `alphaOf_gaussian` are
satisfiable at the degenerate law `ρ* = δ_0`, whose variance proxy is `0`. -/
example (d n : ℕ) (β : ℝ) :
    IsGaussianHeadLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      IsVarianceProxy (d + 1) (n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) 0 :=
  ⟨isGaussianHeadLaw_dirac_zero _,
    isVarianceProxy_dirac_zero d n β (fun _ => (basePoint d : EucSpace (d + 1)))
      (fun _ => by simp [basePoint, PiLp.norm_single])⟩

end Homogenized
end Transformer

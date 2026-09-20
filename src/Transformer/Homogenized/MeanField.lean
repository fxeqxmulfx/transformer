/-
# Homogenized Transformers — the fields at a general measure

The vocabulary of §3 of arXiv:2604.01978v1, *Homogenized Transformers*: the
attention field `EQ:VELOCITY_FIELD_SELF_ATTENTION`, the drift-fluctuation
splitting, and the kernel `eq:G_def`, all read at a general measure `μ` on
`ℝ^d` rather than at the empirical measure of a finite configuration, together
with the covariance kernel of `rem:shared_noise_rewrite` and the
cross-variation identity `eq:cross_variation_kernel` it is there for.

`Defs.lean` carries the same fields at the empirical measure `μ_X` of a
configuration `X ∈ (𝕊^{d-1})^n`, which is what the particle statements of §2
are written in.  The two agree — `attnFieldOf_empMeasure` and
`Gfield_eq_GfieldOf` identify them — so the mean-field statements of §3 are
written in the same fields as §2 and not in a parallel vocabulary.

Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`, `eq:G_def`,
`eq:covariance_kernel`, `rem:shared_noise_rewrite`.
-/

import Transformer.Homogenized.Generator
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.MeasureTheory.Function.L2Space

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- The orthogonal projection onto `T_x 𝕊^{d-1}` is self-adjoint:
`⟨Proj_x u, v⟩ = ⟨u, Proj_x v⟩`, both sides being `⟨u,v⟩ - ⟨x,u⟩⟨x,v⟩`.  This
is what lets `eq:cross_variation_kernel` move the two projections of
`Proj_x K[μ](x,y) Proj_y` onto the vectors they are tested against. -/
theorem real_inner_proj_left {d : ℕ} (x u v : EucSpace d) :
    inner (𝕜 := ℝ) (proj d x u) v = inner (𝕜 := ℝ) u (proj d x v) := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right]
  rw [real_inner_comm u x]
  ring

/-- The attention-induced velocity field at a general measure `μ` on `ℝ^d`:

  `B_θ[μ](x) = 𝒵_A[μ](x)⁻¹ ∫ e^{β⟨A x, y⟩} V y μ(dy)`,
  `𝒵_A[μ](x) = ∫ e^{β⟨A x, y⟩} μ(dy)`.

Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
noncomputable def attnFieldOf {d : ℕ} (β : ℝ) (θ : HeadParam d)
    (μ : Measure (EucSpace d)) (x : EucSpace d) : EucSpace d :=
  (∫ y, attnWeight β θ x y ∂μ)⁻¹ • ∫ y, attnWeight β θ x y • valueMap θ y ∂μ

/-- The mean field `b_{ρ*}[μ](x) = E_{θ∼ρ*} B_θ[μ](x)` at a general measure.

Source: arXiv:2604.01978v1, §2.2.1. -/
noncomputable def meanFieldOf {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x : EucSpace d) : EucSpace d :=
  ∫ θ, attnFieldOf β θ μ x ∂ρ

/-- The fluctuation `ξ_θ[μ](x) = B_θ[μ](x) - b_{ρ*}[μ](x)` at a general
measure.

Source: arXiv:2604.01978v1, §2.2.1. -/
noncomputable def fluctOf {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (θ : HeadParam d) (x : EucSpace d) : EucSpace d :=
  attnFieldOf β θ μ x - meanFieldOf β ρ μ x

/-- **Equation (eq:G_def) at a general measure.**  The diffusion kernel
`G_μ(x,θ) = Proj_x ξ_θ[μ](x)`.

Source: arXiv:2604.01978v1, `eq:G_def`. -/
noncomputable def GfieldOf {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x : EucSpace d) (θ : HeadParam d) : EucSpace d :=
  proj d x (fluctOf β ρ μ θ x)

/-- The empirical measure `μ_X = (1/n) Σ_k δ_{x_k}` of a configuration.

Source: arXiv:2604.01978v1, §2.1. -/
noncomputable def empMeasure {d n : ℕ} (x : Idx n → EucSpace d) : Measure (EucSpace d) :=
  ((n : ℝ≥0∞))⁻¹ • ∑ k : Idx n, Measure.dirac (x k)

/-- Integration against the empirical measure is the normalized sum.  At
`n = 0` both sides are `0`, the empirical measure of no tokens being the zero
measure. -/
theorem integral_empMeasure {d n : ℕ} {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] (x : Idx n → EucSpace d) (f : EucSpace d → E) :
    ∫ y, f y ∂(empMeasure x) = ((n : ℝ))⁻¹ • ∑ k : Idx n, f (x k) := by
  rw [empMeasure, integral_smul_measure,
    integral_finsetSum_measure (fun k _ => integrable_dirac (enorm_lt_top))]
  simp [ENNReal.toReal_inv, integral_dirac]

/-- The attention field at the empirical measure of a configuration is the
attention field of that configuration: the `1/n` of `μ_X` cancels between the
numerator and the normalizer.  This is what makes the mean-field vocabulary of
§3 an extension of the particle vocabulary of §2 and not a second one. -/
theorem attnFieldOf_empMeasure {d n : ℕ} (β : ℝ) (θ : HeadParam d)
    (x : Idx n → EucSpace d) (z : EucSpace d) :
    attnFieldOf β θ (empMeasure x) z = attnField β θ x z := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    simp [attnFieldOf, attnField, integral_empMeasure]
  · have hc : ((n : ℝ))⁻¹ ≠ 0 := inv_ne_zero (Nat.cast_ne_zero.mpr hn.ne')
    rw [attnFieldOf, attnField, integral_empMeasure, integral_empMeasure, smul_eq_mul,
      mul_inv, smul_smul]
    congr 1
    field_simp

/-- The mean field at the empirical measure is the mean field of the
configuration. -/
theorem meanFieldOf_empMeasure {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (z : EucSpace d) :
    meanFieldOf β ρ (empMeasure x) z = meanField β ρ x z := by
  simp only [meanFieldOf, meanField, attnFieldOf_empMeasure]

/-- The fluctuation at the empirical measure is the fluctuation of the
configuration. -/
theorem fluctOf_empMeasure {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (θ : HeadParam d) (x : Idx n → EucSpace d) (z : EucSpace d) :
    fluctOf β ρ (empMeasure x) θ z = fluct β ρ θ x z := by
  simp only [fluctOf, fluct, attnFieldOf_empMeasure, meanFieldOf_empMeasure]

/-- **Equation (eq:G_def), the two readings agree.**  `G(X,θ)_i` of §2 is
`G_{μ_X}(x_i,θ)` of §3. -/
theorem Gfield_eq_GfieldOf {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (θ : HeadParam d) (i : Idx n) :
    Gfield β ρ x θ i = GfieldOf β ρ (empMeasure x) (x i) θ := by
  simp only [Gfield, GfieldOf, fluctOf_empMeasure]

/-- **Equation (eq:covariance_kernel).**  The covariance kernel

  `K[μ](x,y) = E[ξ_θ[μ](x) ξ_θ[μ](y)ᵀ] ∈ ℝ^{d×d}`,

written as the linear map it is: `K[μ](x,y) v = E[⟨ξ_θ[μ](y), v⟩ ξ_θ[μ](x)]`.

Source: arXiv:2604.01978v1, `eq:covariance_kernel`. -/
noncomputable def covKernel {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y v : EucSpace d) : EucSpace d :=
  ∫ θ, (inner (𝕜 := ℝ) (fluctOf β ρ μ θ y) v) • fluctOf β ρ μ θ x ∂ρ

/-- **Equation (eq:cross_variation_kernel) of `rem:shared_noise_rewrite`.**
Two particles driven by the same cylindrical Wiener process have martingale
parts with cross-variation

  `d⟨M_i,M_j⟩_t = α Proj_{x_i} K[μ](x_i,x_j) Proj_{x_j} dt`.

**What the source says and what is written here.**  Mathlib has no stochastic
integral, so the cross-variation of `M_i = √α ∫_Θ ξ_θ[μ](x_i) W(dθ,dt)` cannot
be written.  Its integrand can, and it is the whole content of the identity:
the cross-variation of two Itô integrals against the same cylindrical noise is
the `L²(ρ*)` pairing of their integrands, so `eq:cross_variation_kernel` says
exactly that for all `u, v`

  `∫_Θ ⟨G_μ(x,θ), u⟩ ⟨G_μ(y,θ), v⟩ ρ*(dθ) = ⟨Proj_x K[μ](x,y) Proj_y v, u⟩`.

The scalar `α` multiplies both martingales and is dropped with them.  The
right side is written as `⟨K[μ](x,y) Proj_y v, Proj_x u⟩`, which is the same
number by `real_inner_proj_left`; that is where the source's two projections
`Proj_{x_i} · Proj_{x_j}` go.

Source: arXiv:2604.01978v1, `eq:cross_variation_kernel`. -/
theorem inner_covKernel_proj {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y u v : EucSpace d)
    (hint : Integrable (fun θ =>
      (inner (𝕜 := ℝ) (fluctOf β ρ μ θ y) (proj d y v)) • fluctOf β ρ μ θ x) ρ) :
    inner (𝕜 := ℝ) (covKernel β ρ μ x y (proj d y v)) (proj d x u) =
      ∫ θ, inner (𝕜 := ℝ) (GfieldOf β ρ μ x θ) u *
        inner (𝕜 := ℝ) (GfieldOf β ρ μ y θ) v ∂ρ := by
  rw [covKernel, real_inner_comm, ← integral_inner hint]
  refine integral_congr_ae (Filter.Eventually.of_forall fun θ => ?_)
  have h1 : inner (𝕜 := ℝ) (GfieldOf β ρ μ x θ) u
      = inner (𝕜 := ℝ) (proj d x u) (fluctOf β ρ μ θ x) := by
    rw [GfieldOf, real_inner_proj_left, real_inner_comm]
  have h2 : inner (𝕜 := ℝ) (GfieldOf β ρ μ y θ) v
      = inner (𝕜 := ℝ) (fluctOf β ρ μ θ y) (proj d y v) := by
    rw [GfieldOf, real_inner_proj_left]
  show inner (𝕜 := ℝ) (proj d x u)
      ((inner (𝕜 := ℝ) (fluctOf β ρ μ θ y) (proj d y v)) • fluctOf β ρ μ θ x)
    = inner (𝕜 := ℝ) (GfieldOf β ρ μ x θ) u * inner (𝕜 := ℝ) (GfieldOf β ρ μ y θ) v
  rw [h1, h2, real_inner_smul_right]
  ring

/-- The hypothesis of `inner_covKernel_proj` is satisfiable: under `ρ* = δ_0`
every function is integrable, the fluctuation being evaluated at one head. -/
example {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d)) (x y v : EucSpace d) :
    Integrable (fun θ =>
      (inner (𝕜 := ℝ) (fluctOf β (Measure.dirac (0 : HeadParam d)) μ θ y) (proj d y v)) •
        fluctOf β (Measure.dirac (0 : HeadParam d)) μ θ x)
      (Measure.dirac (0 : HeadParam d)) :=
  integrable_dirac enorm_lt_top

end Homogenized
end Transformer

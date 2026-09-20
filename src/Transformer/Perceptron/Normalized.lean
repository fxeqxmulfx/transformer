/-
# Perceptrons and attention's mean-field landscape — normalized self-attention

Formalization of `eq:full-WGF-main` and `eq: fulltrans.stat` of
arXiv:2601.21366v2: the stationarity condition of the *normalized* coupled
dynamics, in which the attention field is the first variation of `E_β`
rescaled by its own strictly positive weight.

**What the source says and what is carried here.**

* `w[μ] := δE_β/δμ[μ]` is the weight of the conformally equivalent metric.
  The tree has no abstract first variation, so `attentionWeight` is the
  concrete function that variation *is* — `β⁻¹ Z_{β,μ}`, where
  `Z_{β,μ}(x) = ∫ e^{β x·y} dμ(y)` is `Perspective.partitionMu` — and
  `proj_gradient_attentionWeight` proves the relation that fixes the
  normalization: its spherical gradient is the attention half of `energyGrad`,
  the field of `eq: steady.state`.

* `eq: fulltrans.stat` is `∇δE_β/δμ[μ] + w[μ] u_ϑ = 0` on `supp μ`, which is
  the source's own rewriting of the weighted stationarity
  `Grad E_β[μ] + u_ϑ = 0`: the two are equivalent because `w > 0`
  (`attentionWeight_pos`).  `IsNormalizedStationary` carries the displayed
  equation, which is the one the results of §6 are stated for.

* `Grad E ≔ w[μ]⁻¹ ∇δE/δμ[μ]` is, on the sphere, `β` times the mean-field
  self-attention field `𝒳[μ]` of arXiv:2312.10794v5 —
  `proj_inv_smul_gradient_attentionWeight`.  That is the sense in which
  `eq:full-WGF-main` is *normalized* attention: dividing by the partition
  function is what the weight does.

Source: arXiv:2601.21366v2, `eq:full-WGF-main`, `eq: fulltrans.stat`.
-/

import Transformer.Perceptron.Dirac
import Transformer.Perspective.PartitionGradient

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The weight and the attention field -/

/-- **The weight `w[μ] = δE_β/δμ[μ]`** of `eq:full-WGF-main`: the first
variation of `E_β[μ] = (2β)⁻¹ ∬ e^{β x·y} dμ dμ`, which is `β⁻¹ Z_{β,μ}`.

Source: arXiv:2601.21366v2, §2.3. -/
noncomputable def attentionWeight (β : ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  β⁻¹ * Perspective.partitionMu d β μ x

/-- **The attention half of `energyGrad`**: `∫ e^{β x·y} Proj_x y dμ(y)`, the
spherical gradient of `w[μ]`.

Source: arXiv:2601.21366v2, §2.3. -/
noncomputable def attentionGrad (β : ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    EucSpace d :=
  ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d)
    ∂(μ : Measure (SSphere d))

/-- `∇δE_{β,ϑ}/δμ[μ] = ∇δE_β/δμ[μ] + u_ϑ`: the field of `eq: steady.state` is
the attention field plus the perceptron drift. -/
theorem energyGrad_eq_attentionGrad_add_drift (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    energyGrad β σ ω a μ x = attentionGrad β μ x + drift σ ω a x := rfl

/-- **`w[μ] > 0`**, the positivity the source uses to pass between the
weighted stationarity and `eq: fulltrans.stat`. -/
theorem attentionWeight_pos (β : ℝ) (hβ : 0 < β) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) : 0 < attentionWeight β μ x :=
  mul_pos (inv_pos.mpr hβ) (Perspective.partitionMu_pos d β μ x)

/-- The hypothesis of `attentionWeight_pos` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-- **`Proj_x` commutes with the attention integral**: the projection is
affine in the integrand, and both pieces are integrable.  This is the one
analytic step behind the two identities below. -/
theorem proj_integral_expInner_smul (β : ℝ) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    proj d x (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
        ∂(μ : Measure (SSphere d))) = attentionGrad β μ x := by
  have hint := Perspective.integrable_expInner_smul d β μ x
  have hscal : Integrable (fun y : SSphere d =>
      Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
        inner (𝕜 := ℝ) x (y : EucSpace d)) (μ : Measure (SSphere d)) := by
    simpa [real_inner_smul_right] using (innerSL ℝ x).integrable_comp hint
  have hsplit : attentionGrad β μ x
      = (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
            ∂(μ : Measure (SSphere d)))
        - (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
            inner (𝕜 := ℝ) x (y : EucSpace d) ∂(μ : Measure (SSphere d))) • x := by
    rw [attentionGrad]
    simp only [proj, smul_sub, smul_smul]
    rw [integral_sub hint (hscal.smul_const x), integral_smul_const]
  rw [hsplit, proj]
  congr 2
  simpa [real_inner_smul_right] using ((innerSL ℝ x).integral_comp_comm hint).symm

/-- **The normalization of `attentionWeight`**: its spherical gradient is the
attention field.  This is what makes `β⁻¹ Z_{β,μ}` — and not `Z_{β,μ}` — the
first variation of `E_β`: the `β` produced by differentiating the kernel is
exactly what the `(2β)⁻¹` in front of the energy cancels. -/
theorem proj_gradient_attentionWeight (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) :
    proj d x (gradient (attentionWeight β μ) x) = attentionGrad β μ x := by
  rw [show gradient (attentionWeight β μ) x
      = ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
          ∂(μ : Measure (SSphere d)) from Perspective.gradient_partitionMu d β hβ μ x]
  exact proj_integral_expInner_smul β μ x

/-- The hypothesis of `proj_gradient_attentionWeight` is satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **`Grad E_β[μ] = β 𝒳[μ]` on the sphere.**  The weighted gradient
`w[μ]⁻¹ ∇δE_β/δμ[μ]` of `eq:full-WGF-main` is `β` times the mean-field
self-attention field of arXiv:2312.10794v5 — normalized attention, with the
partition function in the denominator. -/
theorem proj_inv_smul_gradient_attentionWeight (β : ℝ) (hβ : β ≠ 0)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    (attentionWeight β μ x)⁻¹ • proj d x (gradient (attentionWeight β μ) x)
      = β • Perspective.vectorField d β μ x := by
  have hps : ∀ (c : ℝ) (v : EucSpace d), proj d x (c • v) = c • proj d x v := by
    intro c v
    simp only [proj, real_inner_smul_right, smul_sub, smul_smul]
  have hvec : Perspective.vectorField d β μ x
      = (Perspective.partitionMu d β μ x)⁻¹ • attentionGrad β μ x := by
    rw [Perspective.vectorField, hps, proj_integral_expInner_smul]
  rw [proj_gradient_attentionWeight β hβ μ x, hvec, smul_smul, attentionWeight, mul_inv, inv_inv]

/-- The hypothesis of `proj_inv_smul_gradient_attentionWeight` is
satisfiable. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-! ### Stationarity -/

/-- **`eq: fulltrans.stat`.**  `μ` is stationary for the normalized dynamics
`eq:full-WGF-main`:

  `∇δE_β/δμ[μ] + w[μ] u_ϑ = 0`  on `supp μ`.

Source: arXiv:2601.21366v2, `eq: fulltrans.stat`. -/
def IsNormalizedStationary (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : Prop :=
  ∀ x ∈ (μ : Measure (SSphere d)).support,
    attentionGrad β μ (x : EucSpace d)
      + attentionWeight β μ (x : EucSpace d) • drift σ ω a (x : EucSpace d) = 0

/-- The attention field of a Dirac mass vanishes at its own atom: there the
kernel sees only `Proj_x x = 0`. -/
theorem attentionGrad_diracProb (β : ℝ) (x : SSphere d) :
    attentionGrad β (Perspective.diracProb d x) (x : EucSpace d) = 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  show (∫ z, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (z : EucSpace d)) •
    proj d (x : EucSpace d) (z : EucSpace d) ∂(Measure.dirac x)) = 0
  rw [integral_dirac, show proj d (x : EucSpace d) (x : EucSpace d)
      = proj d (x : EucSpace d) ((1 : ℝ) • (x : EucSpace d)) by rw [one_smul],
    proj_smul_self hx, smul_zero]

/-- **A Dirac mass is stationary for normalized attention whenever the
perceptron's raw drift at its atom is radial** — the weight multiplies a drift
that is already `0`.  This is the inhabitant every satisfiability witness of
§6 is built from.

Source: arXiv:2601.21366v2, `eq: fulltrans.stat`. -/
theorem isNormalizedStationary_diracProb_of_radial (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (x : SSphere d) (c : ℝ)
    (h : ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) (x : EucSpace d))) • a j
      = c • (x : EucSpace d)) :
    IsNormalizedStationary β σ ω a (Perspective.diracProb d x) := by
  intro y hy
  have hyx : y = x := Interpolation.eq_of_mem_support_dirac hy
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  subst hyx
  rw [drift, h, proj_smul_self hx, smul_zero, add_zero, attentionGrad_diracProb]

/-- **`δ_x` is stationary for normalized attention with the ReLU perceptron**
pinned at `x`: the neuron `a_{j₀} = -x` is inactive at `x`.

This is the witness for the stationarity hypothesis of `prop: unified.log`.

Source: arXiv:2601.21366v2, `eq: fulltrans.stat`. -/
theorem isNormalizedStationary_relu_pin (β : ℝ) (j₀ : Idx d) (x : SSphere d) :
    IsNormalizedStationary β (fun s => max s 0) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (Perspective.diracProb d x) := by
  have hxx : inner (𝕜 := ℝ) (x : EucSpace d) (x : EucSpace d) = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2]; norm_num
  refine isNormalizedStationary_diracProb_of_radial β _ _ _ x 0 ?_
  rw [Finset.sum_eq_single j₀]
  · rw [Pi.single_eq_same, Pi.single_eq_same, inner_neg_left, hxx]
    norm_num
  · intro j _ hj
    simp [Pi.single_eq_of_ne hj]
  · intro h
    exact absurd (Finset.mem_univ j₀) h

end Perceptron
end Transformer

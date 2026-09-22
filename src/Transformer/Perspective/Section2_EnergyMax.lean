/-
# §3.2 — The maximisers of the interaction energy

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2,
`prop: existence.uniqueness.energy`.

That proposition has two halves.  The minimiser half — `σ_d` is the unique
global minimiser of `𝖤_β` — is `existence_uniqueness_energy_min`, in
`Perspective.Section2_EnergyMin`: it rests on the positive definiteness of the
kernel `exp(β⟨x, y⟩)` on the sphere (`Perspective.PositiveDefinite`).

The maximiser half is elementary, and is proved here.  On the sphere
`⟨x, x'⟩ ≤ 1`, with equality exactly at `x = x'`, so

  `𝖤_β[μ] ≤ (2β)⁻¹ e^β = 𝖤_β[δ_x]`   for every `μ` and every `x`,

which already says that the Dirac masses are global maximisers
(`isMaxEnergy_diracProb`).  Conversely, write

  `g(x) = e^β - Z_{β,μ}(x) = ∫ (e^β - e^{β⟨x,x'⟩}) dμ(x')`

for the deficit at `x`.  The bound says `g ≥ 0`, and a maximiser attains it,
which says `∫ g dμ = 0`; so `g` vanishes `μ`-almost everywhere, and in
particular at some point `x⋆`.  There `∫ (e^β - e^{β⟨x⋆,x'⟩}) dμ(x') = 0`
with a non-negative integrand, so `⟨x⋆, x'⟩ = 1` for `μ`-almost every `x'`,
i.e. `x' = x⋆`; and a probability measure carried by a single point is the
Dirac mass there.

The paper's `d ≥ 2` is not needed for this half and is not assumed; the
dimension enters only through the minimiser.
-/

import Transformer.Perspective.Section2_EnergyKernel

open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-! ### The bound, and its cases of equality -/

/-- `eq: interaction.energy` is `(2β)⁻¹` times the mean of the partition
function `eq: partition.function` — the inner integral of the double integral
*is* `Z_{β,μ}`. -/
theorem interactionEnergy_eq_partition (β : ℝ) (μ : ProbSphere d) :
    interactionEnergy d β μ
      = (2 * β)⁻¹ * ∫ x, partitionMu d β μ (x : EucSpace d)
          ∂(μ : Measure (SSphere d)) := rfl

/-- `𝖤_β[μ] ≤ (2β)⁻¹ e^β` for every probability measure on the sphere. -/
theorem interactionEnergy_le (β : ℝ) (hβ : 0 < β) (μ : ProbSphere d) :
    interactionEnergy d β μ ≤ (2 * β)⁻¹ * Real.exp β := by
  rw [interactionEnergy_eq_partition]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  calc ∫ x, partitionMu d β μ (x : EucSpace d) ∂(μ : Measure (SSphere d))
      ≤ ∫ _ : SSphere d, Real.exp β ∂(μ : Measure (SSphere d)) :=
        integral_mono (integrable_partitionMu d β hβ.le μ) (integrable_const _)
          (fun x => partitionMu_le d β hβ.le μ x)
    _ = Real.exp β := by simp

/-- A Dirac mass attains the bound: `𝖤_β[δ_x] = (2β)⁻¹ e^β`. -/
theorem interactionEnergy_diracProb (β : ℝ) (x : SSphere d) :
    interactionEnergy d β (diracProb d x) = (2 * β)⁻¹ * Real.exp β := by
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  have hxx : inner (𝕜 := ℝ) (x : EucSpace d) (x : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2, one_pow]
  rw [interactionEnergy, hμ, integral_dirac, integral_dirac, hxx, mul_one]

/-- **Every Dirac mass is a global maximiser of `𝖤_β`.** -/
theorem isMaxEnergy_diracProb (β : ℝ) (hβ : 0 < β) (x : SSphere d) (μ : ProbSphere d) :
    interactionEnergy d β μ ≤ interactionEnergy d β (diracProb d x) := by
  rw [interactionEnergy_diracProb]
  exact interactionEnergy_le d β hβ μ

/-- A probability measure concentrated at a point *is* the Dirac mass there:
push forward along `id =ᵐ const x₀`. -/
theorem eq_dirac_of_ae_eq (μ : ProbSphere d) (x₀ : SSphere d)
    (h : ∀ᵐ x ∂(μ : Measure (SSphere d)), x = x₀) :
    (μ : Measure (SSphere d)) = Measure.dirac x₀ := by
  have hmap : (μ : Measure (SSphere d)).map id
      = (μ : Measure (SSphere d)).map (fun _ => x₀) := Measure.map_congr h
  rwa [Measure.map_id, Measure.map_const, measure_univ, one_smul] at hmap

/-! ### The maximiser half of `prop: existence.uniqueness.energy` -/

/-- **Proposition (prop: existence.uniqueness.energy), second half.**  For
`β > 0`, every global maximiser of the interaction energy `𝖤_β` over
`𝒫(𝕊^{d-1})` is a Dirac mass.

Together with `isMaxEnergy_diracProb` this identifies the maximisers exactly:
they are the Dirac masses and nothing else.

Source: arXiv:2312.10794v5, §3.2, `prop: existence.uniqueness.energy`. -/
theorem exists_eq_dirac_of_isMaxEnergy (β : ℝ) (hβ : 0 < β) (μ₁ : ProbSphere d)
    (hmax : ∀ μ : ProbSphere d, interactionEnergy d β μ ≤ interactionEnergy d β μ₁) :
    ∃ x : SSphere d, (μ₁ : Measure (SSphere d)) = Measure.dirac x := by
  set m := (μ₁ : Measure (SSphere d)) with hm
  have hconst : ∫ _ : SSphere d, Real.exp β ∂m = Real.exp β := by simp
  -- The sphere is nonempty: it carries a probability measure.
  obtain ⟨z⟩ : Nonempty (SSphere d) := by
    by_contra hcon
    rw [not_nonempty_iff] at hcon
    have h1 : m Set.univ = 1 := measure_univ
    rw [Set.univ_eq_empty_iff.mpr hcon, measure_empty] at h1
    exact zero_ne_one h1
  -- A maximiser attains the bound.
  have htop : interactionEnergy d β μ₁ = (2 * β)⁻¹ * Real.exp β :=
    le_antisymm (interactionEnergy_le d β hβ μ₁)
      (by rw [← interactionEnergy_diracProb d β z]; exact hmax _)
  have hmean : ∫ x, partitionMu d β μ₁ (x : EucSpace d) ∂m = Real.exp β := by
    rw [interactionEnergy_eq_partition] at htop
    exact mul_left_cancel₀ (by positivity) htop
  -- Hence the deficit `e^β - Z_{β,μ₁}` vanishes μ₁-almost everywhere.
  have hdef :
      (fun x : SSphere d => Real.exp β - partitionMu d β μ₁ (x : EucSpace d)) =ᵐ[m] 0 := by
    refine (integral_eq_zero_iff_of_nonneg
      (fun x => sub_nonneg.mpr (partitionMu_le d β hβ.le μ₁ x))
      ((integrable_const _).sub (integrable_partitionMu d β hβ.le μ₁))).mp ?_
    rw [integral_sub (integrable_const _) (integrable_partitionMu d β hβ.le μ₁),
      hconst, hmean, sub_self]
  have : (ae m).NeBot := ae_neBot.mpr (IsProbabilityMeasure.ne_zero m)
  obtain ⟨x₀, hx₀⟩ := hdef.exists
  refine ⟨x₀, ?_⟩
  have hP : ∫ x', Real.exp (β * inner (𝕜 := ℝ) (x₀ : EucSpace d) (x' : EucSpace d)) ∂m
      = Real.exp β := by
    simp only [Pi.zero_apply, sub_eq_zero] at hx₀
    exact hx₀.symm
  -- At `x₀` the kernel is `e^β` almost everywhere, so almost every point is `x₀`.
  have hker :
      (fun x' : SSphere d =>
          Real.exp β - Real.exp (β * inner (𝕜 := ℝ) (x₀ : EucSpace d) (x' : EucSpace d)))
        =ᵐ[m] 0 := by
    refine (integral_eq_zero_iff_of_nonneg
      (fun x' => sub_nonneg.mpr (exp_inner_le d β hβ.le x₀ x'))
      ((integrable_const _).sub (integrable_expInner d β hβ.le μ₁ x₀))).mp ?_
    rw [integral_sub (integrable_const _) (integrable_expInner d β hβ.le μ₁ x₀),
      hconst, hP, sub_self]
  refine eq_dirac_of_ae_eq d μ₁ x₀ ?_
  filter_upwards [hker] with x' hx'
  simp only [Pi.zero_apply, sub_eq_zero] at hx'
  have h1 : β = β * inner (𝕜 := ℝ) (x₀ : EucSpace d) (x' : EucSpace d) :=
    Real.exp_eq_exp.mp hx'
  have h2 : inner (𝕜 := ℝ) (x₀ : EucSpace d) (x' : EucSpace d) = 1 :=
    (mul_left_cancel₀ (ne_of_gt hβ) (by rw [mul_one]; exact h1)).symm
  exact (eq_of_inner_sphere_eq_one d h2).symm

/-- The hypotheses of `exists_eq_dirac_of_isMaxEnergy` are satisfiable: on the
circle, at `β = 1`, the Dirac mass at the base point is a global maximiser. -/
example : ∀ μ : ProbSphere 1,
    interactionEnergy 1 1 μ ≤ interactionEnergy 1 1 (diracProb 1 (basePoint 0)) :=
  isMaxEnergy_diracProb 1 1 one_pos (basePoint 0)

end Perspective
end Transformer

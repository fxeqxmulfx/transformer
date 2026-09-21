/-
# Measure-to-measure interpolation — `lem: hyp.propagation` is false

`lem: hyp.propagation` of arXiv:2411.04551v3 asks for an invertible map
carrying `Φ_1(μ_0^i)` to `Φ_3(μ_1^i)` from nothing more than a transport map
`T^i` with `T^i_# μ_0^i = μ_1^i`.  When `T^i` merges atoms no injective map
can do it: `not_hyp_propagation`, built on the even two-atom mixture
`halfDirac`.

Source: arXiv:2411.04551v3, §5, `lem: hyp.propagation`.
-/

import Transformer.Interpolation.Basic

open MeasureTheory

namespace Transformer
namespace Interpolation

open Perspective

variable (d : ℕ)

/-- The even mixture `½δ_x + ½δ_y`, as a point of `𝒫(𝕊^{d-1})`. -/
noncomputable def halfDirac (x y : SSphere d) : ProbSphere d :=
  ⟨(2⁻¹ : ENNReal) • Measure.dirac x + (2⁻¹ : ENNReal) • Measure.dirac y,
    ⟨by simp [ENNReal.inv_two_add_inv_two]⟩⟩

/-- The underlying measure of `halfDirac`. -/
@[simp] theorem coe_halfDirac (x y : SSphere d) :
    (halfDirac d x y : Measure (SSphere d))
      = (2⁻¹ : ENNReal) • Measure.dirac x + (2⁻¹ : ENNReal) • Measure.dirac y :=
  rfl

/-- **Lemma (lem: hyp.propagation) is false as written.**

The source's claim: if `T^i_# μ_0^i = μ_1^i` for every `i`, then there is a
Lipschitz-continuous, invertible `ψ : 𝕊^{d-1} → 𝕊^{d-1}` with
`(ψ ∘ T^i_{Φ_1})_# μ_0^i = Φ_3(μ_1^i)`, where `Φ_1(μ_0^i) = (T^i_{Φ_1})_# μ_0^i`
and `Φ_3(μ_1^i) = (T^i_{Φ_3})_# μ_1^i` for Lipschitz-continuous, invertible
`T^i_{Φ_1}, T^i_{Φ_3}` — the flow maps of `prop: separation`, which the proof
represents this way in its first line.

Refuted here for *every* such representation at once, and with less asked of
the maps than the source gives: `T^i_{Φ_1}` measurable and bijective,
`T^i_{Φ_3}` arbitrary, `ψ` measurable and injective.  The data: `d = 3`,
`N = 1`, `μ_0 = ½δ_x + ½δ_{-x}`, `μ_1 = δ_x`, `T ≡ x`.  An injective map
sends the two atoms of `μ_0` to two distinct atoms, of mass `½` each, while
`Φ_3(δ_x)` is a Dirac mass.

What breaks is the step `ψ^i = T^i_{Φ_3} ∘ T^i ∘ (T^i_{Φ_1})^{-1}`: it is
invertible, let alone Lipschitz, only when the transport map `T^i` is, and the
lemma asks of `T^i` only that it lie in `L²`.

Source: arXiv:2411.04551v3, §5, `lem: hyp.propagation` and its proof. -/
theorem not_hyp_propagation :
    ∃ μ₀ μ₁ : Idx 1 → ProbSphere 3,
      (∀ i : Idx 1, ∃ Tr : SSphere 3 → SSphere 3, Measurable Tr ∧
        Measure.map Tr (μ₀ i : Measure (SSphere 3)) = (μ₁ i : Measure (SSphere 3))) ∧
      ∀ T₁ T₃ : Idx 1 → SSphere 3 → SSphere 3,
        (∀ i, Measurable (T₁ i) ∧ Function.Bijective (T₁ i)) →
        ¬ ∃ ψ : SSphere 3 → SSphere 3, Measurable ψ ∧ Function.Injective ψ ∧
          ∀ i : Idx 1,
            Measure.map ψ (Measure.map (T₁ i) (μ₀ i : Measure (SSphere 3)))
              = Measure.map (T₃ i) (μ₁ i : Measure (SSphere 3)) := by
  set x := basePoint 2
  set y := antipode 3 x
  refine ⟨fun _ => halfDirac 3 x y, fun _ => diracProb 3 x,
    fun _ => ⟨fun _ => x, measurable_const, ?_⟩, fun T₁ T₃ hT₁ ⟨ψ, hψ, hinj, h⟩ => ?_⟩
  · have hd : (diracProb 3 x : Measure (SSphere 3)) = Measure.dirac x := rfl
    rw [coe_halfDirac, hd, Measure.map_add _ _ measurable_const]
    simp (disch := fun_prop) only [Measure.map_smul, Measure.map_dirac]
    rw [← add_smul, ENNReal.inv_two_add_inv_two, one_smul]
  have h0 := h 0
  have hd : (diracProb 3 x : Measure (SSphere 3)) = Measure.dirac x := rfl
  rw [coe_halfDirac, hd, Measure.map_add _ _ (hT₁ 0).1, Measure.map_add _ _ hψ] at h0
  simp (disch := fun_prop) only [Measure.map_smul, Measure.map_dirac] at h0
  set p := ψ (T₁ 0 x)
  set q := ψ (T₁ 0 y)
  have hpq : p ≠ q := fun e => antipode_ne 3 x ((hT₁ 0).2.1 (hinj e)).symm
  have := congrArg (fun m : Measure (SSphere 3) => m {p}) h0
  simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, Measure.dirac_apply_of_mem
    (Set.mem_singleton p)] at this
  rw [Measure.dirac_apply' _ (measurableSet_singleton p), Measure.dirac_apply' _
    (measurableSet_singleton p)] at this
  simp [Set.indicator, hpq.symm] at this
  split_ifs at this <;> norm_num at this

end Interpolation
end Transformer

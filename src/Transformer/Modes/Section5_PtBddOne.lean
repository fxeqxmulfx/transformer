import Transformer.Modes.Section5_PtBdd
import Transformer.Modes.Section3_Cumulants
import Mathlib.Topology.MetricSpace.HausdorffDimension
import Mathlib.MeasureTheory.Measure.Hausdorff

/-
# The number of modes of a Gaussian KDE — `ν_t` has no density

§5.5 of arXiv:2412.09080v3, `sec: proof.pt.bdd`: the remark that for `n = 1`
the law `μ_t = ν_t` of `(G(t), G'(t))` has no continuous density.

**What the source says and what is carried here.**

* "For `n = 1`, `ν_t` cannot have a continuous density on `ℝ²`, since both
  components are functions of the same one-dimensional Gaussian" is proved in a
  stronger form: `ν_t` has no density at all (`not_isDensityOf_one`).  It is
  carried by the range of the differentiable curve `x ↦ (G(t), G'(t))`, of
  Hausdorff dimension `≤ 1 < 2`, hence Lebesgue-null.

Source: arXiv:2412.09080v3, §5.5, the parenthetical remark after `eq: Gt-prime`.
-/

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer
namespace Modes

/-- The range of a differentiable curve in `ℝ²` is Lebesgue-null. -/
theorem volume_range_eq_zero {f : ℝ → ℝ × ℝ} (hf : Differentiable ℝ f) :
    volume (Set.range f) = 0 := by
  set f' : ℝ → Fin 2 → ℝ := fun x => ![(f x).1, (f x).2]
  have hf' : Differentiable ℝ f' := by
    refine differentiable_pi.2 fun i => ?_
    fin_cases i
    · exact hf.fst
    · exact hf.snd
  have hvol : (volume : Measure (Fin 2 → ℝ)) = μH[((2 : ℕ) : NNReal)] := by
    have h := hausdorffMeasure_pi_real (ι := Fin 2)
    rw [Fintype.card_fin] at h
    exact h.symm
  have hdim : dimH (Set.range f') < ((2 : ℕ) : NNReal) := by
    refine (hf'.dimH_range_le).trans_lt ?_
    simp only [Module.finrank_self]
    norm_num
  have h0 : volume (Set.range f') = 0 :=
    measure_zero_of_dimH_lt (hvol ▸ Measure.AbsolutelyContinuous.rfl) hdim
  rw [← (volume_preserving_finTwoArrow ℝ).measure_preimage_equiv]
  refine measure_mono_null ?_ h0
  rintro g ⟨x, hx⟩
  refine ⟨x, funext fun i => ?_⟩
  fin_cases i
  · simp [f', hx, MeasurableEquiv.finTwoArrow]
  · simp [f', hx, MeasurableEquiv.finTwoArrow]

/-- The hypothesis of `volume_range_eq_zero` is satisfiable. -/
example : Differentiable ℝ fun x : ℝ => (x, x) := by fun_prop

/-- **`ν_t` has no density** on `ℝ²`: the law of `(G(t), G'(t))`, for one
standard Gaussian sample point, is not `q dλ` for any `q`.  In particular it
has no continuous density, which is what the source claims.

Source: arXiv:2412.09080v3, §5.5, the parenthetical remark after `eq: Gt-prime`. -/
theorem not_isDensityOf_one (β t : ℝ) :
    ¬ ∃ q, IsDensityOf (gaussianReal 0 1) (fun x => (bigG β t x, bigG' β t x)) q := by
  rintro ⟨q, hq⟩
  set ψ : ℝ → ℝ × ℝ := fun x => (bigG β t x, bigG' β t x)
  have hψ : Differentiable ℝ ψ := by
    unfold ψ bigG bigG'
    fun_prop
  have h0 := volume_range_eq_zero hψ
  have h1 : gaussianReal 0 1 (ψ ⁻¹' Set.range ψ) ≤ (gaussianReal 0 1).map ψ (Set.range ψ) :=
    Measure.le_map_apply hψ.continuous.measurable.aemeasurable _
  rw [Set.preimage_range, measure_univ, hq.map_eq,
    withDensity_absolutelyContinuous _ _ h0] at h1
  exact one_ne_zero (le_zero_iff.1 h1)

end Modes
end Transformer

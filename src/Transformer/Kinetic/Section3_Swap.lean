module

public import Transformer.Kinetic.Defs
public import Mathlib.Probability.Moments.Covariance
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.MeasureTheory.Integral.Prod

/-!
# The swap identity for covariances

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 2, `eq:est-cov-phi`: for
independent coordinates the covariance of two observables is bounded by the sum
over tokens of the products of their Glauber derivatives. The source cites the
Glauber calculus. Here the bound is proved with an independent copy `x'` of the
coordinates (`covariance_eq_sum_swap`):

  `Cov(U, H) = ½ Σ_l 𝔼[(U(x) - U(x^l)) (H(h_l) - H(h_{l+1}))]`,

where `x^l` takes the `l`-th coordinate from `x'` and the hybrid `h_l` takes the
coordinates before `l` from `x'` and the others from `x`. Exchanging the `l`-th
coordinates of `x` and `x'` preserves the law (`measurePreserving_swapAt`) and
exchanges `h_l` and `h_{l+1}`, which gives the factor `½`.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ}

/-- The hybrid whose coordinates before `l` are those of `x'`, the others those of `x`. -/
def hybrid (l : ℕ) (x x' : Idx N → ℝ) : Idx N → ℝ := fun m => if (m : ℕ) < l then x' m else x m

/-- Exchange the `l`-th coordinates of `x` and `x'`. -/
def swapAt (l : Idx N) (p : (Idx N → ℝ) × (Idx N → ℝ)) : (Idx N → ℝ) × (Idx N → ℝ) :=
  (Function.update p.1 l (p.2 l), Function.update p.2 l (p.1 l))

theorem hybrid_zero (x x' : Idx N → ℝ) : hybrid 0 x x' = x := by
  funext m
  simp [hybrid]

theorem hybrid_last (x x' : Idx N → ℝ) : hybrid N x x' = x' := by
  funext m
  simp [hybrid, m.2]

theorem hybrid_succ (l : Idx N) (x x' : Idx N → ℝ) :
    hybrid ((l : ℕ) + 1) x x' = Function.update (hybrid l x x') l (x' l) := by
  funext m
  by_cases hm : m = l
  · subst hm
    simp [hybrid]
  · have : (m : ℕ) ≠ l := fun h => hm (Fin.ext h)
    rw [Function.update_of_ne hm]
    unfold hybrid
    split_ifs <;> first | rfl | omega

theorem hybrid_swapAt (l : Idx N) (p : (Idx N → ℝ) × (Idx N → ℝ)) :
    hybrid l (swapAt l p).1 (swapAt l p).2 = hybrid ((l : ℕ) + 1) p.1 p.2 := by
  rw [hybrid_succ]
  funext m
  by_cases hm : m = l
  · subst hm
    simp [hybrid, swapAt]
  · have : (m : ℕ) ≠ l := fun h => hm (Fin.ext h)
    simp [hybrid, swapAt, hm]

theorem hybrid_succ_swapAt (l : Idx N) (p : (Idx N → ℝ) × (Idx N → ℝ)) :
    hybrid ((l : ℕ) + 1) (swapAt l p).1 (swapAt l p).2 = hybrid l p.1 p.2 := by
  rw [hybrid_succ]
  funext m
  by_cases hm : m = l
  · subst hm
    simp [hybrid, swapAt]
  · have : (m : ℕ) ≠ l := fun h => hm (Fin.ext h)
    simp [hybrid, swapAt, hm]

theorem measurable_hybrid (l : ℕ) :
    Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) => hybrid l p.1 p.2 := by
  refine measurable_pi_iff.2 fun m => ?_
  unfold hybrid
  split_ifs
  · exact (measurable_pi_apply m).comp measurable_snd
  · exact (measurable_pi_apply m).comp measurable_fst

/-- **Exchanging one coordinate of two independent copies preserves their law.** -/
theorem measurePreserving_swapAt (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]
    (l : Idx N) :
    MeasurePreserving (swapAt l) ((Measure.pi μs).prod (Measure.pi μs))
      ((Measure.pi μs).prod (Measure.pi μs)) := by
  set E := MeasurableEquiv.arrowProdEquivProdArrow ℝ ℝ (Idx N)
  have hE := measurePreserving_arrowProdEquivProdArrow ℝ ℝ (Idx N) μs μs
  have hS : MeasurePreserving (fun (p : Idx N → ℝ × ℝ) i => (if i = l then Prod.swap else id) (p i))
      (Measure.pi fun i => (μs i).prod (μs i)) (Measure.pi fun i => (μs i).prod (μs i)) :=
    measurePreserving_pi _ _ fun _ => by
      split_ifs
      · exact Measure.measurePreserving_swap
      · exact MeasurePreserving.id _
  have := hE.comp (hS.comp (MeasurePreserving.symm E hE))
  convert this using 1
  funext p
  ext i <;> by_cases hi : i = l <;>
    simp [swapAt, E, MeasurableEquiv.arrowProdEquivProdArrow, Equiv.arrowProdEquivProdArrow, hi]

/-- The hypotheses of `measurePreserving_swapAt` are satisfiable: Dirac masses. -/
example : MeasurePreserving (swapAt (0 : Idx 2))
    ((Measure.pi fun _ => Measure.dirac (0 : ℝ)).prod (Measure.pi fun _ => Measure.dirac 0))
    ((Measure.pi fun _ => Measure.dirac (0 : ℝ)).prod (Measure.pi fun _ => Measure.dirac 0)) :=
  measurePreserving_swapAt _ 0

end Kinetic
end Transformer

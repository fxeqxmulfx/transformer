/-
# Both actual neighbors converge to the genuine Euler limit

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The proved neighboring-state estimate is applied to the true floor and
ceiling selectors. The finite horizon can exceed the observation time.
-/

import Transformer.BatchSize.Section4_EulerObservationNeighbors
import Transformer.BatchSize.Section4_DyadicGenerator

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Both true neighboring dyadic states converge in mean square
to the actual interpolated Euler limit, Section 4.3 (2)--(3).
The right neighbor is adapted after the observation; the left
neighbor supplies the genuine frozen generator coefficients. -/
theorem dyadicEuler_neighbors_meanSquare {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : NNReal)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (u T : NNReal) (huT : u ≤ T)
    (Y : BrownianSample d → EucSpace d) (hY : MemLp Y 2 (brownianNoiseLaw d))
    (hlim : Tendsto (fun m => ∫ ω, ‖dyadicEuler b a x₀ u m ω - Y ω‖ ^ 2
      ∂brownianNoiseLaw d) atTop (𝓝 0)) :
    ((∀ m, MemLp (dyadicLeftState b a x₀ T m u) 2 (brownianNoiseLaw d)) ∧
      Tendsto (fun m => ∫ ω, ‖dyadicLeftState b a x₀ T m u ω - Y ω‖ ^ 2
        ∂brownianNoiseLaw d) atTop (𝓝 0)) ∧
    ((∀ m, MemLp (eulerChain b a (dyadicGrid T m) x₀ (dyadicRightIndex u m)) 2
      (brownianNoiseLaw d)) ∧
      Tendsto (fun m => ∫ ω,
        ‖eulerChain b a (dyadicGrid T m) x₀ (dyadicRightIndex u m) ω - Y ω‖ ^ 2
        ∂brownianNoiseLaw d) atTop (𝓝 0)) := by
  have hl := dyadicEuler_neighbor_meanSquare b a Kb Ka M A hb ha hbM haA x₀ u T huT
    (dyadicLeftIndex u) (fun m => (dyadicObservation_indices_le_endpoint u T huT m).1)
    (fun m => by
      rw [abs_of_nonpos (sub_nonpos.mpr (NNReal.coe_le_coe.mpr (dyadicLeftTime_bounds u T huT m).1)),
        neg_sub]
      exact (dyadicLeftTime_bounds u T huT m).2) Y hY hlim
  have hr := dyadicEuler_neighbor_meanSquare b a Kb Ka M A hb ha hbM haA x₀ u T huT
    (dyadicRightIndex u) (fun m => (dyadicObservation_indices_le_endpoint u T huT m).2)
    (fun m => by
      rw [abs_of_nonneg (sub_nonneg.mpr (NNReal.coe_le_coe.mpr (dyadicRightTime_bounds u T huT m).1))]
      exact (dyadicRightTime_bounds u T huT m).2) Y hY hlim
  have heq (m : ℕ) : dyadicLeftState b a x₀ T m u =
      eulerChain b a (dyadicGrid T m) x₀ (dyadicLeftIndex u m) := by
    funext ω
    simp only [dyadicLeftState, Real.toNNReal_coe]
  refine ⟨⟨?_, ?_⟩, hr⟩
  · intro m
    rw [heq]
    exact hl.1 m
  · simpa only [heq] using hl.2

/-- Joint nonvacuity of the actual neighboring-limit hypotheses,
Section 4.3: bounded constant coefficients and their zero-time
limit at a nonzero initial state, within a positive horizon. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : NNReal)) ∧ (0 : NNReal) ≤ 1 ∧
    MemLp (fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 2
      (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1,
      ‖(fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ω -
        EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ ^ 2 ∂brownianNoiseLaw 1) atTop (𝓝 0) :=
  ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by simp, by simp, by norm_num,
    memLp_const _, by simp⟩

end Transformer.BatchSize

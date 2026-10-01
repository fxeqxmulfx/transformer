/-
# Vector stochastic sums and their exact energy

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The energy identity uses the joint past for every coefficient. The
Euclidean norm introduces a sum of coordinate energies, without assuming
independence of the random coefficients themselves.
-/

import Transformer.BatchSize.Section4_ItoSums

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- A finite vector stochastic sum for diagonal state-dependent
amplitudes, Section 4.3 (2)--(3). -/
def vectorItoSum {d : ℕ} (t : ℕ → ℝ≥0)
    (H : ℕ → BrownianSample d → EucSpace d) (n : ℕ) (ω : BrownianSample d) : EucSpace d :=
  WithLp.toLp 2 (fun k => brownianItoSum k t (fun j ω => H j ω k) n ω)

/-- The finite vector sum is square integrable, Section 4.3 (2)--(3). -/
theorem vectorItoSum_memLp {d : ℕ} (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (H : ℕ → BrownianSample d → EucSpace d)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d)) (n : ℕ) :
    MemLp (vectorItoSum t H n) 2 (brownianNoiseLaw d) := by
  apply MemLp.of_eval_piLp
  intro k
  exact brownianItoSum_memLp k t hmono (fun j ω => H j ω k)
    (fun j => (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable (hH j))
    (fun j => (hHL2 j).eval_piLp k) n

/-- Exact vector Itô isometry, Section 4.3 (2)--(3), including
coefficients depending on multiple state coordinates. -/
theorem vectorItoSum_isometry {d : ℕ} (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (H : ℕ → BrownianSample d → EucSpace d)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d)) (n : ℕ) :
    (∫ ω, ‖vectorItoSum t H n ω‖ ^ 2 ∂brownianNoiseLaw d) =
      ∑ j ∈ Finset.range n, (∫ ω, ‖H j ω‖ ^ 2 ∂brownianNoiseLaw d) *
        ((t (j + 1) : ℝ) - t j) := by
  have hcoord j k : MemLp (fun ω => H j ω k) 2 (brownianNoiseLaw d) :=
    (hHL2 j).eval_piLp k
  have hs k := brownianItoSum_memLp k t hmono (fun j ω => H j ω k)
    (fun j => (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable (hH j))
    (fun j => hcoord j k) n
  simp only [EuclideanSpace.real_norm_sq_eq, vectorItoSum]
  rw [integral_finsetSum _ (fun k _ => (hs k).integrable_sq)]
  simp_rw [brownianItoSum_isometry _ t hmono _
    (fun j => (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) _).comp_stronglyMeasurable (hH j))
    (fun j => hcoord j _) n]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  rw [integral_finsetSum _ (fun k _ => (hcoord j k).integrable_sq), Finset.sum_mul]

/-- Adaptedness of the actual finite vector stochastic sum at its
terminal time, Section 4.3 (2)--(3). -/
theorem vectorItoSum_adapted {d : ℕ} (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (H : ℕ → BrownianSample d → EucSpace d)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n)) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (vectorItoSum t H n) := by
  let : MeasurableSpace (BrownianSample d) := brownianFiltration d (t n)
  apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
  apply Measurable.stronglyMeasurable
  apply Measurable.of_eval
  intro k
  apply StronglyMeasurable.measurable
  apply Finset.stronglyMeasurable_fun_sum
  intro j hj
  have hjn : j < n := Finset.mem_range.mp hj
  have ha := ((hH j).mono ((brownianFiltration d).mono (hmono hjn.le)))
  have hc := (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable ha
  have hd1 : StronglyMeasurable[brownianFiltration d (t n)] (coordinateBrownian k (t (j + 1))) :=
    ((coordinateBrownian_filtered k).stronglyAdapted (t (j + 1))).mono
      ((brownianFiltration d).mono (hmono (by omega)))
  have hd0 : StronglyMeasurable[brownianFiltration d (t n)] (coordinateBrownian k (t j)) :=
    ((coordinateBrownian_filtered k).stronglyAdapted (t j)).mono
      ((brownianFiltration d).mono (hmono hjn.le))
  exact hc.mul (hd1.sub hd0)

/-- Joint nonvacuity of the vector isometry hypotheses, Section 4.3:
the integrand is the whole vector driver at each left endpoint. -/
example : Monotone (fun n : ℕ => (n : ℝ≥0)) ∧
    (∀ n : ℕ, StronglyMeasurable[brownianFiltration 2 n] (vectorBrownian 2 n)) ∧
    (∀ n : ℕ, MemLp (vectorBrownian 2 n) 2 (brownianNoiseLaw 2)) := by
  refine ⟨fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h, ?_, ?_⟩
  · intro n
    exact Filtration.stronglyAdapted_natural
      (fun t => (vectorBrownian_measurable 2 t).stronglyMeasurable) n
  · intro n
    apply MemLp.of_eval_piLp
    intro k
    exact ((coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_eval n).memLp_two

end Transformer.BatchSize

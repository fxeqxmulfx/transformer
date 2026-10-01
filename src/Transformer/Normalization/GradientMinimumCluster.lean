/-
# Critical limits of compact energy-level gradient minima

Energy levels approaching a critical point have gradient minimizers with
a critical accumulation point at the same limiting energy. This supplies
the base point for analytic curve selection in the general case of
Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.GradientMinimumPolar
import Mathlib.Topology.Sequences

open Filter Set

namespace Transformer.Normalization

/-- Points minimizing gradient norm on their energy fiber in `K`, with
energy restricted to `S`. This is a genuine predicate on points; analytic
parametrization of this set is not assumed. Auxiliary selection locus for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def energyFiberGradientMinima {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (S : Set ℝ) : Set (EucSpace N) :=
  {w | w ∈ K ∧ E w ∈ S ∧
    ∀ y ∈ K, E y = E w → ‖gradient E w‖ ≤ ‖gradient E y‖}

/-- If energies in `S` occur arbitrarily close to a critical point inside
a compact analytic domain, their gradient-minimizing representatives
accumulate at some critical point of the same energy. The limit need not
be the original point. This is the compactness input for the remaining
analytic selection step in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem compact_gradient_minimum_critical_cluster {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (K : Set (EucSpace N))
    (hK : IsCompact K) (hE : AnalyticOnNhd ℝ E K)
    (hcritical : gradient E z = 0) (S : Set ℝ)
    (hacc : z ∈ closure {y | y ∈ K ∧ E y ∈ S}) :
    ∃ w ∈ K, E w = E z ∧ gradient E w = 0 ∧
      w ∈ closure (energyFiberGradientMinima E K S) := by
  classical
  have hzK : z ∈ K := hK.isClosed.closure_subset
    (closure_mono (fun y hy => hy.1) hacc)
  obtain ⟨y, hy, hyt⟩ := mem_closure_iff_seq_limit.mp hacc
  have hmin (n : ℕ) := compact_gradient_minimum_on_energy_fiber E K hK hE
    (E (y n)) ⟨y n, (hy n).1, rfl⟩
  choose v hvK hvE hvmin using hmin
  have hyG : Tendsto (fun n => ‖gradient E (y n)‖) atTop (nhds 0) := by
    simpa only [Function.comp_def, hcritical, norm_zero] using
      (((analytic_gradient_analyticAt E z (hE z hzK)).continuousAt.tendsto).comp hyt).norm
  have hvG : Tendsto (fun n => ‖gradient E (v n)‖) atTop (nhds 0) :=
    tendsto_const_nhds.squeeze hyG (fun n => norm_nonneg _)
      (fun n => hvmin n (y n) (hy n).1 rfl)
  obtain ⟨w, hwK, phi, hphi, hvt⟩ := hK.isSeqCompact hvK
  have hvEt : Tendsto (fun n => E (v (phi n))) atTop (nhds (E z)) := by
    simpa only [Function.comp_def, hvE] using
      ((hE z hzK).continuousAt.tendsto.comp hyt).comp hphi.tendsto_atTop
  have hwE : E w = E z := tendsto_nhds_unique
    ((hE w hwK).continuousAt.tendsto.comp hvt) hvEt
  have hwG : gradient E w = 0 := norm_eq_zero.mp (tendsto_nhds_unique
    (((analytic_gradient_analyticAt E w (hE w hwK)).continuousAt.tendsto.comp hvt).norm)
    (hvG.comp hphi.tendsto_atTop))
  refine ⟨w, hwK, hwE, hwG, mem_closure_of_tendsto hvt ?_⟩
  exact Eventually.of_forall (fun n =>
    ⟨hvK (phi n), by simpa only [Function.comp_def, hvE] using (hy (phi n)).2,
      fun a ha haE => hvmin (phi n) a ha (haE.trans (hvE (phi n)))⟩)

/-- For the squared norm on a closed unit ball, positive energies
accumulate at the critical origin. All compactness, analyticity,
criticality and accumulation hypotheses are jointly satisfiable.
Auxiliary example for Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ w ∈ Metric.closedBall (0 : EucSpace 1) 1,
    ‖w‖ ^ 2 = 0 ∧ gradient (fun y : EucSpace 1 => ‖y‖ ^ 2) w = 0 ∧
      w ∈ closure (energyFiberGradientMinima (fun y : EucSpace 1 => ‖y‖ ^ 2)
        (Metric.closedBall (0 : EucSpace 1) 1) (Ioi 0)) := by
  have hacc : (0 : EucSpace 1) ∈
      closure {y : EucSpace 1 | y ∈ Metric.closedBall 0 1 ∧ ‖y‖ ^ 2 ∈ Ioi 0} := by
    rw [Metric.mem_closure_iff]
    intro epsilon he
    let t : ℝ := min (epsilon / 2) (1 / 2)
    have ht : 0 < t := lt_min (half_pos he) (by norm_num)
    have ht1 : t ≤ 1 := (min_le_right _ _).trans (by norm_num)
    have hte : t < epsilon := (min_le_left _ _).trans_lt (half_lt_self he)
    let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) t
    have hnorm : ‖v‖ = t := by simp [v, abs_of_pos ht]
    refine ⟨v, ⟨?_, ?_⟩, ?_⟩
    · simpa only [Metric.mem_closedBall, dist_zero_right, hnorm] using ht1
    · change 0 < ‖v‖ ^ 2
      rw [hnorm]
      exact sq_pos_of_pos ht
    · simpa only [dist_zero_left, hnorm] using hte
  simpa only [norm_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0)] using
    compact_gradient_minimum_critical_cluster (fun y : EucSpace 1 => ‖y‖ ^ 2) 0
    (Metric.closedBall 0 1) (isCompact_closedBall _ _)
    (fun y hy => quadratic_energy_analytic y (mem_univ _))
    (by simp [quadratic_energy_gradient]) (Ioi 0) hacc

end Transformer.Normalization

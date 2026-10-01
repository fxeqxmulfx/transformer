/-
# Reducing the general gradient inequality to analytic curve selection

Compact energy-level minima have critical accumulation points. One-sided
analytic curve selection at those points implies the gradient inequality.
All compactness, energy lifting and scalar estimates in this reduction
are proved. Selection itself remains an explicit geometric hypothesis
for the general analytic input of Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.MinimumCurveLifting
import Transformer.Normalization.AnalyticComparisonArcs

open Filter Set

namespace Transformer.Normalization

/-- The usual one-sided analytic curve-selection conclusion for positive
and negative gradient-minimizer sets gives energy-level comparison arcs.
Only critical accumulation points at the base energy need be selected.
This completes the reduction of the general analytic input of
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2 to that geometric selection
property; the property itself is an explicit hypothesis. -/
theorem analytic_comparison_arcs_of_minimum_curve_selection {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (K : Set (EucSpace N))
    (hK : IsCompact K) (hnear : K ∈ nhds z) (hE : AnalyticOnNhd ℝ E K)
    (hcritical : gradient E z = 0)
    (hselect : ∀ sign : ℝ, sign = 1 ∨ sign = -1 → ∀ w ∈ K,
      E w = E z → gradient E w = 0 →
      w ∈ closure (energyFiberGradientMinima E K {c | 0 < sign * (c - E z)}) →
      ∃ (gamma : ℝ → EucSpace N) (epsilon : ℝ), 0 < epsilon ∧
        AnalyticAt ℝ gamma 0 ∧ gamma 0 = w ∧
          ∀ t ∈ Ioo 0 epsilon,
            gamma t ∈ energyFiberGradientMinima E K {c | 0 < sign * (c - E z)}) :
    HasAnalyticGradientComparisonArcs E z := by
  classical
  let sign : Fin 2 → ℝ := fun j => if j = 0 then 1 else -1
  have hsign (j : Fin 2) : sign j = 1 ∨ sign j = -1 := by
    fin_cases j
    · exact Or.inl (by simp [sign])
    · exact Or.inr (by simp [sign])
  have hsign0 (j : Fin 2) : sign j ≠ 0 := by
    rcases hsign j with h | h <;> rw [h] <;> norm_num
  have hzK : z ∈ K := mem_of_mem_nhds hnear
  have hcurves (j : Fin 2) : ∃ gamma : ℝ → EucSpace N,
      AnalyticAt ℝ gamma 0 ∧ AnalyticAt ℝ E (gamma 0) ∧
        E (gamma 0) = E z ∧ gradient E (gamma 0) = 0 ∧
          ∀ delta : ℝ, 0 < delta → ∀ᶠ y in nhds z,
            0 < sign j * (E y - E z) → ∃ t : ℝ, |t| < delta ∧
              E (gamma t) = E y ∧ ‖gradient E (gamma t)‖ ≤ ‖gradient E y‖ := by
    let S : Set ℝ := {c | 0 < sign j * (c - E z)}
    let A : Set (EucSpace N) := {y | y ∈ K ∧ E y ∈ S}
    by_cases hacc : z ∈ closure A
    · obtain ⟨w, hwK, hwE, hwG, hwacc⟩ :=
        compact_gradient_minimum_critical_cluster E z K hK hE hcritical S hacc
      obtain ⟨gamma, epsilon, he, hg, hg0, hmin⟩ :=
        hselect (sign j) (hsign j) w hwK hwE hwG hwacc
      have hEg : AnalyticAt ℝ E (gamma 0) := by rw [hg0]; exact hE w hwK
      have hgE : E (gamma 0) = E z := by simpa only [hg0] using hwE
      refine ⟨gamma, hg, hEg, hgE, by simpa only [hg0] using hwG, ?_⟩
      exact minimum_curve_lifts_energy_side E z K hnear (hE z hzK).continuousAt
        (sign j) (hsign0 j) gamma hg hEg hgE epsilon he hmin
    · have haway : ∀ᶠ y in nhds z, y ∉ closure A :=
        isClosed_closure.isOpen_compl.mem_nhds hacc
      refine ⟨(fun _ => z), analyticAt_const, hE z hzK, rfl, hcritical, ?_⟩
      intro delta hd
      have hKnear : ∀ᶠ y in nhds z, y ∈ K := hnear
      filter_upwards [haway, hKnear] with y hyaway hyK
      intro hypos
      exact False.elim (hyaway (subset_closure (show y ∈ A from ⟨hyK, hypos⟩)))
  choose gamma hgamma hEgamma henergy hgradient hlift using hcurves
  refine ⟨gamma, hgamma, hEgamma, henergy, ?_⟩
  intro delta hd
  filter_upwards [hlift 0 delta hd, hlift 1 delta hd] with y hy0 hy1
  by_cases hyE : E y = E z
  · exact ⟨0, 0, by simpa only [abs_zero] using hd, (henergy 0).trans hyE.symm,
      by rw [hgradient 0, norm_zero]; exact norm_nonneg _⟩
  rcases lt_or_gt_of_ne hyE with hylt | hygt
  · obtain ⟨t, ht, hEt, hGt⟩ := hy1 (by dsimp [sign]; linarith)
    exact ⟨1, t, ht, hEt, hGt⟩
  · obtain ⟨t, ht, hEt, hGt⟩ := hy0 (by simpa [sign] using sub_pos.mpr hygt)
    exact ⟨0, t, ht, hEt, hGt⟩

/-- A constant energy on the closed unit ball satisfies the compact,
analytic and critical hypotheses jointly. The positive and negative
minimizer sets are empty, so the selection hypothesis also holds.
Auxiliary example for Appendix D.1 of arXiv:2510.22026v2. -/
example : HasAnalyticGradientComparisonArcs (fun _ : EucSpace 1 => (0 : ℝ)) 0 := by
  apply analytic_comparison_arcs_of_minimum_curve_selection
    (fun _ : EucSpace 1 => (0 : ℝ)) 0 (Metric.closedBall 0 1)
    (isCompact_closedBall _ _) (Metric.closedBall_mem_nhds _ zero_lt_one)
    analyticOnNhd_const (by simp)
  intro sign hsign w hw henergy hgradient hacc
  have hempty : energyFiberGradientMinima (fun _ : EucSpace 1 => (0 : ℝ))
      (Metric.closedBall 0 1) {c | 0 < sign * (c - 0)} = ∅ := by
    ext y
    simp [energyFiberGradientMinima]
  rw [hempty, closure_empty] at hacc
  exact False.elim hacc

end Transformer.Normalization

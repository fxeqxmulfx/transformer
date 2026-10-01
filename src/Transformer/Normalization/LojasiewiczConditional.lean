/-
# Compact uniformization and conditional modulated convergence

All steps of Appendix D.1 of arXiv:2510.22026v2 after the local analytic
gradient inequality are proved, without extra time regularity assumptions.
-/

import Transformer.Normalization.GradientInequalityCover
import Mathlib.Data.Finset.Lattice.Fold
import Transformer.Normalization.ModulatedConvergence

namespace Transformer.Normalization

/-- Local inequalities at points of the limiting energy level yield a
uniform inequality along a compact trajectory whose energy converges.
This proves the uniformization used in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 without invoking an unproved analytic theorem. -/
theorem uniform_gradient_inequality_of_local {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (K : Set (EucSpace N)) (L : ℝ)
    (hK : IsCompact K) (hE : ContinuousOn E K)
    (hxK : ∀ t, 0 ≤ t → x t ∈ K)
    (hlim : Filter.Tendsto (fun t => E (x t)) Filter.atTop (nhds L))
    (hlocal : ∀ z ∈ K, E z = L → ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖) :
    ∃ a alpha k : ℝ, 0 ≤ a ∧ 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧
      ∀ t, a ≤ t → |E (x t) - L| ^ alpha ≤ k * ‖gradient E (x t)‖ := by
  classical
  have hcover (z : K) := local_gradient_inequality_cover E K L z
    (hE z z.property) (hlocal z z.property)
  choose alpha k delta V ha0 ha1 hk hd hV hzV hineq using hcover
  obtain ⟨s, hs⟩ := hK.elim_finite_subcover V hV (by
    intro z hz
    exact Set.mem_iUnion.mpr ⟨⟨z, hz⟩, hzV ⟨z, hz⟩⟩)
  have hs0 : s.Nonempty := by
    have h := hs (hxK 0 le_rfl)
    obtain ⟨z, hz, _⟩ := Set.mem_iUnion₂.mp h
    exact ⟨z, hz⟩
  let A := s.sup' hs0 alpha
  let B := s.sup' hs0 k
  let D := min 1 (s.inf' hs0 delta)
  let z0 : K := hs0.choose
  have hz0 : z0 ∈ s := hs0.choose_spec
  have hA0 : 0 < A := (ha0 z0).trans_le (Finset.le_sup' alpha hz0)
  have hA1 : A < 1 := (Finset.sup'_lt_iff hs0).2 (fun z _ => ha1 z)
  have hB : 0 < B := (hk z0).trans_le (Finset.le_sup' k hz0)
  have hD : 0 < D := lt_min zero_lt_one
    ((Finset.lt_inf'_iff hs0).2 (fun z _ => hd z))
  have hsmall : ∀ᶠ t in Filter.atTop, |E (x t) - L| < D := by
    have h := (Metric.tendsto_nhds.1 hlim) D hD
    simpa only [Real.dist_eq] using h
  obtain ⟨a, ha⟩ := Filter.eventually_atTop.1
    ((Filter.eventually_ge_atTop (0 : ℝ)).and hsmall)
  refine ⟨a, A, B, (ha a le_rfl).1, hA0, hA1, hB, ?_⟩
  intro t ht
  have ht0 := (ha t ht).1
  have htD := (ha t ht).2
  obtain ⟨z, hz, htz⟩ := Set.mem_iUnion₂.mp (hs (hxK t ht0))
  have hlocalt := hineq z (x t) (hxK t ht0) htz
    (htD.trans_le ((min_le_right _ _).trans (Finset.inf'_le delta hz)))
  have hA := Finset.le_sup' alpha hz
  have hkB := Finset.le_sup' k hz
  by_cases heq : E (x t) - L = 0
  · rw [heq, abs_zero, Real.zero_rpow hA0.ne']
    exact mul_nonneg hB.le (norm_nonneg _)
  · calc
      |E (x t) - L| ^ A ≤ |E (x t) - L| ^ alpha z :=
        Real.rpow_le_rpow_of_exponent_ge (abs_pos.mpr heq)
          (htD.le.trans (min_le_left _ _)) hA
      _ ≤ k z * ‖gradient E (x t)‖ := hlocalt
      _ ≤ B * ‖gradient E (x t)‖ := mul_le_mul_of_nonneg_right hkB (norm_nonneg _)

/-- Constant zero energy on the singleton trajectory satisfies the
compact uniformization hypotheses with local exponent `1/2`;
Appendix D.1 of arXiv:2510.22026v2. -/
example : IsCompact ({0} : Set (EucSpace 1)) ∧
    ContinuousOn (fun _ : EucSpace 1 => (0 : ℝ)) {0} ∧
    (∀ t : ℝ, 0 ≤ t → (0 : EucSpace 1) ∈ ({0} : Set (EucSpace 1))) ∧
    Filter.Tendsto (fun _ : ℝ => (0 : ℝ)) Filter.atTop (nhds 0) ∧
    (∀ z ∈ ({0} : Set (EucSpace 1)), (0 : ℝ) = 0 →
      ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
        0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
          ∀ y ∈ V, |(0 : ℝ) - 0| ^ alpha ≤ k * ‖gradient (fun _ : EucSpace 1 => (0 : ℝ)) y‖) := by
  refine ⟨isCompact_singleton, continuousOn_const, fun _ _ => Set.mem_singleton 0,
    tendsto_const_nhds, fun z _ _ => ?_⟩
  refine ⟨1 / 2, 1, Set.univ, by norm_num, by norm_num, zero_lt_one,
    isOpen_univ, Set.mem_univ _, fun y _ => ?_⟩
  norm_num

/-- Conditional version of Appendix D.1, `lem: loj`, in
arXiv:2510.22026v2: supplying the local analytic gradient inequality at
critical points proves the manuscript's complete modulated convergence
conclusion. The additional input is explicit and is not claimed to follow
from analyticity by a theorem already proved in this repository.

The proof preserves the source's matrix bounds and divergent clock,
and requires no continuity of `M` or `λ`. Compactness uniformizes the
local inequalities. A desingularizing power potential controls displacement
directly; the divergent clock then forces the limiting point to be critical. -/
theorem lojasiewicz_modulated_of_local_gradient_inequality
    (N : ℕ) (U : Set (EucSpace N)) (hU : IsOpen U)
    (E : EucSpace N → ℝ) (hE : AnalyticOnNhd ℝ E U)
    (C : ℝ) (lam : ℝ → ℝ) (M : ℝ → ParamMatrix N)
    (hlam : ∀ t : ℝ, 0 ≤ t → 0 < lam t)
    (hMsymm : ∀ t : ℝ, 0 ≤ t → ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (M t v) w = inner (𝕜 := ℝ) v (M t w))
    (hMlower : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      lam t * ‖v‖ ^ 2 < inner (𝕜 := ℝ) (M t v) v)
    (hMupper : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      inner (𝕜 := ℝ) (M t v) v < C * lam t * ‖v‖ ^ 2)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (x : ℝ → EucSpace N) (K : Set (EucSpace N)) (hK : IsCompact K) (hKU : K ⊆ U)
    (hxK : ∀ t : ℝ, 0 ≤ t → x t ∈ K)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hlocal : ∀ z ∈ U, gradient E z = 0 →
      ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
        0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
          ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖) :
    ∃ xstar ∈ U, Filter.Tendsto x Filter.atTop (nhds xstar) ∧
      gradient E xstar = 0 := by
  have hdiff t (ht : 0 ≤ t) : DifferentiableAt ℝ E (x t) :=
    (hE.differentiableOn (x t) (hKU (hxK t ht))).differentiableAt
      (hU.mem_nhds (hKU (hxK t ht)))
  have hlower t (ht : 0 ≤ t) (v : EucSpace N) :
      lam t * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (M t v) v := by
    by_cases hv : v = 0
    · simp [hv]
    · exact (hMlower t ht v hv).le
  let D := max C 1
  have hD : 0 ≤ D := zero_le_one.trans (le_max_right _ _)
  have hupper t (ht : 0 ≤ t) (v : EucSpace N) :
      inner (𝕜 := ℝ) (M t v) v ≤ D * lam t * ‖v‖ ^ 2 := by
    by_cases hv : v = 0
    · simp [hv]
    · apply (hMupper t ht v hv).le.trans
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right (le_max_left C 1) (hlam t ht).le) (sq_nonneg _)
  have hanti := modulated_energy_antitoneOn E x M 0 hdiff hx
    (fun t ht => (mul_nonneg (hlam t ht).le (sq_nonneg _)).trans
      (hlower t ht (gradient E (x t))))
  have hcont : ContinuousOn E K := hE.continuousOn.mono hKU
  obtain ⟨L, hlimE, hL⟩ := compact_energy_limit E x K 0 hK hcont hxK hanti
  have hlocalK : ∀ z ∈ K, E z = L → ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
    intro z hz _
    by_cases hcritical : gradient E z = 0
    · exact hlocal z (hKU hz) hcritical
    · exact local_gradient_inequality_of_noncritical E z
        (hE z (hKU hz)).continuousAt
        (analytic_gradient_continuousAt E z (hE z (hKU hz))) hcritical
  obtain ⟨a, alpha, k, ha, ha0, ha1, hk, hKL⟩ :=
    uniform_gradient_inequality_of_local E x K L hK hcont hxK hlimE hlocalK
  obtain ⟨z, hz⟩ := modulated_converges_of_gradient_inequality E x M lam a L D k alpha
    hD hk ha1 (fun t ht => hdiff t (ha.trans ht)) (fun t ht => hx t (ha.trans ht))
    (fun t ht => hlam t (ha.trans ht)) (fun t ht => hlower t (ha.trans ht))
    (fun t ht => modulated_angle (M t) (lam t) D (hlam t (ha.trans ht)) hD
      (hMsymm t (ha.trans ht)) (hlower t (ha.trans ht)) (hupper t (ha.trans ht))
      (gradient E (x t)))
    (fun t ht => hL t (ha.trans ht))
    (fun t ht => by
      have habs : |E (x t) - L| = E (x t) - L :=
        abs_of_nonneg (sub_nonneg.mpr (hL t (ha.trans ht)))
      simpa only [habs] using hKL t ht) hlimE
  have hzK : z ∈ K := hK.isClosed.mem_of_tendsto hz (by
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with t ht
    exact hxK t ht)
  exact ⟨z, hKU hzK, hz, modulated_limit_critical E x M lam L z hdiff hx hlam hlower
    hint hL (analytic_gradient_continuousAt E z (hE z (hKU hzK))) hz⟩

end Transformer.Normalization

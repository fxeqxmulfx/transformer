/-
# Local scalar attraction under a converging original prefix

ArXiv:2411.04990v2, §5 and Appendix A, the local stability
step for the smooth frozen specialization of `thm:fixed_centers`.
A strictly negative actual own-angle derivative at a critical
angle gives small intervals with inward fields at both ends.
Continuity in the actual prefix keeps those boundary signs
for all sufficiently nearby original prefix states.

The scalar forward barriers then trap actual trajectories.
On an interval where the limiting scalar field is strictly
antitone and has a zero, a trapped actual trajectory converges
to that zero when its actual prefix converges. Every other
level inside the interval has an eventual strict derivative
sign; outside the interval it is eventually never reached.
The proved countable-level convergence theorem gives a limit,
its derivative limit vanishes, and strict antitonicity gives
the unique critical angle. No convergence rate or general
printed C1 sequential convergence lemma is assumed here.
-/

import Transformer.Causal.SectionA_ScalarConvergence

open Real Set Filter

namespace Transformer.Causal

/-- A smooth scalar row with continuous negative own-angle derivative at a genuine critical point has arbitrarily small inward intervals, uniformly over a nearby actual parameter neighborhood.
Source: arXiv:2411.04990v2, §5 and Appendix A, the local scalar stability step for the original frozen specialization of `thm:fixed_centers`. -/
theorem frozenScalar_inward_intervals {E : Type*} [NormedAddCommGroup E]
    (f d : E × ℝ → ℝ) (hF : Continuous f) (hD : Continuous d)
    (hdf : ∀ p u, HasDerivAt (fun v => f (p, v)) (d (p, u)) u)
    (pstar : E) (l : ℝ) (hcrit : f (pstar, l) = 0) (hneg : d (pstar, l) < 0) :
    ∃ R : ℝ, 0 < R ∧ ∀ ρ : ℝ, 0 < ρ → ρ ≤ R → ∃ δ : ℝ, 0 < δ ∧
      (∀ p : E, ‖p - pstar‖ < δ → 0 < f (p, l - ρ) ∧ f (p, l + ρ) < 0) ∧
      StrictAntiOn (fun u => f (pstar, u)) (Icc (l - ρ) (l + ρ)) := by
  have hd0 : Continuous (fun u : ℝ => d (pstar, u)) := hD.comp (continuous_const.prodMk continuous_id)
  obtain ⟨η, hη, hball⟩ := Metric.mem_nhds_iff.mp (hd0.continuousAt.eventually (eventually_lt_nhds hneg))
  refine ⟨η / 2, by positivity, ?_⟩
  intro ρ hρ hρη
  have hanti : StrictAntiOn (fun u => f (pstar, u)) (Icc (l - ρ) (l + ρ)) := by
    apply strictAntiOn_of_deriv_neg (convex_Icc _ _)
      (fun u hu => (hdf pstar u).continuousAt.continuousWithinAt)
    intro u hu
    rw [interior_Icc, mem_Ioo] at hu
    rw [(hdf pstar u).deriv]
    apply hball
    rw [Metric.mem_ball, Real.dist_eq, abs_lt]
    constructor <;> linarith [hu.1, hu.2]
  have hl : l ∈ Icc (l - ρ) (l + ρ) := ⟨by linarith, by linarith⟩
  have hlo : 0 < f (pstar, l - ρ) := by
    have h := hanti (show l - ρ ∈ Icc (l - ρ) (l + ρ) from ⟨le_rfl, by linarith⟩) hl (by linarith)
    simpa only [hcrit] using h
  have hhi : f (pstar, l + ρ) < 0 := by
    have h := hanti hl (show l + ρ ∈ Icc (l - ρ) (l + ρ) from ⟨by linarith, le_rfl⟩) (by linarith)
    simpa only [hcrit] using h
  have hleft : Continuous (fun p : E => f (p, l - ρ)) := hF.comp (continuous_id.prodMk continuous_const)
  have hright : Continuous (fun p : E => f (p, l + ρ)) := hF.comp (continuous_id.prodMk continuous_const)
  have he := (hleft.continuousAt.eventually (eventually_gt_nhds hlo)).and
    (hright.continuousAt.eventually (eventually_lt_nhds hhi))
  obtain ⟨δ, hδ, hδball⟩ := Metric.mem_nhds_iff.mp he
  refine ⟨δ, hδ, ?_, hanti⟩
  intro p hp
  apply hδball
  simpa only [Metric.mem_ball, dist_eq_norm] using hp

/-- Strict actual inward derivative signs at both endpoints keep an actual scalar trajectory in its initial closed interval for all forward times.
Source: arXiv:2411.04990v2, §5 and Appendix A, the forward-barrier part of local original frozen stability in `thm:fixed_centers`. -/
theorem frozenScalar_trapped_by_inward_levels (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (A B : ℝ) (hinit : r 0 ∈ Icc A B)
    (hlo : ∀ t, 0 ≤ t → r t = A → 0 < r' t)
    (hhi : ∀ t, 0 ≤ t → r t = B → r' t < 0) : ∀ t, 0 ≤ t → r t ∈ Icc A B := by
  have hl := csaScalar_forward_barrier r r' hd A 0 0 le_rfl hinit.1 hlo
  have hh := csaScalar_forward_barrier (fun t => -r t) (fun t => -r' t) (fun t => (hd t).neg)
    (-B) 0 0 le_rfl (neg_le_neg hinit.2) (fun t ht he => by
      have h := hhi t ht (by linarith)
      linarith)
  intro t ht
  exact ⟨hl t ht, neg_le_neg_iff.mp (hh t ht)⟩

/-- A trapped actual scalar row driven by a converging actual parameter converges to the unique zero of its strictly antitone limiting field on the trapping interval.
Source: arXiv:2411.04990v2, §5 and Appendix A, the local scalar attraction step for the smooth original frozen system in `thm:fixed_centers`. -/
theorem frozenScalar_tendsto_of_trapped_antitone {E : Type*} [TopologicalSpace E]
    (f : E × ℝ → ℝ) (hF : Continuous f) (p : ℝ → E) (pstar : E) (hp : Tendsto p atTop (nhds pstar))
    (r : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (f (p t, r t)) t) (A B l : ℝ)
    (hbound : ∀ᶠ t in atTop, r t ∈ Icc A B) (hl : l ∈ Icc A B)
    (hcrit : f (pstar, l) = 0) (hanti : StrictAntiOn (fun u => f (pstar, u)) (Icc A B)) :
    Tendsto r atTop (nhds l) := by
  obtain ⟨v, hv⟩ : ∃ v : ℝ, Tendsto r atTop (nhds v) := by
    apply frozenScalar_tendsto_of_countable_levels r (fun t => f (p t, r t)) hd A B hbound {l} (countable_singleton l)
    intro u hu
    by_cases hUI : u ∈ Icc A B
    · have hne : f (pstar, u) ≠ 0 := by
        intro hz
        apply hu
        rw [mem_singleton_iff]
        exact hanti.injOn hUI hl (hz.trans hcrit.symm)
      have hlim : Tendsto (fun t => f (p t, u)) atTop (nhds (f (pstar, u))) :=
        (hF.tendsto (pstar, u)).comp (hp.prodMk_nhds tendsto_const_nhds)
      rcases lt_or_gt_of_ne hne with hneg | hpos
      · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_lt_const hneg)
        exact ⟨T, Or.inr (fun t ht he => by rw [he]; exact hT t ht)⟩
      · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_const_lt hpos)
        exact ⟨T, Or.inl (fun t ht he => by rw [he]; exact hT t ht)⟩
    · obtain ⟨T, hT⟩ := eventually_atTop.mp hbound
      exact ⟨T, Or.inl (fun t ht he => False.elim (hUI (he ▸ hT t ht)))⟩
  have hfield : Tendsto (fun t => f (p t, r t)) atTop (nhds (f (pstar, v))) :=
    (hF.tendsto (pstar, v)).comp (hp.prodMk_nhds hv)
  have hz := frozenScalar_derivative_limit_eq_zero r _ hd A B hbound _ hfield
  have hvI := isClosed_Icc.mem_of_tendsto hv hbound
  have heq : v = l := hanti.injOn hvI hl (hz.trans hcrit.symm)
  simpa only [heq] using hv

/-- The genuinely parameter-dependent field `p-u` jointly realizes the local smoothness, criticality, and strictly negative derivative premises.
Source: arXiv:2411.04990v2, §5 and Appendix A, an actual parameter-dependent witness for the local scalar stability argument. -/
example : ∃ R : ℝ, 0 < R ∧ ∀ ρ : ℝ, 0 < ρ → ρ ≤ R → ∃ δ : ℝ, 0 < δ ∧
    (∀ p : ℝ, ‖p - 0‖ < δ → 0 < p - (0 - ρ) ∧ p - (0 + ρ) < 0) ∧
    StrictAntiOn (fun u : ℝ => 0 - u) (Icc (0 - ρ) (0 + ρ)) := by
  apply frozenScalar_inward_intervals (fun q : ℝ × ℝ => q.1 - q.2) (fun _ => (-1 : ℝ))
    (by fun_prop) continuous_const _ 0 0 (by norm_num) (by norm_num)
  intro p u
  simpa! only [zero_sub, id_eq] using (hasDerivAt_const u p).fun_sub (hasDerivAt_id u)

/-- A genuine nonconstant actual scalar solution realizes the convergence and trapping premises jointly.
Source: arXiv:2411.04990v2, §5 and Appendix A, the auxiliary smooth scalar attraction argument for the original frozen specialization. -/
example : Tendsto (fun t : ℝ => Real.exp (-t)) atTop (nhds 0) := by
  apply frozenScalar_tendsto_of_trapped_antitone (fun q : ℝ × ℝ => q.1 - q.2) (by fun_prop)
    (fun _ => 0) 0 tendsto_const_nhds (fun t => Real.exp (-t)) _ 0 1 0 _ ⟨le_rfl, zero_le_one⟩ (by norm_num) _
  · intro t
    simpa only [id_eq, mul_neg, mul_one, zero_sub, Pi.neg_apply] using (hasDerivAt_id t).neg.exp
  · apply (eventually_ge_atTop (0 : ℝ)).mono
    intro t ht
    exact ⟨(Real.exp_pos _).le, Real.exp_le_one_iff.mpr (by linarith)⟩
  · intro u hu v hv huv
    dsimp only
    linarith

/-- The same genuine nonconstant scalar trajectory satisfies the actual strict inward boundary conditions of the trapping argument.
Source: arXiv:2411.04990v2, §5 and Appendix A, the scalar forward barriers used for local original frozen stability. -/
example : ∀ t : ℝ, 0 ≤ t → Real.exp (-t) ∈ Icc (-1 : ℝ) 1 := by
  apply frozenScalar_trapped_by_inward_levels (fun t => Real.exp (-t)) (fun t => -Real.exp (-t))
    (fun t => by simpa only [id_eq, mul_neg, mul_one, Pi.neg_apply] using (hasDerivAt_id t).neg.exp)
    (-1) 1 (by norm_num)
  · intro t ht he
    rw [he]
    norm_num
  · intro t ht he
    rw [he]
    norm_num

end Transformer.Causal

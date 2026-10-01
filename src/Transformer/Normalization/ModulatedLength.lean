/-
# Displacement bounds and the constant-energy branch

The finite-length argument of Appendix D.1 of arXiv:2510.22026v2 uses
the mean value theorem, without extra regularity of the modulation.
-/

import Transformer.Normalization.ModulatedEnergy
import Mathlib.Topology.MetricSpace.Cauchy
import Mathlib.Analysis.Calculus.TangentCone.Real

namespace Transformer.Normalization

/-- If the decrease of a differentiable scalar potential dominates speed,
it controls the trajectory's displacement. This is the finite-length
argument of Appendix D.1, `lem: loj`, in arXiv:2510.22026v2, expressed
without an integral or continuity of the derivatives. -/
theorem displacement_le_potential_drop {N : ℕ}
    (x x' : ℝ → EucSpace N) (p p' : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b)
    (hx : ∀ t ∈ Set.Icc a b, HasDerivAt x (x' t) t)
    (hp : ∀ t ∈ Set.Icc a b, HasDerivAt p (p' t) t)
    (hbound : ∀ t ∈ Set.Icc a b, ‖x' t‖ ≤ -p' t) :
    ‖x b - x a‖ ≤ p a - p b := by
  have hanti : AntitoneOn p (Set.Icc a b) := by
    apply antitoneOn_of_hasDerivWithinAt_nonpos (convex_Icc a b)
    · intro t ht
      exact (hp t ht).continuousAt.continuousWithinAt
    · intro t ht
      exact (hp t (interior_subset ht)).hasDerivWithinAt
    · intro t ht
      have hb := hbound t (interior_subset ht)
      have hn := norm_nonneg (x' t)
      linarith
  have hdrop : 0 ≤ p a - p b := sub_nonneg.mpr
    (hanti (Set.left_mem_Icc.mpr hab) (Set.right_mem_Icc.mpr hab) hab)
  let v := x b - x a
  by_cases hv : v = 0
  · change ‖v‖ ≤ p a - p b
    rw [hv, norm_zero]
    exact hdrop
  have hd t (ht : t ∈ Set.Icc a b) :
      HasDerivAt (fun s => inner (𝕜 := ℝ) v (x s) + ‖v‖ * p s)
        (inner (𝕜 := ℝ) v (x' t) + ‖v‖ * p' t) t := by
    convert ((innerSL ℝ v).hasFDerivAt.comp_hasDerivAt t (hx t ht)).add
      ((hp t ht).const_mul ‖v‖) using 1
    · ext s
      rfl
    · rfl
  have hprojection : AntitoneOn
      (fun t => inner (𝕜 := ℝ) v (x t) + ‖v‖ * p t) (Set.Icc a b) := by
    apply antitoneOn_of_hasDerivWithinAt_nonpos (convex_Icc a b)
    · intro t ht
      exact (hd t ht).continuousAt.continuousWithinAt
    · intro t ht
      exact (hd t (interior_subset ht)).hasDerivWithinAt
    · intro t ht
      have hb := mul_le_mul_of_nonneg_left (hbound t (interior_subset ht))
        (norm_nonneg v)
      have hi := real_inner_le_norm v (x' t)
      linarith
  have h := hprojection (Set.left_mem_Icc.mpr hab) (Set.right_mem_Icc.mpr hab) hab
  have hi : inner (𝕜 := ℝ) v (x b) - inner (𝕜 := ℝ) v (x a) = ‖v‖ ^ 2 := by
    rw [← inner_sub_right]
    exact real_inner_self_eq_norm_sq v
  have hn : 0 < ‖v‖ := norm_pos_iff.mpr hv
  have hmul : ‖v‖ * ‖v‖ ≤ ‖v‖ * (p a - p b) := by nlinarith
  exact (mul_le_mul_iff_right₀ hn).mp hmul

/-- The displacement hypotheses have a nonconstant witness: a straight
line and the potential `p(t) = -t` on `[0,1]`, with unit speed; this is the
estimate used in Appendix D.1 of arXiv:2510.22026v2. -/
example (v : EucSpace 1) (hv : ‖v‖ = 1) :
    (∀ t ∈ Set.Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s • v) v t) ∧
    (∀ t ∈ Set.Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => -s) (-1) t) ∧
    (∀ t ∈ Set.Icc (0 : ℝ) 1, ‖v‖ ≤ -(-1 : ℝ)) := by
  refine ⟨fun t _ => ?_, fun t _ => hasDerivAt_neg' t,
    fun _ _ => by simp [hv]⟩
  simpa using (hasDerivAt_id t).smul_const v

/-- The displacement bound makes a complete-space trajectory converge
when the potential tends to zero. This closes the finite-length argument
of Appendix D.1 of arXiv:2510.22026v2 once a desingularizing potential is
available. The proof uses only the stated displacement bound and convergence
of the potential. -/
theorem converges_of_displacement_bound {N : ℕ}
    (x : ℝ → EucSpace N) (p : ℝ → ℝ) (a : ℝ)
    (hp : Filter.Tendsto p Filter.atTop (nhds 0))
    (hbound : ∀ s t : ℝ, a ≤ s → s ≤ t → ‖x t - x s‖ ≤ p s) :
    ∃ z : EucSpace N, Filter.Tendsto x Filter.atTop (nhds z) := by
  let y : ℝ → EucSpace N := fun t => x (max a t)
  have hc : CauchySeq y := by
    apply cauchySeq_of_le_tendsto_0' (fun t => p (max a t))
    · intro s t hst
      rw [dist_eq_norm]
      simpa [y, norm_sub_rev] using
        hbound (max a s) (max a t) (le_max_left _ _) (max_le_max_left a hst)
    · apply hp.comp (Filter.tendsto_atTop_mono (fun t => le_max_right a t)
        Filter.tendsto_id)
  obtain ⟨z, hz⟩ := cauchySeq_tendsto_of_complete hc
  refine ⟨z, hz.congr' ?_⟩
  filter_upwards [Filter.eventually_ge_atTop a] with t ht
  simp [y, max_eq_right ht]

/-- A constant path and the zero potential satisfy the convergence
estimate from Appendix D.1 of arXiv:2510.22026v2. -/
example : Filter.Tendsto (fun _ : ℝ => (0 : ℝ)) Filter.atTop (nhds 0) ∧
    (∀ s t : ℝ, 0 ≤ s → s ≤ t → ‖(0 : EucSpace 1) - 0‖ ≤ (0 : ℝ)) := by
  exact ⟨tendsto_const_nhds, fun _ _ _ _ => by simp⟩

/-- Under the positive lower matrix bound in Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2, a flow with constant energy on a half-line is
constant there. One-sided derivative uniqueness covers its initial time. -/
theorem modulated_constant_of_energy_plateau {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N) (lam : ℝ → ℝ) (a L : ℝ)
    (hE : ∀ t, a ≤ t → DifferentiableAt ℝ E (x t))
    (hx : ∀ t, a ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hlam : ∀ t, a ≤ t → 0 < lam t)
    (hM : ∀ t, a ≤ t → ∀ v : EucSpace N,
      lam t * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (M t v) v)
    (hconst : ∀ t, a ≤ t → E (x t) = L) :
    ∀ t, a ≤ t → x t = x a := by
  have hvel t (ht : a ≤ t) : HasDerivAt x 0 t := by
    have hd := modulated_energy_hasDerivAt E x M t (hE t ht) (hx t ht)
    have hc : HasDerivWithinAt (fun s => E (x s)) 0 (Set.Ici a) t :=
      (hasDerivWithinAt_const t (Set.Ici a) L).congr hconst (hconst t ht)
    have hu := uniqueDiffOn_Ici a t ht
    have hq : -inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t)) = 0 :=
      (hd.hasDerivWithinAt.derivWithin hu).symm.trans (hc.derivWithin hu)
    have hm := hM t ht (gradient E (x t))
    have hl := hlam t ht
    have hn := sq_nonneg ‖gradient E (x t)‖
    have hz : ‖gradient E (x t)‖ ^ 2 = 0 := by nlinarith
    have hg : gradient E (x t) = 0 := norm_eq_zero.mp (by nlinarith [norm_nonneg (gradient E (x t))])
    simpa only [hg, map_zero, neg_zero] using hx t ht
  intro t ht
  have h := displacement_le_potential_drop x (fun _ => 0) (fun _ => 0) (fun _ => 0)
    a t ht (fun s hs => hvel s hs.1) (fun s _ => hasDerivAt_const s (0 : ℝ))
    (fun _ _ => by simp)
  have hz : ‖x t - x a‖ = 0 := le_antisymm (by simpa only [sub_self] using h) (norm_nonneg _)
  exact sub_eq_zero.mp (norm_eq_zero.mp hz)

/-- The constant-flow hypotheses are satisfied by zero energy and the
identity modulation; Appendix D.1 of arXiv:2510.22026v2. -/
example :
    (∀ t : ℝ, 0 ≤ t → DifferentiableAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun _ : ℝ => (0 : EucSpace 1)) 0 t) ∧
    (∀ t : ℝ, 0 ≤ t → (0 : ℝ) < 1) ∧
    (∀ v : EucSpace 1, (1 : ℝ) * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) v v) ∧
    (∀ t : ℝ, 0 ≤ t → (0 : ℝ) = 0) := by
  exact ⟨fun _ _ => differentiableAt_const 0, fun t _ => hasDerivAt_const t 0,
    fun _ _ => zero_lt_one, fun v => by simp, fun _ _ => rfl⟩

end Transformer.Normalization

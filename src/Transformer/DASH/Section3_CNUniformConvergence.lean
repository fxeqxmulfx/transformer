/-
# DASH — uniform CN convergence on compact spectral sets

arXiv:2602.02016v2, §3.2–3.3. Uniform convergence on compact subsets
of the corrected positive CN domain justifies evaluating the second
inverse-root call on a changing first-call approximation.
-/

import Transformer.DASH.Section3_CNConvergence
import Mathlib.Topology.UniformSpace.Dini

open scoped Topology
open Filter

noncomputable section

namespace Transformer.DASH

/-- Each finite product iterate is continuous in its scalar input.
Source: arXiv:2602.02016v2, §3.2, the polynomial CN recurrence. -/
theorem cnScalarM_continuous_input (p : ℕ) (c : ℝ) (k : ℕ) :
    Continuous (fun a : ℝ => cnScalarM p a c k) := by
  induction k with
  | zero => unfold cnScalarM; fun_prop
  | succ k ih =>
    have hmap : Continuous (cnMap p) := by unfold cnMap cnFactor; fun_prop
    exact hmap.comp ih

/-- Each finite inverse-root iterate is continuous in the input.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
theorem cnScalarX_continuous_input (p : ℕ) (c : ℝ) (k : ℕ) :
    Continuous (fun a : ℝ => cnScalarX p a c k) := by
  induction k with
  | zero => exact continuous_const
  | succ k ih =>
    have hM := cnScalarM_continuous_input p c k
    have hfactor : Continuous (cnFactor p) := by unfold cnFactor; fun_prop
    exact ih.mul (hfactor.comp hM)

/-- After the first step the inverse-root iterates are monotone increasing
throughout the corrected domain. Source: arXiv:2602.02016v2, §3.2, CN convergence. -/
theorem cnScalarX_monotone_after_first (p : ℕ) (a c : ℝ) (hp : 0 < p)
    (ha : 0 < a) (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) :
    Monotone (fun k : ℕ => cnScalarX p a c (k + 1)) := by
  apply monotone_nat_of_le_succ
  intro k
  have hp' : (0 : ℝ) < p := by exact_mod_cast hp
  have hM := (cnScalarM_bounds p a c hp ha hc hupper k).2
  have hfactor : 1 ≤ cnFactor p (cnScalarM p a c (k + 1)) := by
    have hgap : 0 ≤ (1 - cnScalarM p a c (k + 1)) / (p : ℝ) :=
      div_nonneg (sub_nonneg.mpr hM) hp'.le
    unfold cnFactor
    rw [sub_div] at hgap
    linarith
  change cnScalarX p a c (k + 1) ≤
    cnScalarX p a c (k + 1) * cnFactor p (cnScalarM p a c (k + 1))
  simpa only [mul_one] using mul_le_mul_of_nonneg_left hfactor
    (cnScalarX_pos p a c hp ha hc hupper (k + 1)).le

/-- Monotonicity assumptions have a valid order-four instance,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (4 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((4 : ℝ) + 1) * (1 : ℝ) ^ (4 : ℕ) := by norm_num

/-- CN inverse-root convergence is uniform on any compact subset of the
corrected open spectral domain. Dini's theorem applies to the continuous,
monotone iterates after the first step and their continuous inverse-power
limit. Source: arXiv:2602.02016v2, §3.2–3.3, convergence and solver chaining. -/
theorem cnScalarX_uniform_on_compact (p : ℕ) (c : ℝ) (S : Set ℝ)
    (hp : 0 < p) (hc : 0 < c) (hS : IsCompact S)
    (hpos : ∀ a ∈ S, 0 < a) (hupper : ∀ a ∈ S, a < ((p : ℝ) + 1) * c ^ p) :
    TendstoUniformlyOn (fun k a => cnScalarX p a c (k + 1))
      (fun a => a ^ (-(1 / (p : ℝ)))) atTop S := by
  apply Monotone.tendstoUniformlyOn_of_forall_tendsto hS
  · intro k
    exact (cnScalarX_continuous_input p c (k + 1)).continuousOn
  · intro a ha
    exact cnScalarX_monotone_after_first p a c hp (hpos a ha) hc (hupper a ha)
  · intro a ha
    exact (Real.continuousAt_rpow_const a _ (Or.inl (hpos a ha).ne')).continuousWithinAt
  · intro a ha
    exact (tendsto_add_atTop_iff_nat 1).2
      (cnScalarX_tendsto p a c hp (hpos a ha) hc (hupper a ha))

/-- A nonempty compact spectral interval satisfies the uniform hypotheses.
Source: arXiv:2602.02016v2, §3.2–3.3. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ IsCompact (Set.Icc (1 / 2 : ℝ) 1) ∧
    (∀ a ∈ Set.Icc (1 / 2 : ℝ) 1, 0 < a) ∧
    (∀ a ∈ Set.Icc (1 / 2 : ℝ) 1, a < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ)) := by
  refine ⟨by norm_num, by norm_num, isCompact_Icc, ?_, ?_⟩
  · intro a ha
    linarith [ha.1]
  · intro a ha
    norm_num
    linarith [ha.2]

/-- A changing scalar input converging inside the corrected CN domain can
be evaluated at a growing iteration count. Uniform convergence supplies
the interchange needed for two finite solver calls, rather than replacing
the first call by an exact root.
Source: arXiv:2602.02016v2, §3.2–3.3, inverse-root solver composition. -/
theorem cnScalarX_moving_input (p : ℕ) (c μ : ℝ) (a : ℕ → ℝ)
    (hp : 0 < p) (hc : 0 < c) (hμ : 0 < μ) (hμ' : μ < ((p : ℝ) + 1) * c ^ p)
    (ha : Tendsto a atTop (𝓝 μ)) :
    Tendsto (fun k : ℕ => cnScalarX p (a k) c (k + 1)) atTop
      (𝓝 (μ ^ (-(1 / (p : ℝ))))) := by
  let L : ℝ := ((p : ℝ) + 1) * c ^ p
  let S : Set ℝ := Set.Icc (μ / 2) ((μ + L) / 2)
  have hlow : μ / 2 < μ := by linarith
  have hhigh : μ < (μ + L) / 2 := by dsimp [L]; linarith
  have hpos : ∀ x ∈ S, 0 < x := by
    intro x hx
    have hx' : μ / 2 ≤ x := hx.1
    linarith
  have hupper : ∀ x ∈ S, x < ((p : ℝ) + 1) * c ^ p := by
    intro x hx
    have hx' : x ≤ (μ + L) / 2 := hx.2
    dsimp [L] at hx'
    linarith
  have hunif := cnScalarX_uniform_on_compact p c S hp hc isCompact_Icc hpos hupper
  have herror : TendstoUniformlyOn
      (fun k x => cnScalarX p x c (k + 1) - x ^ (-(1 / (p : ℝ))))
      (fun _ => (0 : ℝ)) atTop S := by
    rw [Metric.tendstoUniformlyOn_iff] at hunif ⊢
    intro δ hδ
    filter_upwards [hunif δ hδ] with k hk
    intro x hx
    simpa only [Real.dist_eq, zero_sub, abs_neg, abs_sub_comm] using hk x hx
  have hmem : ∀ᶠ k : ℕ in atTop, a k ∈ S := by
    filter_upwards [ha.eventually_const_lt hlow, ha.eventually_lt_const hhigh] with k hk hk'
    exact ⟨hk.le, hk'.le⟩
  have heprod : Tendsto
      (fun z : ℕ × ℝ => cnScalarX p z.2 c (z.1 + 1) - z.2 ^ (-(1 / (p : ℝ))))
      (atTop ×ˢ 𝓟 S) (𝓝 (0 : ℝ)) := tendsto_prod_principal_iff.2 herror
  have he := heprod.comp (tendsto_id.prodMk (tendsto_principal.2 hmem))
  have htarget :=
    (Real.continuousAt_rpow_const μ (-(1 / (p : ℝ))) (Or.inl hμ.ne')).tendsto.comp ha
  simpa only [Function.comp_def, id_eq, sub_add_cancel, zero_add] using he.add htarget

/-- Moving-input assumptions are satisfiable on a nonempty CN domain,
arXiv:2602.02016v2, §3.2–3.3. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) ∧
    Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, tendsto_const_nhds⟩

end Transformer.DASH

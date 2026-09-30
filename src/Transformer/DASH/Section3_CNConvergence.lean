/-
# DASH — convergence of Coupled Newton

arXiv:2602.02016v2, §3.2. Positive integer orders, positive scaling,
and the open initial spectral interval are explicit. No convergence
premise is assumed: the product is bounded and monotone after one step,
and its positive fixed-point limit is one.
-/

import Transformer.DASH.Section3_CoupledNewton
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

noncomputable section

namespace Transformer.DASH

/-- The CN auxiliary product converges to one under the corrected open
interval condition. The source incorrectly includes zero and the upper
endpoint; `cn_zero_product` and `cn_upper_endpoint` refute those endpoints.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
theorem cnScalarM_tendsto (p : ℕ) (a c : ℝ) (hp : 0 < p) (ha : 0 < a)
    (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) :
    Filter.Tendsto (cnScalarM p a c) Filter.atTop (nhds 1) := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  let u : ℕ → ℝ := fun k => cnScalarM p a c (k + 1)
  have hu : ∀ k, 0 < u k ∧ u k ≤ 1 := cnScalarM_bounds p a c hp ha hc hupper
  have hstep : ∀ k, u (k + 1) = cnMap p (u k) := fun _ => rfl
  have hmono : Monotone u := monotone_nat_of_le_succ fun k => by
    rw [hstep]
    exact cnMap_ge_self p _ hp (hu k).1.le (hu k).2
  have hb : BddAbove (Set.range u) := ⟨1, by rintro x ⟨k, rfl⟩; exact (hu k).2⟩
  let L := ⨆ k, u k
  have hlim : Filter.Tendsto u Filter.atTop (nhds L) := tendsto_atTop_ciSup hmono hb
  have hLpos : 0 < L := lt_of_lt_of_le (hu 0).1 (le_ciSup hb 0)
  have hLle : L ≤ 1 := ciSup_le (fun k => (hu k).2)
  have hcont : Continuous (cnMap p) := by unfold cnMap cnFactor; fun_prop
  have hfixed : cnMap p L = L := by
    have hf := (hcont.tendsto L).comp hlim
    have hs := (Filter.tendsto_add_atTop_iff_nat 1).mpr hlim
    have hf' : Filter.Tendsto (fun k => u (k + 1)) Filter.atTop (nhds (cnMap p L)) := by
      simpa only [Function.comp_def, hstep] using hf
    exact tendsto_nhds_unique hf' hs
  have hLone : L = 1 := by
    apply le_antisymm hLle
    by_contra hn
    have hlt : L < 1 := lt_of_not_ge hn
    have hgap : 0 < (1 - L) / (p : ℝ) := div_pos (sub_pos.mpr hlt) hp'
    have hfactor : 1 < cnFactor p L := by
      unfold cnFactor
      rw [sub_div] at hgap
      linarith
    have hpow := one_lt_pow₀ hfactor hp.ne'
    have hmul := mul_lt_mul_of_pos_left hpow hLpos
    unfold cnMap at hfixed
    nlinarith
  rw [hLone] at hlim
  exact (Filter.tendsto_add_atTop_iff_nat 1).mp hlim

/-- The convergence domain is nonempty, arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by norm_num

/-- The CN inverse-root iterates remain positive,
arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
theorem cnScalarX_pos (p : ℕ) (a c : ℝ) (hp : 0 < p) (ha : 0 < a)
    (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) (k : ℕ) :
    0 < cnScalarX p a c k := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  have hM : ∀ j, cnScalarM p a c j < (p : ℝ) + 1 := by
    intro j
    cases j with
    | zero => exact (div_lt_iff₀ (pow_pos hc p)).mpr hupper
    | succ j => have h := (cnScalarM_bounds p a c hp ha hc hupper j).2; linarith
  induction k with
  | zero => exact inv_pos.mpr hc
  | succ k ih => exact mul_pos ih (cnFactor_pos p _ hp (hM k))

/-- Positive iterates have satisfiable assumptions,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (4 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((4 : ℝ) + 1) * (1 : ℝ) ^ (4 : ℕ) := by norm_num

/-- CN converges to the positive inverse `p`-th root, for every positive
integer order. This is the source's convergence claim with its endpoint
error corrected and the required positivity conditions made explicit.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-init`–`equation:CN-X-M`. -/
theorem cnScalarX_tendsto (p : ℕ) (a c : ℝ) (hp : 0 < p) (ha : 0 < a)
    (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) :
    Filter.Tendsto (cnScalarX p a c) Filter.atTop (nhds (a ^ (-(1 / (p : ℝ))))) := by
  have hpne : (p : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
  have hrepr : ∀ k, cnScalarX p a c k =
      (cnScalarM p a c k / a) ^ (p : ℝ)⁻¹ := by
    intro k
    rw [cnScalar_invariant, mul_div_cancel_left₀ _ ha.ne', ← Real.rpow_natCast,
      Real.rpow_rpow_inv (cnScalarX_pos p a c hp ha hc hupper k).le hpne]
  have hdiv := (cnScalarM_tendsto p a c hp ha hc hupper).div tendsto_const_nhds ha.ne'
  have hroot := hdiv.rpow_const (p := (p : ℝ)⁻¹) (Or.inl (one_div_ne_zero ha.ne'))
  have htarget : (1 / a) ^ (p : ℝ)⁻¹ = a ^ (-(1 / (p : ℝ))) := by
    rw [one_div, ← Real.rpow_neg_eq_inv_rpow, one_div]
  rw [htarget] at hroot
  simpa only [Pi.div_apply, ← hrepr] using hroot

/-- Inverse-root convergence assumptions coexist,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (4 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((4 : ℝ) + 1) * (1 : ℝ) ^ (4 : ℕ) := by norm_num

end Transformer.DASH

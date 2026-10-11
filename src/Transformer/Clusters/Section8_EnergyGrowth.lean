/-
# A convergent energy for each rescaled token

ArXiv:2305.05465v6, §8.1, `e:derivativenorm` and the boundedness
`c:theziunifbdd` used in the proof of `t:Idcase11`. Subtracting
`n exp(-2t)` from `||A z_i(t)||²` absorbs the global softmax error.
The resulting energy is monotone at every real time. It is bounded above
on future times by the initial token hull, and hence on the whole line
by monotonicity. Thus both the corrected energy and `||A z_i(t)||²`
have finite limits as time tends to infinity.

The proof gives a separate limit for each token. It does not yet conclude
that token positions converge, nor assume their convergence as an input.
Positive definiteness is unnecessary for this energy estimate.
-/

import Transformer.Clusters.Section8_SoftmaxEnergy
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.MonotoneConvergence

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The energy in `e:derivativenorm` with the softmax error integrated and
subtracted. Source: arXiv:2305.05465v6, §8.1, Step 2 of `t:Idcase11`.
The source uses direct growth estimates; this correction accounts for the
same exponentially decaying error uniformly in the token configuration. -/
noncomputable def correctedTokenEnergy (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (i : Idx n) (t : ℝ) : ℝ :=
  ‖A (Z t i)‖ ^ 2 - (n : ℝ) * Real.exp (-(2 * t))

/-- The corrected energy derivative dominates twice every competing
attention gap. Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm`.
The derivative is therefore nonnegative by choosing the self token. -/
theorem IdRescaledDynamics.corrected_energy_deriv_bound (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (t : ℝ) (i : Idx n) :
    ∃ v : ℝ, HasDerivAt (correctedTokenEnergy A Z i) v t ∧
      ∀ j : Idx n, 2 * inner (𝕜 := ℝ) (A (Z t i)) (A (Z t j - Z t i)) ≤ v := by
  obtain ⟨v, hv, hbound⟩ := hZ.norm_sq_deriv_lower_bound A Z t i
  have hexp : HasDerivAt (fun s : ℝ => Real.exp (-(2 * s)))
      (Real.exp (-(2 * t)) * (-2)) t := by
    simpa only [id_eq, mul_one, neg_mul] using ((hasDerivAt_id t).const_mul 2).fun_neg.exp
  have hd := hv.fun_sub (hexp.const_mul (n : ℝ))
  refine ⟨v + 2 * (n : ℝ) * Real.exp (-(2 * t)), hd.congr_deriv ?_, ?_⟩
  · ring
  · intro j
    have hj := hbound j
    rw [div_eq_mul_inv, ← Real.exp_neg] at hj
    linarith

/-- The derivative estimate has an actual stationary two-token instance;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

/-- The actual derivative of the corrected energy is nonnegative.
Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm`, with the
self-token comparison and the integrated softmax error. -/
theorem IdRescaledDynamics.corrected_energy_deriv_nonneg (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (t : ℝ) (i : Idx n) :
    0 ≤ deriv (correctedTokenEnergy A Z i) t := by
  obtain ⟨v, hv, hbound⟩ := hZ.corrected_energy_deriv_bound A Z t i
  rw [hv.deriv]
  simpa only [sub_self, map_zero, inner_zero_right, mul_zero] using hbound i

/-- The derivative sign hypothesis holds on an actual stationary
two-token solution, at every time. Source: arXiv:2305.05465v6, §8.1. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

/-- The corrected energy is nondecreasing at all real times.
Source: arXiv:2305.05465v6, §8.1, the energy-growth argument in `t:Idcase11`.
This global monotonicity includes times near candidate points. -/
theorem IdRescaledDynamics.monotone_corrected_energy (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    Monotone (correctedTokenEnergy A Z i) := by
  choose v hv hbound using fun t => hZ.corrected_energy_deriv_bound A Z t i
  apply monotone_of_hasDerivAt_nonneg hv
  intro t
  change 0 ≤ v t
  simpa only [sub_self, map_zero, inner_zero_right, mul_zero] using hbound t i

/-- The monotonicity hypothesis holds for stationary coincident tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

/-- The corrected energy is bounded above, using the initial hull for
future times and monotonicity for negative times.
Source: arXiv:2305.05465v6, §8.1, `c:theziunifbdd` and `e:derivativenorm`. -/
theorem IdRescaledDynamics.bddAbove_corrected_energy (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    BddAbove (Set.range (correctedTokenEnergy A Z i)) := by
  obtain ⟨R, hR⟩ := exists_bound_idRescaled A Z hZ
  have hR0 : 0 ≤ R := (norm_nonneg (Z 0 i)).trans (hR i 0 le_rfl)
  let B := (‖A‖ * R) ^ 2
  have hfuture (t : ℝ) (ht : 0 ≤ t) : correctedTokenEnergy A Z i t ≤ B := by
    have hnorm : ‖A (Z t i)‖ ≤ ‖A‖ * R :=
      A.le_opNorm (Z t i) |>.trans (mul_le_mul_of_nonneg_left (hR i t ht) (norm_nonneg A))
    have hsq : ‖A (Z t i)‖ ^ 2 ≤ B :=
      (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg A) hR0)).mpr hnorm
    exact (sub_le_self _ (mul_nonneg (Nat.cast_nonneg n) (Real.exp_pos _).le)).trans hsq
  refine ⟨B, ?_⟩
  rintro _ ⟨t, rfl⟩
  rcases le_or_gt 0 t with ht | ht
  · exact hfuture t ht
  · exact (hZ.monotone_corrected_energy A Z i ht.le).trans (hfuture 0 le_rfl)

/-- A genuine bounded stationary solution satisfies the energy-boundedness
hypothesis. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

/-- The corrected energy converges to its finite supremum.
Source: arXiv:2305.05465v6, §8.1, the bounded-energy argument in the proof
of `t:Idcase11`, applied to the global softmax correction. -/
theorem IdRescaledDynamics.tendsto_corrected_energy (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    Tendsto (correctedTokenEnergy A Z i) atTop (nhds (⨆ t : ℝ, correctedTokenEnergy A Z i t)) :=
  tendsto_atTop_ciSup (hZ.monotone_corrected_energy A Z i)
    (hZ.bddAbove_corrected_energy A Z i)

/-- The convergence hypothesis holds for the unit stationary solution;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

/-- Each squared token norm in the attention metric has a finite limit.
Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm` and boundedness.
The vanishing correction transfers the corrected-energy limit to the norm. -/
theorem IdRescaledDynamics.exists_limit_norm_sq (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    ∃ L : ℝ, Tendsto (fun t => ‖A (Z t i)‖ ^ 2) atTop (nhds L) := by
  have hexp : Tendsto (fun t : ℝ => Real.exp (-(2 * t))) atTop (nhds 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp
      (tendsto_id.const_mul_atTop (by norm_num : (0 : ℝ) < 2))
  refine ⟨⨆ t : ℝ, correctedTokenEnergy A Z i t, ?_⟩
  simpa only [correctedTokenEnergy, sub_add_cancel, mul_zero, add_zero] using
    (hZ.tendsto_corrected_energy A Z i).add (hexp.const_mul (n : ℝ))

/-- The norm-limit hypotheses are realized by actual nonzero unit tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

end Transformer.Clusters

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Domain

/-!
# Feasible finite cross-entropy queries for learned recurrent mixtures

Source: `review/flat_spectrum.md`, CM95. A unit initial state and a unit
readout taking it to one supply an actual constant class-bias component on the
same state space. The scalar complex space provides an explicit inhabited
example. No assumption that a zero-dimensional state admits this bias is made.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent

variable {O E : Type*} [DecidableEq O] [NormedAddCommGroup E] [NormedSpace ℂ E]

/-- Actual signed class-bias recurrence: learned state/readout parameters
with identity transitions, CM95. This is a feasible loss query, not an answer
rule supplied to the learning model. -/
noncomputable def biasParameters (v : E) (L : E →L[ℂ] ℂ) (target : O) (s : ℝ) : Parameters O E :=
  ⟨v, fun _ => ContinuousLinearMap.id ℂ E, fun c => if c = target then (s : ℂ) • L else 0⟩

/-- The full bias recurrence satisfies the declared operator capacities,
CM95. -/
theorem bias_unit (v : E) (L : E →L[ℂ] ℂ) (target : O) (s : ℝ)
    (hv : ‖v‖ ≤ 1) (hL : ‖L‖ ≤ 1) (hs : |s| ≤ 1) : Unit (biasParameters v L target s) := by
  refine ⟨hv, fun _ => ContinuousLinearMap.norm_id_le, ?_⟩
  intro c
  by_cases hc : c = target
  · simp only [biasParameters, hc, ite_true, norm_smul, Complex.norm_real, Real.norm_eq_abs]
    simpa using mul_le_mul hs hL (norm_nonneg L) (by norm_num : (0 : ℝ) ≤ 1)
  · simp [biasParameters, hc]

/-- Identity transitions preserve the actual query state at any input
length, CM95. -/
theorem bias_run (v : E) (L : E →L[ℂ] ℂ) (target : O) (s : ℝ) (xs : List O) :
    runFrom (biasParameters v L target s) v xs = v := by
  induction xs with
  | nil => rfl
  | cons o xs ih => simpa only [runFrom, biasParameters, ContinuousLinearMap.id_apply] using ih

/-- Actual raw original-class bias score, CM95. The nonzero state/readout
pair is required explicitly and is inhabited below. -/
theorem bias_score (v : E) (L : E →L[ℂ] ℂ) (hLv : L v = 1)
    (target : O) (s : ℝ) (xs : List O) (c : O) :
    score (biasParameters v L target s) xs c = if c = target then s else 0 := by
  change ((biasParameters v L target s).readout c
    (runFrom (biasParameters v L target s) v xs)).re = _
  rw [bias_run]
  by_cases hc : c = target <;> simp [biasParameters, hc, hLv]

/-- Complete recurrent query point within a convex component step, CM95. -/
noncomputable def query (B eta : ℝ) (r : List O → O → ℝ)
    (v : E) (L : E →L[ℂ] ℂ) (target : O) (s : ℝ) : List O → O → ℝ :=
  (1 - eta) • r + eta • (B • score (biasParameters v L target s))

/-- Both signed bias models remain feasible for unit state/readout and
a capacity-reserving convex step, CM95. -/
theorem query_feasible (B eta : ℝ) (r : List O → O → ℝ)
    (v : E) (L : E →L[ℂ] ℂ) (target : O) (s : ℝ)
    (hr : r ∈ domain O E B) (hv : ‖v‖ ≤ 1) (hL : ‖L‖ ≤ 1)
    (hs : |s| ≤ 1) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    query B eta r v L target s ∈ domain O E B :=
  step_feasible B eta r _ hr (bias_unit v L target s hv hL hs) he he1

/-- Exact full-vocabulary logits of the actual feasible recurrent query,
CM95. Only the selected class changes from the common shrunken centre. -/
theorem query_logits (B eta : ℝ) (r : List O → O → ℝ)
    (v : E) (L : E →L[ℂ] ℂ) (hLv : L v = 1) (target : O) (s : ℝ)
    (xs : List O) (c : O) :
    query B eta r v L target s xs c =
      (1 - eta) * r xs c + if c = target then eta * B * s else 0 := by
  simp only [query, Pi.add_apply, Pi.smul_apply, smul_eq_mul, bias_score v L hLv]
  split_ifs <;> ring

/-- Ordinary CE at the two actual recurrent query states equals the
finite class-bias difference, CM95. A meaningful numerical step uses eta*B>0;
this identity does not equate finite differences with analytic derivatives. -/
theorem feasible_difference [Fintype O] (B eta : ℝ) (r : List O → O → ℝ)
    (v : E) (L : E →L[ℂ] ℂ) (hLv : L v = 1) (xs : List O) (target c : O) :
    (Landscape.Neural.crossEntropy (query B eta r v L c 1 xs) target -
      Landscape.Neural.crossEntropy (query B eta r v L c (-1) xs) target) / (2 * (eta * B)) =
    (Landscape.Neural.crossEntropy (fun d => (1 - eta) * r xs d +
      if d = c then eta * B else 0) target -
      Landscape.Neural.crossEntropy (fun d => (1 - eta) * r xs d +
      if d = c then -(eta * B) else 0) target) / (2 * (eta * B)) := by
  have hp : query B eta r v L c 1 xs =
      (fun d => (1 - eta) * r xs d + if d = c then eta * B else 0) := by
    funext d
    simpa only [mul_one] using query_logits B eta r v L hLv c 1 xs d
  have hn : query B eta r v L c (-1) xs =
      (fun d => (1 - eta) * r xs d + if d = c then -(eta * B) else 0) := by
    funext d
    simpa only [mul_neg_one] using query_logits B eta r v L hLv c (-1) xs d
  rw [hp, hn]

example : Unit (biasParameters (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ) true (-1)) :=
  bias_unit _ _ _ _ (by norm_num) (by simp) (by norm_num)

example : score (biasParameters (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ) true (-1))
    [false, true] true = -1 := by
  simpa using bias_score (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ) (by simp)
    true (-1) [false, true] true

example : query 2 (1 / 2) (0 : List Bool → Bool → ℝ) (1 : ℂ)
    (ContinuousLinearMap.id ℂ ℂ) true (-1) ∈ domain Bool ℂ 2 :=
  query_feasible 2 (1 / 2) 0 1 _ true (-1) (zero_mem 2)
    (by norm_num) (by simp) (by norm_num) (by norm_num) (by norm_num)

example : query 2 (1 / 2) (0 : List Bool → Bool → ℝ) (1 : ℂ)
    (ContinuousLinearMap.id ℂ ℂ) true (-1) [false] true = -1 := by
  simpa using query_logits 2 (1 / 2) 0 (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ)
    (by simp) true (-1) [false] true

example :
    (Landscape.Neural.crossEntropy (query 2 (1 / 2) 0 (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ)
      false 1 [true]) true -
    Landscape.Neural.crossEntropy (query 2 (1 / 2) 0 (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ)
      false (-1) [true]) true) / (2 * ((1 / 2 : ℝ) * 2)) =
    (Landscape.Neural.crossEntropy (fun d : Bool => if d = false then 1 else 0) true -
      Landscape.Neural.crossEntropy (fun d : Bool => if d = false then -1 else 0) true) / 2 := by
  simpa using feasible_difference 2 (1 / 2) 0 (1 : ℂ) (ContinuousLinearMap.id ℂ ℂ)
    (by simp) [true] true false

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent

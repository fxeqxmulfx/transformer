import Transformer.GPTMini.Convex.Structured.PointerValues

/-!
# Learned content and raw adjacent-binding controls

New finite controls for the compact structured pointer at 5371b5f.
One given shared parameter table has nonzero query, key and value log
potentials; every field is in the freely trained parameter space proved
convex in PointerTraining. Two queries prefer different keys using those
same learned fields. The actual weighted value mean changes when two
raw neighboring values are swapped while keys and query are retained.

The control has a two-symbol vocabulary and two physical records. It
is not yet the complete 256-symbol/eight/sixteen-write Basis solver.
Raw five-token arrays are interpreted by their actual alternating
key/value positions; no paired-key feature or correct route is provided
to the head. A finite learned-value decoder has strict opposite labels
on equal-bag swapped inputs. This verifies content/binding expressivity,
not AdamW finding the weights or original tied residual integration.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

/-- The complete unrestricted small shared Q/K/value/chronology parameter space.
Source: PointerParameters, specialized to two symbols and one two-channel group of each kind. -/
abbrev PointerControlParameters := PointerParameters 2 (Fin 1) (Fin 2) (Fin 1) (Fin 2)

/-- Given finite learned fields for a content/binding witness, rather than a fixed interaction operator.
Source: the actual freely trained token slots; query/key use log two and value channels log three. -/
def pointerControlParams : PointerControlParameters :=
  (fun token field => match field with
    | .inl (_, _, channel) => if token = channel then Real.log 2 else 0
    | .inr (_, channel) => if token = channel then Real.log 3 else 0, 0)

/-- Actual shared query/key matching favors equal symbols by a strict five-versus-four contracted score.
Source: both trained token tables and the true exponential channel sum, not a declared equality test in attention. -/
theorem pointerControl_matching (query key : Fin 2) :
    channelPartition (fun g c => pointerQuery pointerControlParams query g c + pointerKey pointerControlParams key g c) =
      if query = key then 5 else 4 := by
  fin_cases query <;> fin_cases key <;>
    norm_num [channelPartition, pointerQuery, pointerKey, pointerControlParams,
      Fin.prod_univ_one, Fin.sum_univ_two, Real.exp_add, Real.exp_log]

/-- Both true value symbols have equal partition four, despite different learned output-channel preferences.
Source: actual trained value log fields; this prevents a memorized route bias in the binding witness. -/
theorem pointerControl_value_partition (token : Fin 2) :
    channelPartition (pointerValue pointerControlParams token) = 4 := by
  fin_cases token <;>
    norm_num [channelPartition, pointerValue, pointerControlParams, Fin.prod_univ_one, Fin.sum_univ_two, Real.exp_log]

/-- Actual learned value means distinguish the two symbols through their common fixed output-channel decoder.
Source: the true conditional categorical values, with no target channel passed to the computation. -/
theorem pointerControl_value_mean (token : Fin 2) :
    channelMean (pointerValue pointerControlParams token) 0 (fun d => if d = 0 then 1 else 0) =
      if token = 0 then 3 / 4 else 1 / 4 := by
  fin_cases token <;>
    norm_num [channelMean, channelWeight, pointerValue, pointerControlParams, Fin.sum_univ_two, Real.exp_log]

/-- The actual full compact normalizer is 36 for either query and every two-value assignment.
Source: query-dependent five/four matching and equal learned value partitions, not a selected route assumption. -/
theorem pointerControl_partition (query : Fin 2) (values : Fin 2 → Fin 2) :
    pointerPartition (pointerQuery pointerControlParams query) (fun j => pointerKey pointerControlParams j)
      (fun j => pointerValue pointerControlParams (values j)) (fun _ => 0) = 36 := by
  unfold pointerPartition
  rw [Fin.sum_univ_two]
  simp only [pointerControl_matching, pointerControl_value_partition, Real.exp_zero, one_mul]
  fin_cases query <;> norm_num

/-- With the same shared fields, each query prefers its own physical key, including after values are swapped.
Source: the actual computed joint route/value distribution and derived normalizer. -/
theorem pointerControl_weight (query j : Fin 2) (values : Fin 2 → Fin 2) :
    pointerWeight (pointerQuery pointerControlParams query) (fun k => pointerKey pointerControlParams k)
      (fun k => pointerValue pointerControlParams (values k)) (fun _ => 0) j =
      if query = j then 5 / 9 else 4 / 9 := by
  unfold pointerWeight
  rw [pointerControl_matching, pointerControl_value_partition, pointerControl_partition]
  by_cases he : query = j <;> simp only [he, ite_true, ite_false, Real.exp_zero] <;> norm_num

/-- Actual two-record output mean from arbitrary jointly trained shared parameters.
Source: the same pointer inference formula, including the free chronology scalar and physical record positions. -/
def pointerBindingMean (θ : PointerControlParameters) (query : Fin 2) (values : Fin 2 → Fin 2) : ℝ :=
  pointerMean (pointerQuery θ query) (fun j => pointerKey θ j) (fun j => pointerValue θ (values j))
    (fun j => θ.2 * (j.val : ℝ)) 0 (fun d => if d = 0 then 1 else 0)

/-- The actual binding mean has explicit query-dependent coefficients on both learned physical values.
Source: the computed weights and conditional channel means from the same shared parameter table. -/
theorem pointerControl_binding_formula (query : Fin 2) (values : Fin 2 → Fin 2) :
    pointerBindingMean pointerControlParams query values =
      (if query = 0 then 5 / 9 else 4 / 9) * (if values 0 = 0 then 3 / 4 else 1 / 4) +
      (if query = 1 then 5 / 9 else 4 / 9) * (if values 1 = 0 then 3 / 4 else 1 / 4) := by
  unfold pointerBindingMean
  have hb : (fun j : Fin 2 => pointerControlParams.2 * (j.val : ℝ)) = fun _ => (0 : ℝ) := by
    funext j
    change 0 * (j.val : ℝ) = 0
    ring
  rw [hb]
  unfold pointerMean
  simp only [Fin.sum_univ_two, pointerControl_weight, pointerControl_value_mean]

/-- The same physical binding table yields opposite strict decoded labels for its two learned queries.
Source: true joint trained-value inference, rather than an additive query-independent memory score. -/
theorem pointerControl_queries :
    pointerBindingMean pointerControlParams 0 (fun j => j) = 19 / 36 ∧
      pointerBindingMean pointerControlParams 1 (fun j => j) = 17 / 36 := by
  rw [pointerControl_binding_formula, pointerControl_binding_formula]
  norm_num

/-- A genuine raw five-token input reads key/value adjacency and the final query at their original physical positions.
Source: two-record raw control layout, with shared trained lookups and no prepared paired-key feature. -/
def pointerControlRawMean (θ : PointerControlParameters) (tokens : Fin 5 → Fin 2) : ℝ :=
  pointerMean (pointerQuery θ (tokens 4))
    (fun j : Fin 2 => pointerKey θ (tokens ⟨2 * j.val, by have hj := j.isLt; omega⟩))
    (fun j : Fin 2 => pointerValue θ (tokens ⟨2 * j.val + 1, by have hj := j.isLt; omega⟩))
    (fun j : Fin 2 => θ.2 * (j.val : ℝ)) 0 (fun d => if d = 0 then 1 else 0)

/-- Swapping two raw neighboring values retains the query/token bag and strictly changes the actual head's decoded answer.
Source: original-position raw lookups and the same genuine joint value mean; this repairs the bag-encoder control locally. -/
theorem pointerControl_raw_swap :
    pointerControlRawMean pointerControlParams ![0, 0, 1, 1, 0] = 19 / 36 ∧
      pointerControlRawMean pointerControlParams ![0, 1, 1, 0, 0] = 17 / 36 := by
  norm_num [pointerControlRawMean, pointerMean, pointerWeight, pointerPartition, channelPartition,
    channelMean, channelWeight, pointerQuery, pointerKey, pointerValue, pointerControlParams,
    Fin.sum_univ_two, Fin.prod_univ_one, Real.exp_add, Real.exp_log]

/-- The actual finite raw control has a strict decision margin on both binding orders.
Source: derived true head outputs, with the ordinary midpoint threshold for the two fixed output-channel labels. -/
theorem pointerControl_raw_margin :
    1 / 2 < pointerControlRawMean pointerControlParams ![0, 0, 1, 1, 0] ∧
      pointerControlRawMean pointerControlParams ![0, 1, 1, 0, 0] < 1 / 2 := by
  rw [pointerControl_raw_swap.1, pointerControl_raw_swap.2]
  norm_num

/-- The actual finite two-symbol decoder returns different next tokens on the two raw binding orders.
Source: strictly separated computed means; desired answers are not fed into either forward call. -/
theorem pointerControl_raw_decoding :
    (if 1 / 2 < pointerControlRawMean pointerControlParams ![0, 0, 1, 1, 0] then (0 : Fin 2) else 1) = 0 ∧
      (if 1 / 2 < pointerControlRawMean pointerControlParams ![0, 1, 1, 0, 0] then (0 : Fin 2) else 1) = 1 := by
  rw [pointerControl_raw_swap.1, pointerControl_raw_swap.2]
  norm_num

/-- The genuine unrestricted shared Q/K/value objective stays convex for this same finite content/binding problem.
Source: actual contractedPointerNLL, not convexity of a loss with the control weights frozen. -/
example : ConvexOn ℝ Set.univ (contractedPointerNLL (G := Fin 1) (C := Fin 2) (H := Fin 1) (D := Fin 2)
    (0 : Fin 2) (fun j : Fin 2 => j) (fun j : Fin 2 => j) (fun j => (j.val : ℝ))
    (0, (fun _ => 0), (fun _ => 0))) := contractedPointerNLL_convex _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured

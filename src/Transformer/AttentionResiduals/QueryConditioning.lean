module

public import Transformer.AttentionResiduals.QueryCoordinates
public import Transformer.ScheduleFree.AdamW.Defs

/-!
# Complete Schedule-Free state in query coordinates

Source: `review/residual_connections.md`, *Query coordinate conditioning*,
QC6--QC9; `diff_gpt` ScheduleFreeAdamW at 4e58930. This uses the existing
optimized query/base and reference average/base definitions. A factor divides
both iterates, multiplies gradient moments by its square, and leaves the
averaging clocks unchanged. These are coordinate identities, not equality of
future AdamW trajectories: applying the same scalar learning rate in new
coordinates intentionally changes the physical update size.

The conditioning rule examines all three current RMS magnitudes (query,
base and implied average). It shrinks only oversized coordinates. A threshold
of at least one leaves zero and small states alone and puts each resulting
magnitude below that threshold. This is an immediate bound after recharting,
not a bound on the physical query or a convergence theorem.
-/

@[expose] public section
noncomputable section

namespace Transformer.AttentionResiduals

open ScheduleFree.AdamW

/-- QC6: a common coordinate change transforms raw second moments while
preserving the accumulated averaging weight and maximum learning rate. -/
def rechartHistory {ι : Type*} (factor : ℝ) (history : History ι) : History ι :=
  ⟨fun i => factor ^ 2 * history.second i, history.total, history.maximumRate⟩

/-- QC6: reference average/base state in the new coordinates. -/
def rechartState {ι : Type*} (factor : ℝ) (state : State ι) : State ι :=
  ⟨fun i => factor⁻¹ * state.evaluation i, fun i => factor⁻¹ * state.base i,
    rechartHistory factor state.history⟩

/-- QC6: actual parameter-buffer/base state in the same coordinates. -/
def rechartTrainingState {ι : Type*} (factor : ℝ)
    (state : TrainingState ι) : TrainingState ι :=
  ⟨fun i => factor⁻¹ * state.point i, fun i => factor⁻¹ * state.base i,
    rechartHistory factor state.history⟩

/-- QC6: forming the actual Schedule-Free query commutes with a common
linear coordinate change of the average and base iterates. -/
theorem query_rechart {ι : Type*} (p : Parameters) (factor : ℝ) (state : State ι) :
    query p (rechartState factor state) = fun i => factor⁻¹ * query p state i := by
  funext i
  simp only [query, rechartState]
  ring

/-- QC7: recovering evaluation weights from the two-buffer representation
commutes with the same change. Rescaling only the query would not do this. -/
theorem evaluation_rechart {ι : Type*} (p : Parameters) (factor : ℝ)
    (state : TrainingState ι) :
    evaluation p (rechartTrainingState factor state) =
      fun i => factor⁻¹ * evaluation p state i := by
  funext i
  simp only [evaluation, rechartTrainingState]
  ring

/-- QC7: packing reference state and rescaling state have the same query,
base and history. This links both actual source representations. -/
theorem pack_rechart {ι : Type*} (p : Parameters) (factor : ℝ) (state : State ι) :
    pack p (rechartState factor state) = rechartTrainingState factor (pack p state) := by
  unfold pack rechartTrainingState
  rw [query_rechart]
  rfl

/-- QC8: the existing gradient second-moment update commutes with the
chain-rule change `g_new = factor * g_old`, if history is changed too. -/
theorem nextSecond_rechart {ι : Type*} (p : Parameters) (factor : ℝ)
    (history : History ι) (gradient : ι → ℝ) :
    nextSecond p (rechartHistory factor history) (fun i => factor * gradient i) =
      fun i => factor ^ 2 * nextSecond p history gradient i := by
  funext i
  simp only [nextSecond, rechartHistory]
  ring

/-- QC9: largest of the query, base and implied-average RMS magnitudes. -/
def largestQueryMagnitude (magnitudes : Fin 3 → ℝ) : ℝ :=
  max (magnitudes 0) (max (magnitudes 1) (magnitudes 2))

/-- QC9: oversized states are recharted to maximum RMS one; others stay
unchanged. The threshold is explicit, not a hidden gradient assumption. -/
def queryConditioningFactor (threshold : ℝ) (magnitudes : Fin 3 → ℝ) : ℝ :=
  if threshold < largestQueryMagnitude magnitudes then largestQueryMagnitude magnitudes else 1

/-- QC9: the selected factor is positive for thresholds at least one. -/
theorem queryConditioningFactor_pos (threshold : ℝ) (ht : 1 ≤ threshold)
    (magnitudes : Fin 3 → ℝ) : 0 < queryConditioningFactor threshold magnitudes := by
  unfold queryConditioningFactor
  split_ifs with h
  · linarith
  · norm_num

/-- QC9: each of the three current state magnitudes is bounded immediately
after conditioning, including zero states. Future updates require rechecking. -/
theorem queryConditioningFactor_bound (threshold : ℝ) (ht : 1 ≤ threshold)
    (magnitudes : Fin 3 → ℝ) (i : Fin 3) :
    magnitudes i / queryConditioningFactor threshold magnitudes ≤ threshold := by
  have hle : magnitudes i ≤ largestQueryMagnitude magnitudes := by
    unfold largestQueryMagnitude
    fin_cases i
    · exact le_max_left _ _
    · exact (le_max_left _ _).trans (le_max_right _ _)
    · exact (le_max_right _ _).trans (le_max_right _ _)
  unfold queryConditioningFactor
  split_ifs with h
  · have hp : 0 < largestQueryMagnitude magnitudes := by linarith
    apply le_trans ((div_le_one hp).mpr hle) ht
  · simpa only [div_one] using hle.trans (le_of_not_gt h)

/-- QC9: exact RMS magnitude measured by the stable tensor implementation.
Its scaled implementation avoids squaring huge coordinates; the real value
is the Euclidean norm divided by the square root of width. -/
def queryRMS {d : ℕ} (value : EucSpace d) : ℝ := ‖value‖ / Real.sqrt (d : ℝ)

/-- QC9: RMS magnitudes transform by the same positive inverse factor as
the coordinate vectors. No nonzero-vector assumption is needed. -/
theorem queryRMS_rechart {d : ℕ} (factor : ℝ) (hc : 0 < factor)
    (value : EucSpace d) : queryRMS (factor⁻¹ • value) = queryRMS value / factor := by
  unfold queryRMS
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hc)]
  ring

/-- QC9: the actual query, base and implied-average vectors all have bounded
coordinate RMS immediately after applying the selected common factor. -/
theorem conditioned_queryRMS_bound {d : ℕ} (threshold : ℝ) (ht : 1 ≤ threshold)
    (states : Fin 3 → EucSpace d) (i : Fin 3) :
    queryRMS ((queryConditioningFactor threshold (fun j => queryRMS (states j)))⁻¹ •
      states i) ≤ threshold := by
  rw [queryRMS_rechart _ (queryConditioningFactor_pos threshold ht _)]
  exact queryConditioningFactor_bound threshold ht (fun j => queryRMS (states j)) i

example := queryConditioningFactor_pos 2 (by norm_num) ![1000000, 1200000, 900000]

example := queryConditioningFactor_bound 2 (by norm_num)
  ![1000000, 1200000, 900000] 0

example := queryConditioningFactor_bound 2 (by norm_num) ![0, 0, 0] 1

example := queryRMS_rechart 1000000 (by norm_num)
  (WithLp.toLp 2 ![1000000, 2000000])

example := conditioned_queryRMS_bound (d := 2) 2 (by norm_num)
  (fun i : Fin 3 => WithLp.toLp 2 ![1000000 * (i.val + 1 : ℝ), 2000000]) 1

end Transformer.AttentionResiduals

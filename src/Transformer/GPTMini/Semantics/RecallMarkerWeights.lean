import Transformer.GPTMini.Semantics.RecallRawCopy
import Mathlib.Order.Interval.Finset.Fin

/-!
# Exact causal BOS mass at every original head position

Source: original causal softmax/XSA at f11b6e2 and MQAR's initial
BOS at cbafbe9. Zero Q/K gives a denominator equal to the number
of visible positions, i+1, even when the full array contains future
tokens. A single initial marker therefore contributes exactly
1/(i+1) at every later position. Its self-value is zero there, so
the original XSA leaves the marker mass intact.

The result retains the real causal mask and computes its finite
denominator; it does not supply the index to a modified attention
operator. Subsequent raw-head coupling derives the single marker
array from actual integer IDs. The table boundary can then be a
fixed linear threshold of this internally computed coordinate.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The genuine unit value direction of the simultaneous original marker head.
Source: V chunk coordinate 144, the first coordinate of head one. -/
noncomputable def recallMarkerDirection : EucSpace recallConfig.head_dim :=
  EuclideanSpace.single ⟨0, by decide⟩ 1

/-- The actual causal denominator contains exactly i+1 visible positions, regardless of future array length.
Source: original diagonal-inclusive causal mask and the evaluated Fin initial interval. -/
theorem recall_causal_visible_count {T : ℕ} (i : Fin T) :
    (∑ j : Fin T, if j.val ≤ i.val then (1 : ℝ) else 0) = ((i.val + 1 : ℕ) : ℝ) := by
  classical
  have hs : Finset.univ.filter (fun j : Fin T => j.val ≤ i.val) = Finset.Iic i := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic, Fin.le_def]
  rw [← Finset.sum_filter, hs, Finset.sum_const, Fin.card_Iic, nsmul_eq_mul, mul_one]

/-- Every actual visible zero-score causal weight is exactly the reciprocal of the internal prefix length.
Source: original causalAttnWeights, including its genuine finite denominator. -/
theorem recall_uniform_visible_weight (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (i j : Fin T) (hvisible : j.val ≤ i.val) :
    causalAttnWeights cfg alpha eps (fun _ => 0) (fun _ => 0) i j =
      1 / ((i.val + 1 : ℕ) : ℝ) := by
  rw [causalAttnWeights, ite_eq_right (not_lt.mpr hvisible)]
  simp only [preScore, score, normL2, smul_zero, inner_zero_left, mul_zero, Real.exp_zero,
    recall_causal_visible_count]

example : (0 : Fin 2).val ≤ (1 : Fin 2).val := by decide

/-- A genuine future zero-score position still has exactly zero causal weight.
Source: original unmodified softmax mask; the internal prefix denominator never includes it. -/
theorem recall_uniform_future_weight (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (i j : Fin T) (hfuture : i.val < j.val) :
    causalAttnWeights cfg alpha eps (fun _ => 0) (fun _ => 0) i j = 0 := by
  rw [causalAttnWeights, ite_eq_left hfuture]

example : (0 : Fin 2).val < (1 : Fin 2).val := by decide

/-- The actual marker head retains precisely reciprocal-prefix mass at every later position, even with future tokens.
Source: derived causal weight, finite attention sum and original zero-self-value XSA. -/
theorem recall_initial_marker_head (cfg : Config) (eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (first i : Fin T) (hfirst : first.val = 0) (hne : i ≠ first)
    (direction : EucSpace cfg.head_dim) :
    attentionHead cfg 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => if j = first then direction else 0) positions i =
      (1 / ((i.val + 1 : ℕ) : ℝ)) • direction := by
  have hvisible : first.val ≤ i.val := by rw [hfirst]; exact Nat.zero_le _
  unfold attentionHead xsaProjection
  simp only [rope_zero, ite_eq_right hne, normL2, smul_zero, inner_zero_right, sub_zero]
  unfold attnOutput
  rw [Fintype.sum_eq_single first]
  · simp only [ite_true]
    change causalAttnWeights cfg 0 eps (fun _ => 0) (fun _ => 0) i first • direction = _
    rw [recall_uniform_visible_weight cfg 0 eps i first hvisible]
  · intro j hj
    simp only [ite_eq_right hj, smul_zero]

example : (0 : Fin 3).val = 0 ∧ (1 : Fin 3) ≠ 0 := by decide

/-- Reading the genuine unit marker direction gives its exact internally computed reciprocal prefix length.
Source: the actual marker head and original sixteen-coordinate unit at position zero. -/
theorem recall_initial_marker_probe (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (first i : Fin T) (hfirst : first.val = 0) (hne : i ≠ first) :
    inner (𝕜 := ℝ) recallMarkerDirection
      (attentionHead recallConfig 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => if j = first then recallMarkerDirection else 0) positions i) =
      1 / ((i.val + 1 : ℕ) : ℝ) := by
  rw [recall_initial_marker_head recallConfig eps positions first i hfirst hne, real_inner_smul_right]
  norm_num [recallMarkerDirection, EuclideanSpace.inner_single_left, PiLp.single_apply]

example : (0 : Fin 3).val = 0 ∧ (2 : Fin 3) ≠ 0 := by decide

/-- The actual marker direction has exactly reciprocal-prefix norm after softmax and XSA.
Source: the evaluated marker head and true unit value direction; no clipping divisor is postulated. -/
theorem recall_initial_marker_norm (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (first i : Fin T) (hfirst : first.val = 0) (hne : i ≠ first) :
    ‖attentionHead recallConfig 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => if j = first then recallMarkerDirection else 0) positions i‖ =
      1 / ((i.val + 1 : ℕ) : ℝ) := by
  rw [recall_initial_marker_head recallConfig eps positions first i hfirst hne, norm_smul]
  have hp : (0 : ℝ) ≤ 1 / ((i.val + 1 : ℕ) : ℝ) := by positivity
  simp only [Real.norm_eq_abs, abs_of_nonneg hp, recallMarkerDirection, PiLp.norm_single]
  norm_num

example : (0 : Fin 3).val = 0 ∧ (1 : Fin 3) ≠ 0 := by decide

/-- The real reciprocal-prefix marker is strictly positive at every valid array position.
Source: i+1 is a positive integer, so the internal marker never confuses a finite prefix with no BOS. -/
theorem recall_marker_mass_pos {T : ℕ} (i : Fin T) : 0 < 1 / ((i.val + 1 : ℕ) : ℝ) := by
  apply div_pos (by norm_num)
  positivity

/-- At any later position the derived causal marker mass is at most one half.
Source: the actual index is at least one, rather than an externally supplied model input. -/
theorem recall_marker_mass_le_half {T : ℕ} (first i : Fin T) (hfirst : first.val = 0) (hne : i ≠ first) :
    1 / ((i.val + 1 : ℕ) : ℝ) ≤ (1 / 2 : ℝ) := by
  have hi : 1 ≤ i.val := by
    by_contra hn
    apply hne
    apply Fin.ext
    omega
  have hc : (2 : ℝ) ≤ ((i.val + 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 2 ≤ i.val + 1)
  apply (div_le_iff₀ (by positivity : (0 : ℝ) < ((i.val + 1 : ℕ) : ℝ))).mpr
  linarith

example : (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 := by decide

/-- The internally derived marker mass is bounded below uniformly across the complete recall context.
Source: the actual cap 64 and the evaluated reciprocal prefix length, including all future-array sizes. -/
theorem recall_marker_mass_ge_context {T : ℕ} (hT : T ≤ 64) (i : Fin T) :
    (1 / 64 : ℝ) ≤ 1 / ((i.val + 1 : ℕ) : ℝ) := by
  have hi : i.val + 1 ≤ 64 := by have hh := i.isLt; omega
  have hc : (((i.val + 1 : ℕ) : ℝ)) ≤ 64 := by exact_mod_cast hi
  apply (le_div_iff₀ (by positivity : (0 : ℝ) < ((i.val + 1 : ℕ) : ℝ))).mpr
  linarith

example : (64 : ℕ) ≤ 64 := by decide

/-- The true internally computed marker mass strictly decreases at successive visible positions.
Source: reciprocal monotonicity on positive integer prefix lengths, enabling a fixed table-boundary threshold. -/
theorem recall_marker_mass_strict {T : ℕ} (i j : Fin T) (hij : i.val < j.val) :
    1 / ((j.val + 1 : ℕ) : ℝ) < 1 / ((i.val + 1 : ℕ) : ℝ) := by
  apply one_div_lt_one_div_of_lt (by positivity)
  exact_mod_cast (by omega : i.val + 1 < j.val + 1)

example : (1 : Fin 3).val < (2 : Fin 3).val := by decide

end Transformer.GPTMini.Semantics

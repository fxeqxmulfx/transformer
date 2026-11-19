module

public import Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Witness

/-!
# A shared learned decoder is a further nonconvex constraint

Source: `review/convex/moments.md`, CM74. The positive joint repair is exact
at one input. With two inputs and one common conditional decoder, a fixed
hidden state's input/output slice must have rank one. Two strictly positive
stochastic encoder/decoder models have a joint midpoint with minor 1/16.
Every input row is still in the positive joint simplex, but no common decoder
represents that midpoint. Independent tables per prefix are therefore a
relaxation, not a proof of a shared fully trainable sequence architecture.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Sharing

open scoped BigOperators

/-- Positive conditional tables for a pair of explicit learned endpoints,
CM74. The Boolean argument indexes mathematical witnesses, not a target rule. -/
noncomputable def stageTable (b : Bool) (x y : Bool) : ℝ :=
  if decide (x = y) = b then 3 / 4 else 1 / 4

/-- Every entry of both conditional tables is strictly positive, CM74. -/
theorem stageTable_pos (b x y : Bool) : 0 < stageTable b x y := by
  cases b <;> cases x <;> cases y <;> norm_num [stageTable]

/-- Both actual conditional tables have normalized rows, CM74. -/
theorem stageTable_sum (b x : Bool) : ∑ y, stageTable b x y = 1 := by
  cases b <;> cases x <;> norm_num [stageTable, Fintype.sum_bool]

/-- Full input/hidden/output joint table of each actual two-stage endpoint,
CM74. The decoder is shared across the two different inputs. -/
noncomputable def endpoint (b : Bool) (x h c : Bool) : ℝ :=
  stageTable b x h * stageTable b h c

/-- Exact joint image of strictly positive encoders with one shared positive
decoder, CM74. Membership supplies actual factors and normalization. -/
def domain : Set (Bool → Bool → Bool → ℝ) :=
  {T | ∃ E D : Bool → Bool → ℝ,
    (∀ x h, 0 < E x h) ∧ (∀ x, ∑ h, E x h = 1) ∧
    (∀ h c, 0 < D h c) ∧ (∀ h, ∑ c, D h c = 1) ∧
    ∀ x h c, T x h c = E x h * D h c}

/-- Both displayed endpoints are genuine shared encoder/decoder models,
CM74; their positivity is not a boundary degeneracy. -/
theorem endpoint_mem (b : Bool) : endpoint b ∈ domain := by
  exact ⟨stageTable b, stageTable b, stageTable_pos b, stageTable_sum b,
    stageTable_pos b, stageTable_sum b, fun _ _ _ => rfl⟩

/-- The two different encoder/decoder assignments give the same answers at
both inputs, CM74. Answer probabilities do not identify internal state names. -/
theorem endpoint_answers_eq (x c : Bool) :
    (∑ h, endpoint false x h c) = ∑ h, endpoint true x h c := by
  cases x <;> cases c <;> norm_num [endpoint, stageTable, Fintype.sum_bool]

/-- The identical answers above coexist with different actual stage tables,
CM74; the two parameter assignments are not the same hidden representation. -/
theorem endpoint_stages_differ : stageTable false ≠ stageTable true := by
  intro he
  have h := congrFun (congrFun he false) false
  norm_num [stageTable] at h

/-- An actual input/output minor at one common hidden state, CM74. -/
def minor (T : Bool → Bool → Bool → ℝ) : ℝ :=
  T false false false * T true false true -
    T false false true * T true false false

/-- A shared decoder forces this minor to vanish, CM74. The restriction
couples different inputs and does not follow from separate row normalization. -/
theorem minor_zero {T : Bool → Bool → Bool → ℝ} (hT : T ∈ domain) : minor T = 0 := by
  obtain ⟨E, D, _, _, _, _, he⟩ := hT
  unfold minor
  rw [he, he, he, he]
  ring

/-- The positive endpoints' joint midpoint violates that necessary identity,
CM74. Every quantity is an exact rational number. -/
theorem midpoint_minor :
    minor ((1 / 2 : ℝ) • endpoint false + (1 / 2 : ℝ) • endpoint true) = 1 / 16 := by
  norm_num [minor, endpoint, stageTable]

/-- Keeping one learned decoder shared across inputs makes this exact joint
image nonconvex, CM74. This is not a theorem against all parameterizations. -/
theorem domain_not_convex : ¬ Convex ℝ domain := by
  intro h
  have hm := h (endpoint_mem false) (endpoint_mem true)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hz := minor_zero hm
  rw [midpoint_minor] at hz
  norm_num at hz

/-- Each input of the same midpoint still belongs to the convex joint
repair, CM74. Thus applying that repair independently silently relaxes sharing. -/
theorem midpoint_pointwise (x : Bool) :
    ((1 / 2 : ℝ) • endpoint false + (1 / 2 : ℝ) • endpoint true) x ∈ Joint.domain := by
  exact Joint.domain_convex
    (Joint.ofStages_mem (stageTable false x) (stageTable false)
      (stageTable_pos false x) (stageTable_sum false x)
      (stageTable_pos false) (stageTable_sum false))
    (Joint.ofStages_mem (stageTable true x) (stageTable true)
      (stageTable_pos true x) (stageTable_sum true x)
      (stageTable_pos true) (stageTable_sum true))
    (by norm_num) (by norm_num) (by norm_num)

example := minor_zero (endpoint_mem true)

end Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Sharing

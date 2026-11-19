module

public import Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Recovery

/-!
# Both stages can change while ordinary answer CE improves

Source: `review/convex/moments.md`, CM74. Two literal positive joint points
have different encoder marginals and different conditional decoder rows.
The actual answer-only CE is smaller at the second point. These are feasible
comparison points, not a claim that a native optimizer reaches them or that
an unobserved prefix is identified. All conditional recovery hypotheses are
inhabited explicitly on this same finite model.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Joint

/-- A label-independent positive initialization for the two-state control. -/
noncomputable def uniform : Bool → Bool → ℝ := (1 / 4 : ℝ) • 1

/-- A distinct joint point used only as a mathematical comparison witness. -/
noncomputable def improved : Bool → Bool → ℝ
  | false, false => 1 / 2
  | false, true => 1 / 8
  | true, false => 1 / 4
  | true, true => 1 / 8

/-- The uninformative initialization satisfies the entire domain, CM74. -/
theorem uniform_mem : uniform ∈ domain := by
  constructor
  · intro h c
    norm_num [uniform]
  · norm_num [uniform, Fintype.sum_bool]

/-- The changed encoder/decoder point is also fully feasible, CM74. -/
theorem improved_mem : improved ∈ domain := by
  constructor
  · intro h c
    cases h <;> cases c <;> norm_num [improved]
  · norm_num [improved, Fintype.sum_bool]

/-- Both the internal encoder and conditional output map really change,
CM74. The improvement is not only a scalar confidence multiplier. -/
theorem both_stages_change :
    encoder uniform false = 1 / 2 ∧ encoder improved false = 5 / 8 ∧
    decoder uniform false false = 1 / 2 ∧ decoder improved false false = 4 / 5 := by
  norm_num [encoder, decoder, uniform, improved, Fintype.sum_bool]

/-- Ordinary next-object CE can improve through both changed stages, CM74.
There is only an output target; no hidden-path observation enters the loss. -/
theorem improved_loss : loss false improved < loss false uniform := by
  rw [loss_eq improved_mem, loss_eq uniform_mem,
    forward_eq improved_mem, forward_eq uniform_mem]
  norm_num only [Fintype.sum_bool, improved, uniform, Pi.smul_apply, Pi.one_apply,
    smul_eq_mul, mul_one]
  have h := Real.log_lt_log (by norm_num : (0 : ℝ) < 1 / 2)
    (by norm_num : (1 / 2 : ℝ) < 3 / 4)
  norm_num at *
  linarith

example := encoder_pos uniform_mem false
example := forward_eq uniform_mem false
example := forward_pos uniform_mem false
example := forward_sum uniform_mem
example := loss_eq uniform_mem false
example := loss_convex (H := Bool) false
example := ofStages_recovered uniform_mem
example := encoder_sum uniform_mem
example := decoder_pos uniform_mem false false
example := decoder_sum uniform_mem false
example := ofStages_mem (encoder uniform) (decoder uniform)
  (encoder_pos uniform_mem) (encoder_sum uniform_mem)
  (decoder_pos uniform_mem) (decoder_sum uniform_mem)
example := encoder_ofStages (encoder uniform) (decoder uniform) (decoder_sum uniform_mem)
example := decoder_ofStages (encoder uniform) (decoder uniform)
  (encoder_pos uniform_mem) (decoder_sum uniform_mem)

end Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Joint

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair
public import Mathlib.Analysis.Complex.Trigonometric

/-!
# Learned complex rotary phases and operator transport

Source: `review/convex/moments.md`, CM98, derived from the existing GPTMini RoPE
isometry/relative-position formalization and learned-frequency parametrization.
Both frequencies are trainable real parameters. Bounds refer to actual
operators under the product maximum norm, not a token-table Gram spectrum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary

open Pair

/-- A trainable-frequency rotary phase, CM98. Position is the sequence
index; the original object IDs have no numerical interpretation. -/
noncomputable def phase (frequency position : ℝ) : ℂ :=
  Complex.exp (((position * frequency : ℝ) : ℂ) * Complex.I)

/-- Every learned frequency yields a unit complex phase, CM98. -/
theorem phase_norm (frequency position : ℝ) : ‖phase frequency position‖ = 1 := by
  exact Complex.norm_exp_ofReal_mul_I _

/-- Position zero gives the identity phase for every learned frequency,
CM98. -/
theorem phase_zero (frequency : ℝ) : phase frequency 0 = 1 := by
  simp [phase]

/-- A zero learned frequency is permitted and removes its rotation,
CM98. -/
theorem zero_frequency (position : ℝ) : phase 0 position = 1 := by
  simp [phase]

/-- Rotary phases form an additive-position representation, CM98. -/
theorem phase_add (frequency s t : ℝ) :
    phase frequency (s + t) = phase frequency s * phase frequency t := by
  unfold phase
  rw [show (((s + t) * frequency : ℝ) : ℂ) * Complex.I =
    ((s * frequency : ℝ) : ℂ) * Complex.I + ((t * frequency : ℝ) : ℂ) * Complex.I by
      push_cast; ring]
  exact Complex.exp_add _ _

/-- Actual diagonal complex rotations on both learned state channels,
CM98. No channel or imaginary coordinate is removed. -/
noncomputable def turn (frequency : ℝ × ℝ) (position : ℝ) : State →L[ℂ] State :=
  operator ((phase frequency.1 position, 0), (0, phase frequency.2 position))

/-- Complete executed rotary state, CM98. -/
theorem turn_apply (frequency : ℝ × ℝ) (position : ℝ) (z : State) :
    turn frequency position z =
      (phase frequency.1 position * z.1, phase frequency.2 position * z.2) := by
  simp [turn, operator_apply]

/-- Both rotary pairs preserve their complex norm and hence the maximum
state norm, CM98. This is independent of the input sequence length. -/
theorem turn_isometry (frequency : ℝ × ℝ) (position : ℝ) (z : State) :
    ‖turn frequency position z‖ = ‖z‖ := by
  simp [turn_apply, Prod.norm_def, phase_norm]

/-- Zero-position rotary execution is the identity, CM98. -/
theorem turn_zero (frequency : ℝ × ℝ) (z : State) : turn frequency 0 z = z := by
  simp [turn_apply, phase_zero]

/-- Zero learned frequencies recover the unchanged content execution,
CM98, without using a position lookup table. -/
theorem turn_zero_frequency (position : ℝ) (z : State) : turn (0, 0) position z = z := by
  simp [turn_apply, zero_frequency]

/-- Composition uses the sum of positions, CM98. -/
theorem turn_add (frequency : ℝ × ℝ) (s t : ℝ) (z : State) :
    turn frequency (s + t) z = turn frequency s (turn frequency t z) := by
  simp [turn_apply, phase_add, mul_assoc]

/-- The negative-position rotation exactly inverts the actual state
rotation, CM98. -/
theorem turn_inverse (frequency : ℝ × ℝ) (t : ℝ) (z : State) :
    turn frequency (-t) (turn frequency t z) = z := by
  rw [← turn_add, neg_add_cancel, turn_zero]

/-- Relative rotations depend on position difference, CM98, the rotary
identity underlying query/key relative-position interactions. -/
theorem turn_relative (frequency : ℝ × ℝ) (s t : ℝ) (z : State) :
    turn frequency (-s) (turn frequency t z) = turn frequency (t - s) z := by
  rw [← turn_add, show -s + t = t - s by ring]

/-- RoPE-inspired position-dependent learned matrix execution, CM98. -/
noncomputable def transport (frequency : ℝ × ℝ) (position : ℝ)
    (L : State →L[ℂ] State) : State →L[ℂ] State :=
  (turn frequency position).comp (L.comp (turn frequency (-position)))

/-- Rotary conjugation cannot increase the actual operator norm, CM98. -/
theorem transport_norm_le (frequency : ℝ × ℝ) (position : ℝ) (L : State →L[ℂ] State) :
    ‖transport frequency position L‖ ≤ ‖L‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
  intro z
  simp only [transport, ContinuousLinearMap.comp_apply, turn_isometry]
  simpa only [turn_isometry] using L.le_opNorm (turn frequency (-position) z)

/-- Opposite transport exactly reconstructs every matrix operator,
CM98. Every learned direction remains present. -/
theorem transport_inverse (frequency : ℝ × ℝ) (position : ℝ) (L : State →L[ℂ] State) :
    transport frequency (-position) (transport frequency position L) = L := by
  apply ContinuousLinearMap.ext
  intro z
  simp only [transport, ContinuousLinearMap.comp_apply, neg_neg, turn_inverse]

/-- The actual operator capacity is unchanged by every learned rotary
conjugation, CM98. This does not assert a token-table Gram identity. -/
theorem transport_norm (frequency : ℝ × ℝ) (position : ℝ) (L : State →L[ℂ] State) :
    ‖transport frequency position L‖ = ‖L‖ := by
  apply le_antisymm (transport_norm_le frequency position L)
  calc
    ‖L‖ = ‖transport frequency (-position) (transport frequency position L)‖ := by
      rw [transport_inverse]
    _ ≤ ‖transport frequency position L‖ := transport_norm_le _ _ _

/-- This particular rotary conjugation has no effect in a scalar state,
CM98. Other position-dependent scalar families are not ruled out. -/
theorem scalar_conjugation (frequency position : ℝ) (a z : ℂ) :
    phase frequency position * (a * (phase frequency (-position) * z)) = a * z := by
  calc
    _ = (phase frequency position * phase frequency (-position)) * (a * z) := by ring
    _ = a * z := by rw [← phase_add, add_neg_cancel, phase_zero, one_mul]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary

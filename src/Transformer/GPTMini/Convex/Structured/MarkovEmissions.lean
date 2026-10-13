import Transformer.GPTMini.Convex.Structured.MarkovMarginals
import Transformer.GPTMini.Convex.Structured.ChannelMarginals

/-!
# Jointly learned conditional output channels

Source: the compact causal state/path model at f1f6c9c and the exact
factorial output-channel contraction at 5371b5f. Every state's output
log potentials remain free parameters. The real encoder distribution
is mixed with their local channel means, never with supplied correct
states or values. Complete configurations exist only in the proof.

The resulting compact value output is proved exactly the expectation
of the positive normalized initial/path/output-channel joint model.
This supplies the actual inference distribution for the subsequent
convex complete-likelihood objective. Ordinary marginal output-label
CE, full Basis capability and residual/tied decoding are not claimed.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {S A H D : Type*} [Fintype S] [Nonempty S] [Fintype D] [Nonempty D]

/-- A complete configuration comprises initial state, post-token history and output-channel assignment.
Source: the proposed learned causal model; this type is not an inference parameter or stored feature bank. -/
abbrev MarkovConfiguration (S H D : Type*) (n : ℕ) := S × ((Fin n → S) × (H → D))

/-- The actual complete learned state/output probability, with output conditioned on the physical last state.
Source: the normalized causal rows and independent freely learned conditional output channels. -/
def markovJoint [Fintype H] (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ)
    (z : MarkovConfiguration S H D tokens.length) : ℝ :=
  stateRow initial z.1 * conditionalStatePath transition z.1 tokens z.2.1 *
    channelProbability (emission (statePathEnd z.1 tokens.length z.2.1)) z.2.2

/-- Compact output expectation from actual causal encoder probabilities and learned conditional channel means.
Source: the proposed state head's true forward computation, without intermediate semantic targets at inference. -/
def markovValueMean (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (code : D → ℝ) : ℝ :=
  ∑ state, markovRun initial transition tokens state * channelMean (emission state) h code

/-- Every complete actual state/path/value configuration has strictly positive mass at finite trainable parameters.
Source: the same genuine positive initial, transition and conditional emission factors. -/
theorem markovJoint_pos [Fintype H] (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (z : MarkovConfiguration S H D tokens.length) :
    0 < markovJoint initial transition tokens emission z :=
  mul_pos (mul_pos (stateRow_pos _ _) (conditionalStatePath_pos _ _ _ _)) (channelProbability_pos _ _)

/-- The complete conditional-emission model is normalized at all free raw initial, transition and value parameters.
Source: exact output-channel contraction and previously derived full causal-path normalization. -/
theorem markovJoint_sum [Fintype H] (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) :
    ∑ z : MarkovConfiguration S H D tokens.length, markovJoint initial transition tokens emission z = 1 := by
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type]
  simp only [markovJoint, ← Finset.mul_sum, channelProbability_sum, mul_one,
    conditionalStatePath_sum, stateRow_sum]

/-- No complete learned state/path/value probability exceeds one.
Source: genuine positivity and the proved normalization of that same complete model. -/
theorem markovJoint_le_one [Fintype H] (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (z : MarkovConfiguration S H D tokens.length) :
    markovJoint initial transition tokens emission z ≤ 1 := by
  have h := Finset.single_le_sum
    (fun cfg (_ : cfg ∈ Finset.univ) => (markovJoint_pos initial transition tokens emission cfg).le)
    (Finset.mem_univ z)
  rw [markovJoint_sum] at h
  exact h

omit [Nonempty S] in
/-- Compact causal state/channel output equals the exact complete joint-model value expectation.
Source: local channel marginal identity and exact full-path/forward state marginalization, for arbitrary learned values. -/
theorem markovJoint_mean [Fintype H] (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (code : D → ℝ) :
    (∑ z : MarkovConfiguration S H D tokens.length,
      markovJoint initial transition tokens emission z * code (z.2.2 h)) =
      markovValueMean initial transition tokens emission h code := by
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type]
  simp only [markovJoint, mul_assoc, ← Finset.mul_sum, channelProbability_mean]
  simpa only [mul_assoc, ← Finset.mul_sum, markovValueMean] using
    markovRun_path_mean initial transition tokens (fun state => channelMean (emission state) h code)

/-- The actual state/channel mean preserves every constant decoder coordinate.
Source: normalized actual encoder mass and normalized learned conditional channel distributions. -/
theorem markovValueMean_const (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (a : ℝ) :
    markovValueMean initial transition tokens emission h (fun _ => a) = a := by
  unfold markovValueMean
  simp_rw [channelMean_const]
  rw [← Finset.sum_mul, markovRun_sum, one_mul]

omit [Nonempty S] [Nonempty D] in
/-- Decoder-coordinate addition commutes with the actual learned state/channel expectation.
Source: linearity only in the fixed decoder statistic, without an affine claim about the nonlinear inferred state probabilities. -/
theorem markovValueMean_add (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (a b : D → ℝ) :
    markovValueMean initial transition tokens emission h (fun d => a d + b d) =
      markovValueMean initial transition tokens emission h a + markovValueMean initial transition tokens emission h b := by
  simp only [markovValueMean, channelMean_add, mul_add, Finset.sum_add_distrib]

omit [Nonempty S] [Nonempty D] in
/-- Fixed decoder scaling commutes with the true learned state/channel expectation.
Source: finite expectation linearity; all trainable transition and value potentials remain arbitrary. -/
theorem markovValueMean_scale (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (a : ℝ) (code : D → ℝ) :
    markovValueMean initial transition tokens emission h (fun d => a * code d) =
      a * markovValueMean initial transition tokens emission h code := by
  simp only [markovValueMean, channelMean_scale, ← mul_assoc]
  simp only [mul_comm _ a, mul_assoc, ← Finset.mul_sum]

/-- Every real compact output coordinate lies in the fixed decoder-code interval at arbitrary learned parameters.
Source: positive computed state masses and bounded true conditional channel means, rather than a supplied correct value. -/
theorem markovValueMean_bounds (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (emission : S → H → D → ℝ) (h : H) (code : D → ℝ)
    (lower upper : ℝ) (hcode : ∀ d, lower ≤ code d ∧ code d ≤ upper) :
    lower ≤ markovValueMean initial transition tokens emission h code ∧
      markovValueMean initial transition tokens emission h code ≤ upper := by
  have hlo := Finset.sum_le_sum (fun state (_ : state ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (channelMean_bounds (emission state) h code lower upper hcode).1
      (markovRun_pos initial transition tokens state).le)
  have hup := Finset.sum_le_sum (fun state (_ : state ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (channelMean_bounds (emission state) h code lower upper hcode).2
      (markovRun_pos initial transition tokens state).le)
  rw [← Finset.sum_mul, markovRun_sum, one_mul] at hlo hup
  exact ⟨hlo, hup⟩

example : ∀ d : Fin 2, (-1 : ℝ) ≤ (if d = 0 then (1 : ℝ) else -1) ∧
    (if d = 0 then (1 : ℝ) else -1) ≤ 1 := by
  intro d
  fin_cases d <;> norm_num

/-- Arbitrary six-state inference with all-zero untrained binary emissions produces zero signed output.
Source: the actual causal head mixed with real uniform local channel rows, with no desired value supplied. -/
example (initial : Fin 6 → ℝ) (transition : Fin 3 → Fin 6 → Fin 6 → ℝ) :
    markovValueMean initial transition [0, 1, 2]
      (fun _ : Fin 6 => fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) 0
      (fun d => if d = 0 then (1 : ℝ) else -1) = 0 := by
  have h : channelMean
      (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) 0
      (fun d => if d = 0 then (1 : ℝ) else -1) = 0 := by
    norm_num [channelMean, channelWeight, Fin.sum_univ_two]
  simp only [markovValueMean, h, mul_zero, Finset.sum_const_zero]

/-- The proposed five four-channel value groups remain normalized on a genuine three-token six-state computation.
Source: actual arbitrary finite transition/emission tables, with the full distribution implicit rather than stored. -/
example (initial : Fin 6 → ℝ) (transition : Fin 3 → Fin 6 → Fin 6 → ℝ)
    (emission : Fin 6 → Fin 5 → Fin 4 → ℝ) :
    (∑ z : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) 3,
      markovJoint initial transition [0, 1, 2] emission z) = 1 := by
  convert markovJoint_sum initial transition [0, 1, 2] emission

end
end Transformer.GPTMini.Convex.Structured

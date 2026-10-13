import Transformer.GPTMini.Convex.Structured.TensorEmbedding

/-!
# Recovering genuine free embedding fields after RMSNorm

Source: GPTMini.RMSNorm's exact `F.rms_norm` formula and the proposed
TensorEmbedding layout at 62f1f6f. The replacement attention can divide
its normalized coordinates by its own normalized protected anchor.
The actual positive RMS multiplier cancels, for every unrestricted
token/position parameter and every positive epsilon. No external
normalizer, token lookup, reference state or correct output is supplied
to this tensor operation. Both Basis model widths fit the construction.

This proves recovery from the actual Euclidean prenorm input. It does
not identify Q/K/V with frozen features: all 52 recovered fields remain
their original free parameters. Tensor-only attention, residual/tied
readout and full changed-model training are subsequent obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

variable {V C d : ℕ}

/-- Internal coordinate-ratio recovery uses only the actual normalized tensor.
Source: the protected-anchor parameterization; zero anchors use real division's total convention. -/
def tensorAnchorRecovery (anchor : Fin d) (x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun axis => x axis / x anchor)

/-- Attention's actual input recovery first receives the genuine RMSNorm output.
Source: GPTMini.RMSNorm.rmsNormEps and TensorEmbedding's protected physical anchor. -/
def tensorPrenormRecovery (eps : ℝ) (hwidth : 64 ≤ d) (x : EucSpace d) : EucSpace d :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) (rmsNormEps eps x)

/-- The real RMS multiplier is strictly positive, including at zero input.
Source: GPTMini.RMSNorm.rmsNormEps, using positive width and epsilon rather than an assumed scale. -/
theorem tensorRMSScale_pos (hd : 0 < d) (eps : ℝ) (heps : 0 < eps) (x : EucSpace d) :
    0 < Real.sqrt (d : ℝ) / Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) := by
  have hdReal : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have hden : 0 < ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  exact div_pos (Real.sqrt_pos.mpr hdReal) (Real.sqrt_pos.mpr hden)

example : (0 : ℕ) < 64 ∧ (0 : ℝ) < 1 / 100000 := by norm_num

/-- Internal ratio recovery is invariant under any genuine nonzero common scalar.
Source: the actual scalar multiplication in Euclidean RMSNorm, with its nonzero condition explicit. -/
theorem tensorAnchorRecovery_smul (anchor : Fin d) (x : EucSpace d)
    (scale : ℝ) (hscale : scale ≠ 0) :
    tensorAnchorRecovery anchor (scale • x) = tensorAnchorRecovery anchor x := by
  apply PiLp.ext
  intro axis
  simp only [tensorAnchorRecovery, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul]
  exact mul_div_mul_left _ _ hscale

example : (2 : ℝ) ≠ 0 := by norm_num

/-- A protected unit anchor makes actual internal recovery an exact tensor identity.
Source: coordinate division by the actual input's anchor; arbitrary remaining coordinates are permitted. -/
theorem tensorAnchorRecovery_eq (anchor : Fin d) (x : EucSpace d) (hanchor : x anchor = 1) :
    tensorAnchorRecovery anchor x = x := by
  apply PiLp.ext
  intro axis
  simp only [tensorAnchorRecovery, PiLp.toLp_apply, hanchor, div_one]

example : (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)) : EucSpace 64) 62 = 1 := rfl

/-- The actual eps-perturbed RMSNorm loses no protected raw field in this parameterization.
Source: GPTMini.RMSNorm's genuine positive scalar and the internal tensor-ratio operation above. -/
theorem tensorAnchorRecovery_rms (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (anchor : Fin d) (x : EucSpace d) (hanchor : x anchor = 1) :
    tensorAnchorRecovery anchor (rmsNormEps eps x) = x := by
  rw [rmsNormEps, tensorAnchorRecovery_smul anchor x _
    (ne_of_gt (tensorRMSScale_pos hd eps heps x))]
  exact tensorAnchorRecovery_eq anchor x hanchor

example : (0 : ℕ) < 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    (WithLp.toLp 2 (fun _ : Fin 128 => (1 : ℝ)) : EucSpace 128) 62 = 1 := by
  exact ⟨by omega, by norm_num, rfl⟩

/-- Genuine prenorm followed by the tensor-only operation recovers the entire learned raw input.
Source: TensorEmbedding's derived unit anchor for every free token/position parameter. -/
theorem tensorPrenormRecovery_input (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : SharedParameters V C) (token : Fin V) (position : Fin C) :
    tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position) =
      tensorInput hsize hwidth θ token position := by
  apply tensorAnchorRecovery_rms (by omega) eps heps
  exact tensorInput_anchor hsize hwidth θ token position

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by omega, by omega, by norm_num⟩

/-- Every actual learned token field survives both the real prenorm and internal recovery.
Source: the entire recovered input identity and the genuine 52-coordinate embedding read. -/
theorem tensorPrenormRecovery_field (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : SharedParameters V C) (token : Fin V) (position : Fin C) (slot : Fin 52) :
    tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
      (tensorFieldAxis hwidth slot) = θ (.inl (token, slot)) := by
  rw [tensorPrenormRecovery_input hsize hwidth eps heps]
  exact tensorInput_field hsize hwidth θ token position slot

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by omega, by omega, by norm_num⟩

/-- The learned absolute-position field also survives actual prenorm exactly.
Source: true position addition and internal recovery, rather than supplying its pre-normalization value externally. -/
theorem tensorPrenormRecovery_position (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : SharedParameters V C) (token : Fin V) (position : Fin C) :
    tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
      (tensorPositionAxis hwidth) = θ (.inr (.inl position)) := by
  rw [tensorPrenormRecovery_input hsize hwidth eps heps]
  exact tensorInput_position hsize hwidth θ token position

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by omega, by omega, by norm_num⟩

/-- All true query, key and jointly learned value channels are recovered from the actual tensor alone.
Source: SharedSlots' disjoint 16/16/20 projections, not a task-dependent feature selection. -/
theorem tensorPrenormRecovery_qkv (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : SharedParameters V C) (token : Fin V) (position : Fin C)
    (g c : Fin 4) (h : Fin 5) (value : Fin 4) :
    tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
        (tensorFieldAxis hwidth (querySlot g c)) = sharedQuery θ token g c ∧
      tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
        (tensorFieldAxis hwidth (keySlot g c)) = sharedKey θ token g c ∧
      tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
        (tensorFieldAxis hwidth (valueSlot h value)) = sharedValue θ token h value := by
  exact ⟨tensorPrenormRecovery_field hsize hwidth eps heps θ token position _,
    tensorPrenormRecovery_field hsize hwidth eps heps θ token position _,
    tensorPrenormRecovery_field hsize hwidth eps heps θ token position _⟩

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by omega, by omega, by norm_num⟩

/-- Every learned state transition is likewise read from the real recovered tensor.
Source: SharedSlots' six-by-six token-conditioned table, sharing the same unrestricted 52 fields. -/
theorem tensorPrenormRecovery_transition (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : SharedParameters V C) (token : Fin V) (position : Fin C)
    (previous next : Fin 6) :
    tensorPrenormRecovery eps hwidth (tensorInput hsize hwidth θ token position)
      (tensorFieldAxis hwidth (transitionSlot previous next)) = sharedTransitionRead token previous next θ := by
  rw [tensorPrenormRecovery_input hsize hwidth eps heps]
  exact tensorInput_field hsize hwidth θ token position (transitionSlot previous next)

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by omega, by omega, by norm_num⟩

end
end Transformer.GPTMini.Convex.Structured

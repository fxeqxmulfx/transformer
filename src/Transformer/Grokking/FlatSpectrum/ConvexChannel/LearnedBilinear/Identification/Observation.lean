module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Duality

/-!
# Observed directions, capacity, and unobservable parameter changes

Source: `review/convex/moments.md`, CM87. The evaluation operator maps complete
real parameter coordinates to observed correct-versus-competitor margins.
Labels impose inequalities on this operator; they do not reveal its output
values. Its rank bounds the independent observed directions. A certificate
annihilates its kernel, while common-logit and equal real/imaginary changes
lie in the kernel of every raw margin. Physical width and flat coefficient
capacity alone do not supply the required nonnegative directional certificate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- The complete observed-margin evaluation operator, CM87. Its rows are
indexed by records and original competitors, including repeated records. -/
noncomputable def observationOperator (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O) :
    (RealCoordinate O T → ℝ) →ₗ[ℝ] (ObservedIndex labels → ℝ) :=
  LinearMap.pi fun j => form (coefficient (tokens j.1) (labels j.1) j.2.1)

/-- The observation operator evaluates actual learned margins, CM87. -/
theorem observation_encode (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (p : SequenceParameters O T) (j : ObservedIndex labels) :
    observationOperator tokens labels (encode p) j = margin p (tokens j.1) (labels j.1) j.2.1 := by
  exact form_encode (tokens j.1) (labels j.1) j.2.1 p

/-- Independent observed directions are bounded by both full coordinate
capacity and the number of observed comparisons, CM87. This is not a claim
that either dimension or rank determines unobserved labels. -/
theorem observation_rank_bound (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O) :
    Module.finrank ℝ (LinearMap.range (observationOperator tokens labels)) ≤
      min (2 * Fintype.card O * (1 + Fintype.card O) *
        (1 + T * Fintype.card O + (if 2 ≤ T then 1 else 0) * Fintype.card O ^ 2))
        (Fintype.card (ObservedIndex labels)) := by
  have hd := LinearMap.finrank_range_le (observationOperator tokens labels)
  have hc := Submodule.finrank_le (LinearMap.range (observationOperator tokens labels))
  rw [Module.finrank_pi, sequence_real_coordinate_count] at hd
  rw [Module.finrank_pi] at hc
  exact le_min hd hc

/-- Every certified query margin vanishes on every unobserved direction,
CM87. This is a necessary directional condition, not merely a count. -/
theorem certificate_kills_kernel (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (bound : ℝ) (w : Fin N → O → ℝ)
    (hw : Certificate tokens labels query a b bound w) (r : RealCoordinate O T → ℝ)
    (hr : observationOperator tokens labels r = 0) :
    form (coefficient query a b) r = 0 := by
  rw [form_decode, certificate_identity tokens labels query a b w hw.2.2.2 (decode r)]
  apply Finset.sum_eq_zero
  intro i hi
  apply Finset.sum_eq_zero
  intro c hc
  by_cases he : c = labels i
  · subst c
    simp [hw.2.1]
  · have h := congrFun hr ⟨i, ⟨c, he⟩⟩
    change form (coefficient (tokens i) (labels i) c) r = 0 at h
    rw [← form_decode, h, mul_zero]

/-- A common matrix across original output classes is unobservable to every
raw margin, CM87. This accounts for common-logit gauge freedom. -/
theorem common_class_null (M : Matrix (QueryCoordinate O) (ContextCoordinate O T) ℂ)
    (tokens : Prefix O (T + 1)) (a b : O) :
    form (coefficient tokens a b) (encode (fun _ => M)) = 0 := by
  rw [form_encode]
  simp [margin, sequenceLogits]

/-- Equal real and imaginary increments in every joint entry vanish from
all real outputs, CM87. Their coordinates remain in the state and checker. -/
theorem complex_gauge_null (v : O → QueryCoordinate O → ContextCoordinate O T → ℝ)
    (tokens : Prefix O (T + 1)) (a b : O) :
    form (coefficient tokens a b) (encode (fun c q d => ⟨v c q d, v c q d⟩)) = 0 := by
  let p : SequenceParameters O T := fun c q d => ⟨v c q d, v c q d⟩
  change form (coefficient tokens a b) (encode p) = 0
  rw [form_encode, margin, logit_expansion p tokens a, logit_expansion p tokens b]
  simp [p]

/-- Positive certificates against every original competitor force the
actual next-object output, CM87. No query label is supplied to inference. -/
theorem nextObject_of_certificates (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a witness : O)
    (hc : ∀ b, b ≠ a → ∃ bound w, 0 < bound ∧ Certificate tokens labels query a b bound w)
    (p : SequenceParameters O T) (hp : FitsObserved tokens labels p) :
    nextObject p query witness = a := by
  apply nextObject_of_unique
  intro b hb
  obtain ⟨bound, w, hpos, hw⟩ := hc b hb
  have h := certificate_sound tokens labels query a b bound w hw p hp
  unfold margin at h
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

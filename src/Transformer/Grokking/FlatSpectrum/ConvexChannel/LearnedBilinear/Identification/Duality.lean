module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Forms
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative.Certificate

/-!
# Exact observed-margin duality for the learned sequence model

Source: `review/flat_spectrum.md`, CM87. The statement is the same user-requested
equivalence, proved for its unchanged complete complex parameter family.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open LearnedHinge (Prefix)
open scoped BigOperators

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- A checked coefficient identity transfers to every actual learned
margin, CM87. This statement does not require training feasibility. -/
theorem certificate_identity (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (query : Prefix O (T + 1)) (a b : O)
    (w : Fin N → O → ℝ)
    (heq : ∀ z, coefficient query a b z =
      ∑ i, ∑ c, w i c * coefficient (tokens i) (labels i) c z)
    (p : SequenceParameters O T) :
    margin p query a b = ∑ i, ∑ c, w i c * margin p (tokens i) (labels i) c := by
  rw [← form_encode]
  have hv : coefficient query a b =
      fun z => ∑ j : Fin N × O, w j.1 j.2 * coefficient (tokens j.1) (labels j.1) j.2 z := by
    funext z
    simpa only [Fintype.sum_prod_type] using heq z
  rw [hv, form_combination]
  simp only [Fintype.sum_prod_type, form_encode]

/-- A finite observed-label certificate forces its actual query margin,
CM87. All original competitors and all repeated records remain in the sums. -/
theorem certificate_sound (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (query : Prefix O (T + 1)) (a b : O) (bound : ℝ)
    (w : Fin N → O → ℝ) (hw : Certificate tokens labels query a b bound w)
    (p : SequenceParameters O T) (hp : FitsObserved tokens labels p) :
    bound ≤ margin p query a b := by
  rw [certificate_identity tokens labels query a b w hw.2.2.2 p]
  apply hw.2.2.1.trans
  apply Finset.sum_le_sum
  intro i hi
  apply Finset.sum_le_sum
  intro c hc
  by_cases h : c = labels i
  · subst c
    simp [hw.2.1]
  · simpa only [mul_one] using mul_le_mul_of_nonneg_left (hp i c h) (hw.1 i c)

/-- **CM87 bridge.** In the full actual learned rectangular family,
every parameter satisfying the observed unit margins has the requested query
margin iff a finite nonnegative observed-label certificate exists. Feasibility
prevents a vacuous universal conclusion from inconsistent training labels.
This is identification relative to the stated family, not discovery of the
true task family or convergence of an optimizer. Positive bounds for every
competitor would force the original-object prediction at the query. -/
theorem forced_margin_iff_certificate (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (query : Prefix O (T + 1)) (a b : O) (bound : ℝ)
    (hfeasible : ∃ p : SequenceParameters O T, FitsObserved tokens labels p) :
    (∀ p : SequenceParameters O T, FitsObserved tokens labels p →
      bound ≤ margin p query a b) ↔
      ∃ w : Fin N → O → ℝ, Certificate tokens labels query a b bound w := by
  classical
  constructor
  · intro hforced
    let A : ObservedIndex labels → (RealCoordinate O T → ℝ) →ₗ[ℝ] ℝ :=
      fun j => form (coefficient (tokens j.1) (labels j.1) j.2.1)
    let f := form (coefficient query a b)
    obtain ⟨p₀, hp₀⟩ := hfeasible
    have h₀ : ∀ j, 1 ≤ A j (encode p₀) := by
      intro j
      change 1 ≤ form _ (encode p₀)
      rw [form_encode]
      exact hp₀ j.1 j.2.1 j.2.2
    have hf : ∀ r, (∀ j, 1 ≤ A j r) → bound ≤ f r := by
      intro r hr
      change bound ≤ form _ r
      rw [form_decode]
      apply hforced (decode r)
      intro i c hc
      have h := hr ⟨i, ⟨c, hc⟩⟩
      simpa only [A, form_decode] using h
    obtain ⟨u, hu, hmass, heq⟩ :=
      Alternative.unit_margin_certificate A f bound (encode p₀) h₀ hf
    refine ⟨extendWeights labels u, extend_nonneg labels u hu,
      extend_self labels u, ?_, ?_⟩
    · simpa only [sum_extend] using hmass
    · intro z
      rw [sum_observed labels _ (by intro i; simp only [extend_self, zero_mul])]
      simp only [extend_competitor]
      have h := heq (Pi.single z 1)
      simpa only [f, A, form_single] using h
  · rintro ⟨w, hw⟩ p hp
    exact certificate_sound tokens labels query a b bound w hw p hp

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

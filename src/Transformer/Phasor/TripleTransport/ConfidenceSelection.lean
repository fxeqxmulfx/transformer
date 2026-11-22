module

public import Transformer.Phasor.TripleTransport

/-!
# Selecting a trusted prototype in parallel

In the “Triple transport” experiment a dense value contributes an edge only
when one prototype has a unique cosine maximum strictly above the confidence
threshold. A device warp first finds that maximum and then counts its winners.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport

/-- Source: the damped cosine classifier in the experiment's “Triple transport”
section. A slot supplies a trusted image when its score clears the threshold
and strictly exceeds every competing prototype score. -/
def strictWinner {ι : Type*} (score : ι → ℝ) (cut : ℝ) (i : ι) : Prop :=
  cut < score i ∧ ∀ j, j ≠ i → score j < score i

/-- Source: “Triple transport”, parallel extraction of the trusted bank.
At any maximizing slot the bank decision reduces to two independent tests:
the maximum clears the threshold, and exactly one index attains it. This
covers both a tie above the threshold and a maximum on the threshold. -/
theorem strictWinner_iff_unique_peak {ι : Type*} (score : ι → ℝ) (cut : ℝ)
    (i peak : ι) (hmax : ∀ j, score j ≤ score peak) :
    strictWinner score cut i ↔
      cut < score peak ∧ score i = score peak ∧ ∀ j, score j = score peak → j = i := by
  constructor
  · rintro ⟨above, unique⟩
    have h : peak = i := by
      by_contra unequal
      exact (not_lt_of_ge (hmax i)) (unique peak unequal)
    subst peak
    refine ⟨above, rfl, fun j eq => ?_⟩
    by_contra unequal
    exact (ne_of_lt (unique j unequal)) eq
  · rintro ⟨above, eq, only⟩
    refine ⟨eq ▸ above, fun j unequal => ?_⟩
    rw [eq]
    exact lt_of_le_of_ne (hmax j) (fun same => unequal (only j same))

example : strictWinner (fun b : Bool => if b then 3 else 1) 2 true := by
  apply (strictWinner_iff_unique_peak _ _ _ true (by intro j; cases j <;> norm_num)).2
  refine ⟨by norm_num, rfl, ?_⟩
  intro j same
  cases j <;> norm_num at *

end Transformer.Phasor.TripleTransport

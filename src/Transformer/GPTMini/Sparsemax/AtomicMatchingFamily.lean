import Transformer.GPTMini.Sparsemax.AtomicMatchingContinuity
import Transformer.GPTMini.Sparsemax.AtomicMatchingCompression
import Mathlib.Data.Fintype.EquivFin

/-!
# Compact finite charts of actual matching-mixture predictions

New attainment construction following arXiv:2211.11052v1, Appendix A.4.
A chart stores N freely selected Q/K/value heads and N probability weights.
It computes their actual causal sparsemax responses. The chart is compact;
its raw head-coordinate criterion is not asserted convex.

A finite stored mixture with at most N active heads embeds into this chart
by assigning zero weight to unused slots. All matching geometries remain
eligible: the slot identities are storage locations, not a preset atom bank.
The later R+1 chart covers the complete observed convex prediction class.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- N storage slots for freely selected physical heads and their probability weights.
Source: the finite atomic reconstruction after Appendix A.4. -/
abbrev MatchingFamily (V H D N : ℕ) :=
  (Fin N → MatchingHead V H D) × (Fin N → ℝ)

/-- Numerical head bounds and unit total mass for a finite physical chart.
Source: the bounded genuine-head variant of Appendix A.4. -/
def matchingFamilyDomain (V H D N : ℕ) (cap : ℝ) : Set (MatchingFamily V H D N) :=
  {p | (∀ i, p.1 i ∈ matchingHeadBox V H D cap) ∧ p.2 ∈ simplexOn Set.univ}

/-- Actual weighted sums of the selected causal attention/value outputs.
Source: sparsemax Eq. (1) inside the finite Appendix A.4 head family. -/
def matchingFamilySample {V H D N R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (p : MatchingFamily V H D N) : Fin R → ℝ :=
  ∑ i, p.2 i • matchingHeadSample tokens rows channels (p.1 i)

/-- The chart is compact even when empty; no fixed attention support is required.
Source: bounded physical heads and probability weights after Appendix A.4. -/
theorem matchingFamilyDomain_compact (V H D N : ℕ) (cap : ℝ) :
    IsCompact (matchingFamilyDomain V H D N cap) := by
  change IsCompact ({heads : Fin N → MatchingHead V H D | ∀ i, heads i ∈ matchingHeadBox V H D cap} ×ˢ
    simplexOn (Set.univ : Set (Fin N)))
  exact (isCompact_pi_infinite (fun _ => matchingHeadBox_compact V H D cap)).prod
    (simplexOn_compact _)

/-- Every chart's genuine physical forward is continuous across support and weight changes.
Source: sparsemax Eq. (1) and the finite head sum following Appendix A.4. -/
theorem matchingFamilySample_continuous {V H D N R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    Continuous (matchingFamilySample (H := H) (N := N) tokens rows channels) := by
  unfold matchingFamilySample
  apply continuous_finsetSum
  intro i hi
  exact ((continuous_apply i).comp continuous_snd).smul
    ((matchingHeadSample_continuous tokens rows channels).comp ((continuous_apply i).comp continuous_fst))

/-- Every valid finite chart reconstructs a feasible distributional matching state.
Source: exact finite-head reconstruction following Appendix A.4. -/
theorem matchingFamily_mem {V H D N : ℕ} (cap : ℝ) (p : MatchingFamily V H D N)
    (hp : p ∈ matchingFamilyDomain V H D N cap) :
    matchingMixtureOfFamily p.1 p.2 ∈ matchingMixtureDomain V H D cap := by
  apply matchingMixtureOfFamily_mem cap p.1 p.2 hp.1 hp.2.1 hp.2.2.1

/-- Two distinct learned scalar heads inhabit every chart-reconstruction premise. -/
example : matchingMixtureOfFamily (fun i : Fin 2 => matchingScalarHead i.val)
    (fun _ => (1 / 2 : ℝ)) ∈ matchingMixtureDomain 2 1 1 1 := by
  apply matchingFamily_mem (p := ((fun i : Fin 2 => matchingScalarHead i.val), fun _ => 1 / 2))
  refine ⟨?_, ?_⟩
  · intro i
    fin_cases i <;> apply matchingScalarHead_mem <;> norm_num
  · refine ⟨fun _ => by norm_num, by norm_num [Fin.sum_univ_two], ?_⟩
    intro j hj
    exact False.elim (hj (Set.mem_univ j))

/-- The reconstructed finite state computes exactly the chart's actual physical forward.
Source: finite attention-head summation following Appendix A.4. -/
theorem matchingFamily_reconstruct {V H D N R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (p : MatchingFamily V H D N) :
    matchingMixtureSample tokens rows channels (matchingMixtureOfFamily p.1 p.2) =
      matchingFamilySample tokens rows channels p := by
  exact matchingMixtureOfFamily_sample tokens rows channels p.1 p.2

/-- Every feasible state with at most N active heads fits the compact N-slot physical chart.
Source: padding of the actual finite-support representation after Appendix A.4, with zero unused mass. -/
theorem matchingMixture_pad_family {V H D N R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (μ : MatchingMixture V H D) (hμ : μ ∈ matchingMixtureDomain V H D cap) (hn : μ.support.card ≤ N) :
    ∃ p ∈ matchingFamilyDomain V H D N cap,
      matchingFamilySample tokens rows channels p = matchingMixtureSample tokens rows channels μ := by
  classical
  obtain ⟨e⟩ : Nonempty (μ.support ↪ Fin N) :=
    Function.Embedding.nonempty_of_card_le (by simpa only [Fintype.card_coe, Fintype.card_fin] using hn)
  let heads := Function.extend e (fun h : μ.support => h.val) (fun _ => (0 : MatchingHead V H D))
  let w := Function.extend e (fun h : μ.support => μ h.val) (fun _ => (0 : ℝ))
  have hh (i : Fin N) : heads i ∈ matchingHeadBox V H D cap := by
    by_cases hi : ∃ h : μ.support, e h = i
    · obtain ⟨h, rfl⟩ := hi
      rw [show heads (e h) = h.val from e.injective.extend_apply _ _ h]
      exact hμ.2.1 h.val (Finsupp.mem_support_iff.mp h.property)
    · rw [show heads i = 0 from Function.extend_apply' _ _ _ hi]
      exact matchingHeadBox_zero_mem V H D cap hc
  have hw (i : Fin N) : 0 ≤ w i := by
    by_cases hi : ∃ h : μ.support, e h = i
    · obtain ⟨h, rfl⟩ := hi
      rw [show w (e h) = μ h.val from e.injective.extend_apply _ _ h]
      exact hμ.1 h.val
    · rw [show w i = 0 from Function.extend_apply' _ _ _ hi]
  have hs : ∑ i, w i = 1 := by
    have he := Fintype.sum_of_injective e e.injective (fun h : μ.support => μ h.val) w
      (fun i hi => Function.extend_apply' _ _ _ hi)
      (fun h => (show w (e h) = μ h.val from
        e.injective.extend_apply (fun h : μ.support => μ h.val) (fun _ => (0 : ℝ)) h).symm)
    rw [← he]
    calc
      (∑ h : μ.support, μ h.val) = ∑ h ∈ μ.support, μ h :=
        Finset.sum_attach μ.support (fun h : MatchingHead V H D => μ h)
      _ = 1 := by rw [← matchingMixtureMass_eq_sum]; exact hμ.2.2
  refine ⟨(heads, w), ⟨hh, hw, hs, ?_⟩, ?_⟩
  · intro j hj
    exact False.elim (hj (Set.mem_univ j))
  · have he := Fintype.sum_of_injective e e.injective
      (fun h : μ.support => μ h.val • matchingHeadSample tokens rows channels h.val)
      (fun i => w i • matchingHeadSample tokens rows channels (heads i)) ?_ ?_
    · change (∑ i, w i • matchingHeadSample tokens rows channels (heads i)) = _
      rw [← he]
      calc
        (∑ h : μ.support, μ h.val • matchingHeadSample tokens rows channels h.val) =
            ∑ h ∈ μ.support, μ h • matchingHeadSample tokens rows channels h :=
          Finset.sum_attach μ.support
            (fun h : MatchingHead V H D => μ h • matchingHeadSample tokens rows channels h)
        _ = matchingMixtureSample tokens rows channels μ := rfl
    · intro i hi
      rw [show w i = 0 from Function.extend_apply' _ _ _ hi, zero_smul]
    · intro h
      rw [show w (e h) = μ h.val from e.injective.extend_apply _ _ h,
        show heads (e h) = h.val from e.injective.extend_apply _ _ h]

/-- A genuine single learned head satisfies every compact-chart padding premise. -/
example : ∃ p ∈ matchingFamilyDomain 2 1 1 2 1,
    matchingFamilySample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) p =
      matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 1) 1) := by
  apply matchingMixture_pad_family (cap := 1) (by norm_num)
  · exact matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))
  · norm_num [Finsupp.support_single]

end Transformer.GPTMini.Sparsemax

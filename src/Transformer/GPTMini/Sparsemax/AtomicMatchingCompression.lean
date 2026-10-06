import Transformer.GPTMini.Sparsemax.AtomicMatchingHull
import Mathlib.Analysis.Convex.Caratheodory
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Finite sample compression of freely learned matching heads

New genuine-sparsemax extension of arXiv:2211.11052v1, Appendix A.4.
Caratheodory is applied to actual observed responses of all bounded Q/K/
value heads, not to a fixed bit-interaction dictionary. R scalar observed
outputs admit an equivalent feasible mixture with at most R+1 active
heads. If every row/channel is observed, R counts all those coordinates.

The same sample criterion and numerical atom bounds are preserved. This
does not preserve responses on unobserved texts, a fixed head count,
the original four-matrix weight decay or a universal constant memory
bound. The theorem supplies finite existence, not an efficient atom-search
algorithm or a generalization guarantee.

The criterion reads observed physical outputs; it does not select head identities.
Every compressed head retains its independent original values and learned Q/K.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Merging a finite family cannot require more active heads than the family contains.
Source: literal finite atomic reconstruction after Appendix A.4. -/
theorem matchingMixtureOfFamily_card {ι : Type*} [Fintype ι] {V H D : ℕ}
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) :
    (matchingMixtureOfFamily heads w).support.card ≤ Fintype.card ι := by
  classical
  calc
    _ ≤ (Finset.univ.image heads).card := Finset.card_le_card (matchingMixtureOfFamily_support heads w)
    _ ≤ Fintype.card ι := by simpa only [Finset.card_univ] using (Finset.card_image_le (s := Finset.univ) (f := heads))

/-- Every feasible free-matching model has a bounded-size physical representative on the samples.
Source: Appendix A.4's Caratheodory argument, with the exact genuine-head hull proved first. -/
theorem matchingMixture_compress {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (μ : MatchingMixture V H D) (hμ : μ ∈ matchingMixtureDomain V H D cap) :
    ∃ ν : MatchingMixture V H D, ν ∈ matchingMixtureDomain V H D cap ∧
      matchingMixtureSample tokens rows channels ν = matchingMixtureSample tokens rows channels μ ∧
      ν.support.card ≤ R + 1 := by
  classical
  have hx : matchingMixtureSample tokens rows channels μ ∈
      convexHull ℝ (matchingSampleAtoms H cap tokens rows channels) := by
    rw [← matchingPredictionSet_eq_hull]
    exact ⟨μ, hμ, rfl⟩
  obtain ⟨ι, hι, z, w, hz, hAI, hw, hs, hy⟩ := eq_pos_convex_span_of_mem_convexHull hx
  let heads (i : ι) : MatchingHead V H D := Classical.choose (hz ⟨i, rfl⟩)
  have hh (i : ι) : heads i ∈ matchingHeadBox V H D cap := (Classical.choose_spec (hz ⟨i, rfl⟩)).1
  have he (i : ι) : matchingHeadSample tokens rows channels (heads i) = z i :=
    (Classical.choose_spec (hz ⟨i, rfl⟩)).2
  have hc : Fintype.card ι ≤ R + 1 := by
    have ha := hAI.card_le_finrank_succ
    have hb := Submodule.finrank_le (vectorSpan ℝ (Set.range z))
    have hd : Module.finrank ℝ ↥(vectorSpan ℝ (Set.range z)) ≤ R := by
      simpa only [Module.finrank_pi, Fintype.card_fin] using hb
    exact le_trans ha (Nat.add_le_add_right hd 1)
  refine ⟨matchingMixtureOfFamily heads w, matchingMixtureOfFamily_mem cap heads w hh (fun i => (hw i).le) hs, ?_, ?_⟩
  · rw [matchingMixtureOfFamily_sample]
    simp_rw [he]
    exact hy
  · exact le_trans (matchingMixtureOfFamily_card heads w) hc

/-- A real nonuniform matching head inhabits every sample-compression premise. -/
example : ∃ ν : MatchingMixture 2 1 1, ν ∈ matchingMixtureDomain 2 1 1 1 ∧
    matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν =
      matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 1) 1) ∧ ν.support.card ≤ 2 :=
  matchingMixture_compress _ _ _ _ _
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))

/-- Compression preserves every criterion that reads precisely the observed outputs.
Source: exact sample equality in the new Appendix A.4 matching construction; no task loss is chosen. -/
theorem matchingMixture_compress_criterion {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (μ : MatchingMixture V H D)
    (hμ : μ ∈ matchingMixtureDomain V H D cap) :
    ∃ ν : MatchingMixture V H D, ν ∈ matchingMixtureDomain V H D cap ∧
      criterion (matchingMixtureSample tokens rows channels ν) =
        criterion (matchingMixtureSample tokens rows channels μ) ∧ ν.support.card ≤ R + 1 := by
  obtain ⟨ν, hν, he, hc⟩ := matchingMixture_compress cap tokens rows channels μ hμ
  exact ⟨ν, hν, congrArg criterion he, hc⟩

/-- Ordinary nonconstant squared prediction error inhabits every criterion-compression premise. -/
example : ∃ ν : MatchingMixture 2 1 1, ν ∈ matchingMixtureDomain 2 1 1 1 ∧
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν 0 - 1) ^ 2 =
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 0) 1) 0 - 1) ^ 2 ∧ ν.support.card ≤ 2 :=
  matchingMixture_compress_criterion _ _ _ _ (fun y => (y 0 - 1) ^ 2) _
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))

/-- Storage for each selected head counts Q, K, values and its one mixture coefficient.
Source: literal parameter types in the new matching model following Appendix A.4. -/
def matchingStoredScalars (V H D heads : ℕ) : ℕ := heads * (2 * V * H + V * D + 1)

/-- Compressed learned scalar storage grows with observed outputs, not the number of possible texts.
Source: the proved active-head bound after Appendix A.4; this is not constant-size universal fitting. -/
theorem matchingMixture_compress_storage {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (μ : MatchingMixture V H D) (hμ : μ ∈ matchingMixtureDomain V H D cap) :
    ∃ ν : MatchingMixture V H D, ν ∈ matchingMixtureDomain V H D cap ∧
      matchingMixtureSample tokens rows channels ν = matchingMixtureSample tokens rows channels μ ∧
      matchingStoredScalars V H D ν.support.card ≤ matchingStoredScalars V H D (R + 1) := by
  obtain ⟨ν, hν, he, hc⟩ := matchingMixture_compress cap tokens rows channels μ hμ
  exact ⟨ν, hν, he, Nat.mul_le_mul_right _ hc⟩

/-- One observed scalar with two-token one-channel heads has a finite fourteen-scalar storage bound. -/
example : ∃ ν : MatchingMixture 2 1 1, ν ∈ matchingMixtureDomain 2 1 1 1 ∧
    matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν =
      matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 1) 1) ∧ matchingStoredScalars 2 1 1 ν.support.card ≤ 14 :=
  matchingMixture_compress_storage _ _ _ _ _
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))

/-- An attained optimum, if supplied, has an equally optimal physical representative with R+1 heads.
Source: Appendix A.4's sparse-optimum idea; attainment is an explicit hypothesis here. -/
theorem matchingMixture_optimum_compress {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (μ : MatchingMixture V H D)
    (hμ : μ ∈ matchingMixtureDomain V H D cap)
    (hmin : ∀ η ∈ matchingMixtureDomain V H D cap,
      criterion (matchingMixtureSample tokens rows channels μ) ≤ criterion (matchingMixtureSample tokens rows channels η)) :
    ∃ ν : MatchingMixture V H D, ν ∈ matchingMixtureDomain V H D cap ∧ ν.support.card ≤ R + 1 ∧
      ∀ η ∈ matchingMixtureDomain V H D cap,
        criterion (matchingMixtureSample tokens rows channels ν) ≤ criterion (matchingMixtureSample tokens rows channels η) := by
  obtain ⟨ν, hν, he, hc⟩ := matchingMixture_compress_criterion cap tokens rows channels criterion μ hμ
  refine ⟨ν, hν, hc, ?_⟩
  intro η hη
  rw [he]
  exact hmin η hη

/-- A nonconstant true prediction criterion can have an attained zero optimum on the genuine domain. -/
example : ∃ ν : MatchingMixture 2 1 1, ν ∈ matchingMixtureDomain 2 1 1 1 ∧ ν.support.card ≤ 2 ∧
    ∀ η ∈ matchingMixtureDomain 2 1 1 1,
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν 0 - 1) ^ 2 ≤
        (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) η 0 - 1) ^ 2 := by
  apply matchingMixture_optimum_compress _ _ _ _ (fun y => (y 0 - 1) ^ 2)
    (Finsupp.single (matchingScalarHead 1) 1)
  · exact matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))
  · intro η hη
    rw [matchingMixture_single_output]
    simp only [matchingHeadSample, matchingScalarHead_one_output, sub_self, zero_pow (by decide : 2 ≠ 0)]
    exact sq_nonneg _

end Transformer.GPTMini.Sparsemax

import Transformer.Modes.Section1_FixedPoint
import Mathlib.MeasureTheory.Group.MeasurableEquiv

/-
# The number of modes of a Gaussian KDE — endpoints, reflection and interval counts

The first bullet of `thm:mammen`, arXiv:2412.09080v3, §1.1, assigns limit one
both to an interval containing zero in its interior and to an interval with
zero as an endpoint. Reflection and the absence of fixed-point modes show
that these two conclusions cannot hold simultaneously.

These lemmas keep the actual mode count and its expectation. Reflecting a
Gaussian sample preserves its law and reflects every local maximum. When
the shared endpoint is not a mode, the counts of two adjacent intervals
add without double-counting. Their lower Lebesgue integrals satisfy the
same lower bound without assuming measurability of the count functions.
Fixed endpoints also have zero contribution in expectation.

Source: arXiv:2412.09080v3, §1.1, `thm:mammen`; §4.2, the reflection argument
following `lem:scale-space`; §1.1, `eq:gkde`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- Adjacent closed intervals have disjoint sets of modes when their shared
endpoint is not a mode. Source: arXiv:2412.09080v3, §1.1, `thm:mammen`. -/
theorem modeCount_adjacent_Icc_le (f : ℝ → ℝ) {a m b : ℝ}
    (ham : a ≤ m) (hmb : m ≤ b) (hm : ¬ IsLocalMax f m) :
    modeCount f (Set.Icc a m) + modeCount f (Set.Icc m b) ≤ modeCount f (Set.Icc a b) := by
  have hd : Disjoint (modeSet f (Set.Icc a m)) (modeSet f (Set.Icc m b)) := by
    apply Set.disjoint_left.mpr
    intro t ht ht'
    have htm : t = m := le_antisymm ht.1.2 ht'.1.1
    exact hm (htm ▸ ht.2)
  have hsub : modeSet f (Set.Icc a m) ∪ modeSet f (Set.Icc m b) ⊆
      modeSet f (Set.Icc a b) := by
    rintro t (ht | ht)
    · exact ⟨⟨ht.1.1, ht.1.2.trans hmb⟩, ht.2⟩
    · exact ⟨⟨ham.trans ht.1.1, ht.1.2⟩, ht.2⟩
  have hcard := Set.encard_le_encard hsub
  rw [Set.encard_union_eq hd] at hcard
  simpa only [modeCount, ENat.toENNReal_add] using ENat.toENNReal_le.mpr hcard

/-- Increasing affine functions witness the endpoint and order hypotheses. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ ¬ IsLocalMax (fun x : ℝ => x) 1 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro h
  have hz := h.hasDerivAt_eq_zero (hasDerivAt_id (1 : ℝ))
  norm_num at hz

/-- Reflection of the sample reflects the count in any closed interval.
Source: arXiv:2412.09080v3, §4.2, symmetry after `lem:scale-space`. -/
theorem modeCount_kde_Icc_reflect {n : ℕ} (β : ℝ) (X : Fin n → ℝ) (a b : ℝ) :
    modeCount (kde β (-X)) (Set.Icc (-b) (-a)) = modeCount (kde β X) (Set.Icc a b) := by
  have hk : kde β (-X) = fun t => kde β X (-t) := by
    funext t
    unfold kde
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    congr 2
    simp only [Pi.neg_apply]
    ring
  have hmax (t : ℝ) : IsLocalMax (kde β (-X)) (-t) ↔ IsLocalMax (kde β X) t := by
    rw [hk]
    constructor
    · intro h
      have hc := h.comp_continuous (g := fun x : ℝ => -x) (b := t)
        continuous_neg.continuousAt
      simpa only [Function.comp_def, neg_neg] using hc
    · intro h
      have h' : IsLocalMax (kde β X) (-(-t)) := by simpa only [neg_neg] using h
      have hc := h'.comp_continuous (g := fun x : ℝ => -x) (b := -t)
        continuous_neg.continuousAt
      simpa only [Function.comp_def, neg_neg] using hc
  have hs : modeSet (kde β (-X)) (Set.Icc (-b) (-a)) =
      (fun t : ℝ => -t) '' modeSet (kde β X) (Set.Icc a b) := by
    ext t
    simp only [modeSet, Set.mem_ofPred_eq, Set.mem_image, Set.mem_Icc]
    constructor
    · rintro ⟨ht, hm⟩
      refine ⟨-t, ⟨⟨by linarith [ht.2], by linarith [ht.1]⟩, ?_⟩, neg_neg t⟩
      exact (hmax (-t)).mp (by simpa only [neg_neg] using hm)
    · rintro ⟨s, ⟨hs, hm⟩, rfl⟩
      exact ⟨⟨by linarith [hs.2], by linarith [hs.1]⟩, (hmax s).mpr hm⟩
  rw [modeCount, hs, neg_injective.encard_image, modeCount]

/-- The Gaussian sample law is invariant under simultaneous reflection.
Source: arXiv:2412.09080v3, §4.2, symmetry after `lem:scale-space`. -/
theorem map_gaussianSample_neg (n : ℕ) :
    (gaussianSample n).map (fun X => -X) = gaussianSample n := by
  change (Measure.pi fun _ : Fin n => gaussianReal 0 1).map (fun X i => -X i) = _
  rw [Measure.pi_map_pi (fun _ => measurable_neg.aemeasurable)]
  simp only [gaussianReal_map_neg, neg_zero, gaussianSample]

/-- Reflection invariance of the expected count requires no measurability
assumption on the count: a measurable equivalence transports its lower integral.
Source: arXiv:2412.09080v3, §4.2, symmetry after `lem:scale-space`. -/
theorem expectedModes_Icc_reflect (β : ℝ) (n : ℕ) (a b : ℝ) :
    expectedModes β n (Set.Icc (-b) (-a)) = expectedModes β n (Set.Icc a b) := by
  have hpres : MeasurePreserving (MeasurableEquiv.neg (Fin n → ℝ)) (gaussianSample n)
      (gaussianSample n) := ⟨measurable_neg, map_gaussianSample_neg n⟩
  calc
    expectedModes β n (Set.Icc (-b) (-a)) =
        ∫⁻ X, modeCount (kde β (-X)) (Set.Icc (-b) (-a)) ∂gaussianSample n :=
      hpres.lintegral_map_equiv
        (fun X => modeCount (kde β X) (Set.Icc (-b) (-a)))
        (MeasurableEquiv.neg (Fin n → ℝ))
    _ = expectedModes β n (Set.Icc a b) := by
      apply lintegral_congr
      intro X
      exact modeCount_kde_Icc_reflect β X a b

/-- The two halves of a symmetric interval each have the same expectation.
Since zero is almost surely not a mode, their sum is at most the full count.
Source: arXiv:2412.09080v3, §1.1, first bullet of `thm:mammen`. -/
theorem twice_expectedModes_Icc_zero_le {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (hn : 0 < n) {b : ℝ} (hb : 0 ≤ b) :
    expectedModes β n (Set.Icc 0 b) + expectedModes β n (Set.Icc 0 b) ≤
      expectedModes β n (Set.Icc (-b) b) := by
  have hleft : expectedModes β n (Set.Icc (-b) 0) = expectedModes β n (Set.Icc 0 b) := by
    simpa only [neg_zero] using expectedModes_Icc_reflect β n 0 b
  calc
    expectedModes β n (Set.Icc 0 b) + expectedModes β n (Set.Icc 0 b) =
        expectedModes β n (Set.Icc (-b) 0) + expectedModes β n (Set.Icc 0 b) := by
      rw [hleft]
    _ ≤ ∫⁻ X, modeCount (kde β X) (Set.Icc (-b) 0) +
        modeCount (kde β X) (Set.Icc 0 b) ∂gaussianSample n :=
      le_lintegral_add _ _
    _ ≤ expectedModes β n (Set.Icc (-b) b) := by
      apply lintegral_mono_ae
      filter_upwards [ae_not_isLocalMax_kde hβ hn 0] with X hX
      exact modeCount_adjacent_Icc_le (kde β X) (by linarith) hb hX

/-- The symmetric-interval bound's hypotheses hold at positive finite values. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) ∧ (0 : ℝ) ≤ 1 :=
  ⟨one_pos, one_pos, zero_le_one⟩

/-- Including the left endpoint does not change the expected count.
Source: arXiv:2412.09080v3, §1.1, endpoint cases of `thm:mammen`. -/
theorem expectedModes_Icc_eq_Ioc {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (a b : ℝ) : expectedModes β n (Set.Icc a b) = expectedModes β n (Set.Ioc a b) := by
  apply lintegral_congr_ae
  filter_upwards [ae_not_isLocalMax_kde hβ hn a] with X hX
  have hs : modeSet (kde β X) (Set.Icc a b) = modeSet (kde β X) (Set.Ioc a b) := by
    ext t
    constructor
    · rintro ⟨ht, hm⟩
      rcases lt_or_eq_of_le ht.1 with hlt | rfl
      · exact ⟨⟨hlt, ht.2⟩, hm⟩
      · exact (hX hm).elim
    · rintro ⟨ht, hm⟩
      exact ⟨⟨ht.1.le, ht.2⟩, hm⟩
  rw [modeCount, modeCount, hs]

/-- Positive bandwidth and a nonempty sample satisfy the endpoint theorem. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) := ⟨one_pos, one_pos⟩

/-- Including the right endpoint also leaves the expectation unchanged.
Source: arXiv:2412.09080v3, §1.1, endpoint cases of `thm:mammen`. -/
theorem expectedModes_Icc_eq_Ico {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (a b : ℝ) : expectedModes β n (Set.Icc a b) = expectedModes β n (Set.Ico a b) := by
  apply lintegral_congr_ae
  filter_upwards [ae_not_isLocalMax_kde hβ hn b] with X hX
  have hs : modeSet (kde β X) (Set.Icc a b) = modeSet (kde β X) (Set.Ico a b) := by
    ext t
    constructor
    · rintro ⟨ht, hm⟩
      rcases lt_or_eq_of_le ht.2 with hlt | rfl
      · exact ⟨⟨ht.1, hlt⟩, hm⟩
      · exact (hX hm).elim
    · rintro ⟨ht, hm⟩
      exact ⟨⟨ht.1, ht.2.le⟩, hm⟩
  rw [modeCount, modeCount, hs]

/-- The two endpoint-removal theorems have the same positive witnesses. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) := ⟨one_pos, one_pos⟩

end Transformer.Modes

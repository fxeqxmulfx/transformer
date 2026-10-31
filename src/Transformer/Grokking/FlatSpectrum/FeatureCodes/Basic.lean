module

public import Transformer.Grokking.FlatSpectrum.Defs

/-!
# Joint answer codes and their classification gap

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*, items 1–2.
The coordinates may be individually noninjective. Nonnegative weights select the active
coordinates. For unit-modulus codes, the self-score minus a competitor's score is half
the weighted squared distance. It is positive exactly when an active coordinate separates
the two answers. These statements concern scores, with no rank or optimizer assumption.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open scoped BigOperators

variable {A R : Type*} [Fintype R]

/-- Linear readout from a feature vector, source item 1. Complex coordinates represent
two real coordinates; the real part makes the classifier's scores real. -/
noncomputable def score (w : R → ℝ) (C : A → R → ℂ) (z : R → ℂ) (c : A) : ℝ :=
  ∑ r, w r * (z r * star (C c r)).re

/-- The ideal score when the feature vector is the true answer's code, source item 1. -/
noncomputable def codeScore (w : R → ℝ) (C : A → R → ℂ) (a b : A) : ℝ :=
  score w C (C a) b

/-- Squared separation in the weighted code geometry, source item 1. -/
noncomputable def distanceSq (w : R → ℝ) (C : A → R → ℂ) (a b : A) : ℝ :=
  ∑ r, w r * Complex.normSq (C a r - C b r)

/-- A unit-modulus code's self-score is its total weight, source item 1. -/
theorem codeScore_self (w : R → ℝ) (C : A → R → ℂ)
    (hC : ∀ a r, Complex.normSq (C a r) = 1) (a : A) :
    codeScore w C a a = ∑ r, w r := by
  unfold codeScore score
  apply Finset.sum_congr rfl
  intro r _
  rw [Complex.star_def, Complex.mul_conj, Complex.ofReal_re, hC, mul_one]

/-- **Exact classification gap**, source item 1: twice the score gap is the weighted
squared code distance. No positivity or normalization of the weights is needed here. -/
theorem twice_score_gap (w : R → ℝ) (C : A → R → ℂ)
    (hC : ∀ a r, Complex.normSq (C a r) = 1) (a b : A) :
    2 * (codeScore w C a a - codeScore w C a b) = distanceSq w C a b := by
  rw [codeScore_self w C hC]
  unfold codeScore score distanceSq
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r _
  rw [Complex.normSq_sub, hC, hC, Complex.star_def]
  ring

/-- Squared code separation is nonnegative for nonnegative weights, source item 1. -/
theorem distanceSq_nonneg {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (a b : A) : 0 ≤ distanceSq w C a b :=
  Finset.sum_nonneg fun r _ => mul_nonneg (hw r) (Complex.normSq_nonneg _)

/-- Positive separation means that some active coordinate distinguishes the answers,
source item 2. Zero-weight coordinates are explicitly excluded. -/
theorem distanceSq_pos_iff {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (a b : A) :
    0 < distanceSq w C a b ↔ ∃ r, 0 < w r ∧ C a r ≠ C b r := by
  rw [distanceSq, Finset.sum_pos_iff_of_nonneg
    (fun r _ => mul_nonneg (hw r) (Complex.normSq_nonneg _))]
  simp only [Finset.mem_univ, true_and]
  constructor
  · rintro ⟨r, hr⟩
    refine ⟨r, ?_, ?_⟩
    · by_contra h
      have hz : w r = 0 := le_antisymm (le_of_not_gt h) (hw r)
      simp [hz] at hr
    · intro h
      simp [h] at hr
  · rintro ⟨r, hr, hne⟩
    exact ⟨r, mul_pos hr (Complex.normSq_pos.mpr (sub_ne_zero.mpr hne))⟩

/-- **Joint separation criterion**, source item 2: the correct class strictly beats a
competitor exactly when a positively weighted feature distinguishes them. -/
theorem codeScore_lt_self_iff {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1) (a b : A) :
    codeScore w C a b < codeScore w C a a ↔ ∃ r, 0 < w r ∧ C a r ≠ C b r := by
  rw [← distanceSq_pos_iff hw C a b]
  have hgap := twice_score_gap w C hC a b
  constructor <;> intro h <;> linarith

/-- A unit-modulus code never scores above the true code's self-score, source item 1. -/
theorem codeScore_le_self {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1) (a b : A) :
    codeScore w C a b ≤ codeScore w C a a := by
  have hgap := twice_score_gap w C hC a b
  have hdist := distanceSq_nonneg hw C a b
  linarith

/-- With positive weights, unique classification of every answer is equivalent to
injectivity of the whole code, source item 2. No individual coordinate need be injective. -/
theorem unique_classification_iff_injective {w : R → ℝ} (hw : ∀ r, 0 < w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1) :
    (∀ a b, a ≠ b → codeScore w C a b < codeScore w C a a) ↔ Function.Injective C := by
  constructor
  · intro h a b hab
    by_contra hne
    obtain ⟨r, _, hr⟩ := (codeScore_lt_self_iff (fun r => (hw r).le) C hC a b).mp
      (h a b hne)
    exact hr (congrFun hab r)
  · intro hinj a b hne
    apply (codeScore_lt_self_iff (fun r => (hw r).le) C hC a b).mpr
    have hdiff : ∃ r, C a r ≠ C b r := by
      by_contra h
      push Not at h
      exact hne (hinj (funext h))
    obtain ⟨r, hr⟩ := hdiff
    exact ⟨r, hw r, hr⟩

/-- A four-answer code with two binary coordinates, source item 2's example. -/
def binaryCode (a : Bool × Bool) (r : Bool) : ℂ :=
  if (if r then a.2 else a.1) then -1 else 1

theorem binaryCode_unit (a : Bool × Bool) (r : Bool) :
    Complex.normSq (binaryCode a r) = 1 := by
  rcases a with ⟨a, b⟩
  cases a <;> cases b <;> cases r <;> norm_num [binaryCode]

theorem binaryCode_injective : Function.Injective binaryCode := by
  rintro ⟨a, b⟩ ⟨c, d⟩ h
  have h0 := congrFun h false
  have h1 := congrFun h true
  cases a <;> cases b <;> cases c <;> cases d <;> norm_num [binaryCode] at *

/-- Both coordinates lose information individually, although the bank separates all four
answers. Source item 2: joint separation is strictly more general than a faithful feature. -/
theorem binaryCode_coordinate_not_injective (r : Bool) :
    ¬Function.Injective (fun a => binaryCode a r) := by
  intro h
  cases r
  · have := h (a₁ := (false, false)) (a₂ := (false, true)) (by norm_num [binaryCode])
    cases this
  · have := h (a₁ := (false, false)) (a₂ := (true, false)) (by norm_num [binaryCode])
    cases this

example := codeScore_self (fun _ : Bool => (1 / 2 : ℝ)) binaryCode binaryCode_unit (false, true)
example := twice_score_gap (fun _ : Bool => (1 / 2 : ℝ)) binaryCode binaryCode_unit
  (false, true) (true, false)
example := distanceSq_nonneg (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2))
  binaryCode (false, true) (true, false)
example := distanceSq_pos_iff (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2))
  binaryCode (false, true) (true, false)
example := codeScore_lt_self_iff (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2))
  binaryCode binaryCode_unit (false, true) (true, false)
example := codeScore_le_self (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2))
  binaryCode binaryCode_unit (false, true) (true, false)
example := (unique_classification_iff_injective
  (fun _ : Bool => (by norm_num : (0 : ℝ) < 1 / 2)) binaryCode binaryCode_unit).mpr
  binaryCode_injective

end Transformer.Grokking.FlatSpectrum.FeatureCodes

import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# Double descent and sample non-monotonicity

arXiv:1912.02292v1, Sections 2, 5, 6, and 7. These predicates express
finite, strict witnesses of the reported qualitative behavior. They do not
claim global monotonicity between sampled points, or identify a population
test-risk curve with a finite experimental log.
-/

namespace Transformer.DoubleDescent

variable {I : Type*} [Preorder I]

/-- Sections 2, 5, and 6: a first descent, an ascent, and a second descent
at four increasing complexity or epoch values. -/
def HasDoubleDescent (error : I → ℝ) : Prop :=
  ∃ a b c d, a < b ∧ b < c ∧ c < d ∧
    error b < error a ∧ error b < error c ∧ error d < error c

/-- Section 6, last paragraph: training after overfitting can improve on
the first minimum, a stronger property than double descent alone. -/
def CorrectsOverfitting (error : I → ℝ) : Prop :=
  ∃ a b c d, a < b ∧ b < c ∧ c < d ∧
    error b < error a ∧ error b < error c ∧ error d < error b

/-- Section 7: two sample sizes at which more data gives strictly worse loss. -/
def MoreDataHurts (error : I → ℝ) : Prop := ∃ a b, a < b ∧ error a < error b

/-- Sections 5 and 6: double descent rules out monotonically improving error. -/
theorem doubleDescent_not_antitone {error : I → ℝ} (h : HasDoubleDescent error) :
    ¬ Antitone error := by
  rcases h with ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdc⟩
  intro hm
  exact (not_lt_of_ge (hm (le_of_lt hbc))) hcb

/-- Sections 5 and 6: a strict double-descent witness exists. -/
example : HasDoubleDescent (fun n : ℕ =>
    if n = 0 then (3 : ℝ) else if n = 1 then 1 else if n = 2 then 4 else 0) := by
  refine ⟨0, 1, 2, 3, ?_⟩
  norm_num

/-- Section 6: improvement beyond the first minimum implies a second descent. -/
theorem correction_implies_doubleDescent {error : I → ℝ}
    (h : CorrectsOverfitting error) : HasDoubleDescent error := by
  rcases h with ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdb⟩
  exact ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdb.trans hcb⟩

/-- Section 6: the stronger overfitting-correction premise is satisfiable. -/
example : CorrectsOverfitting (fun n : ℕ =>
    if n = 0 then (3 : ℝ) else if n = 1 then 1 else if n = 2 then 4 else 0) := by
  refine ⟨0, 1, 2, 3, ?_⟩
  norm_num

/-- Section 4, label noise: a positive affine rescaling preserves every
finite double-descent witness, including its strict inequalities. -/
theorem doubleDescent_affine {error : I → ℝ} {slope offset : ℝ}
    (hs : 0 < slope) :
    HasDoubleDescent (fun i => offset + slope * error i) ↔ HasDoubleDescent error := by
  constructor
  · rintro ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdc⟩
    refine ⟨a, b, c, d, hab, hbc, hcd, ?_, ?_, ?_⟩ <;> nlinarith
  · rintro ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdc⟩
    refine ⟨a, b, c, d, hab, hbc, hcd, ?_, ?_, ?_⟩ <;> nlinarith

/-- Section 4: the positive-slope premise holds for the identity rescaling. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Section 7: a monotone improvement with sample size excludes more-data harm. -/
theorem moreDataHurts_not_antitone {error : I → ℝ} (h : MoreDataHurts error) :
    ¬ Antitone error := by
  rcases h with ⟨a, b, hab, he⟩
  intro hm
  exact (not_lt_of_ge (hm (le_of_lt hab))) he

/-- Section 7: the sample-harm premise has a concrete increasing-risk witness. -/
example : MoreDataHurts (fun n : ℕ => (n : ℝ)) := by
  exact ⟨0, 1, by norm_num, by norm_num⟩

/-- Sections 2 and 5: double descent supplies a pair where increased
complexity hurts; with a sample-size axis it supplies more-data harm. -/
theorem doubleDescent_has_increase {error : I → ℝ} (h : HasDoubleDescent error) :
    MoreDataHurts error := by
  rcases h with ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdc⟩
  exact ⟨b, c, hbc, hcb⟩

/-- Section 5: the double-descent premise is realized by a four-point curve. -/
example : HasDoubleDescent (fun n : ℕ =>
    if n = 0 then (3 : ℝ) else if n = 1 then 1 else if n = 2 then 4 else 0) := by
  refine ⟨0, 1, 2, 3, ?_⟩
  norm_num

end Transformer.DoubleDescent

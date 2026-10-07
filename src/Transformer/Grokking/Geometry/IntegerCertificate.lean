import Transformer.Grokking.Geometry.IntegerEncoding

/-!
# The exact integer inequality certifies observed decisions

Source: exact_cell in geometry_certificates.py; the margin theorem from
the fixed-orbit interpretation of Nanda et al., arXiv:2301.05217v1,
section 5.1, proved in Geometry.Decisions at 3661a44.

For an integer column-sum target gap g and m classes, the sufficient test
is 2*sum(aligned_numerator^2) < (m*g)^2 with g positive. The averaging
counts and common positive grid scale cancel exactly; the theorem derives
this rather than defining correctness through the inequality. It then
proves the strict prediction of the original observed raw row.

The input is an integer representation of observed logits, not integer
model weights. Neither Python's max loop, Float extraction nor an AdamW
trajectory is proved here. Those are distinct implementation bridges.

The certificate concerns one current observed row. Applying it to a finite
list gives a checked subset of current correct decisions, not a law that
its size must increase during training. Positive scale invariance also
does not supply a margin for the correct target; that premise is explicit.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators
open Transformer.Grokking.NaiveLoss

variable {I C : Type*}

/-- The exact integer test equals the real sufficient cleanup inequality.
Source: exact_cell in geometry_certificates.py, derived from restricted
logits of arXiv:2301.05217v1, section 5.1. A separate positive-gap condition
is necessary for correctness; squaring a wrong margin does not fix it. -/
theorem integer_cleanup_inequality_iff (s : Finset I) (cs : Finset C)
    (a : I → C → ℤ) (scale : ℝ) (d : I) (gap : ℤ)
    (hs : s.Nonempty) (hc : cs.Nonempty) (hp : 0 < scale) :
    (2 * energyOver cs (fun c => alignedLogits s cs (gridLogits a scale) d c -
      columnMean s (gridLogits a scale) c) < ((gap : ℝ) / ((s.card : ℝ) * scale)) ^ 2) ↔
    (2 * (∑ c ∈ cs, (integerAlignedNumerator s cs a d c) ^ 2) <
      ((cs.card : ℤ) * gap) ^ 2) := by
  have hn : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
  have hm : (0 : ℝ) < cs.card := by exact_mod_cast Finset.card_pos.mpr hc
  have hnz : (s.card : ℝ) ≠ 0 := by linarith
  have hmz : (cs.card : ℝ) ≠ 0 := by linarith
  have hsz : scale ≠ 0 := by linarith
  have heq : ((gap : ℝ) / ((s.card : ℝ) * scale)) ^ 2 =
      ((cs.card : ℝ) * (gap : ℝ)) ^ 2 /
        ((s.card : ℝ) * (cs.card : ℝ) * scale) ^ 2 := by
    field_simp
  rw [grid_aligned_energy s cs a scale d hs hc hp, heq, ← mul_div_assoc,
    div_lt_div_iff_of_pos_right (by positivity)]
  constructor <;> intro h <;> exact_mod_cast h

example : ({0, 1} : Finset ℕ).Nonempty ∧ ({false, true} : Finset Bool).Nonempty ∧
    0 < (2 : ℝ) := by
  refine ⟨⟨0, by norm_num⟩, ⟨false, by norm_num⟩, by norm_num⟩

/-- A common positive rescaling cannot fake this cleanup inequality.
Source: the exact arithmetic of geometry_certificates.py and confidence
versus decision changes in arXiv:2501.04697v1, section 4.2. This concerns
the inequality, not Python's separate fixed numerical energy floor. -/
theorem integer_cleanup_scale_invariant (s : Finset I) (cs : Finset C)
    (a : I → C → ℤ) (d : I) (gap : ℤ) (scale₁ scale₂ : ℝ)
    (hs : s.Nonempty) (hc : cs.Nonempty) (h₁ : 0 < scale₁) (h₂ : 0 < scale₂) :
    (2 * energyOver cs (fun c => alignedLogits s cs (gridLogits a scale₁) d c -
      columnMean s (gridLogits a scale₁) c) < ((gap : ℝ) / ((s.card : ℝ) * scale₁)) ^ 2) ↔
    (2 * energyOver cs (fun c => alignedLogits s cs (gridLogits a scale₂) d c -
      columnMean s (gridLogits a scale₂) c) < ((gap : ℝ) / ((s.card : ℝ) * scale₂)) ^ 2) := by
  rw [integer_cleanup_inequality_iff s cs a scale₁ d gap hs hc h₁,
    integer_cleanup_inequality_iff s cs a scale₂ d gap hs hc h₂]

example : ({false, true} : Finset Bool).Nonempty ∧ ({0, 1} : Finset ℕ).Nonempty ∧
    0 < (1 : ℝ) ∧ 0 < (2 : ℝ) := by
  refine ⟨⟨true, by norm_num⟩, ⟨1, by norm_num⟩, by norm_num, by norm_num⟩

/-- The reader's positive integer gap and exact squared-numerator test
certify the strict raw prediction. Source: exact_cell in
geometry_certificates.py, using the margin condition derived from
restricted logits of arXiv:2301.05217v1, section 5.1. The target gap must
bound every competing column, not only a selected convenient class. -/
theorem integer_current_decision_certifies [Fintype C] (s : Finset I)
    (a : I → C → ℤ) (scale : ℝ) (d : I) (y : C) (gap : ℤ)
    (hs : s.Nonempty) (hp : 0 < scale) (hg : 0 < gap)
    (hm : ∀ k, k ≠ y → gap ≤ (∑ i ∈ s, a i y) - (∑ i ∈ s, a i k))
    (he : 2 * (∑ c : C, (integerAlignedNumerator s Finset.univ a d c) ^ 2) <
      (((Finset.univ : Finset C).card : ℤ) * gap) ^ 2) :
    StrictCorrect (gridLogits a scale d) y := by
  have hn : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
  have hden : 0 < (s.card : ℝ) * scale := by positivity
  have hgap : (0 : ℝ) < gap := by exact_mod_cast hg
  have hc : (Finset.univ : Finset C).Nonempty := ⟨y, Finset.mem_univ y⟩
  apply row_aligned_cleanup_certifies s (gridLogits a scale) d y
    ((gap : ℝ) / ((s.card : ℝ) * scale))
  · intro k hk
    rw [grid_target_gap]
    apply (div_le_div_iff_of_pos_right hden).mpr
    exact_mod_cast hm k hk
  · exact div_pos hgap hden
  · exact (integer_cleanup_inequality_iff s Finset.univ a scale d gap hs hc hp).mpr he

example : ({0, 1} : Finset ℕ).Nonempty ∧ 0 < (2 : ℝ) ∧ 0 < (2 : ℤ) ∧
    (∀ k : Bool, k ≠ true → (2 : ℤ) ≤ 2 - 2 * (if k then (1 : ℤ) else -1)) ∧
    2 * (∑ c : Bool, (integerAlignedNumerator ({0, 1} : Finset ℕ) Finset.univ
      (fun _ (b : Bool) => if b then (1 : ℤ) else -1) 0 c) ^ 2) <
        (((Finset.univ : Finset Bool).card : ℤ) * 2) ^ 2 := by
  refine ⟨⟨0, by norm_num⟩, by norm_num, by norm_num, ?_, ?_⟩
  · intro k hk
    cases k
    · norm_num
    · exact False.elim (hk rfl)
  · norm_num [integerAlignedNumerator, Fintype.sum_bool]

/-- A failed strict raw prediction against a positive reference gap needs
at least the integer threshold energy. Source: exact_cell in
geometry_certificates.py and the restricted-logit interpretation of
arXiv:2301.05217v1, section 5.1. Failure of the sufficient test by itself
does not imply an incorrect prediction; the converse is not claimed. -/
theorem integer_failed_decision_requires_energy [Fintype C] (s : Finset I)
    (a : I → C → ℤ) (scale : ℝ) (d : I) (y : C) (gap : ℤ)
    (hs : s.Nonempty) (hp : 0 < scale) (hg : 0 < gap)
    (hm : ∀ k, k ≠ y → gap ≤ (∑ i ∈ s, a i y) - (∑ i ∈ s, a i k))
    (hw : ¬StrictCorrect (gridLogits a scale d) y) :
    (((Finset.univ : Finset C).card : ℤ) * gap) ^ 2 ≤
      2 * (∑ c : C, (integerAlignedNumerator s Finset.univ a d c) ^ 2) := by
  by_contra h
  have he : 2 * (∑ c : C, (integerAlignedNumerator s Finset.univ a d c) ^ 2) <
      (((Finset.univ : Finset C).card : ℤ) * gap) ^ 2 := by omega
  exact hw (integer_current_decision_certifies s a scale d y gap hs hp hg hm he)

-- Rows (0, 0) and (4, 0) have positive target column-sum gap. The first
-- observed row is still tied, and therefore fails strict correctness.
example : ({0, 1} : Finset ℕ).Nonempty ∧ 0 < (2 : ℝ) ∧ 0 < (2 : ℤ) ∧
    (∀ k : Bool, k ≠ true → (2 : ℤ) ≤
      (∑ i ∈ ({0, 1} : Finset ℕ), (if i = 0 then (0 : ℤ) else 4)) -
        (∑ i ∈ ({0, 1} : Finset ℕ), (if k then (if i = 0 then (0 : ℤ) else 4) else 0))) ∧
    ¬StrictCorrect (gridLogits
      (fun (i : ℕ) (b : Bool) => if b then (if i = 0 then (0 : ℤ) else 4) else 0) 2 0) true := by
  refine ⟨⟨0, by norm_num⟩, by norm_num, by norm_num, ?_, ?_⟩
  · intro k hk
    cases k
    · norm_num
    · exact False.elim (hk rfl)
  · intro h
    have he := h false (by decide)
    norm_num [gridLogits] at he

end Transformer.Grokking.Geometry

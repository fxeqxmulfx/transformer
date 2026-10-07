import Mathlib.Algebra.Field.Rat
import Mathlib.Algebra.GroupWithZero.Units.Basic
import Mathlib.Tactic

/-!
# The division task's actual common-scaling orbits

Empirical source: Power et al., arXiv:2201.02177v1, section 3.1 and
Figure 1, division modulo 97. Numeric source: openai/grok at
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, grok/data.py, as ported in
lab.infrastructure.benchmarks.modular.complete_rows. Diagnostic source:
lab.infrastructure.engine.grokking at 43d4d66.

The generator uses input (denominator * quotient, denominator) with a
nonzero denominator. Here operations are in a field; the finite prime
case specializes this algebra. The correct answer is actual division,
not an assumed label attached to an arbitrary partition. Equality of
answers is equivalent to common nonzero scaling of both operands, and
every valid input is recovered by the generator's formula.

These identities justify which examples share a task answer. They do
not imply that a learned logit map respects that symmetry or that its
invariant component decodes the correct label. Python's integer inverse,
token encoding and NumPy split remain additional implementation bridges.
-/

namespace Transformer.Grokking.DivisionOrbits

variable {F : Type*} [Field F]

/-- Numeric row schema of complete_rows. Source: the author-code port
above and arXiv:2201.02177v1, section 3.1; token wrappers are omitted. -/
def divisionInput (q d : F) : F × F := (d * q, d)

/-- Common scaling of both operands. Source: the division-orbit observer
at 43d4d66, adapting arXiv:2301.05217v1, section 5.1. -/
def commonScale (u : F) (p : F × F) : F × F := (u * p.1, u * p.2)

/-- Valid inputs with a specified actual division answer. Source: the
numeric task of arXiv:2201.02177v1, section 3.1, and modular.answer. -/
def InDivisionCell (q : F) (p : F × F) : Prop := p.2 ≠ 0 ∧ p.1 / p.2 = q

/-- The generated operands actually divide to their specified quotient.
Source: complete_rows and its independent inverse check, implementing
arXiv:2201.02177v1, section 3.1. Nonzero denominator is essential. -/
theorem generated_quotient (q d : F) (hd : d ≠ 0) :
    (divisionInput q d).1 / (divisionInput q d).2 = q := by
  unfold divisionInput
  field_simp

example : (3 : ℚ) ≠ 0 := by norm_num

/-- Generated valid rows belong to the cell of their actual answer.
Source: complete_rows, implementing arXiv:2201.02177v1, section 3.1. -/
theorem generated_in_cell (q d : F) (hd : d ≠ 0) :
    InDivisionCell q (divisionInput q d) := by
  exact ⟨hd, generated_quotient q d hd⟩

example : (2 : ℚ) ≠ 0 := by norm_num

/-- Every valid operand pair is reconstructed using its actual quotient
and denominator. Source: the exhaustive corpus schema in complete_rows,
implementing arXiv:2201.02177v1, section 3.1. -/
theorem input_recovered_by_quotient (p : F × F) (hp : p.2 ≠ 0) :
    divisionInput (p.1 / p.2) p.2 = p := by
  apply Prod.ext
  · change p.2 * (p.1 / p.2) = p.1
    field_simp
  · rfl

example : (3 : ℚ) ≠ 0 := by norm_num

/-- Nonzero common scaling leaves the division answer unchanged.
Source: the orbit interpretation at 43d4d66, adapting restricted logits
of arXiv:2301.05217v1, section 5.1, to the actual division task. -/
theorem common_scale_quotient (u : F) (p : F × F) (hu : u ≠ 0) :
    (commonScale u p).1 / (commonScale u p).2 = p.1 / p.2 := by
  exact mul_div_mul_left p.1 p.2 hu

example : (2 : ℚ) ≠ 0 := by norm_num

/-- Nonzero scaling keeps valid operands in the task domain. Source:
the nonzero-denominator restriction of arXiv:2201.02177v1, section 3.1,
implemented by modular.answer and complete_rows. -/
theorem common_scale_valid (u : F) (p : F × F) (hu : u ≠ 0) (hp : p.2 ≠ 0) :
    (commonScale u p).2 ≠ 0 := by
  exact mul_ne_zero hu hp

example : (2 : ℚ) ≠ 0 ∧ (3 : ℚ) ≠ 0 := by norm_num

/-- The actual answer cell is preserved in both directions by common
nonzero scaling. Source: the full-orbit observer at 43d4d66, adapting
arXiv:2301.05217v1, section 5.1. No model logits enter this identity. -/
theorem division_cell_scale_iff (q u : F) (p : F × F) (hu : u ≠ 0) :
    InDivisionCell q (commonScale u p) ↔ InDivisionCell q p := by
  constructor
  · intro h
    refine ⟨(mul_ne_zero_iff.mp h.1).2, ?_⟩
    rw [← common_scale_quotient u p hu]
    exact h.2
  · intro h
    refine ⟨common_scale_valid u p hu h.1, ?_⟩
    rw [common_scale_quotient u p hu]
    exact h.2

example : (2 : ℚ) ≠ 0 := by norm_num

/-- Two valid inputs have the same answer exactly when one nonzero
factor scales both operands. Source: completeness of the division-orbit
partition at 43d4d66, for the task in arXiv:2201.02177v1, section 3.1.
The converse is derived from field division, not assumed in a predicate. -/
theorem equal_quotients_iff_common_scale (a b c d : F) (hb : b ≠ 0) (hd : d ≠ 0) :
    a / b = c / d ↔ ∃ u : F, u ≠ 0 ∧ c = u * a ∧ d = u * b := by
  constructor
  · intro h
    have hcross := (div_eq_div_iff hb hd).mp h
    refine ⟨d / b, div_ne_zero hd hb, ?_, ?_⟩
    · field_simp
      linear_combination -hcross
    · field_simp
  · rintro ⟨u, hu, hc, he⟩
    rw [hc, he]
    exact (common_scale_quotient u (a, b) hu).symm

example : (2 : ℚ) ≠ 0 ∧ (3 : ℚ) ≠ 0 ∧ (4 : ℚ) / 2 = 6 / 3 := by
  norm_num

/-- On any valid input, a common scaling factor is uniquely determined.
Source: the complete division orbit at 43d4d66, adapting
arXiv:2301.05217v1, section 5.1. Zero-numerator inputs also satisfy this. -/
theorem common_scale_factor_unique (p : F × F) (u v : F) (hp : p.2 ≠ 0)
    (he : commonScale u p = commonScale v p) : u = v := by
  have hden := congrArg Prod.snd he
  change u * p.2 = v * p.2 at hden
  exact mul_right_cancel₀ hp hden

example : (3 : ℚ) ≠ 0 ∧ commonScale (2 : ℚ) (6, 3) = commonScale 2 (6, 3) := by
  exact ⟨by norm_num, rfl⟩

/-- Dropping valid denominators breaks the equivalence between division
answers and common scaling. Source: the domain guard in modular.answer,
implementing arXiv:2201.02177v1, section 3.1. Lean's total division sends
both invalid quotients to zero; Python rejects them instead. This refutes
an unrestricted extension, not the paper's valid-domain task. -/
theorem zero_denominator_breaks_orbit_equivalence :
    (1 : ℚ) / 0 = 0 / 0 ∧
      ¬∃ u : ℚ, u ≠ 0 ∧ 0 = u * 1 ∧ 0 = u * 0 := by
  constructor
  · norm_num
  · rintro ⟨u, hu, hn, _⟩
    have hz : u = 0 := by simpa using hn.symm
    exact hu hz

end Transformer.Grokking.DivisionOrbits

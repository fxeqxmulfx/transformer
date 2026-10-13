import Transformer.Grokking.DivisionOrbits.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Finset.Prod
import Mathlib.Data.Fintype.Card

/-!
# Exhaustive division cells and the modulo-97 corpus size

Source: Power et al., arXiv:2201.02177v1, section 3.1 and Figure 1;
the author-code port in modular.complete_rows; GrokkingObserver at
43d4d66. Enumerate each actual answer cell as (d*q, d) over every nonzero
field denominator. Membership equals actual division correctness, the
cells are nonempty and disjoint, and each has field-cardinality minus one
points. This derives the observer's quotient-by-denominator shape.

For modulo 97 there are 96 points per full cell and 9,312 valid inputs.
These are full-corpus counts. They do not imply that a held-out mask has
two points per cell; that empirical coverage is checked separately.
The zero-quotient cell is included in the numeric task even though the
main energy diagnostic omits it. Numeric field semantics do not verify
Python's inverse implementation, token dictionary or train/test shuffle.
-/

namespace Transformer.Grokking.DivisionOrbits

variable {F : Type*} [Field F] [Fintype F]

/-- Primality needed for the study's field semantics. Source:
arXiv:2201.02177v1, section 3.1 and Figure 1, division modulo 97. -/
theorem study_prime97 : Nat.Prime 97 := by norm_num

instance studyPrimeFact : Fact (Nat.Prime 97) := ⟨study_prime97⟩

/-- Exhaustive generator cell. Source: modular.complete_rows, implementing
arXiv:2201.02177v1, section 3.1. Zero denominators are removed. -/
noncomputable def fullCell (q : F) : Finset (F × F) := by
  classical
  exact ((Finset.univ : Finset F).erase 0).image (divisionInput q)

/-- Exhaustive valid numeric input domain. Source: modular.complete_rows,
implementing arXiv:2201.02177v1, section 3.1; numerator zero is allowed. -/
noncomputable def fullDomain : Finset (F × F) := by
  classical
  exact (Finset.univ : Finset F).product ((Finset.univ : Finset F).erase 0)

/-- Generator-cell membership is exactly the actual division answer
on the valid domain. Source: modular.complete_rows and the division
task in arXiv:2201.02177v1, section 3.1. -/
theorem mem_full_cell_iff (q : F) (p : F × F) :
    p ∈ fullCell q ↔ InDivisionCell q p := by
  classical
  unfold fullCell
  rw [Finset.mem_image]
  constructor
  · rintro ⟨d, hd, rfl⟩
    exact generated_in_cell q d (Finset.mem_erase.mp hd).1
  · rintro ⟨hd, hq⟩
    refine ⟨p.2, Finset.mem_erase.mpr ⟨hd, Finset.mem_univ p.2⟩, ?_⟩
    rw [← hq]
    exact input_recovered_by_quotient p hd

/-- Every full answer cell has a valid point, including quotient zero.
Source: complete_rows, implementing arXiv:2201.02177v1, section 3.1. -/
theorem full_cell_nonempty (q : F) : (fullCell q).Nonempty := by
  refine ⟨divisionInput q 1, ?_⟩
  apply (mem_full_cell_iff q _).mpr
  exact generated_in_cell q 1 one_ne_zero

/-- Retaining the denominator in each generated pair makes the map
injective, so each full cell has exactly |F|-1 points. Source:
complete_rows and the orbit shape at 43d4d66, adapting
arXiv:2301.05217v1, section 5.1, to division. -/
theorem full_cell_card (q : F) : (fullCell q).card = Fintype.card F - 1 := by
  classical
  have hinj : Function.Injective (divisionInput q) := by
    intro d e he
    exact congrArg Prod.snd he
  unfold fullCell
  rw [Finset.card_image_of_injective _ hinj,
    Finset.card_erase_of_mem (Finset.mem_univ (0 : F)), Finset.card_univ]

/-- A valid pair cannot occur under two different answer labels.
Source: the independent answer check in modular.complete_rows,
implementing arXiv:2201.02177v1, section 3.1. -/
theorem full_cell_unique_label (q r : F) (p : F × F)
    (hq : p ∈ fullCell q) (hr : p ∈ fullCell r) : q = r := by
  have hpq := (mem_full_cell_iff q p).mp hq
  have hpr := (mem_full_cell_iff r p).mp hr
  exact hpq.2.symm.trans hpr.2

example : (2, 1) ∈ fullCell (2 : ZMod 97) ∧ (2, 1) ∈ fullCell (2 : ZMod 97) := by
  have h : (2, 1) ∈ fullCell (2 : ZMod 97) := by
    apply (mem_full_cell_iff _ _).mpr
    norm_num [InDivisionCell]
  exact ⟨h, h⟩

/-- Different quotient cells are disjoint. Source: the exhaustive
quotient-indexed table in modular.complete_rows, implementing
arXiv:2201.02177v1, section 3.1. -/
theorem distinct_full_cells_disjoint (q r : F) (hqr : q ≠ r) :
    Disjoint (fullCell q) (fullCell r) := by
  classical
  apply Finset.disjoint_left.mpr
  intro p hq hr
  exact hqr (full_cell_unique_label q r p hq hr)

example : (2 : ZMod 97) ≠ 3 := by decide

/-- A full answer cell is exactly the nonzero-scaling orbit of any
of its points. Source: GrokkingObserver at 43d4d66, adapting
arXiv:2301.05217v1, section 5.1. Validity of the other point follows
from membership or from the nonzero scale, rather than being assumed. -/
theorem mem_full_cell_iff_common_scale (q : F) (p r : F × F)
    (hp : p ∈ fullCell q) :
    r ∈ fullCell q ↔ ∃ u : F, u ≠ 0 ∧ r = commonScale u p := by
  have hpc := (mem_full_cell_iff q p).mp hp
  constructor
  · intro hr
    have hrc := (mem_full_cell_iff q r).mp hr
    have he := hpc.2.trans hrc.2.symm
    obtain ⟨u, hu, hn, hd⟩ :=
      (equal_quotients_iff_common_scale p.1 p.2 r.1 r.2 hpc.1 hrc.1).mp he
    refine ⟨u, hu, ?_⟩
    exact Prod.ext hn hd
  · rintro ⟨u, hu, rfl⟩
    apply (mem_full_cell_iff q _).mpr
    exact (division_cell_scale_iff q u p hu).mpr hpc

example : (2, 1) ∈ fullCell (2 : ZMod 97) := by
  apply (mem_full_cell_iff _ _).mpr
  norm_num [InDivisionCell]

/-- The full numeric domain has |F|*(|F|-1) distinct inputs. Source:
the exhaustive disjoint two-way corpus in modular.complete_rows,
implementing arXiv:2201.02177v1, section 3.1. -/
theorem full_domain_card : (fullDomain (F := F)).card =
    Fintype.card F * (Fintype.card F - 1) := by
  classical
  unfold fullDomain
  rw [Finset.product_eq_sprod, Finset.card_product,
    Finset.card_erase_of_mem (Finset.mem_univ (0 : F)),
    Finset.card_univ]

/-- Every modulo-97 quotient cell has the observer's 96 denominator
points. Source: the task in arXiv:2201.02177v1, section 3.1 and Figure 1,
and the actual GrokkingObserver shape at 43d4d66. -/
theorem modulus97_cell_card (q : ZMod 97) : (fullCell q).card = 96 := by
  calc
    (fullCell q).card = Fintype.card (ZMod 97) - 1 := full_cell_card q
    _ = 96 := by rw [ZMod.card]

/-- The actual modulus-97 numeric corpus has 9,312 operand pairs.
Source: arXiv:2201.02177v1, section 3.1 and Figure 1; complete_rows
includes the zero quotient and excludes only zero denominators. -/
theorem modulus97_domain_card : (fullDomain (F := ZMod 97)).card = 9312 := by
  calc
    (fullDomain (F := ZMod 97)).card =
      Fintype.card (ZMod 97) * (Fintype.card (ZMod 97) - 1) := full_domain_card
    _ = 9312 := by rw [ZMod.card]

end Transformer.Grokking.DivisionOrbits

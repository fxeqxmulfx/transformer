/-
# Finite synthetic dataset counts

arXiv:2402.19449v2, Section 2.2 and Appendix A.2 (Custom datasets).
The displayed m groups contain 1,2,...,2^(m-1) classes, totaling 2^m-1.
The source instead states c=2^(m+1)-1. Sample and feature counts agree
with its m-group construction; the class-count discrepancy is explicit.
-/

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

open scoped BigOperators

namespace Transformer.Imbalance

/-- Classes in group g, indexed from zero; Appendix A.2's displayed groups. -/
def groupClasses (g : ℕ) : ℕ := 2 ^ g

/-- Samples per class in group g; Appendix A.2's displayed groups. -/
def groupSamples (m g : ℕ) : ℕ := 2 ^ (m - g)

/-- Number of classes in the displayed m-group construction; Appendix A.2. -/
def syntheticClasses (m : ℕ) : ℕ := ∑ g ∈ Finset.range m, groupClasses g

/-- Number of samples in the displayed m-group construction; Appendix A.2. -/
def syntheticSamples (m : ℕ) : ℕ :=
  ∑ g ∈ Finset.range m, groupClasses g * groupSamples m g

/-- Feature dimension of random heavy-tailed labels; Appendix A.2. -/
def syntheticDimensions (m : ℕ) : ℕ := (m + 1) * 2 ^ m

/-- Source's separately reported class-count formula; Appendix A.2.
It disagrees with the number of classes in its displayed m groups. -/
def reportedSyntheticClasses (m : ℕ) : ℕ := 2 ^ (m + 1) - 1

/-- Every displayed group has the same sample mass 2^m; Appendix A.2. -/
theorem group_sample_mass (m g : ℕ) (hg : g ≤ m) :
    groupClasses g * groupSamples m g = 2 ^ m := by
  unfold groupClasses groupSamples
  rw [← pow_add, Nat.add_sub_of_le hg]

/-- Nonvacuity of the group-index hypothesis; Appendix A.2. -/
example : 3 ≤ 11 := by decide

/-- Corrected class count of the literal displayed construction;
Appendix A.2. The paper states 2^(m+1)-1 instead of 2^m-1. -/
theorem syntheticClasses_corrected (m : ℕ) : syntheticClasses m = 2 ^ m - 1 := by
  have h : syntheticClasses m + 1 = 2 ^ m := by
    induction m with
    | zero => simp [syntheticClasses]
    | succ m ih =>
      simp only [syntheticClasses, Finset.sum_range_succ, groupClasses] at ih ⊢
      rw [pow_succ]
      omega
  omega

/-- Exact sample count of Appendix A.2's m groups. -/
theorem syntheticSamples_eq (m : ℕ) : syntheticSamples m = m * 2 ^ m := by
  unfold syntheticSamples
  have he : (∑ g ∈ Finset.range m, groupClasses g * groupSamples m g) =
      ∑ _g ∈ Finset.range m, 2 ^ m := by
    exact Finset.sum_congr rfl fun g hg => group_sample_mass m g (Nat.le_of_lt (Finset.mem_range.1 hg))
  rw [he]
  simp

/-- The paper's two sample and feature totals, verified exactly;
Appendix A.2: m=8 and m=11. -/
theorem synthetic_reported_sample_dimensions :
    syntheticSamples 8 = 2048 ∧ syntheticDimensions 8 = 2304 ∧
    syntheticSamples 11 = 22528 ∧ syntheticDimensions 11 = 24576 := by
  norm_num [syntheticSamples_eq, syntheticDimensions]

/-- Explicit discrepancy in Appendix A.2's reported class counts.
For m=8 and m=11 the literal displayed groups give 255 and 2047, while
the source reports 511 and 4095. This concerns the written construction,
not the unknown labels used in the authors' experimental code. -/
theorem paper_synthetic_class_counts_inconsistent :
    syntheticClasses 8 = 255 ∧ reportedSyntheticClasses 8 = 511 ∧
    syntheticClasses 11 = 2047 ∧ reportedSyntheticClasses 11 = 4095 := by
  norm_num [syntheticClasses_corrected, reportedSyntheticClasses]

/-- Barcoded-only MNIST classes in Appendix A.2: digit times 10-bit barcode. -/
def barcodedClasses : ℕ := 10 * 2 ^ 10

/-- Samples in the barcoded-only dataset, five per class; Appendix A.2. -/
def barcodedSamples : ℕ := 5 * barcodedClasses

/-- Verification of the reported Barcoded MNIST construction;
Appendix A.2: 10,240 new classes, 51,200 new images, 10,250 combined
classes, and 101,200 combined images after adding the 50,000 originals. -/
theorem barcoded_mnist_counts :
    barcodedClasses = 10240 ∧ barcodedSamples = 51200 ∧
    barcodedClasses + 10 = 10250 ∧ barcodedSamples + 50000 = 101200 := by
  norm_num [barcodedClasses, barcodedSamples]

end Transformer.Imbalance

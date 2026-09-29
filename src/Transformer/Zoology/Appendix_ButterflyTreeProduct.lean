/-
# Recursive butterfly trees in the primary product representation

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`.
Extracting the binary-level coefficients produces exactly the primary
factor product. This bridge uses a single row for the scalar vector;
reindexing it into a general n × d layout is a separate transport step.
-/

import Transformer.Zoology.Appendix_ButterflyTreeLevels

namespace Transformer.Zoology

/-- First t binary factors extracted from a recursive tree.
Source: Appendix `def: butterfly`, increasing block sizes in product order. -/
def butterflyTreePrefixStages {k : ℕ} (tree : ButterflyTree k)
    (t : ℕ) (ht : t ≤ k) : List (BinaryButterflyStage 0 k) :=
  List.ofFn fun j : Fin t => {
    axis := .inr ⟨j.val, lt_of_lt_of_le j.isLt ht⟩
    main := fun _ q => tree.mainCoefficient j.val q
    off := fun _ q => tree.offCoefficient j.val q
  }

/-- Executing the extracted prefix equals the recursive level evaluator.
Source: Appendix `def: butterfly`, factorization of the recursive matrix. -/
theorem butterflyTreePrefixStages_correct {k : ℕ} (tree : ButterflyTree k)
    (t : ℕ) (ht : t ≤ k) (u : Fin (butterflyWidth k) → ℝ) :
    applyBinaryButterflyStages (butterflyTreePrefixStages tree t ht)
      (fun _ q => u q) = fun _ q => tree.applyLevels t u q := by
  induction t with
  | zero => simp [butterflyTreePrefixStages, applyBinaryButterflyStages, ButterflyTree.applyLevels]
  | succ t ih =>
      let last : BinaryButterflyStage 0 k := {
        axis := .inr ⟨t, by omega⟩
        main := fun _ q => tree.mainCoefficient t q
        off := fun _ q => tree.offCoefficient t q
      }
      unfold butterflyTreePrefixStages
      rw [List.ofFn_succ_last]
      change applyBinaryButterflyStages
        (butterflyTreePrefixStages tree t (by omega) ++ [last]) (fun _ q => u q) = _
      rw [applyBinaryButterflyStages_append, ih]
      funext i q
      rfl

/-- The primary coefficient arrays of a recursively represented matrix.
Source: Appendix `def: butterfly`, equivalence of both definitions. -/
def ButterflyTree.toRowMajor {k : ℕ} (tree : ButterflyTree k) : RowMajorButterfly 0 k := {
  main := fun t _ q => tree.mainCoefficient t.val q
  off := fun t _ q => tree.offCoefficient t.val q
}

/-- The primary product stores exactly the extracted binary factors.
Source: Appendix `def: butterfly`, B_n ... B_2 factor list. -/
theorem ButterflyTree.toRowMajor_stages {k : ℕ} (tree : ButterflyTree k) :
    tree.toRowMajor.stages = butterflyTreePrefixStages tree k (le_refl k) := by
  unfold ButterflyTree.toRowMajor RowMajorButterfly.stages butterflyTreePrefixStages
  congr 1
  funext j
  have hj : j.val < k := j.isLt
  simp [hj]

/-- The extracted primary product and recursive matrix have identical
action on every scalar vector. Source: Appendix `def: butterfly`,
the paper's recursive/product equivalence, with a one-row vector layout. -/
theorem ButterflyTree.toRowMajor_correct {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth k) → ℝ) :
    tree.toRowMajor.apply (fun _ q => u q) = fun _ q => tree.apply u q := by
  rw [RowMajorButterfly.apply, tree.toRowMajor_stages,
    butterflyTreePrefixStages_correct, tree.applyLevels_eq]

/-- The previously represented recursive matrices therefore have an
actual K-constrained compiler, with nine rows and unchanged vector width.
Source: Appendix `prop: butterfly-hyena`, a proved extension to a one-row
vector encoding; the source lemma itself assumes n,d ≥ 2. -/
theorem ButterflyTree.productModel_correct {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth k) → ℝ) :
    (binaryButterflyProgramModel tree.toRowMajor.stages).run (fun _ q => u q) =
      fun _ q => tree.apply u q := by
  rw [tree.toRowMajor.model_correct, tree.toRowMajor_correct]

/-- The prefix theorem's dimension bound has a nonempty instance.
Source: Appendix `def: butterfly`, a full depth-one product. -/
example : (1 : ℕ) ≤ 1 := by decide

end Transformer.Zoology

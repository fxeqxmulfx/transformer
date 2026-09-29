/-
# Orientation of a binary butterfly pair

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`.
The partner exchanges upper and lower halves; choosing the upper endpoint
therefore gives a canonical index shared by both coordinates of a pair.
-/

import Transformer.Zoology.Appendix_ExpandedKaleidoscopeGrid

namespace Transformer.Zoology

/-- Reading the opposite endpoint reverses the half-block orientation.
Source: Appendix `def: butterfly`, two-half diagonal blocks. -/
theorem butterflyToggleUpper_toggle (k : ℕ) (t : Fin k)
    (q : Fin (butterflyWidth k)) :
    butterflyToggleUpper k t.val (butterflyToggleIndex k t.val q) =
      !(butterflyToggleUpper k t.val q) := by
  induction k with
  | zero => exact Fin.elim0 t
  | succ k ih =>
      have hq := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply q
      generalize hp : finProdFinEquiv.symm q = h at hq
      rcases h with ⟨half, j⟩
      rw [← hq]
      by_cases ht : t.val = k
      · fin_cases half <;>
          simp [butterflyToggleIndex, butterflyToggleUpper, ht]
      · have hsmall : t.val < k := by have hb := t.isLt; omega
        simpa [butterflyToggleIndex, butterflyToggleUpper, ht] using ih ⟨t.val, hsmall⟩ j

/-- Upper endpoint of the binary pair containing q.
Source: Appendix `def: butterfly`, canonical block coefficient index. -/
def butterflyPairUpper (k : ℕ) (t : Fin k) (q : Fin (butterflyWidth k)) :
    Fin (butterflyWidth k) :=
  if butterflyToggleUpper k t.val q then q else butterflyToggleIndex k t.val q

/-- Both coordinates select the same upper endpoint.
Source: Appendix `def: butterfly`, one matrix per coordinate pair. -/
theorem butterflyPairUpper_toggle (k : ℕ) (t : Fin k)
    (q : Fin (butterflyWidth k)) :
    butterflyPairUpper k t (butterflyToggleIndex k t.val q) = butterflyPairUpper k t q := by
  unfold butterflyPairUpper
  rw [butterflyToggleUpper_toggle]
  cases butterflyToggleUpper k t.val q <;>
    simp [butterflyToggleIndex_involutive k t.val q]

/-- The canonical endpoint is always in the upper half.
Source: Appendix `def: butterfly`, orientation of the four diagonal blocks. -/
theorem butterflyPairUpper_orientation (k : ℕ) (t : Fin k)
    (q : Fin (butterflyWidth k)) :
    butterflyToggleUpper k t.val (butterflyPairUpper k t q) = true := by
  unfold butterflyPairUpper
  cases h : butterflyToggleUpper k t.val q <;>
    simp [h, butterflyToggleUpper_toggle]

end Transformer.Zoology

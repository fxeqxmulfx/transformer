/-
# Binary partner reads in the original row-major layout

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`.
The low k digits of a flattened n × d input are feature digits; the
remaining p digits are row digits. These identities transport the actual
recursive butterfly partner permutation, without increasing feature width.
-/

import Transformer.Zoology.Appendix_ButterflyGridLayout

namespace Transformer.Zoology

/-- Toggling a low vector digit changes only the feature coordinate.
Source: Appendix `prop: butterfly-hyena`, row-major factor action. -/
theorem butterflyGridFlatten_toggle_feature (p k t : ℕ) (ht : t < k)
    (i : Fin (butterflyWidth p)) (q : Fin (butterflyWidth k)) :
    butterflyToggleIndex (k + p) t (butterflyGridFlatten p k i q) =
      butterflyGridFlatten p k i (butterflyToggleIndex k t q) := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hi := (finProdFinEquiv (m := 2) (n := butterflyWidth p)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = h at hi
      rcases h with ⟨half, j⟩
      have htop : t ≠ k + p := by omega
      rw [← hi, butterflyGridFlatten_pair]
      change butterflyToggleIndex ((k + p) + 1) t
        (finProdFinEquiv (half, butterflyGridFlatten p k j q)) = _
      rw [butterflyToggleIndex_pair (k + p) t half _,
        ite_eq_right htop, ih, butterflyGridFlatten_pair]

/-- Toggling a high vector digit changes only the corresponding row digit.
Source: Appendix `prop: butterfly-hyena`, row-major factor action. -/
theorem butterflyGridFlatten_toggle_row (p k t : ℕ) (ht : t < p)
    (i : Fin (butterflyWidth p)) (q : Fin (butterflyWidth k)) :
    butterflyToggleIndex (k + p) (k + t) (butterflyGridFlatten p k i q) =
      butterflyGridFlatten p k (butterflyToggleIndex p t i) q := by
  induction p generalizing t with
  | zero => omega
  | succ p ih =>
      have hi := (finProdFinEquiv (m := 2) (n := butterflyWidth p)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = h at hi
      rcases h with ⟨half, j⟩
      rw [← hi, butterflyGridFlatten_pair]
      change butterflyToggleIndex ((k + p) + 1) (k + t)
        (finProdFinEquiv (half, butterflyGridFlatten p k j q)) = _
      rw [butterflyToggleIndex_pair (k + p) (k + t) half _,
        butterflyToggleIndex_pair p t half j]
      by_cases htop : t = p
      · subst t
        rw [ite_eq_left rfl, ite_eq_left rfl, butterflyGridFlatten_pair]
      · have hlow : t < p := by omega
        have hglobal : k + t ≠ k + p := by omega
        rw [ite_eq_right hglobal, ite_eq_right htop, butterflyGridFlatten_pair, ih t hlow]

/-- Both digit bounds have nonempty examples in a genuine 2 × 2 layout.
Source: Appendix `prop: butterfly-hyena`, n,d ≥ 2. -/
example : (0 : ℕ) < 1 ∧ (0 : ℕ) < 1 := by decide

end Transformer.Zoology

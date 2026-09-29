/-
# Exact row-major layout of a butterfly vector

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`:
the scalar vector is the row-major form of the n × d input. A recursive
coordinate equivalence exposes row digits above the feature digits, and
its numeric value agrees with the usual i*d+q row-major formula.
-/

import Transformer.Zoology.Appendix_ButterflyTreeProduct

namespace Transformer.Zoology

/-- Full vector width is the product of row and feature dimensions.
Source: Appendix `prop: butterfly-hyena`, vector dimension nd. -/
theorem butterflyGrid_width (p k : ℕ) :
    butterflyWidth (k + p) = butterflyWidth p * butterflyWidth k := by
  induction p with
  | zero => simp [butterflyWidth]
  | succ p ih =>
      change 2 * butterflyWidth (k + p) = (2 * butterflyWidth p) * butterflyWidth k
      rw [ih]
      ring

/-- Assemble row digits followed by feature digits into a scalar index.
Source: Appendix `prop: butterfly-hyena`, row-major input vector. -/
def butterflyGridFlatten : (p k : ℕ) → Fin (butterflyWidth p) →
    Fin (butterflyWidth k) → Fin (butterflyWidth (k + p))
  | 0, _, _, q => q
  | p + 1, k, i, q =>
      let h := finProdFinEquiv.symm i
      finProdFinEquiv (h.1, butterflyGridFlatten p k h.2 q)

/-- Separate a flat index into its row and feature coordinates.
Source: Appendix `prop: butterfly-hyena`, n × d layout. -/
def butterflyGridUnflatten : (p k : ℕ) → Fin (butterflyWidth (k + p)) →
    ButterflyGrid p k
  | 0, _, j => (0, j)
  | p + 1, k, j =>
      let h := finProdFinEquiv.symm j
      let c := butterflyGridUnflatten p k h.2
      (finProdFinEquiv (h.1, c.1), c.2)

/-- Flattening an explicitly split row preserves its top binary digit.
Source: Appendix `prop: butterfly-hyena`, row-major binary layout. -/
theorem butterflyGridFlatten_pair (p k : ℕ) (half : Fin 2)
    (i : Fin (butterflyWidth p)) (q : Fin (butterflyWidth k)) :
    butterflyGridFlatten (p + 1) k (finProdFinEquiv (half, i)) q =
      finProdFinEquiv (half, butterflyGridFlatten p k i q) := by
  dsimp only [butterflyGridFlatten]
  rw [Equiv.symm_apply_apply]

/-- Decoding an encoded pair returns precisely its original coordinates.
Source: Appendix `prop: butterfly-hyena`, bijective scalar layout. -/
theorem butterflyGridUnflatten_flatten (p k : ℕ)
    (i : Fin (butterflyWidth p)) (q : Fin (butterflyWidth k)) :
    butterflyGridUnflatten p k (butterflyGridFlatten p k i q) = (i, q) := by
  induction p with
  | zero =>
      have hi : i = (0 : Fin 1) := Subsingleton.elim (α := Fin 1) _ _
      subst i
      rfl
  | succ p ih =>
      have hi := (finProdFinEquiv (m := 2) (n := butterflyWidth p)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = h at hi
      rcases h with ⟨half, j⟩
      rw [← hi, butterflyGridFlatten_pair]
      simp only [butterflyGridUnflatten, Equiv.symm_apply_apply, ih]

/-- Encoding a decoded index returns precisely its original scalar index.
Source: Appendix `prop: butterfly-hyena`, bijective scalar layout. -/
theorem butterflyGridFlatten_unflatten (p k : ℕ)
    (j : Fin (butterflyWidth (k + p))) :
    butterflyGridFlatten p k (butterflyGridUnflatten p k j).1
      (butterflyGridUnflatten p k j).2 = j := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hj := (finProdFinEquiv (m := 2) (n := butterflyWidth (k + p))).apply_symm_apply j
      generalize hp : finProdFinEquiv.symm j = h at hj
      rcases h with ⟨half, i⟩
      rw [← hj]
      simp only [butterflyGridUnflatten, Equiv.symm_apply_apply,
        butterflyGridFlatten_pair, ih]
      rfl

/-- Exact equivalence between all n*d scalar entries and the n × d grid.
Source: Appendix `prop: butterfly-hyena`, row-major correspondence. -/
def butterflyGridEquiv (p k : ℕ) : ButterflyGrid p k ≃
    Fin (butterflyWidth (k + p)) := {
  toFun := fun c => butterflyGridFlatten p k c.1 c.2
  invFun := butterflyGridUnflatten p k
  left_inv := fun c => butterflyGridUnflatten_flatten p k c.1 c.2
  right_inv := butterflyGridFlatten_unflatten p k
}

/-- The recursive index is exactly the ordinary row-major numeric index.
Source: Appendix `prop: butterfly-hyena`, x is the row-major form of u. -/
theorem butterflyGridFlatten_val (p k : ℕ) (i : Fin (butterflyWidth p))
    (q : Fin (butterflyWidth k)) :
    (butterflyGridFlatten p k i q).val = i.val * butterflyWidth k + q.val := by
  induction p with
  | zero =>
      have hi : i = (0 : Fin 1) := Subsingleton.elim (α := Fin 1) _ _
      subst i
      simp [butterflyGridFlatten]
  | succ p ih =>
      have hi := (finProdFinEquiv (m := 2) (n := butterflyWidth p)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = h at hi
      rcases h with ⟨half, j⟩
      rw [← hi, butterflyGridFlatten_pair]
      simp only [finProdFinEquiv_apply_val, ih, butterflyGrid_width]
      ring

end Transformer.Zoology

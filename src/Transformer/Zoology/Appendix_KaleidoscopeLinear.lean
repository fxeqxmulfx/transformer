/-
# Linearity of butterfly and kaleidoscope operators

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`,
`def: kaleidoscope`, and Theorem `thm: k-linear-ac`. This module verifies
that the stored butterfly operators and their products are linear maps,
as required when they represent weight matrices in `def: W-kmat`.
-/

import Transformer.Zoology.Appendix_Kaleidoscope

namespace Transformer.Zoology

/-- A butterfly factor preserves vector addition.
Source: Appendix `def: butterfly`, four diagonal blocks. -/
theorem ButterflyFactor.apply_add {m : ℕ} (f : ButterflyFactor m)
    (u v : HalfVector m) : f.apply (u + v) = f.apply u + f.apply v := by
  funext half t
  fin_cases half <;>
    simp [ButterflyFactor.apply, Pi.add_apply] <;> ring

/-- A butterfly factor preserves scalar multiplication.
Source: Appendix `def: butterfly`, four diagonal blocks. -/
theorem ButterflyFactor.apply_smul {m : ℕ} (f : ButterflyFactor m)
    (r : ℝ) (u : HalfVector m) : f.apply (r • u) = r • f.apply u := by
  funext half t
  fin_cases half <;>
    simp [ButterflyFactor.apply, Pi.smul_apply] <;> ring

/-- Every recursively represented butterfly matrix preserves addition.
Source: Appendix `def: butterfly`, recursive display. -/
theorem ButterflyTree.apply_add : ∀ {k : ℕ} (tree : ButterflyTree k)
    (u v : Fin (butterflyWidth k) → ℝ),
    tree.apply (u + v) = tree.apply u + tree.apply v := by
  intro k tree
  induction tree with
  | leaf =>
      intro u v
      rfl
  | @node k factor left right ihLeft ihRight =>
      intro u v
      funext j
      simp only [ButterflyTree.apply]
      have hleft : left.apply (fun t => (u + v) (finProdFinEquiv (0, t))) =
          left.apply (fun t => u (finProdFinEquiv (0, t))) +
            left.apply (fun t => v (finProdFinEquiv (0, t))) := by
        have hinput : (fun t => (u + v) (finProdFinEquiv (0, t))) =
            (fun t => u (finProdFinEquiv (0, t))) +
              (fun t => v (finProdFinEquiv (0, t))) := by
          funext t
          rfl
        rw [hinput, ihLeft]
      have hright : right.apply (fun t => (u + v) (finProdFinEquiv (1, t))) =
          right.apply (fun t => u (finProdFinEquiv (1, t))) +
            right.apply (fun t => v (finProdFinEquiv (1, t))) := by
        have hinput : (fun t => (u + v) (finProdFinEquiv (1, t))) =
            (fun t => u (finProdFinEquiv (1, t))) +
              (fun t => v (finProdFinEquiv (1, t))) := by
          funext t
          rfl
        rw [hinput, ihRight]
      rw [hleft, hright]
      by_cases h : (finProdFinEquiv.symm j).1 = 0 <;>
        simp [ButterflyFactor.apply, Pi.add_apply, h] <;> ring

/-- Every recursively represented butterfly matrix preserves scalar
multiplication. Source: Appendix `def: butterfly`, recursive display. -/
theorem ButterflyTree.apply_smul : ∀ {k : ℕ} (tree : ButterflyTree k)
    (r : ℝ) (u : Fin (butterflyWidth k) → ℝ),
    tree.apply (r • u) = r • tree.apply u := by
  intro k tree
  induction tree with
  | leaf =>
      intro r u
      rfl
  | @node k factor left right ihLeft ihRight =>
      intro r u
      funext j
      simp only [ButterflyTree.apply]
      have hleft : left.apply (fun t => (r • u) (finProdFinEquiv (0, t))) =
          r • left.apply (fun t => u (finProdFinEquiv (0, t))) := by
        have hinput : (fun t => (r • u) (finProdFinEquiv (0, t))) =
            r • (fun t => u (finProdFinEquiv (0, t))) := by
          funext t
          rfl
        rw [hinput, ihLeft]
      have hright : right.apply (fun t => (r • u) (finProdFinEquiv (1, t))) =
          r • right.apply (fun t => u (finProdFinEquiv (1, t))) := by
        have hinput : (fun t => (r • u) (finProdFinEquiv (1, t))) =
            r • (fun t => u (finProdFinEquiv (1, t))) := by
          funext t
          rfl
        rw [hinput, ihRight]
      rw [hleft, hright]
      by_cases h : (finProdFinEquiv.symm j).1 = 0 <;>
        simp [ButterflyFactor.apply, Pi.smul_apply, h] <;> ring

/-- Multiplication by the transpose of the right butterfly matrix
preserves vector addition. Source: Appendix `def: kaleidoscope`, `BB*`. -/
private theorem butterflyTranspose_add {k : ℕ} (tree : ButterflyTree k)
    (u v : Fin (butterflyWidth k) → ℝ) :
    (fun j => ∑ i, tree.matrix i j * (u + v) i) =
      (fun j => ∑ i, tree.matrix i j * u i) +
      (fun j => ∑ i, tree.matrix i j * v i) := by
  funext j
  simp [Pi.add_apply, mul_add, Finset.sum_add_distrib]

/-- Multiplication by the transpose of the right butterfly matrix
preserves scalar multiplication. Source: Appendix `def: kaleidoscope`. -/
private theorem butterflyTranspose_smul {k : ℕ} (tree : ButterflyTree k)
    (r : ℝ) (u : Fin (butterflyWidth k) → ℝ) :
    (fun j => ∑ i, tree.matrix i j * (r • u) i) =
      r • (fun j => ∑ i, tree.matrix i j * u i) := by
  funext j
  simp [Pi.smul_apply, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- A `BB*` factor preserves addition. Source: Appendix
`def: kaleidoscope`, composition of two linear butterfly maps. -/
theorem BBStar.apply_add {k : ℕ} (factor : BBStar k)
    (u v : Fin (butterflyWidth k) → ℝ) :
    factor.apply (u + v) = factor.apply u + factor.apply v := by
  unfold BBStar.apply
  rw [butterflyTranspose_add, factor.left.apply_add]

/-- A `BB*` factor preserves scalar multiplication. Source: Appendix
`def: kaleidoscope`, composition of two linear butterfly maps. -/
theorem BBStar.apply_smul {k : ℕ} (factor : BBStar k)
    (r : ℝ) (u : Fin (butterflyWidth k) → ℝ) :
    factor.apply (r • u) = r • factor.apply u := by
  unfold BBStar.apply
  rw [butterflyTranspose_smul, factor.left.apply_smul]

/-- Products of `BB*` factors preserve addition. Source: Appendix
`def: kaleidoscope`, class `(BB*)^w`. -/
theorem Kaleidoscope.apply_add {k : ℕ} (K : Kaleidoscope k)
    (u v : Fin (butterflyWidth k) → ℝ) :
    K.apply (u + v) = K.apply u + K.apply v := by
  rcases K with ⟨factors⟩
  change factors.foldl (fun state factor => factor.apply state) (u + v) =
    factors.foldl (fun state factor => factor.apply state) u +
      factors.foldl (fun state factor => factor.apply state) v
  induction factors generalizing u v with
  | nil => rfl
  | cons factor rest ih =>
      simp only [List.foldl_cons]
      rw [factor.apply_add]
      exact ih (factor.apply u) (factor.apply v)

/-- Products of `BB*` factors preserve scalar multiplication. Source:
Appendix `def: kaleidoscope`, class `(BB*)^w`. -/
theorem Kaleidoscope.apply_smul {k : ℕ} (K : Kaleidoscope k)
    (r : ℝ) (u : Fin (butterflyWidth k) → ℝ) :
    K.apply (r • u) = r • K.apply u := by
  rcases K with ⟨factors⟩
  change factors.foldl (fun state factor => factor.apply state) (r • u) =
    r • factors.foldl (fun state factor => factor.apply state) u
  induction factors generalizing u with
  | nil => rfl
  | cons factor rest ih =>
      simp only [List.foldl_cons]
      rw [factor.apply_smul]
      exact ih (factor.apply u)

end Transformer.Zoology

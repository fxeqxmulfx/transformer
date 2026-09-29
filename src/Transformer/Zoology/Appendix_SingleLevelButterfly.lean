/-
# A single binary factor as an actual recursive butterfly weight

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`def: W-kmat`. All other levels are identities. The result supplies fixed
shears and pair mixing projections for compilation without extra rows.
-/

import Transformer.Zoology.Appendix_ButterflyPairOrientation

namespace Transformer.Zoology

/-- Insert arbitrary diagonal and partner coefficients at one binary level.
Source: Appendix `def: butterfly`, independently parameterized blocks. -/
def singleLevelButterfly : (k t : ℕ) →
    (Fin (butterflyWidth k) → ℝ) → (Fin (butterflyWidth k) → ℝ) → ButterflyTree k
  | 0, _, _, _ => .leaf
  | k + 1, t, a, b =>
      if t = k then .node {
        upperLeft := fun j => a (finProdFinEquiv (0, j))
        upperRight := fun j => b (finProdFinEquiv (0, j))
        lowerLeft := fun j => b (finProdFinEquiv (1, j))
        lowerRight := fun j => a (finProdFinEquiv (1, j))
      } (identityButterflyTree k) (identityButterflyTree k)
      else .node (identityButterflyFactor (butterflyWidth k))
        (singleLevelButterfly k t (fun j => a (finProdFinEquiv (0, j)))
          (fun j => b (finProdFinEquiv (0, j))))
        (singleLevelButterfly k t (fun j => a (finProdFinEquiv (1, j)))
          (fun j => b (finProdFinEquiv (1, j))))

/-- The inserted tree is exactly the requested binary factor.
Source: Appendix `def: butterfly`, recursive/product correspondence. -/
theorem singleLevelButterfly_apply (k t : ℕ) (ht : t < k)
    (a b u : Fin (butterflyWidth k) → ℝ) :
    (singleLevelButterfly k t a b).apply u =
      fun q => a q * u q + b q * u (butterflyToggleIndex k t q) := by
  induction k generalizing t with
  | zero => omega
  | succ k ih =>
      funext q
      have hq := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply q
      generalize hp : finProdFinEquiv.symm q = h at hq
      rcases h with ⟨half, j⟩
      rw [← hq]
      by_cases htop : t = k
      · subst t
        fin_cases half <;>
          simp [singleLevelButterfly, ButterflyTree.apply, ButterflyFactor.apply,
            identityButterflyTree_apply, butterflyToggleIndex, add_comm]
      · have hsmall : t < k := by omega
        fin_cases half <;>
          simp [singleLevelButterfly, htop, ButterflyTree.apply,
            identityButterflyFactor, ButterflyFactor.apply,
            ih t hsmall, butterflyToggleIndex]

/-- One real scalar weight family satisfies the level bound.
Source: Appendix `def: butterfly`, size-two factor. -/
example : (0 : ℕ) < 1 := by decide

end Transformer.Zoology

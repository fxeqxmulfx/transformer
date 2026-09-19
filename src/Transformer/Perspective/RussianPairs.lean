/-
# The Russian trick, with the cross terms

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix B.

`Perspective.russian_trick` produces a family of skew generators with
`Σ_k B_k² = -(d-1) I`, which is all Appendix A needs: `e:helpcl` is linear in
`B²`.  Appendix B sums `eq: dr1`, which is *quadratic* in `B` as well, so the
family must also compute

  `Σ_k ⟨B_k x, y⟩² = ‖x‖² ‖y‖² - ⟨x, y⟩²`,

the squared area of the parallelogram on `x` and `y`.  The cycle of rank-two
generators used there does not: in `d = 4` it misses the pairs `(e_0, e_2)`
and `(e_1, e_3)`, and the left-hand side then depends on the basis.

The family that does is the full set of elementary generators, indexed by
*all* ordered pairs `(p, q)`:

  `B_{pq} x = c (⟨e_p, x⟩ e_q - ⟨e_q, x⟩ e_p)`,   `c = 1/√2`.

The diagonal `p = q` contributes nothing, each off-diagonal pair is counted
twice, `B_{pq}²` is `-c²` times the projection onto `span{e_p, e_q}` — every
basis vector lying in `2(d-1)` of them — and `⟨B_{pq} x, y⟩` is `c` times the
`(p,q)` entry of `x ∧ y`, whose squares add up to Lagrange's identity.  The
normalisation `c² = 1/2` is what makes *both* constants come out right, and it
does not depend on `d`, so — unlike `russian_trick` — this needs no hypothesis
at all: at `d ≤ 1` both sides of both identities are `0`.
-/

import Transformer.Perspective.RussianTrick

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- **Lagrange's identity**, in the form the pair family needs:

  `Σ_p Σ_q (a_p b_q - a_q b_p)² = 2 ((Σ a²)(Σ b²) - (Σ ab)²)`. -/
theorem sum_sq_cross (a b : Idx d → ℝ) :
    ∑ p : Idx d, ∑ q : Idx d, (a p * b q - a q * b p) ^ 2
      = 2 * ((∑ p : Idx d, a p ^ 2) * (∑ p : Idx d, b p ^ 2)
          - (∑ p : Idx d, a p * b p) ^ 2) := by
  have h1 : (∑ p : Idx d, a p ^ 2) * ∑ q : Idx d, b q ^ 2
      = ∑ p : Idx d, ∑ q : Idx d, a p ^ 2 * b q ^ 2 := Fintype.sum_mul_sum _ _
  have h2 : (∑ p : Idx d, a p * b p) * ∑ q : Idx d, a q * b q
      = ∑ p : Idx d, ∑ q : Idx d, a p * b p * (a q * b q) := Fintype.sum_mul_sum _ _
  have hswap : ∑ p : Idx d, ∑ q : Idx d, b p ^ 2 * a q ^ 2
      = ∑ p : Idx d, ∑ q : Idx d, a p ^ 2 * b q ^ 2 := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => by ring
  have hrow : ∀ p : Idx d, ∑ q : Idx d, (a p * b q - a q * b p) ^ 2
      = ((∑ q : Idx d, a p ^ 2 * b q ^ 2) + ∑ q : Idx d, b p ^ 2 * a q ^ 2)
          - 2 * ∑ q : Idx d, a p * b p * (a q * b q) := by
    intro p
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun q _ => by ring
  rw [Finset.sum_congr rfl fun p (_ : p ∈ Finset.univ) => hrow p,
    Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hswap,
    pow_two (∑ p : Idx d, a p * b p), ← h1, ← h2]
  ring

/-- **The Russian trick over all pairs.**  There is a family of skew-symmetric
endomorphisms of `ℝ^d`, indexed by the ordered pairs of indices, with

  `Σ_k B_k² = -(d-1) I`    and    `Σ_k ⟨B_k x, y⟩² = ‖x‖² ‖y‖² - ⟨x, y⟩²`.

The first identity is `e:russiantrick` with the scalar cleared; the second is
what the *quadratic* term of `eq: dr1` is summed against, and it is the reason
this family is used in Appendix B rather than the `d`-element one of
`Perspective.russian_trick`.

Source: arXiv:2312.10794v5, Appendix A, `e:russiantrick`, and Appendix B,
where `eq: dr1` is summed over a family of skew directions. -/
theorem russian_trick_pairs (d : ℕ) :
    ∃ B : Idx d × Idx d → ParamMatrix d,
      (∀ k : Idx d × Idx d, IsSkew d (B k)) ∧
      (∀ x : EucSpace d, ∑ k : Idx d × Idx d, B k (B k x) = (-((d : ℝ) - 1)) • x) ∧
      (∀ x y : EucSpace d, ∑ k : Idx d × Idx d, inner (𝕜 := ℝ) (B k x) y ^ 2
          = ‖x‖ ^ 2 * ‖y‖ ^ 2 - inner (𝕜 := ℝ) x y ^ 2) := by
  classical
  set c : ℝ := Real.sqrt 2⁻¹ with hc
  have hc2 : c ^ 2 = 2⁻¹ := Real.sq_sqrt (by norm_num)
  set E : OrthonormalBasis (Idx d) ℝ (EucSpace d) := EuclideanSpace.basisFun (Idx d) ℝ with hE
  have hon : ∀ i j : Idx d, inner (𝕜 := ℝ) (E i) (E j) = if i = j then (1 : ℝ) else 0 :=
    orthonormal_iff_ite.1 E.orthonormal
  refine ⟨fun k => skewPair d c (E k.1) (E k.2), fun k => skewPair_isSkew _ _ _, ?_, ?_⟩
  · -- `Σ_{p,q} B_{pq}² = -(d-1) I`
    intro x
    set a : Idx d → ℝ := fun p => inner (𝕜 := ℝ) (E p) x with ha
    set u : Idx d → EucSpace d := fun p => (-c ^ 2) • (a p • E p) with hu
    have hU : ∑ p : Idx d, u p = (-c ^ 2) • x := by
      rw [hu, ← Finset.smul_sum]
      exact congrArg _ (E.sum_repr' x)
    have hconst : ∀ v : EucSpace d, ∑ _q : Idx d, v = (d : ℝ) • v := by
      intro v
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        ← Nat.cast_smul_eq_nsmul ℝ]
    have hdiag : ∀ p : Idx d,
        skewPair d c (E p) (E p) (skewPair d c (E p) (E p) x) = 0 := by
      intro p
      have h0 : skewPair d c (E p) (E p) x = 0 := by
        simp [skewPair_apply]
      rw [h0]
      simp [skewPair_apply]
    have hoff : ∀ p q : Idx d, p ≠ q →
        skewPair d c (E p) (E q) (skewPair d c (E p) (E q) x) = u p + u q := by
      intro p q hpq
      rw [skewPair_sq c (by rw [hon]; simp) (by rw [hon]; simp)
        (by rw [hon]; simp [hpq]) x, hu, ha]
      simp [smul_add]
    have hterm : ∀ p q : Idx d,
        skewPair d c (E p) (E q) (skewPair d c (E p) (E q) x)
          = (u p + u q) - (if p = q then u p + u q else 0) := by
      intro p q
      by_cases hpq : p = q
      · subst hpq; rw [hdiag p]; simp
      · rw [hoff p q hpq]; simp [hpq]
    have hrow : ∀ p : Idx d,
        ∑ q : Idx d, skewPair d c (E p) (E q) (skewPair d c (E p) (E q) x)
          = ((d : ℝ) • u p + (-c ^ 2) • x) - (u p + u p) := by
      intro p
      rw [Finset.sum_congr rfl fun q (_ : q ∈ Finset.univ) => hterm p q,
        Finset.sum_sub_distrib, Finset.sum_add_distrib, hconst (u p), hU]
      congr 1
      simp
    rw [Fintype.sum_prod_type,
      Finset.sum_congr rfl fun p (_ : p ∈ Finset.univ) => hrow p,
      Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
      ← Finset.smul_sum, hconst ((-c ^ 2) • x), hU, hc2]
    module
  · -- `Σ_{p,q} ⟨B_{pq} x, y⟩² = ‖x‖² ‖y‖² - ⟨x, y⟩²`
    intro x y
    set a : Idx d → ℝ := fun p => inner (𝕜 := ℝ) (E p) x with ha
    set b : Idx d → ℝ := fun p => inner (𝕜 := ℝ) (E p) y with hb
    have hinner : ∀ p q : Idx d,
        inner (𝕜 := ℝ) (skewPair d c (E p) (E q) x) y = c * (a p * b q - a q * b p) := by
      intro p q
      rw [skewPair_apply, real_inner_smul_left, inner_sub_left, real_inner_smul_left,
        real_inner_smul_left, ha, hb]
    have hax : ∑ p : Idx d, a p ^ 2 = ‖x‖ ^ 2 := by
      rw [← real_inner_self_eq_norm_sq, ← E.sum_inner_mul_inner x x]
      exact Finset.sum_congr rfl fun p _ => by
        simp only [ha, pow_two, real_inner_comm x (E p)]
    have hby : ∑ p : Idx d, b p ^ 2 = ‖y‖ ^ 2 := by
      rw [← real_inner_self_eq_norm_sq, ← E.sum_inner_mul_inner y y]
      exact Finset.sum_congr rfl fun p _ => by
        simp only [hb, pow_two, real_inner_comm y (E p)]
    have hab : ∑ p : Idx d, a p * b p = inner (𝕜 := ℝ) x y := by
      rw [← E.sum_inner_mul_inner x y]
      exact Finset.sum_congr rfl fun p _ => by
        simp only [ha, hb, real_inner_comm x (E p)]
    rw [Fintype.sum_prod_type]
    have hrow : ∀ p : Idx d,
        ∑ q : Idx d, inner (𝕜 := ℝ) (skewPair d c (E p) (E q) x) y ^ 2
          = c ^ 2 * ∑ q : Idx d, (a p * b q - a q * b p) ^ 2 := by
      intro p
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun q _ => by rw [hinner p q]; ring
    rw [Finset.sum_congr rfl fun p (_ : p ∈ Finset.univ) => hrow p,
      ← Finset.mul_sum, sum_sq_cross a b, hax, hby, hab, hc2]
    ring

/-- The pair family exists in every dimension; in `d = 0` both identities read
`0 = 0`. -/
example : ∃ B : Idx 2 × Idx 2 → ParamMatrix 2, ∀ k, IsSkew 2 (B k) :=
  ⟨(russian_trick_pairs 2).choose, (russian_trick_pairs 2).choose_spec.1⟩

end Perspective
end Transformer

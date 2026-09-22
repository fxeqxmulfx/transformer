/-
# Homogenized Transformers — the softmax is smooth with bounded derivatives

`claim:softmax_is_smooth` of arXiv:2604.01978v1, as stated in
`SoftmaxDerivatives.lean`: every `k`-th partial derivative of a softmax weight
`χ_j` is at most `C_k χ_j`, with `C_k` depending on `k` alone.

The proof runs the source's induction on monomials in the weights,
`χ_a = χ_{a_1} ⋯ χ_{a_p}`.  By `fderiv_softmaxWeight`,

  `∂χ_a/∂z_s = #{i : a_i = s} · χ_a - p · χ_a χ_s`,

a combination of two monomials, of degrees `p` and `p + 1`, with coefficients
at most `p`, both keeping the factor `χ_{a_1}`.  Since every weight lies in
`[0,1]`, `|∂^k χ_a| ≤ C(k, p) χ_{a_1}` with `C(0, p) = 1` and
`C(k+1, p) = p · C(k, p) + p · C(k, p+1)` — independent of the number of scores.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth`.
-/

import Transformer.Homogenized.SoftmaxDerivatives

open scoped BigOperators
open Real

namespace Transformer
namespace Homogenized

variable {m : ℕ}

/-- **The derivative of a monomial**, in the direction of the score `s`:
`∂χ_a/∂z_s = #{i : a_i = s} · χ_a - p · χ_{(a, s)}`.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth`, the induction step. -/
theorem fderiv_softmaxMonomial (hm : 0 < m) {p : ℕ} (a : Fin p → Idx m) (s : Idx m) :
    (fun y => fderiv ℝ (softmaxMonomial a) y (Pi.single s 1))
      = ((Finset.univ.filter fun i => a i = s).card : ℝ) • softmaxMonomial a
        - (p : ℝ) • softmaxMonomial (Fin.snoc a s) := by
  funext y
  have hd := HasFDerivAt.finsetProd (u := Finset.univ) (x := y)
    fun i _ => hasFDerivAt_softmaxWeight hm y (a i)
  rw [show softmaxMonomial a = fun v => ∏ i, Perspective.softmaxWeight v (a i) from rfl,
    hd.fderiv]
  have hi : ∀ i : Fin p, ((Perspective.softmaxWeight y (a i) •
      ((ContinuousLinearMap.proj (a i) : (Idx m → ℝ) →L[ℝ] ℝ) -
        ∑ k : Idx m, Perspective.softmaxWeight y k •
          (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ))) (Pi.single s 1))
      = (if s = a i then Perspective.softmaxWeight y (a i) else 0)
        - Perspective.softmaxWeight y (a i) * Perspective.softmaxWeight y s := by
    intro i
    rw [← (hasFDerivAt_softmaxWeight hm y (a i)).fderiv, fderiv_softmaxWeight hm]
  simp only [FunLike.coe_sum, Finset.sum_apply, smul_apply,
    smul_eq_mul, hi, mul_sub, Finset.sum_sub_distrib, Pi.sub_apply, Pi.smul_apply,
    softmaxMonomial, Fin.prod_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last]
  congr 1
  · rw [Finset.card_filter, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : a i = s
    · simp only [h, ite_true, Nat.cast_one, one_mul]
      rw [← h, Finset.prod_erase_mul _ _ (Finset.mem_univ i)]
    · simp [h, Ne.symm h]
  · rw [Finset.sum_congr rfl fun i _ => (by
        rw [← mul_assoc, Finset.prod_erase_mul _ _ (Finset.mem_univ i)] :
        _ = (∏ j, Perspective.softmaxWeight y (a j)) * Perspective.softmaxWeight y s),
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- The hypothesis of `fderiv_softmaxMonomial` is satisfiable. -/
example : 0 < 1 := one_pos

/-- The bound `C(k, p)` on the `k`-th partial derivatives of a monomial of
degree `p`: `C(0, p) = 1`, `C(k+1, p) = p · C(k, p) + p · C(k, p+1)`. -/
noncomputable def monomialBound : ℕ → ℕ → ℝ
  | 0, _ => 1
  | k + 1, p => p * monomialBound k p + p * monomialBound k (p + 1)

/-- The bound `C(k, p)` is nonnegative. -/
theorem monomialBound_nonneg : ∀ k p, 0 ≤ monomialBound k p
  | 0, _ => zero_le_one
  | k + 1, p => by
    have := monomialBound_nonneg k p
    have := monomialBound_nonneg k (p + 1)
    simp only [monomialBound]
    positivity

/-- **The partial derivatives of a monomial are bounded by its first factor**,
uniformly in the scores and in their number:
`|∂^k χ_a / ∂z_{r_1} ⋯ ∂z_{r_k}| ≤ C(k, p) · χ_{a_1}` for a monomial of degree
`p ≥ 1`.  Every monomial the induction produces keeps the factor `χ_{a_1}`.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth`. -/
theorem abs_iteratedFDeriv_softmaxMonomial_le (hm : 0 < m) :
    ∀ (k p : ℕ) (a : Fin (p + 1) → Idx m) (r : Fin k → Idx m) (u : Idx m → ℝ),
      |iteratedFDeriv ℝ k (softmaxMonomial a) u (fun i => Pi.single (r i) (1 : ℝ))|
        ≤ monomialBound k (p + 1) * Perspective.softmaxWeight u (a 0) := by
  intro k
  induction k with
  | zero =>
    intro p a r u
    rw [iteratedFDeriv_zero_apply, softmaxMonomial, Fin.prod_univ_succ, monomialBound, one_mul,
      abs_mul, abs_of_nonneg (Perspective.softmaxWeight_nonneg u (a 0))]
    exact mul_le_of_le_one_right (Perspective.softmaxWeight_nonneg u (a 0))
      (abs_softmaxMonomial_le_one hm (fun i => a i.succ) u)
  | succ k ih =>
    intro p a r u
    have hc := (contDiff_softmaxMonomial hm a).fderiv_right (m := (⊤ : ℕ∞)) le_rfl
    have hC : ∀ {q : ℕ} (b : Fin q → Idx m), ContDiff ℝ k (softmaxMonomial b) :=
      fun b => (contDiff_softmaxMonomial hm b).of_le (mod_cast le_top)
    rw [iteratedFDeriv_succ_apply_right,
      ← iteratedFDeriv_clm_apply_const_apply hc (mod_cast le_top), fderiv_softmaxMonomial hm,
      iteratedFDeriv_sub_apply
        ((hC a).const_smul ((Finset.univ.filter fun i => a i = r (Fin.last k)).card : ℝ)
          : ContDiff ℝ k (_ • softmaxMonomial a)).contDiffAt
        ((hC (Fin.snoc a (r (Fin.last k)))).const_smul ((p + 1 : ℕ) : ℝ)
          : ContDiff ℝ k (_ • softmaxMonomial _)).contDiffAt,
      iteratedFDeriv_const_smul_apply (hC a).contDiffAt,
      iteratedFDeriv_const_smul_apply (hC _).contDiffAt]
    simp only [sub_apply, smul_apply, smul_eq_mul]
    rw [show (Fin.init (α := fun _ => Idx m → ℝ) fun i => Pi.single (r i) (1 : ℝ))
      = fun i : Fin k => Pi.single (r (Fin.castSucc i)) (1 : ℝ) from rfl]
    set c := ((Finset.univ.filter fun i => a i = r (Fin.last k)).card : ℝ)
    have h1 : |iteratedFDeriv ℝ k (softmaxMonomial a) u
          (fun i : Fin k => Pi.single (r (Fin.castSucc i)) (1 : ℝ))|
        ≤ monomialBound k (p + 1) * Perspective.softmaxWeight u (a 0) := ih p a (Fin.init r) u
    have h0 : (Fin.snoc a (r (Fin.last k)) : Fin (p + 1 + 1) → Idx m) 0 = a 0 :=
      Fin.snoc_castSucc (α := fun _ => Idx m) (r (Fin.last k)) a 0
    have h2 : |iteratedFDeriv ℝ k (softmaxMonomial (Fin.snoc a (r (Fin.last k)))) u
          (fun i : Fin k => Pi.single (r (Fin.castSucc i)) (1 : ℝ))|
        ≤ monomialBound k (p + 1 + 1) * Perspective.softmaxWeight u (a 0) := by
      rw [← h0]; exact ih (p + 1) _ (Fin.init r) u
    generalize iteratedFDeriv ℝ k (softmaxMonomial a) u
      (fun i : Fin k => Pi.single (r (Fin.castSucc i)) (1 : ℝ))
      = X at h1 ⊢
    generalize iteratedFDeriv ℝ k (softmaxMonomial (Fin.snoc a (r (Fin.last k)))) u
      (fun i : Fin k => Pi.single (r (Fin.castSucc i)) (1 : ℝ)) = Y at h2 ⊢
    have hcard : c ≤ ((p + 1 : ℕ) : ℝ) := Nat.cast_le.mpr
      ((Finset.card_filter_le _ _).trans_eq (Finset.card_fin (p + 1)))
    have hc0 : 0 ≤ c := Nat.cast_nonneg _
    have hq : (0 : ℝ) ≤ ((p + 1 : ℕ) : ℝ) := Nat.cast_nonneg _
    have hχ := Perspective.softmaxWeight_nonneg u (a 0)
    calc |c * X - ((p + 1 : ℕ) : ℝ) * Y| ≤ c * |X| + ((p + 1 : ℕ) : ℝ) * |Y| := by
          rw [← abs_of_nonneg hc0, ← abs_of_nonneg hq, ← abs_mul, ← abs_mul,
            abs_of_nonneg hc0, abs_of_nonneg hq]
          exact abs_sub _ _
      _ ≤ ((p + 1 : ℕ) : ℝ) * (monomialBound k (p + 1) * Perspective.softmaxWeight u (a 0))
          + ((p + 1 : ℕ) : ℝ) * (monomialBound k (p + 1 + 1)
            * Perspective.softmaxWeight u (a 0)) := by
          have := monomialBound_nonneg k (p + 1)
          gcongr
      _ = monomialBound (k + 1) (p + 1) * Perspective.softmaxWeight u (a 0) := by
          simp only [monomialBound]; ring

/-- The hypothesis of `abs_iteratedFDeriv_softmaxMonomial_le` is satisfiable. -/
example : 0 < 1 := one_pos

/-- **Claim (claim:softmax_is_smooth).**  The softmax is `C^∞`, and for every
order `k` there is a constant `C_k` such that every `k`-th partial derivative of
every weight is at most `C_k` times that weight, at every vector of scores and
for every number of scores:

  `|∂^k χ_j / ∂z_{r_1} … ∂z_{r_k} (z)| ≤ C_k χ_j(z)`.

This is what the claim's displayed formula says once it is made a statement —
the derivative is `χ_j` times a polynomial in the weights with coefficients
depending on `k` alone, and the weights lie in `[0,1]` — and it is the form
`eq:pi_derivative_bound` uses, with the factor `χ_j`; see the docstring of
`SoftmaxDerivatives.lean`.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth`, `eq:pi_derivative_bound`. -/
theorem softmax_is_smooth :
    ∃ C : ℕ → ℝ, (∀ k : ℕ, 0 < C k) ∧
      ∀ (n : ℕ), 0 < n →
        (∀ j : Idx n,
          ContDiff ℝ (⊤ : ℕ∞) (fun v : Idx n → ℝ => Perspective.softmaxWeight v j)) ∧
        ∀ (k : ℕ) (j : Idx n) (r : Fin k → Idx n) (u : Idx n → ℝ),
          |iteratedFDeriv ℝ k (fun v : Idx n → ℝ => Perspective.softmaxWeight v j) u
            (fun i => Pi.single (r i) (1 : ℝ))| ≤ C k * Perspective.softmaxWeight u j := by
  refine ⟨fun k => monomialBound k 1 + 1,
    fun k => by have := monomialBound_nonneg k 1; positivity,
    fun n hn => ⟨contDiff_softmaxWeight hn, fun k j r u => ?_⟩⟩
  have h : (fun v : Idx n → ℝ => Perspective.softmaxWeight v j)
      = softmaxMonomial (fun _ : Fin (0 + 1) => j) := by
    funext v; simp [softmaxMonomial]
  rw [h]
  refine (abs_iteratedFDeriv_softmaxMonomial_le hn k 0 _ r u).trans ?_
  have := Perspective.softmaxWeight_nonneg u j
  dsimp only
  gcongr
  linarith

/-- The quantifiers of `softmax_is_smooth` are not empty: there is a positive
number of scores. -/
example : 0 < 1 := one_pos

end Homogenized
end Transformer

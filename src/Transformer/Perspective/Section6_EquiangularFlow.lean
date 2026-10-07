/-
# §6.2 — Common-angle solutions of the Gram-matrix ODE

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
`thm: orthogonal` and `eq: ybeta`. The calculation retains the full softmax
partition function and the zero contribution of the self-attention term.
-/

import Transformer.Perspective.Section6_AngleFlow
import Transformer.Perspective.Section6_GramFlow

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

/-- A unit-diagonal matrix with common off-diagonal value `c`.

Source: arXiv:2312.10794v5, §6.2, the common angle in `thm: orthogonal`. -/
def equiGram (n : ℕ) (c : ℝ) : Idx n → Idx n → ℝ :=
  fun i j => if i = j then 1 else c

/-- A row with one diagonal entry `a` and `n-1` equal entries `b` sums
to `a + (n-1)b`.

Source: arXiv:2312.10794v5, §6.2, the partition function in `eq: ybeta`. -/
theorem sum_diagonal_const (n : ℕ) (i : Idx n) (a b : ℝ) :
    (∑ k : Idx n, if i = k then a else b) = a + (n - 1 : ℝ) * b := by
  have heq (k : Idx n) : (if i = k then a else b) = b + (if i = k then a - b else 0) := by
    by_cases h : i = k <;> simp [h]
  simp_rw [heq]
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, Fintype.sum_ite_eq]
  ring

/-- The common-angle matrix is symmetric.

Source: arXiv:2312.10794v5, §6.2, `thm: orthogonal`. -/
theorem equiGram_symm (n : ℕ) (c : ℝ) (i j : Idx n) : equiGram n c i j = equiGram n c j i := by
  simp only [equiGram, eq_comm]

/-- The common-angle row sum from the proof of `thm: orthogonal`,
arXiv:2312.10794v5, §6.2. -/
theorem sum_equiGram (n : ℕ) (c : ℝ) (i : Idx n) :
    (∑ k : Idx n, equiGram n c i k) = 1 + (n - 1 : ℝ) * c :=
  sum_diagonal_const n i 1 c

/-- The same sum holds for each column.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. -/
theorem sum_equiGram_column (n : ℕ) (c : ℝ) (j : Idx n) :
    (∑ k : Idx n, equiGram n c k j) = 1 + (n - 1 : ℝ) * c := by
  simp_rw [equiGram_symm n c]
  exact sum_equiGram n c j

/-- The common-angle partition function is `e^β + (n-1)e^(βc)`.

Source: arXiv:2312.10794v5, §6.2, denominator of `eq: ybeta`. -/
theorem gramDen_equiGram (n : ℕ) (β c : ℝ) (i : Idx n) :
    (∑ k : Idx n, exp (β * equiGram n c i k)) = exp β + (n - 1 : ℝ) * exp (β * c) := by
  have heq (k : Idx n) : exp (β * equiGram n c i k) =
      if i = k then exp β else exp (β * c) := by
    by_cases h : i = k <;> simp [equiGram, h]
  simp_rw [heq]
  exact sum_diagonal_const n i (exp β) (exp (β * c))

/-- One half of the common-angle drift. The diagonal weight can be
replaced by the off-diagonal weight only because its projected term is zero.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
theorem gramHalfDrift_equiGram (n : ℕ) (β c : ℝ) (i j : Idx n) (hij : i ≠ j) :
    gramHalfDrift n β (equiGram n c) i j =
      exp (β * c) / (exp β + (n - 1 : ℝ) * exp (β * c)) * (1 - c) * ((n - 1 : ℝ) * c + 1) := by
  unfold gramHalfDrift
  have heq (k : Idx n) : gramWeight n β (equiGram n c) i k *
      (equiGram n c k j - equiGram n c i k * equiGram n c i j) =
      exp (β * c) / (exp β + (n - 1 : ℝ) * exp (β * c)) *
      (equiGram n c k j - equiGram n c i k * equiGram n c i j) := by
    by_cases h : i = k
    · subst k
      simp [equiGram, hij]
    · rw [gramWeight, gramDen_equiGram]
      simp [equiGram, h]
  simp_rw [heq]
  rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul,
    sum_equiGram_column, sum_equiGram]
  have hc : equiGram n c i j = c := by simp [equiGram, hij]
  rw [hc]
  ring

/-- Off the diagonal, the matrix ODE is exactly the scalar angle ODE.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. -/
theorem gramDrift_equiGram_off (n : ℕ) (β c : ℝ) (i j : Idx n) (hij : i ≠ j) :
    gramDrift n β (equiGram n c) i j = saAngleDrift n β c := by
  rw [gramDrift, gramHalfDrift_equiGram n β c i j hij,
    gramHalfDrift_equiGram n β c j i hij.symm]
  unfold saAngleDrift
  ring

/-- The unit diagonal remains fixed under the Gram-matrix ODE.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. -/
theorem gramDrift_equiGram_self (n : ℕ) (β c : ℝ) (i : Idx n) :
    gramDrift n β (equiGram n c) i i = 0 := by
  simp [gramDrift, gramHalfDrift, equiGram, eq_comm]

/-- A scalar angle solution gives an actual solution of the matrix ODE.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
theorem hasDerivAt_equiGram (n : ℕ) (β : ℝ) (γ : ℝ → ℝ)
    (hγ : ∀ t, HasDerivAt γ (saAngleDrift n β (γ t)) t) (t : ℝ) :
    HasDerivAt (fun s => equiGram n (γ s)) (gramDrift n β (equiGram n (γ t))) t := by
  refine hasDerivAt_pi.2 fun i => hasDerivAt_pi.2 fun j => ?_
  by_cases hij : i = j
  · subst j
    simpa only [equiGram, ite_eq_left, gramDrift_equiGram_self] using hasDerivAt_const t (1 : ℝ)
  · simpa [equiGram, hij, gramDrift_equiGram_off n β (γ t) i j hij]
      using hγ t

/-- Distinct indices exist for the two off-diagonal drift statements. -/
example : (0 : Idx 2) ≠ 1 := by norm_num

/-- The scalar hypothesis of the matrix solution lemma is satisfiable by
consensus at two particles and positive temperature. -/
example : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) (saAngleDrift 2 1 1) t := by
  intro t
  simpa only [saAngleDrift_one] using hasDerivAt_const t (1 : ℝ)

/-- An orthogonal sphere configuration has the identity Gram matrix.

Source: arXiv:2312.10794v5, §6.2, initial condition in `thm: orthogonal`. -/
theorem tokenGram_orthogonal (d n : ℕ) (X₀ : SphereTuple d n)
    (h_ortho : ∀ i j : Idx n, i ≠ j → inner ℝ (X₀ i : EucSpace d) (X₀ j : EucSpace d) = 0) :
    tokenGram X₀ = equiGram n 0 := by
  funext i j
  by_cases h : i = j
  · subst j
    simp only [tokenGram, equiGram, ite_eq_left, real_inner_self_eq_norm_sq,
      norm_coe_tuple, one_pow]
  · simpa [tokenGram, equiGram, h] using h_ortho i j h

/-- Uniqueness identifies an attention Gram matrix with a common-angle
solution having the same initial matrix. No angle symmetry is assumed of
the attention trajectory.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. This follows
by comparing the closed Gram equations rather than permuting a basis. -/
theorem tokenGram_eq_equiGram (d n : ℕ) (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : SA d n β X) (γ : ℝ → ℝ)
    (hγ : ∀ t, HasDerivAt γ (saAngleDrift n β (γ t)) t)
    (h0 : tokenGram (X 0) = equiGram n (γ 0)) :
    ∀ t, tokenGram (X t) = equiGram n (γ t) := by
  have heq := Transformer.GlobalFlow.eq_of_hasDerivAt
    (contDiff_gramDrift n β).locallyLipschitz
    (fun t => hasDerivAt_tokenGram d n β X hX t)
    (fun t => hasDerivAt_equiGram n β γ hγ t) h0
  exact fun t => congrFun heq t

/-- The orthogonal initial-matrix hypothesis is witnessed by the standard
basis of the plane, which has two distinct tokens. -/
example : ∀ i j : Idx 2, i ≠ j →
    inner ℝ (sphereBasisConfig 2 i : EucSpace 2) (sphereBasisConfig 2 j : EucSpace 2) = 0 :=
  sphereBasisConfig_orthogonal 2

/-- Both ODE hypotheses and their common initial matrix have actual
witnesses: the two-token consensus and its constant angle one. -/
example : ∃ (X : ℝ → SphereTuple 2 2) (γ : ℝ → ℝ), SA 2 2 1 X ∧
    (∀ t, HasDerivAt γ (saAngleDrift 2 1 (γ t)) t) ∧
    tokenGram (X 0) = equiGram 2 (γ 0) := by
  refine ⟨fun _ _ => basePoint 1, fun _ => 1,
    SA_const_consensus 2 2 (by norm_num) 1 (basePoint 1), ?_, ?_⟩
  · intro t
    simpa only [saAngleDrift_one] using hasDerivAt_const t (1 : ℝ)
  · funext i j
    simp [tokenGram_consensus, equiGram]

end Perspective
end Transformer

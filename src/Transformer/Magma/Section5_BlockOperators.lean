/-
# Positive damping on an orthogonal block decomposition

Formalization of arXiv:2602.15322v1, Section 5, S_t = s_t tensor I.
The block embeddings are linear, sum to the identity, and have mutually
orthogonal images on every input. These are genuine properties of a
disjoint parameter partition, not assumptions about descent.
-/

import Transformer.Magma.Section3_DampingBounds
import Transformer.Magma.Section5_MaskDescent

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {ι E : Type*} [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Linear blocks sum to the identity and are pointwise orthogonal.
Source: arXiv:2602.15322v1, Section 5, disjoint coordinate blocks U_b. -/
def OrthogonalBlocks (P : ι → E →L[ℝ] E) : Prop :=
  (∀ u, ∑ j, P j u = u) ∧ ∀ u j k, j ≠ k → ⟪P j u, P k u⟫_ℝ = 0

/-- The actual block-diagonal damping operator.
Source: arXiv:2602.15322v1, Section 5, S_t = s_t tensor I. -/
def blockOperator (P : ι → E →L[ℝ] E) (scale : ι → ℝ) : E →L[ℝ] E :=
  ∑ j, scale j • P j

omit [DecidableEq ι] in
/-- Inner products with the full input equal each block's energy.
Source: arXiv:2602.15322v1, Section 5, orthogonal block partition. -/
theorem block_inner_self (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P) (u : E) (j : ι) :
    ⟪P j u, u⟫_ℝ = ‖P j u‖ ^ 2 := by
  calc
    _ = ⟪P j u, ∑ k, P k u⟫_ℝ := by rw [hP.1 u]
    _ = _ := by
      rw [inner_sum, Finset.sum_eq_single j]
      · exact real_inner_self_eq_norm_sq _
      · intro k hk hkj
        exact hP.2 u j k (Ne.symm hkj)
      · simp

/-- A nonzero identity block satisfies the decomposition hypotheses.
Source: arXiv:2602.15322v1, Section 5, one-block case. -/
example : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) := by
  constructor
  · simp
  · intro u j k h
    exact (h (Subsingleton.elim j k)).elim

omit [DecidableEq ι] in
/-- Damped directions remain orthogonal. Source:
arXiv:2602.15322v1, Section 5, block-diagonal S_t. -/
theorem blockOperator_orthogonal (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (scale : ι → ℝ) (u : E) (j k : ι) (hjk : j ≠ k) :
    ⟪scale j • P j u, scale k • P k u⟫_ℝ = 0 := by
  simp [real_inner_smul_left, real_inner_smul_right, hP.2 u j k hjk]

/-- Orthogonality and distinctness are jointly satisfiable with a nonzero
scalar block and an empty block. Source: arXiv:2602.15322v1, Section 5,
the algebra of orthogonal blocks; nonempty blocks are not a hypothesis. -/
example :
    let P : Fin 2 → ℝ →L[ℝ] ℝ := fun j =>
      if j = 0 then ContinuousLinearMap.id ℝ ℝ else 0
    OrthogonalBlocks P ∧ (0 : Fin 2) ≠ 1 := by
  dsimp
  constructor
  · constructor
    · intro u
      simp [Fin.sum_univ_two]
    · intro u j k hjk
      fin_cases j <;> fin_cases k <;> simp_all
  · decide

omit [DecidableEq ι] in
/-- Uniform positive block scales give the exact operator hypotheses
used by the corrected convergence theorem. Source:
arXiv:2602.15322v1, Section 5, corrected use of block-diagonal S_t. -/
theorem blockOperator_bounds (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (scale : ι → ℝ) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hs : ∀ j, scale j ∈ Set.Icc a b) : DampingBounds a b (blockOperator P scale) := by
  intro u
  have henergy : ∑ j, ‖P j u‖ ^ 2 = ‖u‖ ^ 2 := by
    rw [← orthogonal_sum_norm_sq (fun j => P j u) (hP.2 u), hP.1 u]
  constructor
  · have hi : ⟪blockOperator P scale u, u⟫_ℝ = ∑ j, scale j * ‖P j u‖ ^ 2 := by
      simp [blockOperator, sum_apply, sum_inner, real_inner_smul_left,
        block_inner_self P hP]
    rw [hi, ← henergy, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro j hj
    exact mul_le_mul_of_nonneg_right (hs j).1 (sq_nonneg ‖P j u‖)
  · have hn : ‖blockOperator P scale u‖ ^ 2 = ∑ j, scale j ^ 2 * ‖P j u‖ ^ 2 := by
      have ho := fun j k hjk => blockOperator_orthogonal P hP scale u j k hjk
      have h := orthogonal_sum_norm_sq (fun j => scale j • P j u) ho
      simpa [blockOperator, sum_apply, norm_smul, mul_pow, Real.norm_eq_abs] using h
    have hsq : ‖blockOperator P scale u‖ ^ 2 ≤ (b * ‖u‖) ^ 2 := by
      rw [hn, mul_pow, ← henergy, Finset.mul_sum]
      apply Finset.sum_le_sum
      intro j hj
      have hs0 : 0 ≤ scale j := ha.trans (hs j).1
      exact mul_le_mul_of_nonneg_right ((sq_le_sq₀ hs0 hb).mpr (hs j).2)
        (sq_nonneg ‖P j u‖)
    exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hb (norm_nonneg _))).mp hsq

/-- Operator-bound hypotheses hold jointly on a nonzero identity block
with scale 1/2. Source: arXiv:2602.15322v1, Section 5. -/
example : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ _ : Unit, (1 / 2 : ℝ) ∈ Set.Icc (1 / 4) 1) := by
  refine ⟨⟨by simp, ?_⟩, by norm_num, by norm_num, by norm_num⟩
  intro u j k h
  exact (h (Subsingleton.elim j k)).elim

omit [DecidableEq ι] in
/-- The paper's actual cosine/sigmoid EMA yields uniformly bounded
block damping under a fixed positive temperature and invariant initial
scales. Source: arXiv:2602.15322v1, Sections 3 and 5; corrected Appendix A.3. -/
theorem magma_block_operator_bounds (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (τ : ℝ) (hτ : 0 < τ) (previous : ι → ℝ) (momentum g : ι → E)
    (hprev : ∀ j, previous j ∈ Set.Icc (Real.sigmoid (-1 / τ)) 1) :
    DampingBounds (Real.sigmoid (-1 / τ)) 1
      (blockOperator P (fun j => damping τ (previous j) (momentum j) (g j))) := by
  apply blockOperator_bounds P hP _ _ _ (Real.sigmoid_nonneg _) (by norm_num)
  exact fun j => damping_uniform_bounds τ (previous j) hτ (momentum j) (g j) (hprev j)

/-- The actual-EMA operator conditions have a nonzero one-block model.
Source: arXiv:2602.15322v1, Sections 3 and 5. -/
example : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) ∧
    (0 : ℝ) < 1 ∧
    (∀ _ : Unit, (1 / 2 : ℝ) ∈ Set.Icc (Real.sigmoid (-1 / 1)) 1) := by
  refine ⟨⟨by simp, ?_⟩, by norm_num, ?_⟩
  · intro u j k h
    exact (h (Subsingleton.elim j k)).elim
  · intro j
    constructor
    · simpa using Real.sigmoid_le (by norm_num : (-1 : ℝ) ≤ 0)
    · norm_num

end Transformer.Magma

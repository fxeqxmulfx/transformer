module

public import Transformer.Residual.Basic
public import Mathlib.Analysis.Convex.Birkhoff

/-!
# mHC: the ideal residual map and its depth composition

arXiv:2512.24880v2, §4.1, and Eq. (4). Double stochasticity is an
explicit hypothesis. Finite Sinkhorn iterates are defined separately.
These statements concern the residual map with coefficients fixed at a
token, not the Jacobian of the entire input-dependent Transformer block.
-/

@[expose] public section

open scoped BigOperators Matrix.Norms.L2Operator
open Transformer.Residual

namespace Transformer.MHC

variable {ι : Type*} [Fintype ι] [DecidableEq ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Chronological residual product: later maps multiply on the left.
Source: arXiv:2512.24880v2, Eq. (4), with starting depth zero. -/
noncomputable def skipProduct (H : ℕ → Matrix ι ι ℝ) : ℕ → Matrix ι ι ℝ
  | 0 => 1
  | t + 1 => H t * skipProduct H t

/-- Compositional closure, arXiv:2512.24880v2, §4.1, property 2.
The source's arbitrarily deep skip product remains doubly stochastic. -/
theorem skipProduct_stochastic (H : ℕ → Matrix ι ι ℝ)
    (hH : ∀ t, H t ∈ doublyStochastic ℝ ι) (t : ℕ) :
    skipProduct H t ∈ doublyStochastic ℝ ι := by
  induction t with
  | zero => exact (doublyStochastic ℝ ι).one_mem
  | succ t ih => exact (doublyStochastic ℝ ι).mul_mem (hH t) ih

/-- The product hypothesis is realized by the four-stream identity path. -/
example : ∀ t : ℕ,
    (fun _ => (1 : Matrix (Fin 4) (Fin 4) ℝ)) t ∈ doublyStochastic ℝ (Fin 4) := by
  intro t
  exact (doublyStochastic ℝ (Fin 4)).one_mem

/-- Spectral nonexpansion, arXiv:2512.24880v2, §4.1, property 1.
The source's heading is “Norm Preservation”; the displayed inequality
means nonexpansion, and does not imply that every vector's norm is kept. -/
theorem skipProduct_spectral_norm_le (H : ℕ → Matrix ι ι ℝ)
    (hH : ∀ t, H t ∈ doublyStochastic ℝ ι) (t : ℕ) :
    ‖skipProduct H t‖ ≤ 1 := by
  apply Matrix.l2_opNorm_le_one_of_mem_doublyStochastic
  exact skipProduct_stochastic H hH t

/-- The spectral theorem's hypothesis is realized at every depth. -/
example : ∀ t : ℕ,
    (fun _ => (1 : Matrix (Fin 4) (Fin 4) ℝ)) t ∈ doublyStochastic ℝ (Fin 4) := by
  intro t
  exact (doublyStochastic ℝ (Fin 4)).one_mem

/-- The identical-stream mode survives every depth product.
Source: arXiv:2512.24880v2, §4.1, conservation interpretation. -/
theorem skipProduct_preserves_constant (H : ℕ → Matrix ι ι ℝ)
    (hH : ∀ t, H t ∈ doublyStochastic ℝ ι) (t : ℕ) (v : E) :
    mix (skipProduct H t) (fun _ => v) = fun _ => v := by
  apply mix_const
  exact sum_row_of_mem_doublyStochastic (skipProduct_stochastic H hH t)

/-- A concrete nonzero constant mode meets the preservation hypothesis. -/
example : ∀ t : ℕ,
    (fun _ => (1 : Matrix (Fin 4) (Fin 4) ℝ)) t ∈ doublyStochastic ℝ (Fin 4) := by
  intro t
  exact (doublyStochastic ℝ (Fin 4)).one_mem

/-- Column normalization conserves the feature sum, and hence the feature
mean for a nonempty fixed stream set. Source: arXiv:2512.24880v2, §1
and §4.1. This conservation applies to the skip map, before write-back. -/
theorem mix_preserves_sum (H : Matrix ι ι ℝ)
    (hH : H ∈ doublyStochastic ℝ ι) (x : ι → E) :
    ∑ i, mix H x i = ∑ j, x j := by
  unfold mix weightedSum
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [← Finset.sum_smul, sum_col_of_mem_doublyStochastic hH, one_smul]

/-- The conservation hypothesis is satisfied by the four-stream identity. -/
example : (1 : Matrix (Fin 4) (Fin 4) ℝ) ∈ doublyStochastic ℝ (Fin 4) :=
  (doublyStochastic ℝ (Fin 4)).one_mem

/-- Mean conservation applies to the full skip product at every depth, on
vector-valued features as well as scalar coordinates. Each product acts
by the `mix_mul` composition law. Source: arXiv:2512.24880v2, §4.1,
compositional closure and the conservation statement in §1. -/
theorem skipProduct_preserves_sum (H : ℕ → Matrix ι ι ℝ)
    (hH : ∀ t, H t ∈ doublyStochastic ℝ ι) (t : ℕ) (x : ι → E) :
    ∑ i, mix (skipProduct H t) x i = ∑ i, x i := by
  apply mix_preserves_sum
  exact skipProduct_stochastic H hH t

/-- The arbitrary-depth conservation hypothesis has a concrete witness. -/
example : ∀ t : ℕ,
    (fun _ => (1 : Matrix (Fin 4) (Fin 4) ℝ)) t ∈ doublyStochastic ℝ (Fin 4) := by
  intro t
  exact (doublyStochastic ℝ (Fin 4)).one_mem

/-- Each output stream stays inside any ball containing all input streams.
Source: arXiv:2512.24880v2, §1, convex-combination interpretation. -/
theorem mix_norm_le (H : Matrix ι ι ℝ) (hH : H ∈ doublyStochastic ℝ ι)
    (x : ι → E) (R : ℝ) (hx : ∀ j, ‖x j‖ ≤ R) (i : ι) :
    ‖mix H x i‖ ≤ R := by
  apply weightedSum_convex_norm_le
  · exact fun j => nonneg_of_mem_doublyStochastic hH
  · exact sum_row_of_mem_doublyStochastic hH i
  · exact hx

/-- The norm hypotheses hold for identity mixing and four unit features. -/
example : (1 : Matrix (Fin 4) (Fin 4) ℝ) ∈ doublyStochastic ℝ (Fin 4) ∧
    ∀ j : Fin 4, ‖(fun _ : Fin 4 => (1 : ℝ)) j‖ ≤ 1 := by
  exact ⟨(doublyStochastic ℝ (Fin 4)).one_mem, by norm_num⟩

/-- The single-stream skip coefficient is exactly one. Input and output
gates still act on the branch; double stochasticity alone does not set
them to one. Source: arXiv:2512.24880v2, §4.1, `n = 1`. -/
theorem single_stream_update (H : Matrix (Fin 1) (Fin 1) ℝ)
    (hH : H ∈ doublyStochastic ℝ (Fin 1)) (a c : Fin 1 → ℝ)
    (F : E → E) (x : Fin 1 → E) :
    hyperStep a c H F x 0 = x 0 + c 0 • F (a 0 • x 0) := by
  have h00 : H 0 0 = 1 := by
    simpa using sum_row_of_mem_doublyStochastic hH 0
  simp [hyperStep, mix, weightedSum, h00]

/-- The single-stream hypothesis holds for the scalar identity. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ) ∈ doublyStochastic ℝ (Fin 1) :=
  (doublyStochastic ℝ (Fin 1)).one_mem

/-- Counterexample to reading §4.1's “Norm Preservation” as no vanishing:
a strictly positive doubly stochastic map can erase a nonzero stream
disagreement. The proven source claim is the upper norm bound above. -/
theorem stochastic_can_erase_disagreement :
    ∃ (H : Matrix (Fin 2) (Fin 2) ℝ) (x : Fin 2 → ℝ),
      H ∈ doublyStochastic ℝ (Fin 2) ∧ (∀ i j, 0 < H i j) ∧
      x ≠ 0 ∧ mix H x = 0 := by
  refine ⟨Matrix.of (fun _ _ => 1 / 2), ![1, -1], ?_, ?_, ?_, ?_⟩
  · rw [mem_doublyStochastic_iff_sum]
    norm_num [Fin.sum_univ_two]
  · norm_num
  · intro h
    have := congrFun h 0
    norm_num at this
  · funext i
    norm_num [mix, weightedSum, Fin.sum_univ_two]

end Transformer.MHC

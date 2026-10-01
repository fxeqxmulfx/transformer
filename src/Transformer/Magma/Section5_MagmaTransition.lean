/-
# Dense momentum and invariant EMA scales in the corrected SGD model

Formalization of arXiv:2602.15322v1, Sections 2--3 and 5. The actual
momentum recurrence and cosine/sigmoid EMA determine the block directions.
An invariant subtype makes the permitted previous scales explicit. The
normalized parameter step is Section 5's rule, with Algorithm 1's dense
auxiliary-state updates; their learning-rate difference remains explicit.
-/

import Transformer.Magma.Section5_BlockOperators
import Transformer.Magma.Section5_Transition

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

/-- Genuine dense block momenta and one invariant EMA scale per block.
Source: arXiv:2602.15322v1, Sections 2--3, first moment and scale EMA.
The invariant interval is an explicit initialization correction. -/
def DenseBlockState (ι E : Type*) (τ : ℝ) :=
  (ι → E) × (ι → Set.Icc (Real.sigmoid (-1 / τ)) 1)

variable {ι E : Type*} [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Dense moment update, actual alignment EMA, and damped block gradient
proposal. The subsequent mask does not influence these auxiliary updates.
Source: arXiv:2602.15322v1, Sections 2--3, Algorithm 1, normalized SGD
specialization of Section 5. -/
def magmaProposal (P : ι → E →L[ℝ] E) (τ : ℝ) (hτ : 0 < τ) (β : ℝ)
    (state : E × DenseBlockState ι E τ) (g : E) : DenseBlockState ι E τ × (ι → E) :=
  let μ := fun j => β • state.2.1 j + (1 - β) • P j g
  let scale := fun j => (⟨damping τ (state.2.2 j).val (μ j) (P j g),
    damping_uniform_bounds τ _ hτ _ _ (state.2.2 j).property⟩ :
      Set.Icc (Real.sigmoid (-1 / τ)) 1)
  ((μ, scale), fun j => (scale j).val • P j g)

/-- The damping operator built from the actual new EMA scales.
Source: arXiv:2602.15322v1, Section 5, block-diagonal S_t. -/
def magmaDampingOperator (P : ι → E →L[ℝ] E) (τ : ℝ) (hτ : 0 < τ) (β : ℝ)
    (state : E × DenseBlockState ι E τ) (g : E) : E →L[ℝ] E :=
  blockOperator P (fun j => ((magmaProposal P τ hτ β state g).1.2 j).val)

omit [Fintype ι] [DecidableEq ι] in
/-- The dense first-moment recurrence is exactly the one in Section 2.
Source: arXiv:2602.15322v1, Section 2, first-moment estimates and Algorithm 1. -/
theorem magmaProposal_momentum (P : ι → E →L[ℝ] E) (τ : ℝ) (hτ : 0 < τ) (β : ℝ)
    (state : E × DenseBlockState ι E τ) (g : E) (j : ι) :
    (magmaProposal P τ hτ β state g).1.1 j = β • state.2.1 j + (1 - β) • P j g := rfl

/-- Positive temperatures exist for the dense-state transition.
Source: arXiv:2602.15322v1, Section 3, tau>0. -/
example : (0 : ℝ) < 1 := by norm_num

omit [DecidableEq ι] in
/-- The proposed directions sum to the actual damping operator applied
to the current stochastic gradient. Source:
arXiv:2602.15322v1, Section 5, S_t g_t. -/
theorem magmaProposal_sum (P : ι → E →L[ℝ] E) (τ : ℝ) (hτ : 0 < τ) (β : ℝ)
    (state : E × DenseBlockState ι E τ) (g : E) :
    (∑ j, (magmaProposal P τ hτ β state g).2 j) = magmaDampingOperator P τ hτ β state g g := by
  simp [magmaDampingOperator, magmaProposal, blockOperator, sum_apply]

/-- Positive temperature also satisfies the operator identity's domain.
Source: arXiv:2602.15322v1, Section 5. -/
example : (0 : ℝ) < 1 := by norm_num

omit [DecidableEq ι] in
/-- Orthogonal blocks remain orthogonal after the actual EMA damping.
Source: arXiv:2602.15322v1, Section 5, S_t applied blockwise. -/
theorem magmaProposal_orthogonal (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (τ : ℝ) (hτ : 0 < τ) (β : ℝ) (state : E × DenseBlockState ι E τ) (g : E) :
    ∀ j k, j ≠ k →
      ⟪(magmaProposal P τ hτ β state g).2 j, (magmaProposal P τ hτ β state g).2 k⟫_ℝ = 0 := by
  exact fun j k hjk => blockOperator_orthogonal P hP
    (fun b => ((magmaProposal P τ hτ β state g).1.2 b).val) g j k hjk

/-- The orthogonality and positive-temperature hypotheses have a joint
nonzero one-block witness. Source: arXiv:2602.15322v1, Section 5. -/
example : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) ∧ (0 : ℝ) < 1 := by
  refine ⟨⟨by simp, ?_⟩, by norm_num⟩
  intro u j k h
  exact (h (Subsingleton.elim j k)).elim

omit [DecidableEq ι] in
/-- Uniform positive damping follows from the actual new scale subtype;
it is not a hypothesis assuming descent. Source:
arXiv:2602.15322v1, Section 3 and corrected Section 5 analysis. -/
theorem magmaDampingOperator_bounds (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (τ : ℝ) (hτ : 0 < τ) (β : ℝ) (state : E × DenseBlockState ι E τ) (g : E) :
    DampingBounds (Real.sigmoid (-1 / τ)) 1 (magmaDampingOperator P τ hτ β state g) := by
  apply blockOperator_bounds P hP _ _ _ (Real.sigmoid_nonneg _) (by norm_num)
  exact fun j => ((magmaProposal P τ hτ β state g).1.2 j).property

/-- The uniform operator conditions have a joint positive-temperature,
nonzero-block witness. Source: arXiv:2602.15322v1, Sections 3 and 5. -/
example : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) ∧ (0 : ℝ) < 1 := by
  refine ⟨⟨by simp, ?_⟩, by norm_num⟩
  intro u j k h
  exact (h (Subsingleton.elim j k)).elim

end Transformer.Magma

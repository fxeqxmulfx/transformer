module

public import Transformer.SequenceModel.Basic
public import Transformer.Landscape.Neural.Tangent.Basic
public import Transformer.ScheduleFree.AdamW.Invariants

/-!
# The actual preconditioned prediction-transfer operator

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T1--T2.
The operator is J_eval P J_train^T at the actual evaluation/query weights.
It need not be symmetric across splits. Only the same-Jacobian operator is
automatically positive. No fixed task readout or nuclear-norm objective is
substituted for the network's loss.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Transfer

open scoped BigOperators RealInnerProductSpace
open ScheduleFree

variable {I A B : Type*} [Fintype I] [Fintype A] [Fintype B]

/-- A diagonal parameter-space map in the actual optimizer coordinates. -/
def diagonal (a : I → ℝ) : Weights I →L[ℝ] Weights I :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 (fun i => a i * x i)
    map_add' := by intro x y; ext i; simp; ring
    map_smul' := by intro c x; ext i; simp; ring }

/-- T1: diagonal action uses every scalar trainable coordinate. -/
theorem diagonal_apply (a : I → ℝ) (x : Weights I) (i : I) :
    diagonal a x i = a i * x i := rfl

/-- T1: the diagonal quadratic form retains the actual coordinate weights. -/
theorem inner_diagonal (a : I → ℝ) (x : Weights I) :
    ⟪x, diagonal a x⟫ = ∑ i, a i * x i ^ 2 := by
  simp only [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Pi.star_apply,
    star_trivial, diagonal_apply]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- Adam's real, history-dependent preconditioner, with current-gradient bias correction. -/
def preconditioner (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I)
    (g : I → ℝ) : Weights I →L[ℝ] Weights I :=
  diagonal (fun i => (AdamW.denominator p t s.history g i)⁻¹)

/-- T1: this operator recovers the source normalized gradient exactly. -/
theorem preconditioner_gradient (p : AdamW.Parameters) (t : ℕ)
    (s : AdamW.TrainingState I) (g : I → ℝ) :
    preconditioner p t s g (WithLp.toLp 2 g) = Rotate90.Learning.normalized p t s g := by
  ext i
  simp only [preconditioner, diagonal_apply, Rotate90.Learning.normalized, WithLp.ofLp_toLp]
  ring

/-- T1: positive optimizer epsilon proves positivity, without postulating P. -/
theorem preconditioner_nonneg (p : AdamW.Parameters) (heps : 0 < p.adam.epsilon)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ) (x : Weights I) :
    0 ≤ ⟪x, preconditioner p t s g x⟫ := by
  rw [preconditioner, inner_diagonal]
  exact Finset.sum_nonneg (fun i _ => mul_nonneg
    (inv_nonneg.mpr (AdamW.denominator_pos p t s.history g heps i).le) (sq_nonneg _))

/-- The prediction transfer between different data splits and parameter points. -/
def cross (eval : Weights I →L[ℝ] Weights A) (train : Weights I →L[ℝ] Weights B)
    (P : Weights I →L[ℝ] Weights I) : Weights B →L[ℝ] Weights A :=
  eval.comp (P.comp train.adjoint)

/-- T2: the output-space transfer is exactly the full parameter-gradient pairing. -/
theorem inner_cross (eval : Weights I →L[ℝ] Weights A)
    (train : Weights I →L[ℝ] Weights B) (P : Weights I →L[ℝ] Weights I)
    (u : Weights A) (v : Weights B) :
    ⟪u, cross eval train P v⟫ = ⟪eval.adjoint u, P (train.adjoint v)⟫ := by
  exact (eval.adjoint_inner_left _ _).symm

/-- T2: the same-point/same-split Gram operator is positive for actual Adam P. -/
theorem gram_nonneg (J : Weights I →L[ℝ] Weights A) (p : AdamW.Parameters)
    (heps : 0 < p.adam.epsilon) (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ)
    (v : Weights A) : 0 ≤ ⟪v, cross J J (preconditioner p t s g) v⟫ := by
  rw [inner_cross]
  exact preconditioner_nonneg p heps t s g _

section ChainRule

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- T2: the adjoint-residual formula is derived by differentiating the actual
composition. The prediction map and outer loss need not be quadratic or linear. -/
theorem hasGradientAt_comp {prediction : E → F} {J : E →L[ℝ] F} {x : E}
    (hJ : HasFDerivAt prediction J x) {loss : F → ℝ} {r : F}
    (hr : HasGradientAt loss r (prediction x)) :
    HasGradientAt (fun w => loss (prediction w)) (J.adjoint r) x := by
  apply hasGradientAt_iff_hasFDerivAt.mpr
  have hdual : InnerProductSpace.toDual ℝ E (J.adjoint r) =
      (InnerProductSpace.toDual ℝ F r).comp J := by
    ext v
    exact J.adjoint_inner_left v r
  rw [hdual]
  exact (hasGradientAt_iff_hasFDerivAt.mp hr).comp x hJ

/-- T2: this is the actual loss gradient, not a frozen-kernel update rule. -/
theorem gradient_comp {prediction : E → F} {J : E →L[ℝ] F} {x : E}
    (hJ : HasFDerivAt prediction J x) {loss : F → ℝ} {r : F}
    (hr : HasGradientAt loss r (prediction x)) :
    gradient (fun w => loss (prediction w)) x = J.adjoint r :=
  (hasGradientAt_comp hJ hr).gradient

example := hasGradientAt_comp
  ((ContinuousLinearMap.id ℝ ℝ).hasFDerivAt (x := 1))
  (hasGradientAt_const (1 : ℝ) (2 : ℝ))
example := gradient_comp
  ((ContinuousLinearMap.id ℝ ℝ).hasFDerivAt (x := 1))
  ((differentiable_id (𝕜 := ℝ) (E := ℝ)).differentiableAt.hasGradientAt)

end ChainRule

example := preconditioner_nonneg (I := Fin 2)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ (by norm_num)
  5 ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ (fun i => i.val + 1)
  (WithLp.toLp 2 ![1, 2])
example := gram_nonneg (I := Fin 2) (ContinuousLinearMap.id ℝ (Weights (Fin 2)))
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ (by norm_num)
  5 ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ (fun i => i.val + 1)
  (WithLp.toLp 2 ![1, 2])

end Transformer.SequenceModel.Transfer

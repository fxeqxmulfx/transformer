/-
# Homogenized Transformers — a rotation head

A deterministic instance of the model of arXiv:2604.01978v1, *Homogenized
Transformers*, on which every object of §2 and §5 can be computed by hand:
`d = 2`, one token, and the weight law `ρ* = δ_θ` at the head `θ = (J, 0)`,
`J` the generator of the rotations of the plane.

With one token the softmax weights cancel, so `B_θ[μ_X](x) = J x`, which is
tangent to the circle; hence `b(X) = J x`, the flow `eq: deterministic` is the
unit-speed rotation `x(t) = (cos t, sin t)`, and `∇_b b = Proj_x(J² x) = 0`, so
the corrected flow `eq: deterministic.modified` is the same rotation.  A point
mass has no fluctuation, so `σ = 0`, `α = 0`, and `ass:high_order_short` holds
with `σ_V = σ_A = 0`.

This is the witness `RegimeRefutation` runs on.

Source: arXiv:2604.01978v1, §2.1, §2.2.1, `ass:high_order_short`,
`eq: deterministic`, `eq: deterministic.modified`.
-/

import Transformer.Homogenized.RandomChain
import Transformer.Homogenized.Generator

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- `J = [[0, -1], [1, 0]]`, the generator of the rotations of the plane.

Source: arXiv:2604.01978v1, §2.1 (a value matrix `V`). -/
def rotGen : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The rotation head `θ = (V, A) = (J, 0)`.

Source: arXiv:2604.01978v1, §2.1. -/
def rotHead : HeadParam 2 := (rotGen, 0)

/-- `J (x₀, x₁) = (-x₁, x₀)`. -/
theorem rotGen_apply (x : EucSpace 2) :
    Matrix.toEuclideanLin rotGen x = !₂[-x 1, x 0] := by
  ext k; fin_cases k <;>
    simp [rotGen, Matrix.toEuclideanLin, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- `J x` is orthogonal to `x`. -/
theorem inner_rotGen (x : EucSpace 2) :
    inner (𝕜 := ℝ) x (Matrix.toEuclideanLin rotGen x) = 0 := by
  rw [rotGen_apply]
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]
  ring

/-- With one token the softmax weight cancels: `B_θ[μ_X](z) = J x₀`.

Source: arXiv:2604.01978v1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem attnField_rotHead (β : ℝ) (x : Idx 1 → EucSpace 2) (z : EucSpace 2) :
    attnField β rotHead x z = Matrix.toEuclideanLin rotGen (x 0) := by
  simp only [attnField, Fin.sum_univ_one, smul_smul]
  rw [inv_mul_cancel₀ (attnWeight_pos _ _ _ _).ne', one_smul]
  rfl

/-- The projected drift of `δ_θ` is `b(X) = J x₀`, on the whole of `(ℝ²)¹`: `J x`
is already tangent.

Source: arXiv:2604.01978v1, §5. -/
theorem bField_rotHead (β : ℝ) :
    bField β (Measure.dirac rotHead) = fun (x : Idx 1 → EucSpace 2) _ =>
      Matrix.toEuclideanLin rotGen (x 0) := by
  funext x i
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  rw [bField, meanField, integral_dirac, attnField_rotHead, proj, inner_rotGen, zero_smul, sub_zero]

/-- `J² = -1`. -/
theorem rotGen_rotGen (x : EucSpace 2) :
    Matrix.toEuclideanLin rotGen (Matrix.toEuclideanLin rotGen x) = -x := by
  rw [rotGen_apply, rotGen_apply]
  ext k; fin_cases k <;> simp

/-- `∇_b b = Proj_x(J² x) = -x + ‖x‖² x = 0` on the circle: the corrector of
`eq: deterministic.modified` vanishes.

Source: arXiv:2604.01978v1, `eq: deterministic.modified`. -/
theorem covDerivB_rotHead (β : ℝ) (x : Idx 1 → EucSpace 2) (hx : ‖x 0‖ = 1) (i : Idx 1) :
    covDerivB β (Measure.dirac rotHead) x i = 0 := by
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  let Lm : (Idx 1 → EucSpace 2) →L[ℝ] EucSpace 2 :=
    (LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin rotGen)).comp
      (ContinuousLinearMap.proj 0)
  have hL : (fun y : Idx 1 → EucSpace 2 => bField β (Measure.dirac rotHead) y 0) = Lm := by
    rw [bField_rotHead]; rfl
  rw [covDerivB, covDeriv, hL, Lm.fderiv, bField_rotHead]
  change proj 2 (x 0) (Matrix.toEuclideanLin rotGen (Matrix.toEuclideanLin rotGen (x 0))) = 0
  rw [rotGen_rotGen, proj, inner_neg_right, real_inner_self_eq_norm_sq, hx]
  simp

/-- The hypothesis of `covDerivB_rotHead` is satisfiable. -/
example : ‖(fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) 0‖ = 1 := by
  simp [PiLp.norm_single]

/-- The unit-speed rotation of the plane, one token: `x(t) = cos t e₀ + sin t e₁`.

Source: arXiv:2604.01978v1, `eq: deterministic`. -/
noncomputable def rotFlow (t : ℝ) : Idx 1 → EucSpace 2 := fun _ =>
  Real.cos t • EuclideanSpace.single 0 1 + Real.sin t • EuclideanSpace.single 1 1

/-- `x(t) = (cos t, sin t)`. -/
theorem rotFlow_apply (t : ℝ) (i : Idx 1) : rotFlow t i = !₂[Real.cos t, Real.sin t] := by
  ext k; fin_cases k <;> simp [rotFlow]

/-- The rotation stays on the circle. -/
theorem norm_rotFlow (t : ℝ) (i : Idx 1) : ‖rotFlow t i‖ = 1 := by
  rw [rotFlow_apply, EuclideanSpace.norm_eq]
  simp [Fin.sum_univ_two]

/-- `ẋ = J x = b(x)`. -/
theorem hasDerivAt_rotFlow (β : ℝ) (t : ℝ) (i : Idx 1) :
    HasDerivAt (fun r => rotFlow r i) (bField β (Measure.dirac rotHead) (rotFlow t) i) t := by
  rw [bField_rotHead]
  have h := ((Real.hasDerivAt_cos t).smul_const (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))).add
    ((Real.hasDerivAt_sin t).smul_const (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)))
  convert h using 1
  · rfl
  · show Matrix.toEuclideanLin rotGen (rotFlow t 0) = _
    rw [rotGen_apply, rotFlow_apply]
    ext k; fin_cases k <;> simp

/-- The rotation solves `eq: deterministic` for `ρ* = δ_θ`.

Source: arXiv:2604.01978v1, `eq: deterministic`. -/
theorem isBallisticFlow_rotFlow (β : ℝ) : IsBallisticFlow β (Measure.dirac rotHead) rotFlow :=
  ⟨norm_rotFlow, hasDerivAt_rotFlow β⟩

/-- The rotation solves `eq: deterministic.modified` for `ρ* = δ_θ`, at every `η`,
since the corrector vanishes.

Source: arXiv:2604.01978v1, `eq: deterministic.modified`. -/
theorem isModifiedFlow_rotFlow (η β : ℝ) :
    IsModifiedFlow η β (Measure.dirac rotHead) rotFlow := by
  refine ⟨norm_rotFlow, fun t i => ?_⟩
  rw [covDerivB_rotHead β _ (norm_rotFlow t 0), smul_zero, sub_zero]
  exact hasDerivAt_rotFlow β t i

/-- The chain of `eq:update_tokens` at the rotation head, one head, from `e₀`.

Source: arXiv:2604.01978v1, `eq:update_tokens`. -/
noncomputable def rotChain (η β : ℝ) : ℕ → Idx 1 → EucSpace 2 :=
  fun ℓ => Nat.rec (fun _ => EuclideanSpace.single 0 1)
    (fun _ x i => layerUpdate (H := 1) η β (fun _ => rotHead) x i) ℓ

/-- The chain stays on the circle: `x + η J x` has inner product `1` with `x`,
so it is not `0` and `N` normalizes it. -/
theorem norm_rotChain (η β : ℝ) (ℓ : ℕ) (i : Idx 1) : ‖rotChain η β ℓ i‖ = 1 := by
  induction ℓ generalizing i with
  | zero => simp [rotChain, PiLp.norm_single]
  | succ ℓ ih =>
    show ‖layerUpdate (H := 1) η β (fun _ => rotHead) (rotChain η β ℓ) i‖ = 1
    apply norm_normalizeLayer
    intro h0
    have h := congrArg (inner (𝕜 := ℝ) (rotChain η β ℓ i)) h0
    rw [Fin.sum_univ_one, attnField_rotHead, Subsingleton.elim (0 : Idx 1) i, inner_add_right,
      inner_smul_right, inner_rotGen, real_inner_self_eq_norm_sq, ih, inner_zero_right] at h
    norm_num at h

/-- The rotation chain is the chain of `eq:update_tokens` under `ρ* = δ_θ`, on a
one-point probability space.

Source: arXiv:2604.01978v1, `eq:update_tokens`. -/
theorem isRandomChain_rotChain (η β : ℝ) :
    IsRandomChain (H := 1) η β (Measure.dirac rotHead) (Measure.dirac ())
      (fun _ _ _ => rotHead) (fun _ => rotChain η β) (fun _ => EuclideanSpace.single 0 1) := by
  refine ⟨inferInstance, fun _ _ => measurable_const, iIndepFun_of_unit _, ?_,
    fun _ => ⟨norm_rotChain η β, fun _ _ => rfl⟩, fun _ => rfl⟩
  intro ℓ h
  simp

/-- `ass:high_order_short` holds at every point mass, with `σ_V = σ_A = 0`: the
means are the point, the centered parts vanish on a one-point space.

Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem hasHighOrderLaw_dirac {d : ℕ} (θ : HeadParam d) :
    HasHighOrderLaw d 0 0 (Measure.dirac θ) := by
  refine ⟨1, Unit, inferInstance, Measure.dirac (), 0, 0, 0, θ.1, θ.2, one_pos,
    inferInstance, measurable_const, measurable_const, measurable_const, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_⟩
  · simp
  · intro i j; simp
  · exact indepFun_const_left _ _
  · exact iIndepFun_of_unit _
  · exact iIndepFun_of_unit _
  · intro i j; simp
  · intro i j; simp
  · intro i j; simp

/-- At a point mass the fluctuation vanishes, so `σ = 0` is the variance proxy.

Source: arXiv:2604.01978v1, `eq: defining.alpha`. -/
theorem isVarianceProxy_rotHead (β : ℝ) :
    IsVarianceProxy 2 1 β (Measure.dirac rotHead) 0 := by
  refine ⟨le_rfl, ⟨⟨fun _ => EuclideanSpace.single 0 1, fun _ => by simp [PiLp.norm_single],
    ⟨0, ?_⟩⟩, ?_⟩⟩
  · simp [Gfield, fluct, meanField, proj]
  · rintro v ⟨x, -, j, rfl⟩
    simp [Gfield, fluct, meanField, proj]

end Homogenized
end Transformer

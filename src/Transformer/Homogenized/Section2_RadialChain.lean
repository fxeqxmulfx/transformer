module

public import Transformer.Homogenized.Section2_RadialRotation

/-!
# The exact discrete chain with a radial value component

Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`.
The deterministic head `radialRotHead = (I+J,0)` is defined in
`Section2_RadialRotation`, together with its actual projected
drift and exact modified SDE solution.

This module verifies the source's discrete process. For one
token and one head, the unnormalized update is
`(1+η)x + η Jx`. Positive rescaling leaves normalization
unchanged, so its exact effective rotation step is
`q = η/(1+η)`, rather than `η`.

The resulting chain is the existing rotation chain at that
effective step. We verify every condition of `IsRandomChain`
at the physical step size `η`, including the actual update,
unit norm, head law, independence, and initial condition.
The source and comparison spaces are both probability spaces.

The explicit angular formula is proved for the full chain,
not only for its first iterate. It supplies the accumulated
phase discrepancy at the grid time `η L = 1` downstream.
No comparison estimate is assumed in any of these calculations.
-/

@[expose] public section

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The exact source update at `I+J` equals the rotation-head update at
`η/(1+η)`. Positivity of `η` makes the removed scale `1+η` positive.
Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`. -/
theorem layerUpdate_radialRotHead {η : ℝ} (hη : 0 < η) (β : ℝ)
    (x : Idx 1 → EucSpace 2) (i : Idx 1) :
    layerUpdate (H := 1) η β (fun _ => radialRotHead) x i =
      layerUpdate (H := 1) (η / (1 + η)) β (fun _ => rotHead) x i := by
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  simp only [layerUpdate, Fin.sum_univ_one, Nat.cast_one, div_one,
    attnField_radialRotHead, attnField_rotHead]
  have hs : x 0 + η • (x 0 + Matrix.toEuclideanLin rotGen (x 0)) =
      (1 + η) • (x 0 + (η / (1 + η)) • Matrix.toEuclideanLin rotGen (x 0)) := by
    rw [smul_add, smul_add, smul_smul, mul_div_cancel₀ _ (by linarith : 1 + η ≠ 0)]
    module
  rw [hs]
  unfold normalizeLayer
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by linarith : 0 < 1 + η), mul_inv, smul_smul]
  congr 1
  field_simp

/-- The actual shifted-head chain, represented by its exact effective step.
Both the physical step and temperature enter the underlying rotation chain.
Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`. -/
noncomputable def radialRotChain (η β : ℝ) : ℕ → Idx 1 → EucSpace 2 :=
  rotChain (η / (1 + η)) β

/-- Every iterate of the shifted-head chain has unit norm.
Source: arXiv:2604.01978v1, §2.1, spherical token configurations. -/
theorem norm_radialRotChain (η β : ℝ) (ℓ : ℕ) (i : Idx 1) :
    ‖radialRotChain η β ℓ i‖ = 1 := norm_rotChain _ _ _ _

/-- The represented chain satisfies the complete source predicate at the
physical step `η`, with actual constant head `I+J` on a probability space.
Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`. -/
theorem isRandomChain_radialRotChain {η : ℝ} (hη : 0 < η) (β : ℝ) :
    IsRandomChain (H := 1) η β (Measure.dirac radialRotHead) (Measure.dirac ())
      (fun _ _ _ => radialRotHead) (fun _ => radialRotChain η β)
      (fun _ => EuclideanSpace.single 0 1) := by
  refine ⟨inferInstance, fun _ _ => measurable_const, iIndepFun_of_unit _, ?_,
    fun _ => ⟨norm_radialRotChain η β, fun ℓ i => ?_⟩, fun _ => rfl⟩
  · intro ℓ h
    simp
  · rw [layerUpdate_radialRotHead hη]
    rfl

/-- The second coordinate after one exact step from `e₀`.
Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens` at `V=I+J`. -/
theorem radialRotChain_one (η β : ℝ) :
    radialRotChain η β 1 0 1 = (η / (1 + η)) / Real.sqrt (1 + (η / (1 + η)) ^ 2) := by
  simp [radialRotChain, rotChain, layerUpdate, attnField_rotHead, rotGen_apply,
    normalizeLayer, EuclideanSpace.norm_eq, Fin.sum_univ_two, div_eq_mul_inv, mul_comm]

/-- Every rotation-head iterate has angle `ℓ arctan q`.
The proof computes the normalization norm before applying angle addition.
Source: arXiv:2604.01978v1, §2.1, the exact rotation-head update. -/
theorem rotChain_eq_rotFlow (q β : ℝ) (ℓ : ℕ) :
    rotChain q β ℓ = rotFlow ((ℓ : ℝ) * Real.arctan q) := by
  induction ℓ with
  | zero =>
    ext i k
    fin_cases k <;> simp [rotChain, rotFlow]
  | succ ℓ ih =>
    change (fun i => layerUpdate (H := 1) q β (fun _ => rotHead) (rotChain q β ℓ) i) = _
    rw [ih]
    have hn : ∀ i : Idx 1,
        ‖rotFlow ((ℓ : ℝ) * Real.arctan q) i +
          q • Matrix.toEuclideanLin rotGen (rotFlow ((ℓ : ℝ) * Real.arctan q) 0)‖ =
          Real.sqrt (1 + q ^ 2) := by
      intro i
      rw [rotGen_apply, rotFlow_apply, rotFlow_apply, EuclideanSpace.norm_eq]
      congr 1
      simp only [Fin.sum_univ_two, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul,
        Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
        Real.norm_eq_abs, sq_abs]
      nlinarith [Real.sin_sq_add_cos_sq ((ℓ : ℝ) * Real.arctan q)]
    ext i k
    simp only [layerUpdate, Fin.sum_univ_one, Nat.cast_one, div_one, attnField_rotHead,
      normalizeLayer]
    rw [hn]
    simp only [rotGen_apply, rotFlow_apply, Nat.cast_add, Nat.cast_one,
      add_mul, one_mul, Real.cos_add, Real.sin_add, Real.cos_arctan, Real.sin_arctan]
    fin_cases k <;> simp [div_eq_mul_inv] <;> ring

/-- The actual shifted chain has angle `ℓ arctan(η/(1+η))` at layer `ℓ`.
Source: arXiv:2604.01978v1, §2.1, the iterates of `eq:update_tokens`. -/
theorem radialRotChain_eq_rotFlow (η β : ℝ) (ℓ : ℕ) :
    radialRotChain η β ℓ = rotFlow ((ℓ : ℝ) * Real.arctan (η / (1 + η))) :=
  rotChain_eq_rotFlow _ _ _

/-- The continuous rotation and exact discrete chain start from the same
configuration at time and layer zero, respectively.
Both use the first standard basis vector as their deterministic initial token.
Source: arXiv:2604.01978v1, §2.1, the initial condition in `eq:update_tokens`. -/
theorem radialRotChain_initial (η β : ℝ) :
    radialRotChain η β 0 = rotFlow 0 := by
  ext i k
  fin_cases k <;> simp [radialRotChain, rotChain, rotFlow]

/-- The concrete source carrying the shifted chain is a probability space.
Its deterministic head has exactly the required high-order law. -/
example : IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧
    HasHighOrderLaw 2 0 0 (Measure.dirac radialRotHead) :=
  ⟨inferInstance, (radialRotHead_law 1).1⟩

/-- A positive physical step jointly witnesses the update and chain hypotheses.
The initial state is a genuine nonzero unit vector. -/
example : (0 : ℝ) < 1 ∧
    ‖radialRotChain 1 1 0 0‖ = 1 ∧
    radialRotChain 1 1 0 0 ≠ 0 := by
  refine ⟨zero_lt_one, norm_radialRotChain 1 1 0 0, ?_⟩
  intro h
  have hn := norm_radialRotChain 1 1 0 0
  rw [h, norm_zero] at hn
  norm_num at hn

end Transformer.Homogenized

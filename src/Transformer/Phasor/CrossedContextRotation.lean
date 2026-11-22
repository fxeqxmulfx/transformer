module

public import Transformer.Phasor.ContextRotation
public import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Two overlapping sets of learned contextual rotations

Source: the crossed contextual rotation experiment in `experiments/phasor_value_transport/README.md`.
The first stage rotates half-split coordinate pairs. The second stage
rotates adjacent pairs. Both angle arrays come from learned linear maps
of the same exclusive prefix of normalized embeddings. A coordinate
permutation expresses the second pairing in the established rotation model.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer SequenceModel

variable {D d H F : ℕ}

/-- Half-split lanes `(side, channel)` become adjacent lanes
`(channel, side)` at any even width. -/
def halfToAdjacent (n : ℕ) : Equiv.Perm (Fin (2*n)) :=
  (((finProdFinEquiv (m:=2) (n:=n)).symm.trans (Equiv.prodComm _ _)).trans
    (finProdFinEquiv (m:=n) (n:=2))).trans (finCongr (Nat.mul_comm n 2))

/-- The physical coordinate used by the adjacent-pair Kernite projection. -/
theorem halfToAdjacent_index (n : ℕ) (s : Fin 2) (k : Fin n) :
    (halfToAdjacent n (finProdFinEquiv (s,k))).val = 2*k.val+s.val := by
  simp [halfToAdjacent, finProdFinEquiv_apply_val, Nat.add_comm]

/-- Rotating planes after reindexing is again a norm-preserving map of
the original contextual vector. The classifier keeps the original coordinates. -/
noncomputable def reindexedRotation (p : Equiv.Perm (Fin D)) (a : Fin (D/2) → ℝ)
    (x : EucSpace D) : EucSpace D :=
  let S := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ p
  S (rotateHead D a (S.symm x))

theorem reindexedRotation_norm (p : Equiv.Perm (Fin D)) (a : Fin (D/2) → ℝ) (x : EucSpace D) :
    ‖reindexedRotation p a x‖ = ‖x‖ := by
  unfold reindexedRotation
  rw [LinearIsometryEquiv.norm_map, rotateHead_isometry, LinearIsometryEquiv.norm_map]

/-- Negating the angle array in the same pairing exactly recovers the input. -/
theorem reindexedRotation_neg (p : Equiv.Perm (Fin D)) (a : Fin (D/2) → ℝ) (x : EucSpace D) :
    reindexedRotation p (fun k => -a k) (reindexedRotation p a x) = x := by
  simp only [reindexedRotation, LinearIsometryEquiv.symm_apply_apply, rotateHead_neg,
    LinearIsometryEquiv.apply_symm_apply]

theorem reindexedRotation_zero (p : Equiv.Perm (Fin D)) (x : EucSpace D) :
    reindexedRotation p (fun _ => 0) x = x := by
  simp only [reindexedRotation, rotateHead_zero_angles, LinearIsometryEquiv.apply_symm_apply]

/-- The second physical lane receives the signed contribution from the
first lane of that same reindexed pair. -/
theorem reindexedRotation_snd (p : Equiv.Perm (Fin D)) (a : Fin (D/2) → ℝ) (x : EucSpace D)
    (k : Fin (D/2)) :
    reindexedRotation p a x (p (GPTMini.ropeSnd D k)) =
      x (p (GPTMini.ropeFst D k))*Real.sin (a k) + x (p (GPTMini.ropeSnd D k))*Real.cos (a k) := by
  let S := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ p
  change rotateHead D a (S.symm x) (p.symm (p (GPTMini.ropeSnd D k))) = _
  rw [Equiv.symm_apply_apply, rotateHead_apply_snd]
  simp only [rotate, splitPairs, Fin.isValue, one_ne_zero, ↓reduceIte,
    S, LinearIsometryEquiv.piLpCongrLeft_symm]
  rfl

/-- Even with the base coordinate system held fixed, a second pairing
can transfer mass from coordinate zero to coordinate one. No single
half-split rotation of this input has any mass in coordinate one. -/
theorem adjacentRotation_outside_halfSplit :
    ∃ x : EucSpace 4, ∀ a : Fin (4/2) → ℝ,
      reindexedRotation (halfToAdjacent 2) (fun _ => Real.pi/2) x ≠ rotateHead 4 a x := by
  let x : EucSpace 4 := WithLp.toLp 2 (fun i => if i = (0:Fin 4) then (1:ℝ) else 0)
  refine ⟨x, fun a h => ?_⟩
  have adj := reindexedRotation_snd (halfToAdjacent 2) (fun _ => Real.pi/2) x (0:Fin (4/2))
  have left : halfToAdjacent 2 (GPTMini.ropeFst 4 0) = 0 := by decide
  have right : halfToAdjacent 2 (GPTMini.ropeSnd 4 0) = 1 := by decide
  rw [left,right, Real.sin_pi_div_two, Real.cos_pi_div_two, mul_one, mul_zero, add_zero] at adj
  have half := rotateHead_apply_fst 4 a x (1:Fin (4/2))
  have fst : GPTMini.ropeFst 4 1 = 1 := by decide
  have snd : GPTMini.ropeSnd 4 1 = 3 := by decide
  simp only [rotate, splitPairs, Fin.isValue, one_ne_zero, ↓reduceIte, fst, snd] at half
  rw [h, half] at adj
  norm_num [x] at adj

/-- Four output cotangents feed the derivative of one first-stage angle
through its two overlapping second-stage planes. The other first-stage
plane contributes an additive constant when varying this angle. -/
theorem crossedRotation_first_angle_deriv (a b c u v g₀ g₁ g₂ g₃ : ℝ) :
    let s := fun t => u*Real.cos t - v*Real.sin t
    let t := fun z => u*Real.sin z + v*Real.cos z
    HasDerivAt (fun z => g₀*(Real.cos b*s z)+g₁*(Real.sin b*s z)+
      g₂*(Real.cos c*t z)+g₃*(Real.sin c*t z))
      (s a*(g₂*Real.cos c+g₃*Real.sin c)-t a*(g₀*Real.cos b+g₁*Real.sin b)) a := by
  dsimp only
  convert contextRotation_angle_deriv a u v (g₀*Real.cos b+g₁*Real.sin b)
    (g₂*Real.cos c+g₃*Real.sin c) using 1
  funext z
  ring

/-- Two shared input projections drive two sequential, overlapping
isometries of the same contextual feature. -/
noncomputable def EmbeddingBackbone.crossedRotationFeature (A : EmbeddingBackbone D d H F)
    (p : Equiv.Perm (Fin D)) (U V : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  reindexedRotation p (contextAngles V eps x i) (A.contextRotationFeature U θ eps scale x l i)

/-- A zero second projection preserves the entire first contextual rotation. -/
theorem EmbeddingBackbone.crossedRotationFeature_zero (A : EmbeddingBackbone D d H F)
    (p : Equiv.Perm (Fin D)) (U : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.crossedRotationFeature p U 0 θ eps scale x l i = A.contextRotationFeature U θ eps scale x l i := by
  have zero : contextAngles (0 : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) eps x i = fun _ => 0 := by
    funext k
    simp [contextAngles, prefixPhase]
  rw [EmbeddingBackbone.crossedRotationFeature, zero, reindexedRotation_zero]

/-- Recover the original contextual read by undoing the two rotations in reverse order. -/
theorem EmbeddingBackbone.crossedRotationFeature_inverse (A : EmbeddingBackbone D d H F)
    (p : Equiv.Perm (Fin D)) (U V : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    rotateHead D (fun k => -contextAngles U eps x i k)
      (reindexedRotation p (fun k => -contextAngles V eps x i k)
        (A.crossedRotationFeature p U V θ eps scale x l i)) = A.feature θ eps scale x l i := by
  rw [EmbeddingBackbone.crossedRotationFeature, reindexedRotation_neg, A.contextRotationFeature_inverse]

theorem EmbeddingBackbone.crossedRotationFeature_norm (A : EmbeddingBackbone D d H F)
    (p : Equiv.Perm (Fin D)) (U V : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    ‖A.crossedRotationFeature p U V θ eps scale x l i‖ = ‖A.feature θ eps scale x l i‖ := by
  rw [EmbeddingBackbone.crossedRotationFeature, reindexedRotation_norm, A.contextRotationFeature_norm]

/-- Both stages see only preceding projected embeddings and the causal
context at the scored position, for every choice of learned projections. -/
theorem EmbeddingBackbone.crossedRotationFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (p : Equiv.Perm (Fin D)) (U V : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (θ : StreamReadWeights D) (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ)
    (h : ∀ j ≤ i, x j = y j) :
    A.crossedRotationFeature p U V θ eps scale x l i = A.crossedRotationFeature p U V θ eps scale y l i := by
  have eqAngle : contextAngles V eps x i = contextAngles V eps y i := by
    funext k
    unfold contextAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change V (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt (Finset.mem_range.mp hj))]
  rw [EmbeddingBackbone.crossedRotationFeature, EmbeddingBackbone.crossedRotationFeature, eqAngle,
    A.contextRotationFeature_congr_prefix U θ eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (U V : EucSpace 128 →ₗ[ℝ] (Fin 64 → ℝ))
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) :
    A.crossedRotationFeature (halfToAdjacent 64) U V θ 1 1 x 2 5 =
      A.crossedRotationFeature (halfToAdjacent 64) U V θ 1 1 (fun j => if j≤5 then x j else fill j) 2 5 :=
  A.crossedRotationFeature_congr_prefix _ _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor

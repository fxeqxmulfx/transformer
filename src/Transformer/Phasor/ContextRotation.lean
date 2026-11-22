module

public import Transformer.Phasor.InputPhaseRead
public import Transformer.SequenceModel.RotateHead

/-!
# Rotating a contextual representation by its learned input prefix

Source: the contextual rotation experiment in `experiments/phasor_value_transport/README.md`.
The projection of each normalized embedding supplies one increment per
coordinate pair. Immediately before position `i`, the accumulated angles
rotate the full learned contextual read at `i`. The same projection and
rotation act at every position. The classifier is external.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer SequenceModel

variable {D d H F : ℕ}

/-- An exclusive sum of learned input projections. In particular the
current embedding affects its own contextual vector but adds to later angles. -/
noncomputable def contextAngles (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) : Fin (D/2) → ℝ :=
  fun k => prefixPhase (fun j => W (GPTMini.rmsNormEps eps (x j)) k) i

/-- The native streaming update reads the old phase before adding the
increment of the token at the current position. -/
theorem contextAngles_succ (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ))
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) (k : Fin (D/2)) :
    contextAngles W eps x (i+1) k =
      contextAngles W eps x i k + W (GPTMini.rmsNormEps eps (x i)) k := by
  exact Finset.sum_range_succ _ _

/-- The rotation replaces the contextual vector by its isometric image
before the ordinary classifier scores it. The native layout uses two halves. -/
noncomputable def EmbeddingBackbone.contextRotationFeature (A : EmbeddingBackbone D d H F)
    (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  rotateHead D (contextAngles W eps x i) (A.feature θ eps scale x l i)

/-- Zero-initializing the input projection recovers the learned contextual
feature on every sequence, including nonzero features at arbitrary depth. -/
theorem EmbeddingBackbone.contextRotationFeature_zero (A : EmbeddingBackbone D d H F)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.contextRotationFeature 0 θ eps scale x l i = A.feature θ eps scale x l i := by
  have angles : contextAngles (0 : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) eps x i = fun _ => 0 := by
    funext k
    simp [contextAngles, prefixPhase]
  rw [EmbeddingBackbone.contextRotationFeature, angles, rotateHead_zero_angles]

/-- For every learned projection and every embedding sequence, all
contextual information can be recovered from the rotated state and its angles. -/
theorem EmbeddingBackbone.contextRotationFeature_inverse (A : EmbeddingBackbone D d H F)
    (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) :
    rotateHead D (fun k => -contextAngles W eps x i k) (A.contextRotationFeature W θ eps scale x l i) =
      A.feature θ eps scale x l i := by
  rw [EmbeddingBackbone.contextRotationFeature, rotateHead_add]
  simpa using rotateHead_zero_angles D (A.feature θ eps scale x l i)

/-- Rephasing preserves the exact norm of every hidden vector, even when
its angles depend on the observations and the contextual features also vary. -/
theorem EmbeddingBackbone.contextRotationFeature_norm (A : EmbeddingBackbone D d H F)
    (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) :
    ‖A.contextRotationFeature W θ eps scale x l i‖ = ‖A.feature θ eps scale x l i‖ :=
  rotateHead_isometry _ _ _

/-- Prefix agreement propagates down both paths. Future changes affect
neither the angles nor the contextual vector at this position. -/
theorem EmbeddingBackbone.contextRotationFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (W : EucSpace D →ₗ[ℝ] (Fin (D/2) → ℝ)) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.contextRotationFeature W θ eps scale x l i = A.contextRotationFeature W θ eps scale y l i := by
  have angles : contextAngles W eps x i = contextAngles W eps y i := by
    funext k
    unfold contextAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change W (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt (Finset.mem_range.mp hj))]
  simp only [EmbeddingBackbone.contextRotationFeature, EmbeddingBackbone.feature, angles,
    A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (W : EucSpace 128 →ₗ[ℝ] (Fin 64 → ℝ))
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.contextRotationFeature W θ 1 1 x l 5 =
      A.contextRotationFeature W θ 1 1 (fun j => if j ≤ 5 then x j else fill j) l 5 :=
  A.contextRotationFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- For one complex pair, the angle derivative of the outgoing dot product
is the signed area of the rotated hidden vector and its cotangent. -/
theorem contextRotation_angle_deriv (a u v g h : ℝ) :
    HasDerivAt (fun t => g*(u*Real.cos t - v*Real.sin t) + h*(u*Real.sin t + v*Real.cos t))
      ((u*Real.cos a-v*Real.sin a)*h - (u*Real.sin a+v*Real.cos a)*g) a := by
  have c := Real.hasDerivAt_cos a
  have s := Real.hasDerivAt_sin a
  convert! (((c.const_mul u).sub (s.const_mul v)).const_mul g).add
    (((s.const_mul u).add (c.const_mul v)).const_mul h) using 1
  ring

/-- The exclusive forward scan has a strictly later suffix sum as adjoint.
In particular the last increment receives zero from this output path. -/
theorem contextRotation_exclusive_adjoint {n : ℕ} (d g : Fin n → ℝ) :
    ∑ i, g i * (∑ j, if j < i then d j else 0) =
      ∑ j, (∑ i, if j < i then g i else 0) * d j := by
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  split_ifs <;> simp

end Transformer.Phasor

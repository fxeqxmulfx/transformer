module

public import Transformer.Phasor.BilinearPhaseRead

/-!
# Learned frequencies on a shared bilinear state

Source: “Shared bilinear phase” in `experiments/phasor_value_transport/README.md`.
The recurrent state has `r` coordinates. Each of the `m` periodic observations
uses a learned linear combination of the same state. The dimensions and
frequencies are independent of the input length and the external vocabulary.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F r m : ℕ}

/-- Additive increments and strictly ordered interactions up to the end of
a segment. All three streams can vary independently at every position. -/
noncomputable def sharedBilinearState (a b c : ℕ → Fin r → ℝ) (n : ℕ) : Fin r → ℝ :=
  fun k => prefixPhase (fun j => a j k) n +
    orderedPhaseIntegral (fun j => b j k) (fun j => c j k) n

/-- The frequency-weighted cross-segment interaction factors through the
same two vectors of segment sums for every observation. The cost of carrying
these summaries is independent of the number of observations. -/
theorem sharedBilinearState_split (a b c : ℕ → Fin r → ℝ) (ω : Fin r → ℝ) (n s : ℕ) :
    ω ⬝ᵥ sharedBilinearState a b c (n+s) =
      ω ⬝ᵥ sharedBilinearState a b c n +
      ω ⬝ᵥ sharedBilinearState (fun j => a (n+j)) (fun j => b (n+j)) (fun j => c (n+j)) s +
      ω ⬝ᵥ (fun k => prefixPhase (fun j => b j k) n * prefixPhase (fun j => c (n+j) k) s) := by
  have split : sharedBilinearState a b c (n+s) = sharedBilinearState a b c n +
      sharedBilinearState (fun j => a (n+j)) (fun j => b (n+j)) (fun j => c (n+j)) s +
      (fun k => prefixPhase (fun j => b j k) n * prefixPhase (fun j => c (n+j) k) s) := by
    funext k
    simp only [sharedBilinearState, Pi.add_apply, orderedPhaseIntegral_split,
      prefixPhase, Finset.sum_range_add]
    ring
  rw [split, dotProduct_add, dotProduct_add]

/-- A single displacement rotates each observation through its own learned
frequency applied to the displacement. The squared chord across the whole
observation bank is independent of the starting state. -/
theorem sharedBilinear_observation_shift (ω : Fin m → Fin r → ℝ) (z δ : Fin r → ℝ) :
    ∑ j, ((Real.sin (ω j ⬝ᵥ (z+δ)) - Real.sin (ω j ⬝ᵥ z))^2 +
      (Real.cos (ω j ⬝ᵥ (z+δ)) - Real.cos (ω j ⬝ᵥ z))^2) =
      ∑ j, (2 - 2 * Real.cos (ω j ⬝ᵥ δ)) := by
  apply Finset.sum_congr rfl
  intro j _
  rw [dotProduct_add, phase_pair_distance]

/-- Simultaneously changing a learned frequency and its input sends
derivatives to both. This is the projection directly before the periodic
observations of the shared bilinear phase architecture. -/
theorem sharedBilinear_frequency_deriv (ω z dω dz : Fin r → ℝ) :
    HasDerivAt (fun t : ℝ => (fun k => ω k+t*dω k) ⬝ᵥ (fun k => z k+t*dz k))
      (dω ⬝ᵥ z + ω ⬝ᵥ dz) 0 := by
  have line (x dx : ℝ) : HasDerivAt (fun t : ℝ => x+t*dx) dx 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dx).const_add x
  simp only [dotProduct, ← Finset.sum_add_distrib]
  apply HasDerivAt.fun_sum
  intro k _
  simpa using (line (ω k) (dω k)).fun_mul (line (z k) (dz k))

/-- All observations contribute to each latent cotangent through the
transpose of the learned frequency matrix. The matrix receives a separate
outer-product gradient for every observation and state coordinate. -/
theorem sharedBilinear_frequency_adjoint (ω dω : Fin m → Fin r → ℝ)
    (z dz : Fin r → ℝ) (g : Fin m → ℝ) :
    (∑ j, g j * (dω j ⬝ᵥ z + ω j ⬝ᵥ dz)) =
      (∑ j, ∑ k, (g j * z k) * dω j k) + ∑ k, (∑ j, g j * ω j k) * dz k := by
  simp only [dotProduct, mul_add, Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_mul]
  congr 1
  · apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro k _
    ring
  · rw [Finset.sum_comm]
    simp only [mul_assoc]

/-- Four learned projections write or read the latent state. Each row of
the learned frequency matrix addresses that state, and two projections map
the periodic observations back into the backbone's feature space. -/
structure SharedBilinearRead (D r m : ℕ) where
  additive : EucSpace D →ₗ[ℝ] (Fin r → ℝ)
  ordered : BilinearPhaseRead D r
  frequency : Fin m → Fin r → ℝ
  sineOut : (Fin m → ℝ) →ₗ[ℝ] EucSpace D
  cosineOut : (Fin m → ℝ) →ₗ[ℝ] EucSpace D

/-- Inclusive bilinear state plus a transient, position-local projection.
The transient value has no effect on any subsequent state. -/
noncomputable def SharedBilinearRead.latent (P : SharedBilinearRead D r m)
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) : Fin r → ℝ :=
  let v := fun j => GPTMini.rmsNormEps eps (x j)
  sharedBilinearState (fun j => P.additive (v j)) (fun j => P.ordered.left (v j))
    (fun j => P.ordered.right (v j)) (i+1) + P.ordered.transient (v i)

/-- Each learned observation sees the entire latent state on any prefix
of dense vectors. Sine and centered cosine are projected separately. -/
noncomputable def EmbeddingBackbone.sharedBilinearFeature (A : EmbeddingBackbone D d H F)
    (P : SharedBilinearRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let z := P.latent eps x i
  A.feature θ eps scale x l i + (P.sineOut (fun j => Real.sin (P.frequency j ⬝ᵥ z)) +
    P.cosineOut (fun j => Real.cos (P.frequency j ⬝ᵥ z)-1))

/-- Zero output projections initialize at the trained stream architecture
for any settings of the four input projections and the frequency matrix. -/
theorem EmbeddingBackbone.sharedBilinearFeature_zero (A : EmbeddingBackbone D d H F)
    (P : SharedBilinearRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) :
    A.sharedBilinearFeature {P with sineOut := 0, cosineOut := 0} θ eps scale x l i =
      A.feature θ eps scale x l i := by
  simp [EmbeddingBackbone.sharedBilinearFeature]

/-- The latent state and its local projection depend only on the dense prefix. -/
theorem SharedBilinearRead.latent_congr_prefix (P : SharedBilinearRead D r m)
    (eps : ℝ) (x y : ℕ → EucSpace D) (i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    P.latent eps x i = P.latent eps y i := by
  let v := fun (seq : ℕ → EucSpace D) j => GPTMini.rmsNormEps eps (seq j)
  have equal (j) (hj : j < i+1) : v x j = v y j := by simp [v, h j (by omega)]
  funext k
  change (prefixPhase (fun j => P.additive (v x j) k) (i+1) + _) + _ =
    (prefixPhase (fun j => P.additive (v y j) k) (i+1) + _) + _
  have additive : prefixPhase (fun j => P.additive (v x j) k) (i+1) =
      prefixPhase (fun j => P.additive (v y j) k) (i+1) :=
    Finset.sum_congr rfl fun j hj => by
      change P.additive (v x j) k = P.additive (v y j) k
      rw [equal j (Finset.mem_range.mp hj)]
  have pairs := orderedPhaseIntegral_congr_prefix
    (fun j => P.ordered.left (v x j) k) (fun j => P.ordered.right (v x j) k)
    (fun j => P.ordered.left (v y j) k) (fun j => P.ordered.right (v y j) k) (i+1)
    (fun j hj => by rw [equal j hj]; exact ⟨rfl, rfl⟩)
  change (_ + orderedPhaseIntegral _ _ _) + P.ordered.transient (v x i) k = _
  rw [additive, pairs, equal i (by omega)]

example (P : SharedBilinearRead 128 1 32) (x fill : ℕ → EucSpace 128) :
    P.latent 1 x 15 = P.latent 1 (fun j => if j ≤ 15 then x j else fill j) 15 :=
  P.latent_congr_prefix _ _ _ _ (fun j hj => by simp [hj])

/-- All recurrent and local projections, the learned frequencies and the
contextual branch depend only on the observed embedding prefix. -/
theorem EmbeddingBackbone.sharedBilinearFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : SharedBilinearRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.sharedBilinearFeature P θ eps scale x l i = A.sharedBilinearFeature P θ eps scale y l i := by
  simp only [EmbeddingBackbone.sharedBilinearFeature, EmbeddingBackbone.feature,
    P.latent_congr_prefix eps x y i h, A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (P : SharedBilinearRead 128 8 32)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.sharedBilinearFeature P θ 1 1 x l 37 =
      A.sharedBilinearFeature P θ 1 1 (fun j => if j ≤ 37 then x j else fill j) l 37 :=
  A.sharedBilinearFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor

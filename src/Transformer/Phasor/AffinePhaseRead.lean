module

public import Transformer.Phasor.InputPhaseRead

/-!
# A learned affine rotation memory on input embeddings

Source: the affine phase memory in `experiments/phasor_value_transport/README.md`.
Each channel rotates its previous complex state and then adds a learned real
write. Both scalars are linear projections of the same normalized input.
The read exposes the imaginary and centered real parts of the new state.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- Rotation of a complex memory channel by a real angle. -/
noncomputable def phaseRotation (a : ℝ) : ℂ := Complex.exp (a * Complex.I)

private theorem phaseRotation_add (a b : ℝ) :
    phaseRotation (a+b) = phaseRotation a * phaseRotation b := by
  simp [phaseRotation, add_mul, Complex.exp_add]

private theorem phaseRotation_cancel (a : ℝ) : phaseRotation a * phaseRotation (-a) = 1 := by
  rw [← phaseRotation_add]
  simp [phaseRotation]

/-- The first `n` projected angle/write pairs applied to initial state `z`. -/
noncomputable def affinePhaseState (a b : ℕ → ℝ) (z : ℂ) : ℕ → ℂ
  | 0 => z
  | n+1 => phaseRotation (a n) * affinePhaseState a b z n + (b n : ℂ)

/-- The moving-frame prefix sum implements the ordered recurrence using
one accumulated angle and one accumulated complex sum per channel. -/
theorem affinePhaseState_closed (a b : ℕ → ℝ) (z : ℂ) (n : ℕ) :
    affinePhaseState a b z n = phaseRotation (prefixPhase a n) *
      (z + ∑ j ∈ Finset.range n, phaseRotation (-prefixPhase a (j+1)) * (b j : ℂ)) := by
  induction n with
  | zero => simp [affinePhaseState, prefixPhase, phaseRotation]
  | succ n ih =>
    have step : phaseRotation (a n) * phaseRotation (prefixPhase a n) =
        phaseRotation (prefixPhase a (n+1)) := by
      rw [show prefixPhase a (n+1) = prefixPhase a n + a n from Finset.sum_range_succ ..,
        phaseRotation_add, mul_comm]
    rw [affinePhaseState, ih, ← mul_assoc, step, Finset.sum_range_succ]
    linear_combination -(b n : ℂ) * phaseRotation_cancel (prefixPhase a (n+1))

/-- Learned writes can change the stored value arbitrarily. Under a common
suffix, the distance between any two incoming states stays exactly constant. -/
theorem affinePhaseState_distance (a b : ℕ → ℝ) (z w : ℂ) (n : ℕ) :
    ‖affinePhaseState a b z n - affinePhaseState a b w n‖ = ‖z-w‖ := by
  rw [affinePhaseState_closed, affinePhaseState_closed, ← mul_sub, norm_mul]
  simp [phaseRotation]

/-- Exchanging two learned writes has an explicit order-dependent effect.
The initial state cancels for every pair of angles and writes. -/
theorem affinePhase_exchange (a b u v : ℝ) (z : ℂ) :
    (phaseRotation b * (phaseRotation a * z + u) + v) -
      (phaseRotation a * (phaseRotation b * z + v) + u) =
      (phaseRotation b - 1) * u - (phaseRotation a - 1) * v := by ring

/-- A half-turn and a unit translation give different outputs in reverse order. -/
theorem affinePhase_order_witness (z : ℂ) :
    (phaseRotation Real.pi * (z + 1)) - (phaseRotation Real.pi * z + 1) = -2 := by
  simp [phaseRotation]
  ring

/-- Prefix agreement of the two projections gives equality of ordered states. -/
theorem affinePhaseState_congr_prefix (a b c d : ℕ → ℝ) (z : ℂ) (n : ℕ)
    (h : ∀ j < n, a j = c j ∧ b j = d j) :
    affinePhaseState a b z n = affinePhaseState c d z n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    obtain ⟨ha, hb⟩ := h n (by omega)
    rw [affinePhaseState, affinePhaseState, ha, hb, ih (fun j hj => h j (by omega))]

example (a b : ℕ → ℝ) (z : ℂ) (n : ℕ) :
    affinePhaseState a b z n =
      affinePhaseState (fun j => if j < n then a j else 1)
        (fun j => if j < n then b j else -2) z n :=
  affinePhaseState_congr_prefix _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- Normalized inputs supply the two learned projections. The output maps
and contextual residual are shared with the embedding-input phase read. -/
noncomputable def EmbeddingBackbone.affinePhaseFeature {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m)
    (W : EucSpace D →ₗ[ℝ] (Fin m → ℝ)) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let v := fun j => GPTMini.rmsNormEps eps (x j)
  let z := fun k => affinePhaseState (fun j => P.increments (v j) k) (fun j => W (v j) k) 1 (i+1)
  A.feature θ eps scale x l i + (P.sineOut (fun k => (z k).im) +
    P.cosineOut (fun k => (z k).re-1))

/-- Zero learned writes recover the previously tested input-phase model at
all depths, including with active learned output maps and arbitrary embeddings. -/
theorem EmbeddingBackbone.affinePhaseFeature_zero_write {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.affinePhaseFeature P 0 θ eps scale x l i = A.inputPhaseFeature P θ eps scale x l i := by
  simp [EmbeddingBackbone.affinePhaseFeature, EmbeddingBackbone.inputPhaseFeature,
    PhasePrefixRead.angleResidual, PhasePrefixRead.inputAngles, affinePhaseState_closed, phaseRotation]

/-- The complete ordered read and contextual path are causal for sequences
of arbitrary dense vectors, at any layer and output position. -/
theorem EmbeddingBackbone.affinePhaseFeature_congr_prefix {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m)
    (W : EucSpace D →ₗ[ℝ] (Fin m → ℝ)) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.affinePhaseFeature P W θ eps scale x l i = A.affinePhaseFeature P W θ eps scale y l i := by
  have state (k : Fin m) := affinePhaseState_congr_prefix
    (fun j => P.increments (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => W (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => P.increments (GPTMini.rmsNormEps eps (y j)) k)
    (fun j => W (GPTMini.rmsNormEps eps (y j)) k) 1 (i+1)
    (fun j hj => by rw [h j (by omega)]; exact ⟨rfl, rfl⟩)
  simp only [EmbeddingBackbone.affinePhaseFeature, EmbeddingBackbone.feature, state,
    A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (W : EucSpace 128 →ₗ[ℝ] (Fin 32 → ℝ)) (θ : StreamReadWeights 128)
    (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.affinePhaseFeature P W θ 1 1 x l 3 =
      A.affinePhaseFeature P W θ 1 1 (fun j => if j ≤ 3 then x j else fill j) l 3 :=
  A.affinePhaseFeature_congr_prefix _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- The one-step pullback rotates the incoming cotangent back, sends its
real coordinate to the write, and contracts the rotated state with the
angular derivative. All four input directions vary simultaneously. -/
theorem affinePhase_step_adjoint (a b u v da db du dv gr gi : ℝ) :
    HasDerivAt (fun t : ℝ =>
      gr * (Real.cos (a+t*da) * (u+t*du) - Real.sin (a+t*da) * (v+t*dv) + (b+t*db)) +
      gi * (Real.sin (a+t*da) * (u+t*du) + Real.cos (a+t*da) * (v+t*dv)))
      ((gr*Real.cos a + gi*Real.sin a)*du + (gi*Real.cos a - gr*Real.sin a)*dv +
       gr*db + (gi*(Real.cos a*u-Real.sin a*v) - gr*(Real.sin a*u+Real.cos a*v))*da) 0 := by
  have line (x dx : ℝ) : HasDerivAt (fun t : ℝ => x+t*dx) dx 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dx).const_add x
  have hs := (Real.hasDerivAt_sin (a+0*da)).comp 0 (line a da)
  have hc := (Real.hasDerivAt_cos (a+0*da)).comp 0 (line a da)
  have realPart := (((hc.mul (line u du)).sub (hs.mul (line v dv))).add (line b db)).const_mul gr
  have imagPart := ((hs.mul (line u du)).add (hc.mul (line v dv))).const_mul gi
  convert! realPart.add imagPart using 1
  simp
  ring

end Transformer.Phasor

module

public import Transformer.Phasor.TwoSidedFrame

/-!
# A one-parameter orbit of conditional operators with a shared center

Source: “Single-coordinate frame orbit” in `experiments/phasor_value_transport/README.md`.
The same strict-prefix scalar drives the two rotation banks with independent
learned frequencies. Another scalar dilates the result around a single learned
point shared by every prefix. This removes the unrestricted prefix bias.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F : ℕ}

/-- Act on an arbitrary vector-valued function by rotating both its input and
output. The two learned banks share the same displacement parameter. -/
noncomputable def frameOrbitAction (ω ν : Fin (D/2) → ℝ) (t : ℝ)
    (f : EucSpace D → EucSpace D) (x : EucSpace D) : EucSpace D :=
  frameRotation ν t (f (frameRotation ω t x))

/-- These transformations compose by addition, even for nonlinear functions
and unrelated input and output frequencies. -/
theorem frameOrbitAction_add (ω ν : Fin (D/2) → ℝ) (s t : ℝ)
    (f : EucSpace D → EucSpace D) :
    frameOrbitAction ω ν s (frameOrbitAction ω ν t f) = frameOrbitAction ω ν (s+t) f := by
  funext x
  simp only [frameOrbitAction, frameRotation_add, add_comm t s]

/-- One learned center can serve as the constant output for any zero-gain
prefix. The two frequency banks have no dependence on the vocabulary. -/
structure SingleFrameOrbitRead (D : ℕ) where
  middle : EucSpace D →ₗ[ℝ] EucSpace D
  inputFrequency : Fin (D/2) → ℝ
  outputFrequency : Fin (D/2) → ℝ
  position : EucSpace D →ₗ[ℝ] ℝ
  gain : EucSpace D →ₗ[ℝ] ℝ
  center : EucSpace D

/-- Expanding the shared coordinate and center into an ordinary two-frame
model duplicates one coordinate and gives its bias matrix rank at most one. -/
noncomputable def SingleFrameOrbitRead.asFrame (P : SingleFrameOrbitRead D) : TwoSidedFrameRead D :=
  ⟨P.middle, P.inputFrequency, P.outputFrequency, P.position, P.position, P.gain,
    -(P.gain.smulRight P.center)⟩

/-- The expanded affine action is precisely a dilation about the same point
for all prefixes and current inputs. -/
theorem SingleFrameOrbitRead.centered (P : SingleFrameOrbitRead D) (h x : EucSpace D) :
    P.asFrame.read h x = P.center + (1 + P.gain h) •
      (frameOrbitAction P.inputFrequency P.outputFrequency (P.position h) P.middle x - P.center) := by
  simp only [asFrame, TwoSidedFrameRead.read, twoSidedFrame, frameOrbitAction,
    LinearMap.neg_apply, LinearMap.smulRight_apply, smul_sub, add_smul, one_smul]
  abel

/-- Undoing a nonzero dilation recovers the common operator in its current
input and output frames. -/
theorem SingleFrameOrbitRead.undo_gain (P : SingleFrameOrbitRead D) (h x : EucSpace D)
    (hg : 1 + P.gain h ≠ 0) :
    P.center + (1 + P.gain h)⁻¹ • (P.asFrame.read h x - P.center) =
      frameOrbitAction P.inputFrequency P.outputFrequency (P.position h) P.middle x := by
  rw [P.centered, add_sub_cancel_left, smul_smul, inv_mul_cancel₀ hg, one_smul, add_sub_cancel]

example (P : SingleFrameOrbitRead 128) (x : EucSpace 128) :
    P.center + (1 + P.gain 0)⁻¹ • (P.asFrame.read 0 x - P.center) =
      frameOrbitAction P.inputFrequency P.outputFrequency (P.position 0) P.middle x :=
  P.undo_gain 0 x (by simp)

/-- A single displacement transports the entire normalized response between
two arbitrary prefixes. Both scalar gains may change independently. -/
theorem SingleFrameOrbitRead.transport (P : SingleFrameOrbitRead D) (h k x : EucSpace D)
    (hh : 1 + P.gain h ≠ 0) (hk : 1 + P.gain k ≠ 0) :
    let s := P.position k - P.position h
    P.center + (1 + P.gain k)⁻¹ • (P.asFrame.read k x - P.center) =
      frameRotation P.outputFrequency s (P.center + (1 + P.gain h)⁻¹ •
        (P.asFrame.read h (frameRotation P.inputFrequency s x) - P.center)) := by
  dsimp only
  rw [P.undo_gain k _ hk, P.undo_gain h _ hh]
  let s := P.position k - P.position h
  change _ = frameOrbitAction P.inputFrequency P.outputFrequency s
    (frameOrbitAction P.inputFrequency P.outputFrequency (P.position h) P.middle) x
  rw [frameOrbitAction_add, show s + P.position h = P.position k by dsimp [s]; ring]

example (P : SingleFrameOrbitRead 128) (h k x : EucSpace 128) :
    let Q := { P with gain := 0 }
    Q.asFrame.read k x = frameRotation Q.outputFrequency (Q.position k - Q.position h)
      (Q.asFrame.read h (frameRotation Q.inputFrequency (Q.position k - Q.position h) x)) := by
  dsimp only
  have h' := ({P with gain := 0} : SingleFrameOrbitRead 128).transport h k x (by simp) (by simp)
  simpa using h'

/-- The gain, center and transformed current value all receive gradients
from the centered affine read. This pairs their complete variation with an
arbitrary upstream real vector. -/
theorem frameCenter_adjoint (p v dp dv g : EucSpace D) (a da : ℝ) :
    inner ℝ g (dp + da • (v-p) + a • (dv-dp)) =
      inner ℝ ((1-a) • g) dp + inner ℝ (a • g) dv + da * inner ℝ g (v-p) := by
  simp only [inner_add_right, inner_sub_right, real_inner_smul_left, real_inner_smul_right]
  ring

/-- Restricting the frame parameters preserves the causal feature on every
prefix of arbitrary dense embeddings. -/
theorem EmbeddingBackbone.singleFrameOrbit_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : SingleFrameOrbitRead D) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.twoFrameFeature P.asFrame θ eps scale x l i = A.twoFrameFeature P.asFrame θ eps scale y l i :=
  A.twoFrameFeature_congr_prefix _ _ _ _ _ _ _ _ h

example (A : EmbeddingBackbone 128 32 4 171) (P : SingleFrameOrbitRead 128)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.twoFrameFeature P.asFrame θ 1 1 x l 19 =
      A.twoFrameFeature P.asFrame θ 1 1 (fun j => if j ≤ 19 then x j else fill j) l 19 :=
  A.singleFrameOrbit_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor

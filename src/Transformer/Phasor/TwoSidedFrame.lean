module

public import Transformer.Phasor.ContextRotation

/-!
# One learned operator between two prefix-conditioned rotating frames

Source: “Two-sided frame read” in `experiments/phasor_value_transport/README.md`.
Two scalar coordinates from the strict prefix rotate the input and output
of one shared real-linear map. Separate learned frequencies govern the two
frames. A scalar gain and a vector bias complete the conditional affine map.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer SequenceModel

variable {D d H F : ℕ}

/-- A single prefix coordinate drives every half-split pair through its own
learned frequency. The action is defined for all real coordinates. -/
noncomputable def frameRotation (ω : Fin (D/2) → ℝ) (u : ℝ) (x : EucSpace D) : EucSpace D :=
  rotateHead D (fun j => ω j * u) x

/-- Adding scalar displacements composes the two learned rotations. -/
theorem frameRotation_add (ω : Fin (D/2) → ℝ) (s t : ℝ) (x : EucSpace D) :
    frameRotation ω s (frameRotation ω t x) = frameRotation ω (s+t) x := by
  simp only [frameRotation, rotateHead_add, mul_add]

/-- The shared operator uses the same weights for every input and output
frame position. All operands are continuous hidden features. -/
noncomputable def twoSidedFrame (L : EucSpace D →ₗ[ℝ] EucSpace D)
    (ω ν : Fin (D/2) → ℝ) (a b : ℝ) (x : EucSpace D) : EucSpace D :=
  frameRotation ν b (L (frameRotation ω a x))

/-- Moving between two conditional operators needs only one input and one
output rotation. The shared map need not commute with either rotation. -/
theorem twoSidedFrame_transport (L : EucSpace D →ₗ[ℝ] EucSpace D)
    (ω ν : Fin (D/2) → ℝ) (a b s t : ℝ) (x : EucSpace D) :
    twoSidedFrame L ω ν (a+s) (b+t) x =
      frameRotation ν t (twoSidedFrame L ω ν a b (frameRotation ω s x)) := by
  simp only [twoSidedFrame, frameRotation_add, add_comm t b]

/-- Every unscaled conditional operator attains the same output norms on
the unit sphere, even when the two frames use unrelated learned frequencies. -/
theorem twoSidedFrame_unit_range (L : EucSpace D →ₗ[ℝ] EucSpace D)
    (ω ν : Fin (D/2) → ℝ) (a b r : ℝ) :
    (∃ x : EucSpace D, ‖x‖ = 1 ∧ ‖twoSidedFrame L ω ν a b x‖ = r) ↔
      ∃ y : EucSpace D, ‖y‖ = 1 ∧ ‖L y‖ = r := by
  constructor
  · rintro ⟨x, hx, hr⟩
    exact ⟨frameRotation ω a x, (rotateHead_isometry ..).trans hx,
      (rotateHead_isometry ..).symm.trans hr⟩
  · rintro ⟨y, hy, hr⟩
    refine ⟨frameRotation ω (-a) y, (rotateHead_isometry ..).trans hy, ?_⟩
    rw [twoSidedFrame, frameRotation_add, add_neg_cancel]
    simp only [frameRotation, mul_zero, rotateHead_zero_angles, rotateHead_isometry, hr]

/-- Learned context maps supply two frame positions, one gain adjustment
and an additive output. The only shared linear map acts between the frames. -/
structure TwoSidedFrameRead (D : ℕ) where
  middle : EucSpace D →ₗ[ℝ] EucSpace D
  inputFrequency : Fin (D/2) → ℝ
  outputFrequency : Fin (D/2) → ℝ
  inputPosition : EucSpace D →ₗ[ℝ] ℝ
  outputPosition : EucSpace D →ₗ[ℝ] ℝ
  gain : EucSpace D →ₗ[ℝ] ℝ
  bias : EucSpace D →ₗ[ℝ] EucSpace D

/-- One affine action on the normalized embedding, selected by the strict prefix. -/
noncomputable def TwoSidedFrameRead.read (P : TwoSidedFrameRead D) (h x : EucSpace D) :
    EucSpace D :=
  P.bias h + (1 + P.gain h) • twoSidedFrame P.middle P.inputFrequency P.outputFrequency
    (P.inputPosition h) (P.outputPosition h) x

/-- Subtracting the prefix bias leaves one global gain on the norm of the
shared operator applied in its input frame. -/
theorem TwoSidedFrameRead.centered_norm (P : TwoSidedFrameRead D) (h x : EucSpace D) :
    ‖P.read h x - P.bias h‖ =
      |1 + P.gain h| * ‖P.middle (frameRotation P.inputFrequency (P.inputPosition h) x)‖ := by
  simp [TwoSidedFrameRead.read, twoSidedFrame, frameRotation, rotateHead_isometry,
    norm_smul, Real.norm_eq_abs]

/-- At the first position the conditional operator is precisely the shared map. -/
theorem TwoSidedFrameRead.read_zero (P : TwoSidedFrameRead D) (x : EucSpace D) :
    P.read 0 x = P.middle x := by
  simp [TwoSidedFrameRead.read, twoSidedFrame, frameRotation, rotateHead_zero_angles]

/-- Varying a frame coordinate and a learned frequency simultaneously
multiplies the rotation's signed-area derivative by their product rule. -/
theorem frameRotation_frequency_deriv (a ω da dω x y g h : ℝ) :
    let θ := a*ω
    HasDerivAt (fun t : ℝ =>
      let z := (a+t*da)*(ω+t*dω)
      g*(x*Real.cos z-y*Real.sin z) + h*(x*Real.sin z+y*Real.cos z))
      (((x*Real.cos θ-y*Real.sin θ)*h-(x*Real.sin θ+y*Real.cos θ)*g) * (da*ω+a*dω)) 0 := by
  dsimp only
  have line (k b : ℝ) : HasDerivAt (fun t : ℝ => k+t*b) b 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add k
  have prod := (line a da).fun_mul (line ω dω)
  simp only [zero_mul, add_zero] at prod
  have c := prod.cos
  have s := prod.sin
  convert! (((c.const_mul x).sub (s.const_mul y)).const_mul g).add
    (((s.const_mul x).add (c.const_mul y)).const_mul h) using 1
  simp only [zero_mul, add_zero]
  ring

/-- A nonempty strict prefix supplies the previous Phasor context. The same
frame, gain and bias projections are used at every sequence length. -/
noncomputable def EmbeddingBackbone.twoFrameFeature (A : EmbeddingBackbone D d H F)
    (P : TwoSidedFrameRead D) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let previous := match i with
    | 0 => 0
    | j+1 => A.feature θ eps scale x l j
  P.read previous (GPTMini.rmsNormEps eps (x i))

/-- Equality of dense prefixes suffices at every depth and position. -/
theorem EmbeddingBackbone.twoFrameFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : TwoSidedFrameRead D) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.twoFrameFeature P θ eps scale x l i = A.twoFrameFeature P θ eps scale y l i := by
  cases i with
  | zero => simp [EmbeddingBackbone.twoFrameFeature, h 0 (by omega)]
  | succ i =>
    simp only [EmbeddingBackbone.twoFrameFeature, EmbeddingBackbone.feature, h (i+1) (by omega),
      A.states_congr_prefix eps scale x y l i fun j hj => h j (by omega)]

example (A : EmbeddingBackbone 128 32 4 171) (P : TwoSidedFrameRead 128)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.twoFrameFeature P θ 1 1 x l 31 =
      A.twoFrameFeature P θ 1 1 (fun j => if j ≤ 31 then x j else fill j) l 31 :=
  A.twoFrameFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor

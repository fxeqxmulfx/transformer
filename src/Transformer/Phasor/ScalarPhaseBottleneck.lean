module

public import Transformer.Phasor.SharedBilinearPhase

/-!
# A scalar periodic bottleneck for a contextual embedding sequence

Source: “Scalar phase bottleneck” in `experiments/phasor_value_transport/README.md`.
One projected contextual coordinate and one shared bilinear state form the entire
input to the learned frequency bank. All hidden features passed to the external
classifier factor through this scalar. Each prefix may have any finite length.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F m : ℕ}

/-- The same four dense input projections are used throughout the sequence.
The contextual vector enters through its own learned scalar projection. -/
structure ScalarPhaseRead (D m : ℕ) where
  memory : SharedBilinearRead D 1 m
  context : EucSpace D →ₗ[ℝ] ℝ

/-- A whole prefix has one scalar coordinate combining the ordered state,
the local input and the contextual feature. -/
noncomputable def EmbeddingBackbone.phaseCoordinate (A : EmbeddingBackbone D d H F)
    (P : ScalarPhaseRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : ℝ :=
  P.memory.latent eps x i 0 + P.context (A.feature θ eps scale x l i)

/-- Arbitrary learned frequencies and output projections decode the scalar. -/
noncomputable def ScalarPhaseRead.decode (P : ScalarPhaseRead D m) (z : ℝ) : EucSpace D :=
  P.memory.sineOut (fun j => Real.sin (P.memory.frequency j 0 * z)) +
    P.memory.cosineOut (fun j => Real.cos (P.memory.frequency j 0 * z)-1)

/-- The only path to the external classifier is the learned periodic decoder. -/
noncomputable def EmbeddingBackbone.scalarPhaseFeature (A : EmbeddingBackbone D d H F)
    (P : ScalarPhaseRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  P.decode (A.phaseCoordinate P θ eps scale x l i)

/-- All logits agree on two arbitrary prefixes of different lengths whenever
their learned scalar coordinates agree, even if the contextual features differ. -/
theorem EmbeddingBackbone.scalarPhase_logit_fiber (A : EmbeddingBackbone D d H F)
    (P : ScalarPhaseRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i j : ℕ) (score : EucSpace D →ₗ[ℝ] ℝ)
    (h : A.phaseCoordinate P θ eps scale x l i = A.phaseCoordinate P θ eps scale y l j) :
    score (A.scalarPhaseFeature P θ eps scale x l i) = score (A.scalarPhaseFeature P θ eps scale y l j) :=
  congrArg (fun z => score (P.decode z)) h

example (A : EmbeddingBackbone 128 32 4 171) (P : ScalarPhaseRead 128 32)
    (θ : StreamReadWeights 128) (x y : ℕ → EucSpace 128) (score : EucSpace 128 →ₗ[ℝ] ℝ) :
    let Q : ScalarPhaseRead 128 32 :=
      { P with
        context := 0
        memory := { P.memory with
          additive := 0
          ordered := { P.memory.ordered with right := 0, transient := 0 } } }
    score (A.scalarPhaseFeature Q θ 1 1 x 1 2) = score (A.scalarPhaseFeature Q θ 1 1 y 1 9) := by
  dsimp only
  apply A.scalarPhase_logit_fiber
  simp [EmbeddingBackbone.phaseCoordinate, SharedBilinearRead.latent,
    sharedBilinearState, prefixPhase, orderedPhaseIntegral]

/-- The contextual and recurrent contributions to the bottleneck are causal
on arbitrary embedding vectors. -/
theorem EmbeddingBackbone.scalarPhaseFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : ScalarPhaseRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.scalarPhaseFeature P θ eps scale x l i = A.scalarPhaseFeature P θ eps scale y l i := by
  simp only [EmbeddingBackbone.scalarPhaseFeature, EmbeddingBackbone.phaseCoordinate,
    EmbeddingBackbone.feature, P.memory.latent_congr_prefix eps x y i h,
    A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (P : ScalarPhaseRead 128 32)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.scalarPhaseFeature P θ 1 1 x l 11 =
      A.scalarPhaseFeature P θ 1 1 (fun j => if j ≤ 11 then x j else fill j) l 11 :=
  A.scalarPhaseFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- The scalar selected by an external linear classifier, expressed in
the learned observation coordinates. -/
noncomputable def bottleneckScore (ω a b : Fin m → ℝ) (z : ℝ) : ℝ :=
  ∑ j, (a j * Real.sin (ω j*z) + b j * (Real.cos (ω j*z)-1))

/-- All frequency channels contribute to one derivative at the bottleneck.
An arbitrary incoming perturbation scales that entire derivative. -/
theorem bottleneckScore_deriv (ω a b : Fin m → ℝ) (z v : ℝ) :
    HasDerivAt (fun t : ℝ => bottleneckScore ω a b (z+t*v))
      ((∑ j, (a j * Real.cos (ω j*z) - b j * Real.sin (ω j*z)) * ω j) * v) 0 := by
  unfold bottleneckScore
  rw [Finset.sum_mul]
  apply HasDerivAt.fun_sum
  intro j _
  have h := phase_pair_adjoint (ω j*z) (ω j*v) (a j) (b j)
  convert! h using 1
  · funext t
    rw [show ω j * (z+t*v) = ω j*z+t*(ω j*v) by ring]
  · ring

/-- Every two-by-two minor of the directional logit Jacobian vanishes.
The scores may use unrelated readout coefficients and the angles, number
of observations and the two scalar directions are arbitrary. -/
theorem bottleneckScore_rank_one (ω a b c e : Fin m → ℝ) (z u v : ℝ) :
    deriv (fun t => bottleneckScore ω a b (z+t*u)) 0 * deriv (fun t => bottleneckScore ω c e (z+t*v)) 0 =
    deriv (fun t => bottleneckScore ω a b (z+t*v)) 0 * deriv (fun t => bottleneckScore ω c e (z+t*u)) 0 := by
  rw [(bottleneckScore_deriv ..).deriv, (bottleneckScore_deriv ..).deriv,
    (bottleneckScore_deriv ..).deriv, (bottleneckScore_deriv ..).deriv]
  ring

end Transformer.Phasor

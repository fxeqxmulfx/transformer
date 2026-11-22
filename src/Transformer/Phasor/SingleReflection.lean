module

public import Transformer.Phasor.ReflectionPhaseRead

/-!
# One signed phase state per channel

Source: the single-reflection experiment in `experiments/phasor_value_transport/README.md`.
At every position the dense input embedding selects the reflection direction.
Keeping only the reflected state removes the rotation, mixing parameters and
their gradients. These equalities hold before the classifier, at arbitrary width.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- Across an even number of embeddings the state displacement is the sum
of adjacent direction differences. This also shows why one angle suffices. -/
theorem reflectedPhase_even_pairs (a : ℕ → ℝ) (z : ℝ) (n : ℕ) :
    reflectedPhase a z (2*n) =
      z + ∑ j ∈ Finset.range n, (a (2*j+1) - a (2*j)) := by
  induction n with
  | zero => simp [reflectedPhase]
  | succ n ih =>
    rw [show 2*(n+1) = (2*n+1)+1 by omega, reflectedPhase, reflectedPhase, ih,
      Finset.sum_range_succ]
    ring

/-- Every input direction, including a perturbation of the initial state,
travels through the same single-state update as the original angles. -/
theorem reflectedPhase_perturb (a da : ℕ → ℝ) (z dz t : ℝ) (n : ℕ) :
    reflectedPhase (fun j => a j + t*da j) (z+t*dz) n =
      reflectedPhase a z n + t * reflectedPhase da dz n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [reflectedPhase, ih]
    ring

/-- Pullback through a complete sine and centered-cosine observation after
simultaneously perturbing every increment and the incoming phase. -/
theorem singleReflection_read_deriv (a da : ℕ → ℝ) (z dz gs gc : ℝ) (n : ℕ) :
    let angle := fun t => reflectedPhase (fun j => a j + t*da j) (z+t*dz) n
    HasDerivAt (fun t => gs*Real.sin (angle t) + gc*(Real.cos (angle t)-1))
      ((gs*Real.cos (reflectedPhase a z n) - gc*Real.sin (reflectedPhase a z n)) *
        reflectedPhase da dz n) 0 := by
  simpa only [reflectedPhase_perturb] using
    phase_pair_adjoint (reflectedPhase a z n) (reflectedPhase da dz n) gs gc

/-- The ordinary contextual read plus the observation of one state per
learned input projection. Its parameter shapes are those of the additive scan. -/
noncomputable def EmbeddingBackbone.singleReflectionFeature {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  A.feature θ eps scale x l i + P.angleResidual
    (fun k => reflectedPhase (fun j => P.increments (GPTMini.rmsNormEps eps (x j)) k) 0 (i+1))

/-- Freezing every mixed gate to one gives precisely this single-state
feature, for all output maps, early and late positions and contextual depths. -/
theorem EmbeddingBackbone.singleReflectionFeature_eq_one_gate {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.singleReflectionFeature P θ eps scale x l i =
      A.reflectionPhaseFeature P (fun _ => 1) θ eps scale x l i := by
  have endpoint (u v : ℝ) : phaseMix 1 u v = v := by unfold phaseMix; ring
  simp only [EmbeddingBackbone.singleReflectionFeature, EmbeddingBackbone.reflectionPhaseFeature,
    PhasePrefixRead.angleResidual, endpoint]

/-- Arbitrary changes after a position preserve its complete single-state
feature and its contextual read. -/
theorem EmbeddingBackbone.singleReflectionFeature_congr_prefix {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.singleReflectionFeature P θ eps scale x l i = A.singleReflectionFeature P θ eps scale y l i := by
  rw [A.singleReflectionFeature_eq_one_gate, A.singleReflectionFeature_eq_one_gate]
  exact A.reflectionPhaseFeature_congr_prefix _ _ _ _ _ _ _ _ _ h

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.singleReflectionFeature P θ 1 1 x l 5 =
      A.singleReflectionFeature P θ 1 1 (fun j => if j ≤ 5 then x j else fill j) l 5 :=
  A.singleReflectionFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor

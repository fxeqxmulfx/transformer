module

public import Transformer.Phasor.Forward

/-!
# A learned readout of Phasor's final four streams

Source: the learned stream readout in `experiments/phasor_value_transport/README.md`.
Each hidden coordinate learns four coefficients, shared across positions and
vocabulary items. Inputs enter Phasor through its embedding and outputs use
that same embedding. The stream readout receives only contextual hidden states.

The zero offsets recover the existing mean at every depth. The coordinate
expansion gives the first-order variation and the exact second-order remainder.
Its adjoint sums the parameter gradient over every active training row.
Relabelling the entire vocabulary commutes with all backbone layers and logits.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC.PeriRMS Transformer.Residual

variable {D d H F V W : ℕ}

/-- Trainable offsets from the original quarter weight on the four final
streams. The same array is used for every input sequence. -/
abbrev StreamReadWeights (D : ℕ) := Fin 4 → Fin D → ℝ

/-- The hidden vector immediately before the final RMS and tied projection. -/
noncomputable def learnedCollapse (θ : StreamReadWeights D) (x : State D) : EucSpace D :=
  collapse x + WithLp.toLp 2 (fun c => ∑ s, θ s c * x s c)

/-- The readout coefficient of a stream and a channel is its original
quarter plus the corresponding learned offset. -/
theorem learnedCollapse_apply (θ : StreamReadWeights D) (x : State D) (c : Fin D) :
    learnedCollapse θ x c = ∑ s, (1 / 4 + θ s c) * x s c := by
  simp [learnedCollapse, collapse, weightedSum, WithLp.ofLp_sum, Finset.sum_apply,
    add_mul, Finset.sum_add_distrib, smul_eq_mul]

/-- Initialization recovers Phasor's original output on every stream state. -/
theorem learnedCollapse_zero (x : State D) : learnedCollapse 0 x = collapse x := by
  apply PiLp.ext
  intro c
  simp [learnedCollapse]

/-- Opposite hidden signals cancel in the original mean. A possible learned
readout recovers their entire first signal, in every dimension. -/
theorem learnedCollapse_opposite (h : EucSpace D) :
    let x : State D := ![h, -h, 0, 0]
    collapse x = 0 ∧ learnedCollapse (fun s _ => if s = 0 then 1 else 0) x = h := by
  dsimp only
  have hz : collapse (![h, -h, 0, 0] : State D) = 0 := by
    simp [collapse, weightedSum, Fin.sum_univ_four]
  refine ⟨hz, ?_⟩
  apply PiLp.ext
  intro c
  simp [learnedCollapse, hz]

/-- The exact simultaneous perturbation in one channel. Its last term is
quadratic in the step and vanishes from the backpropagated derivative. -/
theorem learnedCollapse_perturb (θ δ : StreamReadWeights D) (x dx : State D)
    (c : Fin D) (t : ℝ) :
    learnedCollapse (fun s k => θ s k + t * δ s k) (fun s => x s + t • dx s) c =
      learnedCollapse θ x c +
      t * (∑ s, ((1 / 4 + θ s c) * dx s c + δ s c * x s c)) +
      t ^ 2 * ∑ s, δ s c * dx s c := by
  simp only [learnedCollapse_apply, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul,
    Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro s _
  ring

/-- For each channel the adjoint multiplies a row's upstream gradient by
each stream coefficient and sums its outer product with the states over rows. -/
theorem learnedCollapse_adjoint {n : ℕ} (θ δ : Fin 4 → ℝ)
    (x dx : Fin n → Fin 4 → ℝ) (g : Fin n → ℝ) :
    ∑ i, g i * (∑ s, ((1 / 4 + θ s) * dx i s + δ s * x i s)) =
      (∑ i, ∑ s, (g i * (1 / 4 + θ s)) * dx i s) +
      ∑ s, (∑ i, g i * x i s) * δ s := by
  simp only [Finset.mul_sum, mul_add, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm (f := fun s i => g i * x i s * δ s)]
  congr 1 <;> apply Finset.sum_congr rfl <;> intro i _ <;>
    apply Finset.sum_congr rfl <;> intro s _ <;> ring

/-- Reindex the embedding rows using a bijection of vocabulary names. -/
def Model.renameTokens (M : Model D d H F V) (σ : Fin V ≃ Fin W) : Model D d H F W :=
  { embedding := fun v => M.embedding (σ.symm v), alpha := M.alpha, layer := M.layer }

/-- At arbitrary depth the renamed backbone has the same hidden states when
given the renamed input, including all context read by causal attention. -/
theorem Model.states_renameTokens (M : Model D d H F V) (σ : Fin V ≃ Fin W)
    (eps scale : ℝ) (tokens : ℕ → Fin V) (l : ℕ) :
    (M.renameTokens σ).states eps scale (σ ∘ tokens) l = M.states eps scale tokens l := by
  induction l with
  | zero => funext i; simp [Model.states, Model.renameTokens]
  | succ l ih =>
    funext i
    change (M.layer l).apply eps scale M.alpha ((M.renameTokens σ).states _ _ _ l) i = _
    rw [ih]
    rfl

/-- Apply the learned stream readout at a position of the original backbone,
then use the same final RMS and tied embedding as Phasor. -/
noncomputable def Model.streamReadLogit (M : Model D d H F V) (θ : StreamReadWeights D)
    (eps scale : ℝ) (tokens : ℕ → Fin V) (l i : ℕ) (v : Fin V) : ℝ :=
  inner ℝ (M.embedding v) (GPTMini.rmsNormEps eps (learnedCollapse θ (M.states eps scale tokens l i)))

/-- An arbitrary vocabulary permutation only permutes the logits for every
parameter setting, sequence, position and depth, including unseen symbols. -/
theorem Model.streamReadLogit_renameTokens (M : Model D d H F V) (σ : Fin V ≃ Fin W)
    (θ : StreamReadWeights D) (eps scale : ℝ) (tokens : ℕ → Fin V) (l i : ℕ) (v : Fin V) :
    (M.renameTokens σ).streamReadLogit θ eps scale (σ ∘ tokens) l i (σ v) =
      M.streamReadLogit θ eps scale tokens l i v := by
  simp only [Model.streamReadLogit, Model.states_renameTokens]
  change inner ℝ (M.embedding (σ.symm (σ v))) _ = _
  rw [σ.symm_apply_apply]

/-- The full learned model at initialization has the original tied logits. -/
theorem Model.streamReadLogit_zero (M : Model D d H F V) (eps scale : ℝ)
    (tokens : ℕ → Fin V) (l i : ℕ) (v : Fin V) :
    M.streamReadLogit 0 eps scale tokens l i v = M.logits eps scale tokens l i v := by
  rw [Model.streamReadLogit, learnedCollapse_zero]
  rfl

end Transformer.Phasor

module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters

/-! Recovering the native tensors of the single-stream model in ARGAND.md.
Every complex linear projection is specified by its values on the coordinate
vectors. This lets capacity witnesses be written as layers and then used as
weights in the actual optimizer. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped BigOperators

variable {A E : Type*} {c H d L : ℕ}

private theorem stream_columns [AddCommMonoid E] [Module ℂ E] (f : CSpace c →ₗ[ℂ] E) :
    streamProjection (fun i => f (EuclideanSpace.single i 1)) = f := by
  apply LinearMap.ext
  intro x
  simp only [streamProjection, LinearMap.sum_apply, LinearMap.smulRight_apply,
    PiLp.projₗ_apply, ← f.map_smul, ← map_sum f]
  apply congrArg f
  ext k
  change (PiLp.projₗ 2 (𝕜 := ℂ) (fun _ : Fin c => ℂ) k) (∑ i, x i • EuclideanSpace.single i (1 : ℂ)) = x k
  rw [map_sum]
  simp [EuclideanSpace.single, PiLp.single_apply]

private theorem vector_columns {I : Type*} [Fintype I] [DecidableEq I]
    [AddCommMonoid E] [Module ℂ E] (f : (I → ℂ) →ₗ[ℂ] E) :
    vectorProjection (fun i => f (Pi.single i 1)) = f := by
  apply LinearMap.ext
  intro x
  simp only [vectorProjection, LinearMap.sum_apply, LinearMap.smulRight_apply,
    LinearMap.proj_apply, ← f.map_smul, ← map_sum f]
  apply congrArg f
  ext k
  simp [Finset.sum_apply, Pi.single_apply]

private theorem head_columns (f : Heads H d →ₗ[ℂ] CSpace c) :
    headProjection (fun h j => f (Pi.single h (Pi.single j 1))) = f := by
  apply LinearMap.ext
  intro x
  simp only [headProjection, vectorProjection, LinearMap.sum_apply, LinearMap.comp_apply,
    LinearMap.smulRight_apply, LinearMap.proj_apply, ← f.map_smul, ← map_sum f]
  apply congrArg f
  ext k t
  simp [Finset.sum_apply, Pi.single_apply, ite_apply]

/-- Read the learned complex coordinates of a layer, including both residual gates. -/
noncomputable def Model.tensorEntry (m : Model A H d L) : ComplexParameter A H d L → ℂ
  | .embedding a i => m.embedding a i
  | .attentionIn l k i h j =>
    let p := (m.layer l).attentionProjections
    (![p.WQ, p.WK, p.WV] k) (EuclideanSpace.single i 1) h j
  | .attentionOut l h j i =>
    (m.layer l).attentionProjections.WO (Pi.single h (Pi.single j 1)) i
  | .feedForwardIn l k i j =>
    let p := (m.layer l).feedForward
    (![p.Wg, p.Wu] k) (EuclideanSpace.single i 1) j
  | .feedForwardOut l j i => (m.layer l).feedForward.WF (Pi.single j 1) i
  | .score l => (m.layer l).scoreMultiplier
  | .residualGate l s k i =>
    let g := (![ (m.layer l).attentionGates, (m.layer l).feedForwardGates] s)
    (![g.skip, g.write] k) (EuclideanSpace.single i 1)

/-- Pack a full model into the independent real coordinates from weights.rs. -/
noncomputable def Model.toWeights (m : Model A H d L) : Weights A H d L
  | .inr j => m.alpha j
  | .inl (p, b) => if b then (m.tensorEntry p).im else (m.tensorEntry p).re

private theorem entry_toWeights (m : Model A H d L) (p : ComplexParameter A H d L) :
    complexEntry m.toWeights p = m.tensorEntry p := Complex.ext rfl rfl

/-- The native forward pass reproduces every specified model at the runner's RMS offset. -/
theorem Model.fromWeights_toWeights (m : Model A H d L) (hε : m.eps = 1 / 1000000) :
    Model.fromWeights m.toWeights = m := by
  have hLayer (l : Fin L) : (Model.fromWeights m.toWeights).layer l = m.layer l := by
    simp only [Model.fromWeights, entry_toWeights, Model.tensorEntry, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_fin_one, stream_columns]
    change Layer.mk ⟨_, _, _, headProjection (fun h j =>
        (m.layer l).attentionProjections.WO (Pi.single h (Pi.single j 1)))⟩
      ⟨_, _, vectorProjection (fun j => (m.layer l).feedForward.WF (Pi.single j 1))⟩ _ _ _ = _
    rw [head_columns, vector_columns]
    rfl
  calc
    Model.fromWeights m.toWeights = ⟨m.embedding, m.layer, m.alpha, 1 / 1000000⟩ := by
      change Model.mk _ _ _ _ = _
      congr 1
      exact funext hLayer
    _ = m := by rw [← hε]

end Transformer.Grokking.FiniteRecovery.Argand

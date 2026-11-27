module

public import Transformer.Grokking.FiniteRecovery.ArgandLayerDifferentiability
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-! The complete real-coordinate batch loss of Argand is C¹. In
particular, vanishing minibatch gradients at a finite limit give actual
stationarity, even at zero activations. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization Transformer.AMSGradW
open scoped InnerProductSpace

variable {A : Type*} [Fintype A] {H d layers T : ℕ}

private theorem c1Lipschitz_hidden_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) :
    C1Lipschitz (fun w : Weights A H d layers => (Model.fromWeights w).hidden repeats word i) := by
  let W := Weights A H d layers
  let ε : ℝ := 1 / 1000000
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  let x (w : W) (j : Fin T) := argandRMS ε ((Model.fromWeights w).embedding (word j))
  let slide (w : W) (z : Word (CSpace (H * d)) T) (l : Fin layers) :=
    ((Model.fromWeights w).layer l).forwardWith ε scale (causalGeometry (Model.fromWeights w).alpha) z
  let pass (w : W) (z : Word (CSpace (H * d)) T) := (List.finRange layers).foldl (slide w) z
  have hrms := c1Lipschitz_argandRMS (c := H * d) (by norm_num : 0 < ε)
  have hx (j : Fin T) : C1Lipschitz (fun w => x w j) :=
    hrms.comp (c1Lipschitz_embedding_fromWeights (word j))
  have hpass (ls : List (Fin layers)) (z : W → Word (CSpace (H * d)) T)
      (hz : ∀ j, C1Lipschitz (fun w => z w j)) (j : Fin T) :
      C1Lipschitz (fun w => ls.foldl (slide w) (z w) j) := by
    induction ls generalizing z with
    | nil => exact hz j
    | cons l ls ih =>
      simp only [List.foldl_cons]
      exact ih (fun w => slide w (z w) l) (c1Lipschitz_forward_fromWeights l scale z hz)
  have hrepeat (n : ℕ) (j : Fin T) : C1Lipschitz (fun w => ((pass w)^[n]) (x w) j) := by
    induction n generalizing j with
    | zero => exact hx j
    | succ n ih =>
      simp only [Function.iterate_succ_apply']
      exact hpass _ _ ih j
  simp only [Model.hidden, Model.hiddenWith, List.ofFn_eq_map, List.foldl_map]
  change C1Lipschitz (fun w : W => argandRMS ε ((pass w)^[repeats] (x w) i))
  exact hrms.comp (hrepeat repeats i)

theorem c1Lipschitz_logits_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    C1Lipschitz (fun w : Weights A H d layers => (Model.fromWeights w).logits repeats word i a) :=
  ((C1Lipschitz.of_contDiff Complex.reCLM.contDiff).comp ((c1Lipschitz_embedding_fromWeights a).inner ℂ
    (c1Lipschitz_hidden_fromWeights repeats word i))).div_const 2

theorem c1Lipschitz_labelLoss_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    C1Lipschitz (fun w : Weights A H d layers =>
      -Real.log ((Model.fromWeights w).probability repeats word i a)) := by
  have hp (t : A) := (c1Lipschitz_logits_fromWeights (H := H) (d := d) (layers := layers) repeats word i t).exp
  have hsum (w : Weights A H d layers) : 0 < ∑ t, Real.exp ((Model.fromWeights w).logits repeats word i t) :=
    Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨a, Finset.mem_univ _, Real.exp_pos _⟩
  have hprob : C1Lipschitz (fun w : Weights A H d layers => (Model.fromWeights w).probability repeats word i a) :=
    (hp a).div (C1Lipschitz.sum fun t _ => hp t) (fun w => (hsum w).ne')
  exact (hprob.log fun w => ((Model.fromWeights w).probability_pos repeats word i a).ne').neg

theorem contDiff_logits_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    ContDiff ℝ 1 (fun w : Weights A H d layers => (Model.fromWeights w).logits repeats word i a) :=
  (c1Lipschitz_logits_fromWeights repeats word i a).contDiff

theorem contDiff_labelLoss_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    ContDiff ℝ 1 (fun w : Weights A H d layers =>
      -Real.log ((Model.fromWeights w).probability repeats word i a)) :=
  (c1Lipschitz_labelLoss_fromWeights repeats word i a).contDiff

theorem c1Lipschitz_batchLoss (repeats : ℕ) (batch : List (LabeledPosition A)) :
    C1Lipschitz (batchLoss (H := H) (d := d) (layers := layers) repeats batch) := by
  unfold batchLoss
  apply C1Lipschitz.div_const
  induction batch with
  | nil => exact (C1Lipschitz.const _)
  | cons p batch ih =>
    simp only [List.map_cons, List.sum_cons]
    exact (c1Lipschitz_labelLoss_fromWeights repeats p.2.1 p.2.2.1 p.2.2.2).add ih

theorem contDiff_batchLoss (repeats : ℕ) (batch : List (LabeledPosition A)) :
    ContDiff ℝ 1 (batchLoss (H := H) (d := d) (layers := layers) repeats batch) :=
  (c1Lipschitz_batchLoss repeats batch).contDiff

/-- The gradient used by the exact-real optimizer is continuous in all
parameters, including values at initialization. -/
theorem continuous_batchGradient (repeats : ℕ) (batch : List (LabeledPosition A)) :
    Continuous (batchGradient (H := H) (d := d) (layers := layers) repeats batch) := by
  let E := Kernite.Space (Parameter A H d layers)
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let f : E → ℝ := (batchLoss repeats batch) ∘ coords
  have hf : ContDiff ℝ 1 f := (contDiff_batchLoss repeats batch).comp coords.contDiff
  have hg : Continuous (gradient f) :=
    (InnerProductSpace.toDual ℝ E).symm.continuous.comp (hf.continuous_fderiv one_ne_zero)
  exact coords.continuous.comp (hg.comp coords.symm.continuous)

example (batch : List (LabeledPosition (Fin 16))) :=
  continuous_batchGradient (H := 4) (d := 16) (layers := 2) 2 batch

end Transformer.Grokking.FiniteRecovery.Argand

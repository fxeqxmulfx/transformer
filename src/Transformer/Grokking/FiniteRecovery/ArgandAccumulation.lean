module

public import Transformer.Grokking.FiniteRecovery.ArgandBatchDifferentiability
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-! Gradient accumulation at a single set of weights. Each subbatch
uses a mean over its labeled positions, so its contribution to the
numerator must be multiplied by that number of positions. The full
objective adds a quadratic penalty once across all trainable coordinates. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.AMSGradW Transformer.Optimization

variable {A : Type*} [Fintype A] {H d layers : ℕ}

abbrev WeightSpace (A : Type*) [Fintype A] (H d layers : ℕ) :=
  Kernite.Space (Parameter A H d layers)

private theorem weighted_batchLoss (repeats : ℕ) (batch : List (LabeledPosition A))
    (w : Weights A H d layers) :
    (batch.length : ℝ) * batchLoss repeats batch w =
      (batch.map fun p => -Real.log
        ((Model.fromWeights w).probability repeats p.2.1 p.2.2.1 p.2.2.2)).sum := by
  cases batch with
  | nil => simp [batchLoss]
  | cons p batch =>
    unfold batchLoss
    have h : ((p :: batch).length : ℝ) ≠ 0 := by simp; positivity
    exact mul_div_cancel₀ _ h

private theorem batchGradient_append_weighted (repeats : ℕ) (b c : List (LabeledPosition A))
    (w : Weights A H d layers) :
    ((b ++ c).length : ℝ) • batchGradient repeats (b ++ c) w =
      (b.length : ℝ) • batchGradient repeats b w + (c.length : ℝ) • batchGradient repeats c w := by
  let E := WeightSpace A H d layers
  let B (s : List (LabeledPosition A)) (x : E) := batchLoss repeats s (WithLp.ofLp x)
  let x : E := WithLp.toLp 2 w
  have hd (s : List (LabeledPosition A)) : DifferentiableAt ℝ (B s) x :=
    ((contDiff_batchLoss repeats s).comp
      (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)).contDiff).differentiable one_ne_zero x
  have hsum :
      (fun x => ((b ++ c).length : ℝ) • B (b ++ c) x) =
        (fun x => (b.length : ℝ) • B b x + (c.length : ℝ) • B c x) := by
    funext x
    simp only [smul_eq_mul, B, weighted_batchLoss, List.map_append, List.sum_append]
  have hlin (s : List (LabeledPosition A)) :
      HasFDerivAt (fun x => (s.length : ℝ) • B s x)
        ((s.length : ℝ) • fderiv ℝ (B s) x) x :=
    (hd s).hasFDerivAt.const_smul (s.length : ℝ)
  have hF := hlin (b ++ c)
  rw [hsum] at hF
  have heq := hF.unique ((hlin b).add (hlin c))
  have hh := congrArg (InnerProductSpace.toDual ℝ E).symm heq
  simp only [map_add, map_smul] at hh
  exact congrArg WithLp.ofLp hh

/-- Accumulating the sample-weighted gradients in all subbatches at one
weight vector gives the full gradient numerator. This holds even when
some subbatches contain no labels. -/
theorem accumulated_batchGradient (repeats : ℕ) (chunks : List (List (LabeledPosition A)))
    (w : Weights A H d layers) :
    (chunks.flatten.length : ℝ) • batchGradient repeats chunks.flatten w =
      (chunks.map fun b => (b.length : ℝ) • batchGradient repeats b w).sum := by
  induction chunks with
  | nil => simp
  | cons b chunks ih =>
    simpa only [List.flatten_cons, List.map_cons, List.sum_cons, ← ih] using
      batchGradient_append_weighted repeats b chunks.flatten w

example (w : Weights (Fin 2) 4 16 1) (b c : List (LabeledPosition (Fin 2))) :=
  accumulated_batchGradient 1 [b, c] w

/-- Cross-entropy of all labeled positions plus coupled quadratic decay
on all independent real trainable coordinates. -/
noncomputable def fullGradientObjective (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (x : WeightSpace A H d layers) : ℝ :=
  batchLoss repeats batch (WithLp.ofLp x) + (lam / 2) * ‖x‖ ^ 2

theorem contDiff_fullGradientObjective (repeats : ℕ) (batch : List (LabeledPosition A)) (lam : ℝ) :
    ContDiff ℝ 1 (fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam) :=
  ((contDiff_batchLoss repeats batch).comp
    (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)).contDiff).add
      (contDiff_const.mul (contDiff_norm_sq ℝ))

private theorem batchLoss_nonnegative (repeats : ℕ)
    (batch : List (LabeledPosition A)) (w : Weights A H d layers) : 0 ≤ batchLoss repeats batch w := by
  unfold batchLoss
  apply div_nonneg _ (Nat.cast_nonneg _)
  apply List.sum_nonneg
  intro l hl
  obtain ⟨⟨T, word, i, a⟩, _, rfl⟩ := List.mem_map.mp hl
  let model := Model.fromWeights w
  have : Nonempty A := ⟨a⟩
  have hp : model.probability repeats word i a ≤ 1 := by
    calc
      _ ≤ ∑ b, model.probability repeats word i b :=
        Finset.single_le_sum (fun b _ => (model.probability_pos repeats word i b).le)
          (Finset.mem_univ a)
      _ = 1 := model.sum_probability repeats word i
  exact neg_nonneg.mpr (Real.log_nonpos (model.probability_pos repeats word i a).le hp)

theorem fullGradientObjective_coercive (repeats : ℕ) (batch : List (LabeledPosition A)) (lam : ℝ)
    (x : WeightSpace A H d layers) :
    (lam / 2) * ‖x‖ ^ 2 ≤ fullGradientObjective repeats batch lam x :=
  le_add_of_nonneg_left (batchLoss_nonnegative repeats batch (WithLp.ofLp x))

/-- The full regularized objective has locally Lipschitz gradient on all
native coordinates. Positive RMS offsets include zero activations. -/
theorem locallyLipschitz_fullGradient (repeats : ℕ) (batch : List (LabeledPosition A)) (lam : ℝ) :
    LocallyLipschitz (gradient
      (fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam)) := by
  let E := WeightSpace A H d layers
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  have hdata := (c1Lipschitz_batchLoss (H := H) (d := d) (layers := layers) repeats batch).comp
    (C1Lipschitz.of_contDiff coords.contDiff)
  have hpen : ContDiff ℝ 2 (fun x : E => (lam / 2) * ‖x‖ ^ 2) :=
    contDiff_const.mul (contDiff_norm_sq ℝ)
  have hF : C1Lipschitz f := hdata.add (C1Lipschitz.of_contDiff hpen)
  let dual := (InnerProductSpace.toDual ℝ E).symm
  exact LocallyLipschitz.comp (f := dual) (g := fderiv ℝ f)
    dual.isometry.lipschitzWith.locallyLipschitz hF.fderiv_lipschitz

/-- The penalty is included once in the full gradient before clipping. -/
theorem fullGradientObjective_gradient (repeats : ℕ) (batch : List (LabeledPosition A))
    (lam : ℝ) (x : WeightSpace A H d layers) :
    gradient (fullGradientObjective repeats batch lam) x =
      WithLp.toLp 2 (batchGradient repeats batch (WithLp.ofLp x)) + lam • x := by
  let E := WeightSpace A H d layers
  let data (y : E) := batchLoss repeats batch (WithLp.ofLp y)
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  have hd : HasFDerivAt data (fderiv ℝ data x) x :=
    (((contDiff_batchLoss repeats batch).comp coords.contDiff).differentiable one_ne_zero x).hasFDerivAt
  have hpen : HasFDerivAt (fun y : E => (lam / 2) * ‖y‖ ^ 2) (lam • innerSL ℝ x) x := by
    have h := (hasStrictFDerivAt_norm_sq x).hasFDerivAt.const_mul (lam / 2)
    convert h using 1
    ext z
    simp only [two_smul, smul_apply, add_apply, smul_eq_mul]
    ring
  have h := (hd.add hpen).fderiv
  change gradient (fullGradientObjective repeats batch lam) x = gradient data x + lam • x
  unfold gradient
  rw [show fderiv ℝ (fullGradientObjective repeats batch lam) x =
    fderiv ℝ data x + lam • innerSL ℝ x from h, map_add, map_smul]
  congr 1
  exact congrArg (fun y : E => lam • y) ((InnerProductSpace.toDual ℝ E).symm_apply_apply x)

end Transformer.Grokking.FiniteRecovery.Argand

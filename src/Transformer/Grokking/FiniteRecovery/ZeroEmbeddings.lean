module

public import Transformer.Grokking.FiniteRecovery.ArgandFullGradient
public import Transformer.Grokking.FiniteRecovery.ArgandSamples

/-! A tied, bias-free Argand model with zero embeddings sees zero input
and has zero readout. Its data gradient vanishes on this subspace even
when the score multiplier and other weights are nonzero. -/

@[expose] public section

open Filter
open scoped Topology InnerProductSpace

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.AMSGradW Transformer.Optimization

variable {A : Type*} {H d layers T : ℕ}

/-- Both real coordinates of each input/output embedding vanish. -/
def ZeroEmbeddings (w : Weights A H d layers) : Prop :=
  ∀ a j part, w (.inl (.embedding a j, part)) = 0

private theorem ZeroEmbeddings.embedding {w : Weights A H d layers} (hw : ZeroEmbeddings w) (a : A) :
    (Model.fromWeights w).embedding a = 0 := by
  ext j
  exact Complex.ext (hw a j false) (hw a j true)

private theorem layer_forwardWith_zero (layer : Layer H d) (eps scale : ℝ)
    (geometry : AttentionGeometry (Fin T) d) :
    layer.forwardWith eps scale geometry (0 : Word (CSpace (H * d)) T) = 0 := by
  have ha (i : Fin T) :
      attentionWith geometry layer.attentionProjections eps layer.scoreMultiplier
        (0 : Word (CSpace (H * d)) T) i = 0 := by
    simp only [attentionWith, Pi.zero_apply, map_zero, mul_zero, Finset.sum_const_zero]
    exact layer.attentionProjections.WO.map_zero
  have haf : sublayer layer.attentionGates eps scale
      (attentionWith geometry layer.attentionProjections eps layer.scoreMultiplier)
        (0 : Word (CSpace (H * d)) T) = 0 := by
    funext i
    have ha' (i : Fin T) : attentionWith geometry layer.attentionProjections eps layer.scoreMultiplier
        (fun _ => 0) i = 0 := ha i
    simp [sublayer, argandRMS, ha']
  rw [Layer.forwardWith, haf]
  funext i
  simp [sublayer, argandRMS]

theorem Model.hidden_eq_zero_of_embedding_eq_zero (model : Model A H d layers)
    (he : ∀ a, model.embedding a = 0) (repeats : ℕ) (word : Word A T) (i : Fin T) :
    model.hidden repeats word i = 0 := by
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  let pass (z : Word (CSpace (H * d)) T) :=
    (List.ofFn model.layer).foldl
      (fun x layer => layer.forwardWith model.eps scale (causalGeometry model.alpha) x) z
  have hp : pass 0 = 0 := by
    unfold pass
    induction List.ofFn model.layer with
    | nil => rfl
    | cons l ls ih => simpa only [List.foldl_cons, layer_forwardWith_zero] using ih
  have hi (r : ℕ) : (pass^[r]) 0 = 0 := by
    induction r with
    | zero => rfl
    | succ r ih => rw [Function.iterate_succ_apply', ih, hp]
  have hz : (fun j => argandRMS model.eps (model.embedding (word j))) = 0 := by
    funext j
    simp [he, argandRMS]
  change argandRMS model.eps ((pass^[repeats]) _ i) = 0
  rw [hz, hi]
  simp [argandRMS]

variable [Fintype A]

private theorem logit_hasFDerivAt_zero {w : Weights A H d layers} (hw : ZeroEmbeddings w)
    (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    HasFDerivAt (fun x => (Model.fromWeights x).logits repeats word i a)
      (0 : Weights A H d layers →L[ℝ] ℝ) w := by
  let f (x : Weights A H d layers) := (Model.fromWeights x).embedding a
  let g (x : Weights A H d layers) := (Model.fromWeights x).hidden repeats word i
  have hf : DifferentiableAt ℝ f w :=
    (contDiff_embedding_fromWeights a).differentiable one_ne_zero w
  have hg : DifferentiableAt ℝ g w :=
    (c1Lipschitz_hidden_fromWeights repeats word i).contDiff.differentiable one_ne_zero w
  have hz : g w = 0 := (Model.fromWeights w).hidden_eq_zero_of_embedding_eq_zero
    hw.embedding repeats word i
  have ht := hf.hasFDerivAt.inner ℂ hg.hasFDerivAt
  have hF : HasFDerivAt (fun x => ⟪f x, g x⟫_ℂ) (0 : Weights A H d layers →L[ℝ] ℂ) w := by
    have heq : (fderivInnerCLM ℂ (f w, g w)).comp
        ((fderiv ℝ f w).prod (fderiv ℝ g w)) = 0 := by
      ext x; simp [show f w = 0 from hw.embedding a, hz]
    rwa [heq] at ht
  let output : ℂ →L[ℝ] ℝ := (1 / 2 : ℝ) • Complex.reCLM
  have heq : (fun x => (Model.fromWeights x).logits repeats word i a) =
      output ∘ (fun x => ⟪f x, g x⟫_ℂ) := by
    funext x
    simp [Model.logits, Model.readout, f, g, output, div_eq_mul_inv, mul_comm]
  rw [heq]
  simpa only [ContinuousLinearMap.comp_zero] using output.hasFDerivAt.comp w hF

private theorem label_hasFDerivAt_zero {w : Weights A H d layers} (hw : ZeroEmbeddings w)
    (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    HasFDerivAt (fun x => -Real.log ((Model.fromWeights x).probability repeats word i a))
      (0 : Weights A H d layers →L[ℝ] ℝ) w := by
  let scores (x : Weights A H d layers) : A → ℝ := (Model.fromWeights x).logits repeats word i
  let loss (z : A → ℝ) := -Real.log (Real.exp (z a) / ∑ b, Real.exp (z b))
  have hd : ContDiff ℝ 1 loss := by
    have h (b : A) : ContDiff ℝ 1 (fun z : A → ℝ => Real.exp (z b)) :=
      (contDiff_apply ℝ _ b).exp
    have hp (z : A → ℝ) : 0 < ∑ b, Real.exp (z b) :=
      Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨a, Finset.mem_univ _, Real.exp_pos _⟩
    exact (((h a).div (ContDiff.sum fun b _ => h b) (fun z => (hp z).ne')).log
      (fun z => (div_pos (Real.exp_pos _) (hp z)).ne')).neg
  have hsc : HasFDerivAt scores (0 : Weights A H d layers →L[ℝ] (A → ℝ)) w :=
    hasFDerivAt_pi'' fun j => by simpa only [ContinuousLinearMap.comp_zero] using
      (logit_hasFDerivAt_zero hw repeats word i j)
  change HasFDerivAt (loss ∘ scores) _ w
  simpa only [ContinuousLinearMap.comp_zero] using (hd.differentiable one_ne_zero _).hasFDerivAt.comp w hsc

theorem batchGradient_eq_zero_of_zeroEmbeddings (repeats : ℕ) (batch : List (LabeledPosition A))
    (w : Weights A H d layers) (hw : ZeroEmbeddings w) :
    batchGradient repeats batch w = 0 := by
  let E := WeightSpace A H d layers
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  have hs : HasFDerivAt (batchLoss repeats batch) (0 : Weights A H d layers →L[ℝ] ℝ) w := by
    unfold batchLoss
    have ht (xs : List (LabeledPosition A)) : HasFDerivAt (fun x : Weights A H d layers =>
        (xs.map fun p => -Real.log ((Model.fromWeights x).probability repeats p.2.1 p.2.2.1 p.2.2.2)).sum)
          (0 : Weights A H d layers →L[ℝ] ℝ) w := by
      induction xs with
      | nil => simpa using (hasFDerivAt_const (𝕜 := ℝ) (0 : ℝ) w)
      | cons p xs ih =>
        simp only [List.map_cons, List.sum_cons]
        have hp := (label_hasFDerivAt_zero hw repeats p.2.1 p.2.2.1 p.2.2.2).add ih
        rw [zero_add] at hp
        exact hp
    have hf := (ht batch).const_smul (batch.length : ℝ)⁻¹
    simp only [div_eq_inv_mul]
    rw [smul_zero] at hf
    exact hf
  have hc := hs.comp (C.symm w) C.toContinuousLinearMap.hasFDerivAt
  have hF : HasFDerivAt (batchLoss repeats batch ∘ C)
      (0 : E →L[ℝ] ℝ) (C.symm w) := by
    rw [ContinuousLinearMap.zero_comp] at hc
    exact hc
  change WithLp.ofLp ((InnerProductSpace.toDual ℝ E).symm
    (fderiv ℝ (batchLoss repeats batch ∘ C) (C.symm w))) = 0
  rw [hF.fderiv, map_zero]
  rfl

end Transformer.Grokking.FiniteRecovery.Argand

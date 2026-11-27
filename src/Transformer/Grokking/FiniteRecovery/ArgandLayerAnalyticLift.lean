module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters
public import Transformer.Grokking.FiniteRecovery.ArgandRMSAnalyticLift
public import Transformer.Optimization.AnalyticLiftOperations
public import Mathlib.Analysis.Calculus.ContDiff.WithLp

/-! Analytic lifts of the complete Argand layer, using the exact RMS lift for
head and residual normalization and retaining the fixed causal mask. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization
open scoped ComplexConjugate

variable {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
variable {c H d layers T : ℕ}

private theorem analyticLift_streamProjection {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E] (col : X → Fin c → E)
    (hcol : ∀ j, HasAnalyticLift (fun w => col w j)) {x : X → CSpace c} (hx : HasAnalyticLift x) :
    HasAnalyticLift (fun w => streamProjection (col w) (x w)) := by
  simp only [streamProjection, LinearMap.sum_apply, LinearMap.smulRight_apply]
  exact HasAnalyticLift.sum fun j _ => (hx.projLp j).smul (hcol j)

private theorem analyticLift_vectorProjection {ι : Type*} [Fintype ι]
    (col : X → ι → CSpace c) (hcol : ∀ j, HasAnalyticLift (fun w => col w j))
    {x : X → ι → ℂ} (hx : ∀ j, HasAnalyticLift (fun w => x w j)) :
    HasAnalyticLift (fun w => vectorProjection (col w) (x w)) := by
  simp only [vectorProjection, LinearMap.sum_apply, LinearMap.smulRight_apply, LinearMap.proj_apply]
  exact HasAnalyticLift.sum fun j _ => (hx j).smul (hcol j)

private theorem analyticLift_normalizeHead {ε : ℝ} (hε : 0 < ε)
    {x : X → Fin d → ℂ} (hx : ∀ j, HasAnalyticLift (fun w => x w j)) (m : Fin d) :
    HasAnalyticLift (fun w => normalizeHead ε (x w) m) := by
  let z (w : X) : CSpace d := WithLp.toLp 2 (x w)
  have hz : HasAnalyticLift z := HasAnalyticLift.piLp hx
  have he (w : X) : normalizeHead ε (x w) m = argandRMS ε (z w) m := by
    rw [normalizeHead, argandRMS, PiLp.smul_apply, PiLp.norm_eq_of_L2]
    rfl
  simp_rw [he]
  exact ((analyticLift_argandRMS hε).comp hz).projLp m

private theorem analyticLift_maskedSoftmax {ι : Type*} [Fintype ι] (keep : ι → Bool)
    {x : X → ι → ℝ} (hx : ∀ i, HasAnalyticLift (fun w => x w i)) (j : ι) :
    HasAnalyticLift (fun w => maskedSoftmax keep (x w) j) := by
  simp only [maskedSoftmax]
  split
  · rename_i hj
    exact (hx j).exp.div (HasAnalyticLift.sum fun k _ => (hx k).exp) (fun w =>
      (Finset.sum_pos' (fun k _ => (Real.exp_pos (x w k)).le)
        ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, Real.exp_pos _⟩).ne')
  · exact (HasAnalyticLift.const _)

variable {A : Type*} [Fintype A]

private theorem analyticLift_complexEntry (p : ComplexParameter A H d layers) :
    HasAnalyticLift (fun w : Weights A H d layers => complexEntry w p) := by
  have h (im : Bool) : HasAnalyticLift (fun w : Weights A H d layers => w (.inl (p, im))) :=
    HasAnalyticLift.clm (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Parameter A H d layers => ℝ) (.inl (p, im)))
  exact (HasAnalyticLift.clm Complex.equivRealProdCLM.symm.toContinuousLinearMap).comp ((h false).prodMk (h true))

theorem analyticLift_embedding_fromWeights (a : A) :
    HasAnalyticLift (fun w : Weights A H d layers => (Model.fromWeights w).embedding a) :=
  HasAnalyticLift.piLp fun j => analyticLift_complexEntry (.embedding a j)

private theorem analyticLift_attention (l : Fin layers)
    (params : X → Weights A H d layers) (hparams : HasAnalyticLift params)
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ i, HasAnalyticLift (fun w => x w i)) (i : Fin T) :
    HasAnalyticLift (fun w => attentionWith (causalGeometry (Model.fromWeights (params w)).alpha)
      ((Model.fromWeights (params w)).layer l).attentionProjections (Model.fromWeights (params w)).eps
      ((Model.fromWeights (params w)).layer l).scoreMultiplier (x w) i) := by
  let ε : ℝ := 1 / 1000000
  let proj (p : Fin 3) (j : Fin T) (h : Fin H) (w : X) :=
    streamProjection (fun k h m => complexEntry (params w) (.attentionIn l p k h m)) (x w j) h
  have hp (p j h m) : HasAnalyticLift (fun w => proj p j h w m) := by
    have ht := analyticLift_streamProjection _ (fun k => HasAnalyticLift.pi fun h => HasAnalyticLift.pi fun m =>
      (analyticLift_complexEntry (.attentionIn l p k h m)).comp hparams) (hx j)
    exact (ht.proj h).proj m
  let rotated p j h m (w : X) :=
    (causalGeometry (Model.fromWeights (params w)).alpha).rotate j (normalizeHead ε (proj p j h w)) m
  have hr (p j h m) : HasAnalyticLift (rotated p j h m) := by
    have ha : HasAnalyticLift (fun w : X => params w (.inr m)) :=
      (HasAnalyticLift.clm (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Parameter A H d layers => ℝ) (.inr m))).comp hparams
    have hf : HasAnalyticLift (fun w : X => leropeFrequency 10000 (fun k => params w (.inr k)) m) :=
      ha.exp.mul (HasAnalyticLift.const _)
    have he : HasAnalyticLift (Complex.exp : ℂ → ℂ) :=
      HasAnalyticLift.of_analytic fun _ => analyticAt_cexp.restrictScalars
    exact (he.comp
      (((HasAnalyticLift.clm Complex.ofRealCLM).comp ((HasAnalyticLift.const _).mul hf)).mul (HasAnalyticLift.const _))).mul
      (analyticLift_normalizeHead (by norm_num : 0 < ε) (hp p j h) m)
  let score (w : X) (h : Fin H) (j : Fin T) :=
    complexEntry (params w) (.score l) * ((Real.log ((i.val : ℝ) + 1) / Real.sqrt d : ℝ) : ℂ) *
      ∑ m, conj (rotated 0 i h m w) * rotated 1 j h m w
  have hs (h j) : HasAnalyticLift (fun w => score w h j) :=
    (((analyticLift_complexEntry (.score l)).comp hparams).mul (HasAnalyticLift.const _)).mul
      (HasAnalyticLift.sum fun m _ => ((HasAnalyticLift.clm (RCLike.conjCLE : ℂ ≃L[ℝ] ℂ).toContinuousLinearMap).comp (hr 0 i h m)).mul (hr 1 j h m))
  let y (w : X) (h : Fin H) (m : Fin d) := ∑ j,
    (⟨maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).re) j,
      maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).im) j⟩ : ℂ) *
      proj 2 j h w m
  have hy (h m) : HasAnalyticLift (fun w => y w h m) := by
    apply HasAnalyticLift.sum
    intro j _
    let keep := fun k : Fin T => decide (k ≤ i)
    have hre := analyticLift_maskedSoftmax keep (fun k => (HasAnalyticLift.clm Complex.reCLM).comp (hs h k)) j
    have him := analyticLift_maskedSoftmax keep (fun k => (HasAnalyticLift.clm Complex.imCLM).comp (hs h k)) j
    exact ((HasAnalyticLift.clm Complex.equivRealProdCLM.symm.toContinuousLinearMap).comp (hre.prodMk him)).mul (hp 2 j h m)
  change HasAnalyticLift (fun w =>
    headProjection (fun h m => WithLp.toLp 2 fun k => complexEntry (params w) (.attentionOut l h m k)) (y w))
  simp only [headProjection, LinearMap.sum_apply, LinearMap.comp_apply, LinearMap.proj_apply]
  exact HasAnalyticLift.sum fun h _ => analyticLift_vectorProjection _
    (fun m => HasAnalyticLift.piLp fun k => (analyticLift_complexEntry (.attentionOut l h m k)).comp hparams) (hy h)

private theorem analyticLift_ffn (l : Fin layers)
    (params : X → Weights A H d layers) (hparams : HasAnalyticLift params)
    {x : X → CSpace (H * d)} (hx : HasAnalyticLift x) :
    HasAnalyticLift (fun w => let f := ((Model.fromWeights (params w)).layer l).feedForward
      f.WF (f.Wg (x w) * f.Wu (x w))) := by
  have hp (p : Fin 2) := analyticLift_streamProjection
    (fun w i j => complexEntry (params w) (.feedForwardIn l p i j))
    (fun i => HasAnalyticLift.pi fun j => (analyticLift_complexEntry (.feedForwardIn l p i j)).comp hparams) hx
  exact analyticLift_vectorProjection _
    (fun j => HasAnalyticLift.piLp fun k => (analyticLift_complexEntry (.feedForwardOut l j k)).comp hparams)
    (fun j => ((hp 0).proj j |>.mul ((hp 1).proj j)))

private theorem analyticLift_sublayer (l : Fin layers) (s : Fin 2) (scale : ℝ)
    (params : X → Weights A H d layers) (hparams : HasAnalyticLift params)
    (branch : X → Word (CSpace (H * d)) T → Word (CSpace (H * d)) T)
    (hbranch : ∀ z : X → Word (CSpace (H * d)) T,
      (∀ j, HasAnalyticLift (fun w => z w j)) → ∀ j, HasAnalyticLift (fun w => branch w (z w) j))
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ j, HasAnalyticLift (fun w => x w j)) (i : Fin T) :
    HasAnalyticLift (fun w => sublayer
      ⟨streamProjection fun k => complexEntry (params w) (.residualGate l s 0 k),
       streamProjection fun k => complexEntry (params w) (.residualGate l s 1 k)⟩
      (1 / 1000000) scale (branch w) (x w) i) := by
  have hε : (0 : ℝ) < 1 / 1000000 := by norm_num
  let z := fun w j => argandRMS (1 / 1000000) (x w j)
  have hz (j : Fin T) : HasAnalyticLift (fun w => z w j) := (analyticLift_argandRMS hε).comp (hx j)
  have hg (g : Fin 2) := analyticLift_streamProjection (fun w k => complexEntry (params w) (.residualGate l s g k))
    (fun k => (analyticLift_complexEntry (.residualGate l s g k)).comp hparams) (hz i)
  exact (((HasAnalyticLift.const _).add (hg 0)).smul (hx i)).add
    (((HasAnalyticLift.const _).mul ((HasAnalyticLift.const _).add (hg 1))).smul
      ((analyticLift_argandRMS hε).comp (hbranch z hz i)))

private theorem analyticLift_forward (l : Fin layers) (scale : ℝ)
    (params : X → Weights A H d layers) (hparams : HasAnalyticLift params)
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ j, HasAnalyticLift (fun w => x w j)) (i : Fin T) :
    HasAnalyticLift (fun w => ((Model.fromWeights (params w)).layer l).forwardWith (Model.fromWeights (params w)).eps scale
      (causalGeometry (Model.fromWeights (params w)).alpha) (x w) i) := by
  let attentionBranch (w : X) (z : Word (CSpace (H * d)) T) :=
    attentionWith (causalGeometry (Model.fromWeights (params w)).alpha)
      ((Model.fromWeights (params w)).layer l).attentionProjections (Model.fromWeights (params w)).eps
      ((Model.fromWeights (params w)).layer l).scoreMultiplier z
  let y (w : X) :=
    sublayer ((Model.fromWeights (params w)).layer l).attentionGates (Model.fromWeights (params w)).eps scale (attentionBranch w) (x w)
  let feedForwardBranch (w : X) (z : Word (CSpace (H * d)) T) (j : Fin T) :=
    let f := ((Model.fromWeights (params w)).layer l).feedForward
    f.WF (f.Wg (z j) * f.Wu (z j))
  have hy (j : Fin T) : HasAnalyticLift (fun w => y w j) :=
    analyticLift_sublayer l 0 scale params hparams attentionBranch (analyticLift_attention l params hparams) x hx j
  exact analyticLift_sublayer l 1 scale params hparams feedForwardBranch (fun z h j =>
    analyticLift_ffn l params hparams (x := fun w => z w j) (h j)) y hy i

theorem analyticLift_forward_fromWeights (l : Fin layers) (scale : ℝ)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, HasAnalyticLift (fun w => x w j)) (i : Fin T) :
    HasAnalyticLift (fun w => ((Model.fromWeights w).layer l).forwardWith (Model.fromWeights w).eps scale
      (causalGeometry (Model.fromWeights w).alpha) (x w) i) :=
  analyticLift_forward l scale id HasAnalyticLift.id x hx i


example := analyticLift_forward_fromWeights (A := Fin 16) (H := 4) (d := 16) (layers := 1) (T := 8)
  0 1 (fun _ _ => 0) (fun _ => (HasAnalyticLift.const _)) 0

end Transformer.Grokking.FiniteRecovery.Argand

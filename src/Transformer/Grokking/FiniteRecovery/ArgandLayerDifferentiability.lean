module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters
public import Transformer.Grokking.FiniteRecovery.ArgandRMSDifferentiability
public import Transformer.Optimization.C1LipschitzOperations
public import Mathlib.Analysis.Calculus.ContDiff.WithLp

/-! Locally Lipschitz real derivatives of the trainable Argand layer. Positive offsets
handle zero head or residual activations. The causal mask is fixed. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization
open scoped ComplexConjugate

variable {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
variable {c H d layers T : ℕ}

private theorem c1Lipschitz_streamProjection {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E] (col : X → Fin c → E)
    (hcol : ∀ j, C1Lipschitz (fun w => col w j)) {x : X → CSpace c} (hx : C1Lipschitz x) :
    C1Lipschitz (fun w => streamProjection (col w) (x w)) := by
  simp only [streamProjection, LinearMap.sum_apply, LinearMap.smulRight_apply]
  exact C1Lipschitz.sum fun j _ => (hx.projLp j).smul (hcol j)

private theorem c1Lipschitz_vectorProjection {ι : Type*} [Fintype ι]
    (col : X → ι → CSpace c) (hcol : ∀ j, C1Lipschitz (fun w => col w j))
    {x : X → ι → ℂ} (hx : ∀ j, C1Lipschitz (fun w => x w j)) :
    C1Lipschitz (fun w => vectorProjection (col w) (x w)) := by
  simp only [vectorProjection, LinearMap.sum_apply, LinearMap.smulRight_apply, LinearMap.proj_apply]
  exact C1Lipschitz.sum fun j _ => (hx j).smul (hcol j)

private theorem c1Lipschitz_normalizeHead {ε : ℝ} (hε : 0 < ε)
    {x : X → Fin d → ℂ} (hx : ∀ j, C1Lipschitz (fun w => x w j)) (m : Fin d) :
    C1Lipschitz (fun w => normalizeHead ε (x w) m) := by
  let z (w : X) : CSpace d := WithLp.toLp 2 (x w)
  have hz : C1Lipschitz z := C1Lipschitz.piLp hx
  have he (w : X) : normalizeHead ε (x w) m = argandRMS ε (z w) m := by
    rw [normalizeHead, argandRMS, PiLp.smul_apply, PiLp.norm_eq_of_L2]
    rfl
  simp_rw [he]
  exact ((c1Lipschitz_argandRMS hε).comp hz).projLp m

private theorem c1Lipschitz_maskedSoftmax {ι : Type*} [Fintype ι] (keep : ι → Bool)
    {x : X → ι → ℝ} (hx : ∀ i, C1Lipschitz (fun w => x w i)) (j : ι) :
    C1Lipschitz (fun w => maskedSoftmax keep (x w) j) := by
  simp only [maskedSoftmax]
  split
  · rename_i hj
    exact (hx j).exp.div (C1Lipschitz.sum fun k _ => (hx k).exp) (fun w =>
      (Finset.sum_pos' (fun k _ => (Real.exp_pos (x w k)).le)
        ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, Real.exp_pos _⟩).ne')
  · exact (C1Lipschitz.const _)

variable {A : Type*} [Fintype A]

private theorem c1Lipschitz_complexEntry (p : ComplexParameter A H d layers) :
    C1Lipschitz (fun w : Weights A H d layers => complexEntry w p) := by
  have h (im : Bool) : C1Lipschitz (fun w : Weights A H d layers => w (.inl (p, im))) :=
    C1Lipschitz.of_contDiff (contDiff_apply ℝ _ (Sum.inl (p, im)))
  exact (C1Lipschitz.of_contDiff Complex.equivRealProdCLM.symm.contDiff).comp ((h false).prodMk (h true))

theorem c1Lipschitz_embedding_fromWeights (a : A) :
    C1Lipschitz (fun w : Weights A H d layers => (Model.fromWeights w).embedding a) :=
  C1Lipschitz.piLp fun j => c1Lipschitz_complexEntry (.embedding a j)

theorem contDiff_embedding_fromWeights (a : A) :
    ContDiff ℝ 1 (fun w : Weights A H d layers => (Model.fromWeights w).embedding a) :=
  (c1Lipschitz_embedding_fromWeights a).contDiff

private theorem c1Lipschitz_attention (l : Fin layers)
    (params : X → Weights A H d layers) (hparams : C1Lipschitz params)
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ i, C1Lipschitz (fun w => x w i)) (i : Fin T) :
    C1Lipschitz (fun w => attentionWith (causalGeometry (Model.fromWeights (params w)).alpha)
      ((Model.fromWeights (params w)).layer l).attentionProjections (Model.fromWeights (params w)).eps
      ((Model.fromWeights (params w)).layer l).scoreMultiplier (x w) i) := by
  let ε : ℝ := 1 / 1000000
  let proj (p : Fin 3) (j : Fin T) (h : Fin H) (w : X) :=
    streamProjection (fun k h m => complexEntry (params w) (.attentionIn l p k h m)) (x w j) h
  have hp (p j h m) : C1Lipschitz (fun w => proj p j h w m) := by
    have ht := c1Lipschitz_streamProjection _ (fun k => C1Lipschitz.pi fun h => C1Lipschitz.pi fun m =>
      (c1Lipschitz_complexEntry (.attentionIn l p k h m)).comp hparams) (hx j)
    exact (ht.proj h).proj m
  let rotated p j h m (w : X) :=
    (causalGeometry (Model.fromWeights (params w)).alpha).rotate j (normalizeHead ε (proj p j h w)) m
  have hr (p j h m) : C1Lipschitz (rotated p j h m) := by
    have ha : C1Lipschitz (fun w : X => params w (.inr m)) :=
      (C1Lipschitz.of_contDiff (contDiff_apply ℝ _ _)).comp hparams
    have hf : C1Lipschitz (fun w : X => leropeFrequency 10000 (fun k => params w (.inr k)) m) :=
      ha.exp.mul (C1Lipschitz.const _)
    have heC : ContDiff ℂ 2 (Complex.exp : ℂ → ℂ) := Complex.contDiff_exp
    have he : C1Lipschitz (Complex.exp : ℂ → ℂ) := C1Lipschitz.of_contDiff (heC.restrict_scalars ℝ)
    exact (he.comp
      (((C1Lipschitz.of_contDiff Complex.ofRealCLM.contDiff).comp ((C1Lipschitz.const _).mul hf)).mul (C1Lipschitz.const _))).mul
      (c1Lipschitz_normalizeHead (by norm_num : 0 < ε) (hp p j h) m)
  let score (w : X) (h : Fin H) (j : Fin T) :=
    complexEntry (params w) (.score l) * ((Real.log ((i.val : ℝ) + 1) / Real.sqrt d : ℝ) : ℂ) *
      ∑ m, conj (rotated 0 i h m w) * rotated 1 j h m w
  have hs (h j) : C1Lipschitz (fun w => score w h j) :=
    (((c1Lipschitz_complexEntry (.score l)).comp hparams).mul (C1Lipschitz.const _)).mul
      (C1Lipschitz.sum fun m _ => ((C1Lipschitz.of_contDiff (RCLike.conjCLE : ℂ ≃L[ℝ] ℂ).contDiff).comp (hr 0 i h m)).mul (hr 1 j h m))
  let y (w : X) (h : Fin H) (m : Fin d) := ∑ j,
    (⟨maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).re) j,
      maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).im) j⟩ : ℂ) *
      proj 2 j h w m
  have hy (h m) : C1Lipschitz (fun w => y w h m) := by
    apply C1Lipschitz.sum
    intro j _
    let keep := fun k : Fin T => decide (k ≤ i)
    have hre := c1Lipschitz_maskedSoftmax keep (fun k => (C1Lipschitz.of_contDiff Complex.reCLM.contDiff).comp (hs h k)) j
    have him := c1Lipschitz_maskedSoftmax keep (fun k => (C1Lipschitz.of_contDiff Complex.imCLM.contDiff).comp (hs h k)) j
    exact ((C1Lipschitz.of_contDiff Complex.equivRealProdCLM.symm.contDiff).comp (hre.prodMk him)).mul (hp 2 j h m)
  change C1Lipschitz (fun w =>
    headProjection (fun h m => WithLp.toLp 2 fun k => complexEntry (params w) (.attentionOut l h m k)) (y w))
  simp only [headProjection, LinearMap.sum_apply, LinearMap.comp_apply, LinearMap.proj_apply]
  exact C1Lipschitz.sum fun h _ => c1Lipschitz_vectorProjection _
    (fun m => C1Lipschitz.piLp fun k => (c1Lipschitz_complexEntry (.attentionOut l h m k)).comp hparams) (hy h)

private theorem c1Lipschitz_ffn (l : Fin layers)
    (params : X → Weights A H d layers) (hparams : C1Lipschitz params)
    {x : X → CSpace (H * d)} (hx : C1Lipschitz x) :
    C1Lipschitz (fun w => let f := ((Model.fromWeights (params w)).layer l).feedForward
      f.WF (f.Wg (x w) * f.Wu (x w))) := by
  have hp (p : Fin 2) := c1Lipschitz_streamProjection
    (fun w i j => complexEntry (params w) (.feedForwardIn l p i j))
    (fun i => C1Lipschitz.pi fun j => (c1Lipschitz_complexEntry (.feedForwardIn l p i j)).comp hparams) hx
  exact c1Lipschitz_vectorProjection _
    (fun j => C1Lipschitz.piLp fun k => (c1Lipschitz_complexEntry (.feedForwardOut l j k)).comp hparams)
    (fun j => ((hp 0).proj j |>.mul ((hp 1).proj j)))

private theorem c1Lipschitz_sublayer (l : Fin layers) (s : Fin 2) (scale : ℝ)
    (params : X → Weights A H d layers) (hparams : C1Lipschitz params)
    (branch : X → Word (CSpace (H * d)) T → Word (CSpace (H * d)) T)
    (hbranch : ∀ z : X → Word (CSpace (H * d)) T,
      (∀ j, C1Lipschitz (fun w => z w j)) → ∀ j, C1Lipschitz (fun w => branch w (z w) j))
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ j, C1Lipschitz (fun w => x w j)) (i : Fin T) :
    C1Lipschitz (fun w => sublayer
      ⟨streamProjection fun k => complexEntry (params w) (.residualGate l s 0 k),
       streamProjection fun k => complexEntry (params w) (.residualGate l s 1 k)⟩
      (1 / 1000000) scale (branch w) (x w) i) := by
  have hε : (0 : ℝ) < 1 / 1000000 := by norm_num
  let z := fun w j => argandRMS (1 / 1000000) (x w j)
  have hz (j : Fin T) : C1Lipschitz (fun w => z w j) := (c1Lipschitz_argandRMS hε).comp (hx j)
  have hg (g : Fin 2) := c1Lipschitz_streamProjection (fun w k => complexEntry (params w) (.residualGate l s g k))
    (fun k => (c1Lipschitz_complexEntry (.residualGate l s g k)).comp hparams) (hz i)
  exact (((C1Lipschitz.const _).add (hg 0)).smul (hx i)).add
    (((C1Lipschitz.const _).mul ((C1Lipschitz.const _).add (hg 1))).smul
      ((c1Lipschitz_argandRMS hε).comp (hbranch z hz i)))

private theorem c1Lipschitz_forward (l : Fin layers) (scale : ℝ)
    (params : X → Weights A H d layers) (hparams : C1Lipschitz params)
    (x : X → Word (CSpace (H * d)) T)
    (hx : ∀ j, C1Lipschitz (fun w => x w j)) (i : Fin T) :
    C1Lipschitz (fun w => ((Model.fromWeights (params w)).layer l).forwardWith (Model.fromWeights (params w)).eps scale
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
  have hy (j : Fin T) : C1Lipschitz (fun w => y w j) :=
    c1Lipschitz_sublayer l 0 scale params hparams attentionBranch (c1Lipschitz_attention l params hparams) x hx j
  exact c1Lipschitz_sublayer l 1 scale params hparams feedForwardBranch (fun z h j =>
    c1Lipschitz_ffn l params hparams (x := fun w => z w j) (h j)) y hy i

theorem c1Lipschitz_forward_fromWeights (l : Fin layers) (scale : ℝ)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, C1Lipschitz (fun w => x w j)) (i : Fin T) :
    C1Lipschitz (fun w => ((Model.fromWeights w).layer l).forwardWith (Model.fromWeights w).eps scale
      (causalGeometry (Model.fromWeights w).alpha) (x w) i) :=
  c1Lipschitz_forward l scale id (C1Lipschitz.of_contDiff contDiff_id) x hx i

set_option maxHeartbeats 4500000 in
theorem contDiff_forward_fromWeights (l : Fin layers) (scale : ℝ)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, ContDiff ℝ 1 (fun w => x w j)) (i : Fin T) :
    ContDiff ℝ 1 (fun w => ((Model.fromWeights w).layer l).forwardWith (Model.fromWeights w).eps scale
      (causalGeometry (Model.fromWeights w).alpha) (x w) i) := by
  let P := Weights A H d layers × Word (CSpace (H * d)) T
  have hz : C1Lipschitz (Prod.snd : P → Word (CSpace (H * d)) T) :=
    C1Lipschitz.of_contDiff contDiff_snd
  have h := c1Lipschitz_forward l scale (Prod.fst : P → Weights A H d layers)
    (C1Lipschitz.of_contDiff contDiff_fst) Prod.snd (fun j => hz.proj j) i
  have hpair : ContDiff ℝ 1 (fun w : Weights A H d layers => (w, x w)) :=
    contDiff_id.prodMk (contDiff_pi.mpr hx)
  let forward (q : P) := ((Model.fromWeights q.1).layer l).forwardWith (Model.fromWeights q.1).eps
    scale (causalGeometry (Model.fromWeights q.1).alpha) q.2 i
  have hF : ContDiff ℝ 1 forward := h.contDiff
  change ContDiff ℝ 1 (forward ∘ fun w => (w, x w))
  exact hF.comp hpair

example := c1Lipschitz_forward_fromWeights (A := Fin 16) (H := 4) (d := 16) (layers := 1) (T := 8)
  0 1 (fun _ _ => 0) (fun _ => (C1Lipschitz.const _)) 0

end Transformer.Grokking.FiniteRecovery.Argand

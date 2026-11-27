module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters
public import Transformer.Grokking.FiniteRecovery.ArgandRMSDifferentiability
public import Mathlib.Analysis.Calculus.ContDiff.WithLp

/-! Real C¹ regularity of the trainable Argand layer. Positive offsets
handle zero head or residual activations. The causal mask is fixed. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped ComplexConjugate

variable {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
variable {c H d layers T : ℕ}

private theorem contDiff_streamProjection {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E] (col : X → Fin c → E)
    (hcol : ∀ j, ContDiff ℝ 1 (fun w => col w j)) {x : X → CSpace c} (hx : ContDiff ℝ 1 x) :
    ContDiff ℝ 1 (fun w => streamProjection (col w) (x w)) := by
  simp only [streamProjection, LinearMap.sum_apply, LinearMap.smulRight_apply]
  exact ContDiff.sum fun j _ => (((contDiff_piLp 2).mp hx) j).smul (hcol j)

private theorem contDiff_vectorProjection {ι : Type*} [Fintype ι]
    (col : X → ι → CSpace c) (hcol : ∀ j, ContDiff ℝ 1 (fun w => col w j))
    {x : X → ι → ℂ} (hx : ∀ j, ContDiff ℝ 1 (fun w => x w j)) :
    ContDiff ℝ 1 (fun w => vectorProjection (col w) (x w)) := by
  simp only [vectorProjection, LinearMap.sum_apply, LinearMap.smulRight_apply, LinearMap.proj_apply]
  exact ContDiff.sum fun j _ => (hx j).smul (hcol j)

private theorem contDiff_normalizeHead {ε : ℝ} (hε : 0 < ε)
    {x : X → Fin d → ℂ} (hx : ∀ j, ContDiff ℝ 1 (fun w => x w j)) (m : Fin d) :
    ContDiff ℝ 1 (fun w => normalizeHead ε (x w) m) := by
  let z (w : X) : CSpace d := WithLp.toLp 2 (x w)
  have hz : ContDiff ℝ 1 z := (contDiff_piLp 2).mpr hx
  have he (w : X) : normalizeHead ε (x w) m = argandRMS ε (z w) m := by
    rw [normalizeHead, argandRMS, PiLp.smul_apply, PiLp.norm_eq_of_L2]
    rfl
  simp_rw [he]
  exact (contDiff_piLp 2).mp ((contDiff_argandRMS hε).comp hz) m

private theorem contDiff_maskedSoftmax {ι : Type*} [Fintype ι] (keep : ι → Bool)
    {x : X → ι → ℝ} (hx : ∀ i, ContDiff ℝ 1 (fun w => x w i)) (j : ι) :
    ContDiff ℝ 1 (fun w => maskedSoftmax keep (x w) j) := by
  simp only [maskedSoftmax]
  split
  · rename_i hj
    exact (hx j).exp.div (ContDiff.sum fun k _ => (hx k).exp) (fun w =>
      (Finset.sum_pos' (fun k _ => (Real.exp_pos (x w k)).le)
        ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, Real.exp_pos _⟩).ne')
  · exact contDiff_const

variable {A : Type*} [Fintype A]

private theorem contDiff_complexEntry (p : ComplexParameter A H d layers) :
    ContDiff ℝ 1 (fun w : Weights A H d layers => complexEntry w p) := by
  have h (im : Bool) : ContDiff ℝ 1 (fun w : Weights A H d layers => w (.inl (p, im))) :=
    contDiff_apply ℝ _ (Sum.inl (p, im))
  exact Complex.equivRealProdCLM.symm.contDiff.comp ((h false).prodMk (h true))

theorem contDiff_embedding_fromWeights (a : A) :
    ContDiff ℝ 1 (fun w : Weights A H d layers => (Model.fromWeights w).embedding a) :=
  (contDiff_piLp 2).mpr fun j => contDiff_complexEntry (.embedding a j)

private theorem contDiff_attention (l : Fin layers)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ i, ContDiff ℝ 1 (fun w => x w i)) (i : Fin T) :
    ContDiff ℝ 1 (fun w => attentionWith (causalGeometry (Model.fromWeights w).alpha)
      ((Model.fromWeights w).layer l).attentionProjections (Model.fromWeights w).eps
      ((Model.fromWeights w).layer l).scoreMultiplier (x w) i) := by
  let ε : ℝ := 1 / 1000000
  let proj (p : Fin 3) (j : Fin T) (h : Fin H) (w : Weights A H d layers) :=
    streamProjection (fun k h m => complexEntry w (.attentionIn l p k h m)) (x w j) h
  have hp (p j h m) : ContDiff ℝ 1 (fun w => proj p j h w m) := by
    apply (contDiff_apply ℝ _ m).comp
    apply (contDiff_apply ℝ _ h).comp
    exact contDiff_streamProjection _ (fun k => contDiff_pi.mpr fun h => contDiff_pi.mpr fun m =>
      contDiff_complexEntry (.attentionIn l p k h m)) (hx j)
  let rotated p j h m (w : Weights A H d layers) :=
    (causalGeometry (Model.fromWeights w).alpha).rotate j (normalizeHead ε (proj p j h w)) m
  have hr (p j h m) : ContDiff ℝ 1 (rotated p j h m) := by
    have ha : ContDiff ℝ 1 (fun w : Weights A H d layers => w (.inr m)) := contDiff_apply ℝ _ _
    have hf : ContDiff ℝ 1 (fun w : Weights A H d layers => leropeFrequency 10000 (fun k => w (.inr k)) m) :=
      ha.exp.mul contDiff_const
    have heC : ContDiff ℂ 1 (Complex.exp : ℂ → ℂ) := Complex.contDiff_exp
    have he : ContDiff ℝ 1 (Complex.exp : ℂ → ℂ) := heC.restrict_scalars ℝ
    exact (he.comp
      ((Complex.ofRealCLM.contDiff.comp (contDiff_const.mul hf)).mul contDiff_const)).mul
      (contDiff_normalizeHead (by norm_num : 0 < ε) (hp p j h) m)
  let score (w : Weights A H d layers) (h : Fin H) (j : Fin T) :=
    complexEntry w (.score l) * ((Real.log ((i.val : ℝ) + 1) / Real.sqrt d : ℝ) : ℂ) *
      ∑ m, conj (rotated 0 i h m w) * rotated 1 j h m w
  have hs (h j) : ContDiff ℝ 1 (fun w => score w h j) :=
    ((contDiff_complexEntry (.score l)).mul contDiff_const).mul
      (ContDiff.sum fun m _ => ((RCLike.conjCLE : ℂ ≃L[ℝ] ℂ).contDiff.comp (hr 0 i h m)).mul (hr 1 j h m))
  let y (w : Weights A H d layers) (h : Fin H) (m : Fin d) := ∑ j,
    (⟨maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).re) j,
      maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).im) j⟩ : ℂ) *
      proj 2 j h w m
  have hy (h m) : ContDiff ℝ 1 (fun w => y w h m) := by
    apply ContDiff.sum
    intro j _
    let keep := fun k : Fin T => decide (k ≤ i)
    have hre := contDiff_maskedSoftmax keep (fun k => Complex.reCLM.contDiff.comp (hs h k)) j
    have him := contDiff_maskedSoftmax keep (fun k => Complex.imCLM.contDiff.comp (hs h k)) j
    exact (Complex.equivRealProdCLM.symm.contDiff.comp (hre.prodMk him)).mul (hp 2 j h m)
  change ContDiff ℝ 1 (fun w =>
    headProjection (fun h m => WithLp.toLp 2 fun k => complexEntry w (.attentionOut l h m k)) (y w))
  simp only [headProjection, LinearMap.sum_apply, LinearMap.comp_apply, LinearMap.proj_apply]
  exact ContDiff.sum fun h _ => contDiff_vectorProjection _
    (fun m => (contDiff_piLp 2).mpr fun k => contDiff_complexEntry (.attentionOut l h m k)) (hy h)

private theorem contDiff_ffn (l : Fin layers)
    {x : Weights A H d layers → CSpace (H * d)} (hx : ContDiff ℝ 1 x) :
    ContDiff ℝ 1 (fun w => let f := ((Model.fromWeights w).layer l).feedForward
      f.WF (f.Wg (x w) * f.Wu (x w))) := by
  have hp (p : Fin 2) := contDiff_streamProjection
    (fun w i j => complexEntry w (.feedForwardIn l p i j))
    (fun i => contDiff_pi.mpr fun j => contDiff_complexEntry (.feedForwardIn l p i j)) hx
  exact contDiff_vectorProjection _
    (fun j => (contDiff_piLp 2).mpr fun k => contDiff_complexEntry (.feedForwardOut l j k))
    (fun j => ((contDiff_apply ℝ _ j).comp ((hp 0).mul (hp 1))))

private theorem contDiff_sublayer (l : Fin layers) (s : Fin 2) (scale : ℝ)
    (branch : Weights A H d layers → Word (CSpace (H * d)) T → Word (CSpace (H * d)) T)
    (hbranch : ∀ z : Weights A H d layers → Word (CSpace (H * d)) T,
      (∀ j, ContDiff ℝ 1 (fun w => z w j)) → ∀ j, ContDiff ℝ 1 (fun w => branch w (z w) j))
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, ContDiff ℝ 1 (fun w => x w j)) (i : Fin T) :
    ContDiff ℝ 1 (fun w => sublayer
      ⟨streamProjection fun k => complexEntry w (.residualGate l s 0 k),
       streamProjection fun k => complexEntry w (.residualGate l s 1 k)⟩
      (1 / 1000000) scale (branch w) (x w) i) := by
  have hε : (0 : ℝ) < 1 / 1000000 := by norm_num
  let z := fun w j => argandRMS (1 / 1000000) (x w j)
  have hz (j : Fin T) : ContDiff ℝ 1 (fun w => z w j) := (contDiff_argandRMS hε).comp (hx j)
  have hg (g : Fin 2) := contDiff_streamProjection (fun w k => complexEntry w (.residualGate l s g k))
    (fun k => contDiff_complexEntry (.residualGate l s g k)) (hz i)
  exact ((contDiff_const.add (hg 0)).smul (hx i)).add
    ((contDiff_const.mul (contDiff_const.add (hg 1))).smul
      ((contDiff_argandRMS hε).comp (hbranch z hz i)))

theorem contDiff_forward_fromWeights (l : Fin layers) (scale : ℝ)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, ContDiff ℝ 1 (fun w => x w j)) (i : Fin T) :
    ContDiff ℝ 1 (fun w => ((Model.fromWeights w).layer l).forwardWith (Model.fromWeights w).eps scale
      (causalGeometry (Model.fromWeights w).alpha) (x w) i) := by
  let attentionBranch (w : Weights A H d layers) (z : Word (CSpace (H * d)) T) :=
    attentionWith (causalGeometry (Model.fromWeights w).alpha)
      ((Model.fromWeights w).layer l).attentionProjections (Model.fromWeights w).eps
      ((Model.fromWeights w).layer l).scoreMultiplier z
  let y (w : Weights A H d layers) :=
    sublayer ((Model.fromWeights w).layer l).attentionGates (Model.fromWeights w).eps scale (attentionBranch w) (x w)
  let feedForwardBranch (w : Weights A H d layers) (z : Word (CSpace (H * d)) T) (j : Fin T) :=
    let f := ((Model.fromWeights w).layer l).feedForward
    f.WF (f.Wg (z j) * f.Wu (z j))
  have hy (j : Fin T) : ContDiff ℝ 1 (fun w => y w j) :=
    contDiff_sublayer l 0 scale attentionBranch (contDiff_attention l) x hx j
  exact contDiff_sublayer l 1 scale feedForwardBranch (fun z h j =>
    contDiff_ffn l (x := fun w => z w j) (h j)) y hy i

example := contDiff_forward_fromWeights (A := Fin 16) (H := 4) (d := 16) (layers := 1) (T := 8)
  0 1 (fun _ _ => 0) (fun _ => contDiff_const) 0

end Transformer.Grokking.FiniteRecovery.Argand

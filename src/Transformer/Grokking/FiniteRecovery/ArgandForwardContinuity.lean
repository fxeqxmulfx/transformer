module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters

/-! Continuity of the tied Argand readout in all its independent weights.
Positive RMS offsets and the causal mask preserve continuity at zero
activation, including at the initial zero gates and frequencies. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped ComplexConjugate InnerProductSpace

variable {A X : Type*} {H d layers T c : ℕ}
variable [TopologicalSpace X]

private theorem continuous_streamCoordinate {x : X → CSpace c} (hx : Continuous x) (j : Fin c) :
    Continuous (fun z => x z j) :=
  (continuous_apply j).comp ((PiLp.continuous_ofLp 2 _).comp hx)

private theorem continuous_stream {x : X → CSpace c} (hx : ∀ j : Fin c, Continuous (fun z => x z j)) :
    Continuous x :=
  (PiLp.continuous_toLp 2 _).comp (continuous_pi hx)

private theorem continuous_complexEntry (p : ComplexParameter A H d layers) :
    Continuous (fun w : Weights A H d layers => complexEntry w p) := by
  have he (w : Weights A H d layers) : complexEntry w p =
      (w (.inl (p, false)) : ℂ) + (w (.inl (p, true)) : ℂ) * Complex.I := by
    apply Complex.ext <;> simp [complexEntry]
  simp_rw [he]
  exact (Complex.continuous_ofReal.comp (continuous_apply _)).add
    ((Complex.continuous_ofReal.comp (continuous_apply _)).mul continuous_const)

theorem continuous_argandRMS {ε : ℝ} (hε : 0 < ε) {x : X → CSpace c} (hx : Continuous x) :
    Continuous (fun w => argandRMS ε (x w)) :=
  (Complex.continuous_ofReal.comp (continuous_const.div (hx.norm.add continuous_const)
    (fun w => (add_pos_of_nonneg_of_pos (norm_nonneg (x w)) hε).ne'))).smul hx

theorem continuous_embedding_fromWeights (a : A) :
    Continuous (fun w : Weights A H d layers => (Model.fromWeights w).embedding a) :=
  continuous_stream fun j => continuous_complexEntry (.embedding a j)

private theorem continuous_streamProjection {E : Type*} [AddCommMonoid E] [Module ℂ E]
    [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul ℂ E] (cols : X → Fin c → E)
    (hcols : ∀ j, Continuous (fun w => cols w j)) {x : X → CSpace c} (hx : Continuous x) :
    Continuous (fun w => streamProjection (cols w) (x w)) := by
  simp only [streamProjection, LinearMap.sum_apply, LinearMap.smulRight_apply]
  exact continuous_finsetSum _ fun j _ => (continuous_streamCoordinate hx j).smul (hcols j)

private theorem continuous_vectorProjection {ι : Type*} [Fintype ι]
    (cols : X → ι → CSpace c) (hcols : ∀ j, Continuous (fun w => cols w j))
    {x : X → ι → ℂ} (hx : ∀ j, Continuous (fun w => x w j)) :
    Continuous (fun w => vectorProjection (cols w) (x w)) := by
  simp only [vectorProjection, LinearMap.sum_apply, LinearMap.smulRight_apply, LinearMap.proj_apply]
  exact continuous_finsetSum _ fun j _ => (hx j).smul (hcols j)

private theorem continuous_normalizeHead {ε : ℝ} (hε : 0 < ε)
    {x : X → Fin d → ℂ} (hx : ∀ j, Continuous (fun w => x w j)) (m : Fin d) :
    Continuous (fun w => normalizeHead ε (x w) m) := by
  have hs : Continuous (fun w => Real.sqrt (sqLen (x w))) :=
    (continuous_finsetSum _ fun j _ => (hx j).norm.pow 2).sqrt
  exact (Complex.continuous_ofReal.comp (continuous_const.div (hs.add continuous_const)
    fun _ => (add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε).ne')).mul (hx m)

private theorem continuous_maskedSoftmax {ι : Type*} [Fintype ι] (keep : ι → Bool)
    {x : X → ι → ℝ} (hx : ∀ i, Continuous (fun w => x w i)) (j : ι) :
    Continuous (fun w => maskedSoftmax keep (x w) j) := by
  simp only [maskedSoftmax]
  split
  · rename_i hj
    exact (hx j).rexp.div (continuous_finsetSum _ fun k _ => (hx k).rexp) (fun w =>
      (Finset.sum_pos' (fun k _ => (Real.exp_pos (x w k)).le)
        ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, Real.exp_pos _⟩).ne')
  · exact continuous_const

private theorem continuous_attention (l : Fin layers)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ i, Continuous (fun w => x w i)) (i : Fin T) :
    Continuous (fun w => attentionWith (causalGeometry (Model.fromWeights w).alpha)
      ((Model.fromWeights w).layer l).attentionProjections (Model.fromWeights w).eps
      ((Model.fromWeights w).layer l).scoreMultiplier (x w) i) := by
  let ε : ℝ := 1 / 1000000
  let proj (p : Fin 3) (j : Fin T) (h : Fin H) (w : Weights A H d layers) :=
    streamProjection (fun k h m => complexEntry w (.attentionIn l p k h m)) (x w j) h
  have hp (p j h m) : Continuous (fun w => proj p j h w m) := by
    apply (continuous_apply m).comp
    apply (continuous_apply h).comp
    exact continuous_streamProjection _ (fun k => continuous_pi fun h => continuous_pi fun m =>
      continuous_complexEntry (.attentionIn l p k h m)) (hx j)
  let rotated p j h m (w : Weights A H d layers) :=
    (causalGeometry (Model.fromWeights w).alpha).rotate j (normalizeHead ε (proj p j h w)) m
  have hr (p j h m) : Continuous (rotated p j h m) := by
    have ha : Continuous (fun w : Weights A H d layers => w (.inr m)) := continuous_apply _
    have hf : Continuous (fun w : Weights A H d layers => leropeFrequency 10000 (fun k => w (.inr k)) m) :=
      ha.rexp.mul continuous_const
    exact (Complex.continuous_exp.comp
      ((Complex.continuous_ofReal.comp (hf.const_mul (j.val : ℝ))).mul continuous_const)).mul
      (continuous_normalizeHead (by norm_num : 0 < ε) (hp p j h) m)
  let score (w : Weights A H d layers) (h : Fin H) (j : Fin T) :=
    complexEntry w (.score l) * ((Real.log ((i.val : ℝ) + 1) / Real.sqrt d : ℝ) : ℂ) *
      ∑ m, conj (rotated 0 i h m w) * rotated 1 j h m w
  have hs (h j) : Continuous (fun w => score w h j) :=
    ((continuous_complexEntry (.score l)).mul continuous_const).mul
      (continuous_finsetSum _ fun m _ => (Complex.continuous_conj.comp (hr 0 i h m)).mul (hr 1 j h m))
  let y (w : Weights A H d layers) (h : Fin H) (m : Fin d) := ∑ j,
    (⟨maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).re) j,
      maskedSoftmax (fun k : Fin T => decide (k ≤ i)) (fun k => (score w h k).im) j⟩ : ℂ) *
      proj 2 j h w m
  have hy (h m) : Continuous (fun w => y w h m) := by
    apply continuous_finsetSum
    intro j _
    let keep := fun k : Fin T => decide (k ≤ i)
    have hre := continuous_maskedSoftmax keep (fun k => Complex.continuous_re.comp (hs h k)) j
    have him := continuous_maskedSoftmax keep (fun k => Complex.continuous_im.comp (hs h k)) j
    have hmk := Complex.equivRealProdCLM.symm.continuous.comp (hre.prodMk him)
    exact hmk.mul (hp 2 j h m)
  change Continuous (fun w =>
    headProjection (fun h m => WithLp.toLp 2 fun k => complexEntry w (.attentionOut l h m k)) (y w))
  simp only [headProjection, LinearMap.sum_apply, LinearMap.comp_apply, LinearMap.proj_apply]
  exact continuous_finsetSum _ fun h _ => continuous_vectorProjection _
    (fun m => continuous_stream fun k => continuous_complexEntry (.attentionOut l h m k)) (hy h)

private theorem continuous_ffn (l : Fin layers)
    {x : Weights A H d layers → CSpace (H * d)} (hx : Continuous x) :
    Continuous (fun w => let f := ((Model.fromWeights w).layer l).feedForward
      f.WF (f.Wg (x w) * f.Wu (x w))) := by
  have hp (p : Fin 2) := continuous_streamProjection
    (fun w i j => complexEntry w (.feedForwardIn l p i j))
    (fun i => continuous_pi fun j => continuous_complexEntry (.feedForwardIn l p i j)) hx
  exact continuous_vectorProjection _
    (fun j => continuous_stream fun k => continuous_complexEntry (.feedForwardOut l j k))
    (fun j => (continuous_apply j).comp ((hp 0).mul (hp 1)))

private theorem continuous_sublayer (l : Fin layers) (s : Fin 2) (scale : ℝ)
    (branch : Weights A H d layers → Word (CSpace (H * d)) T → Word (CSpace (H * d)) T)
    (hbranch : ∀ z : Weights A H d layers → Word (CSpace (H * d)) T,
      (∀ j, Continuous (fun w => z w j)) → ∀ j, Continuous (fun w => branch w (z w) j))
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, Continuous (fun w => x w j)) (i : Fin T) :
    Continuous (fun w => sublayer
      ⟨streamProjection fun k => complexEntry w (.residualGate l s 0 k),
       streamProjection fun k => complexEntry w (.residualGate l s 1 k)⟩
      (1 / 1000000) scale (branch w) (x w) i) := by
  have hε : (0 : ℝ) < 1 / 1000000 := by norm_num
  let z := fun w j => argandRMS (1 / 1000000) (x w j)
  have hz (j : Fin T) : Continuous (fun w => z w j) := continuous_argandRMS hε (hx j)
  have hg (g : Fin 2) := continuous_streamProjection (fun w k => complexEntry w (.residualGate l s g k))
    (fun k => continuous_complexEntry (.residualGate l s g k)) (hz i)
  exact (((hg 0).const_add 1).smul (hx i)).add
    ((((hg 1).const_add 1).const_mul (scale : ℂ)).smul
      (continuous_argandRMS hε (hbranch z hz i)))

/-- The forward pass of a full trainable layer is continuous, with both
skip/write sublayers, split causal softmaxes and learned LeRoPE. -/
theorem continuous_forward_fromWeights (l : Fin layers) (scale : ℝ)
    (x : Weights A H d layers → Word (CSpace (H * d)) T)
    (hx : ∀ j, Continuous (fun w => x w j)) (i : Fin T) :
    Continuous (fun w => ((Model.fromWeights w).layer l).forwardWith (Model.fromWeights w).eps scale
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
  have hy (j : Fin T) : Continuous (fun w => y w j) :=
    continuous_sublayer l 0 scale attentionBranch (continuous_attention l) x hx j
  exact continuous_sublayer l 1 scale feedForwardBranch (fun z h j =>
    continuous_ffn l (x := fun w => z w j) (h j)) y hy i

example := continuous_forward_fromWeights (A := Fin 256) (H := 4) (d := 16) (layers := 1) (T := 32)
  0 1 (fun _ _ => 0) (fun _ => continuous_const) 0

example := continuous_argandRMS (c := 64) (ε := 1 / 1000000) (by norm_num)
  (continuous_id : Continuous (id : CSpace 64 → CSpace 64))

end Transformer.Grokking.FiniteRecovery.Argand

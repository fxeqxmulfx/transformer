module

public import Transformer.Grokking.FiniteRecovery.FactorReadout
public import Transformer.Grokking.FiniteRecovery.DivisionProduct

/-! A single native residual layer computes any factored table in the
division prompt format. Its linear query, value and output maps and its
product FFN are shared across the entire class. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open Transformer.PhasorV2

noncomputable def layer : Layer 4 16 :=
  ⟨routeMap common features,multiply,1,⟨0,0⟩,⟨0,0⟩⟩

noncomputable def afterAtt (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput)
    (scale : ℝ) : Fin 5 → Stream :=
  sublayer ⟨0,0⟩ Parity.offset scale (attentionWith (causalGeometry 0) (routeMap common features) Parity.offset 1)
    (input χ r c z x)

noncomputable def operands (χ : Character) (r c : UnitFactor) (x : DivisionInput) (i : Fin 64) : ℂ :=
  if i=5 then rowPhase χ r x.1 else if i=6 then χ (c x.2).toMul.val
    else if i=7 then (zeroValue x.1 : ℂ) else 0

private theorem residual (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput)
    {scale : ℝ} (hs : 0 < scale) :
    ∃ a > (0:ℝ), ∀ i : Fin 64, i ≠ 0 → afterAtt χ r c z x scale 4 i = (a:ℂ)*operands χ r c x i := by
  let out := attentionWith (causalGeometry 0) (routeMap common features) Parity.offset 1 (normalized χ r c z x) (4 : Fin 5)
  have he : afterAtt χ r c z x scale 4 = input χ r c z x 4 + (scale:ℂ) • argandRMS Parity.offset out :=
    sublayer_zero_gates _ _ _ _ _
  have ht : tag z (1:Fin 239) = 0 := by simp [tag]
  have hp := small_features χ (t := 1) (by decide)
  refine ⟨scale * Parity.gain out * numeralGain,by positivity [Parity.gain_pos out,numeralGain_pos],?_⟩
  intro i hi
  have hskip : input χ r c z x 4 i = 0 := by
    change (Parity.gain (vocab χ r c z 1):ℂ) * vocab χ r c z 1 i = 0
    simp [vocab,channels,Channels.vector,EuclideanSpace.single,PiLp.single_apply,unitAt_small (t := 1) (by decide),
      number,residue_small (t := 1) (by decide),hp.1,hp.2,ht,hi]
  rw [he]
  change input χ r c z x 4 i + (scale:ℂ)*((Parity.gain out:ℂ)*out i) = _
  rw [hskip,zero_add,show out i = (numeralGain:ℂ)*operands χ r c x i from routed ..]
  push_cast
  ring

set_option maxHeartbeats 1000000 in
/-- The native product sublayer writes a phase for nonzero numerators
and the constant-row selector for a zero numerator. Both receive the
same positive, input-dependent RMS scale. -/
theorem layer_state (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput)
    {scale : ℝ} (hs : 0 < scale) :
    classifierState χ r c x (layer.forwardWith Parity.offset scale (causalGeometry 0) (input χ r c z x) 4) := by
  let v := afterAtt χ r c z x scale 4
  let out := productValue (argandRMS Parity.offset v)
  obtain ⟨a,ha,hcoord⟩ := residual χ r c z x hs
  have hv (i : Fin 64) (hi : i ≠ 0) : v i = (a:ℂ)*operands χ r c x i := hcoord i hi
  let b := Parity.gain v*a
  have hb : 0 < b := mul_pos (Parity.gain_pos v) ha
  have hz (i : Fin 64) (hi : i ≠ 0) : argandRMS Parity.offset v i = (b:ℂ)*operands χ r c x i := by
    rw [Parity.rms_coord,hv i hi]
    simp [b,mul_assoc]
  have h1 : out 1 = (b:ℂ)^2 * phaseValue χ r c x := by
    change productValue _ 1 = _
    rw [product_coords,hz 5 (by decide),hz 6 (by decide)]
    norm_num [operands,phaseValue]
    ring
  have h3 : out 3 = (b:ℂ)^2 * ((zeroValue x.1)^2 : ℝ) := by
    change productValue _ 3 = _
    rw [product_coords,hz 7 (by decide)]
    norm_num [operands]
    ring
  have hskip (i : Fin 64) (h0 : i ≠ 0) (h5 : i ≠ 5) (h6 : i ≠ 6) (h7 : i ≠ 7) : v i = 0 := by
    simp [hv i h0,operands,h5,h6,h7]
  have he : layer.forwardWith Parity.offset scale (causalGeometry 0) (input χ r c z x) 4 =
      v + (scale:ℂ) • argandRMS Parity.offset out :=
    sublayer_zero_gates _ _ (fun y j => productValue (y j)) (afterAtt χ r c z x scale) 4
  rw [he]
  have hget (i : Fin 64) : (v + (scale:ℂ) • argandRMS Parity.offset out) i =
      v i + (scale:ℂ)*((Parity.gain out:ℂ)*out i) := rfl
  refine ⟨scale*Parity.gain out*b^2,by positivity [Parity.gain_pos out],?_,?_,?_⟩
  · rw [hget,hskip _ (by decide) (by decide) (by decide) (by decide),h1]
    push_cast
    ring
  · rw [hget,hskip _ (by decide) (by decide) (by decide) (by decide),h3]
    push_cast
    ring
  · intro i h0 h1 h3 h5 h6 h7
    rw [hget,hskip i h0 h5 h6 h7]
    simp [out,product_coords,h1,h3]

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

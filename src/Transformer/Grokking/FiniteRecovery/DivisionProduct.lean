module

public import Transformer.Grokking.FiniteRecovery.DivisionOperands

/-! The product FFN in the same layer multiplies the two routed phases
and squares the zero indicator. The attention and FFN residuals use the
ordinary zero gate weights. Their RMS gains are positive on every input. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2
open scoped ComplexConjugate

noncomputable def featureSlot (j : Fin 256) : (Fin 256 → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ : Fin 256 => ℂ) j

noncomputable def multiply : FFN 64 256 where
  Wg := LinearMap.pi fun j => if j = 0 then Parity.coord 5 else Parity.coord 7
  Wu := LinearMap.pi fun j => if j = 0 then Parity.coord 6 else Parity.coord 7
  WF := Parity.assemble fun i => if i = 1 then featureSlot 0 else
    if i = 3 then featureSlot 1 else 0

noncomputable def productValue (v : Stream) : Stream :=
  multiply.WF (multiply.Wg v * multiply.Wu v)

theorem product_coords (v : Stream) (i : Fin 64) :
    productValue v i = if i = 1 then v 5 * v 6 else if i = 3 then v 7 * v 7 else 0 := by
  change (if i = 1 then featureSlot 0 else if i = 3 then featureSlot 1 else
    (0 : (Fin 256 → ℂ) →ₗ[ℂ] ℂ)) (multiply.Wg v * multiply.Wu v) = _
  split
  · rfl
  · split <;> rfl

noncomputable def routingLayer : Layer 4 16 :=
  ⟨attentionMap, multiply, 1, ⟨0,0⟩, ⟨0,0⟩⟩

/-- One native product sublayer writes the tied classifier coordinates for
the quotient. Every one of the 97 possible answers has the same positive
scale in its active coordinate for that input. -/
theorem product_answer {χ : Character} {x : DivisionInput} {v : Stream}
    (hv : operandState χ x v) {scale : ℝ} (hs : 0 < scale) :
    answerState χ (quotient x)
      (v + (scale:ℂ) • argandRMS Parity.offset (productValue (argandRMS Parity.offset v))) := by
  obtain ⟨a,ha,hc⟩ := hv
  let z := argandRMS Parity.offset v
  let out := productValue z
  let b := Parity.gain v * a
  have hb : 0 < b := mul_pos (Parity.gain_pos v) ha
  have hz (i : Fin 64) (hi : i ≠ 0) : z i = (b:ℂ) * scratch χ x i := by
    rw [show z = argandRMS Parity.offset v from rfl,Parity.rms_coord,hc i hi]
    simp [b,mul_assoc]
  have h1 : out 1 = (b:ℂ)^2 * χ (quotient x) := by
    change productValue z 1 = _
    rw [product_coords,hz 5 (by decide),hz 6 (by decide),← phase_product χ x]
    norm_num [scratch]
    ring
  have h3 : out 3 = (b:ℂ)^2 * (flag (residueToken (quotient x)) : ℂ) := by
    change productValue z 3 = _
    rw [product_coords,hz 7 (by decide),← flag_product x]
    norm_num [scratch]
    ring
  have h2 : out 2 = 0 := by change productValue z 2 = 0; norm_num [product_coords]
  have h4 : out 4 = 0 := by change productValue z 4 = 0; norm_num [product_coords]
  have hskip (i : Fin 64) (hi : i ≠ 0) (h5 : i ≠ 5) (h6 : i ≠ 6) (h7 : i ≠ 7) : v i = 0 := by
    simp [hc i hi,scratch,h5,h6,h7]
  have he (i : Fin 64) : (v + (scale:ℂ) • argandRMS Parity.offset out) i =
      v i + (scale:ℂ) * ((Parity.gain out : ℂ) * out i) := rfl
  refine ⟨scale * Parity.gain out * b^2,by positivity [Parity.gain_pos out],?_,?_,?_,?_⟩
  all_goals
    rw [he,hskip _ (by decide) (by decide) (by decide) (by decide),zero_add]
  · rw [h1]; push_cast; ring
  · rw [h2]; ring
  · rw [h3]; push_cast; ring
  · rw [h4]; ring

theorem routingLayer_answer (χ : Character) (x : DivisionInput) {scale : ℝ} (hs : 0 < scale) :
    answerState χ (quotient x)
      (routingLayer.forwardWith Parity.offset scale (causalGeometry 0) (promptInput χ x) 4) := by
  change answerState _ _ (sublayer ⟨0,0⟩ _ _
    (fun z i => productValue (z i)) (afterAttention χ x scale) (4 : Fin 5))
  rw [sublayer_zero_gates]
  exact product_answer (afterAttention_operands χ x hs) hs

example (χ : Character) (x : DivisionInput) := routingLayer_answer χ x (scale := 1) zero_lt_one

end Transformer.Grokking.FiniteRecovery.Argand.Division

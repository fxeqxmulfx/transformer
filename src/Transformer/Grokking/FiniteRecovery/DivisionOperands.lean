module

public import Transformer.Grokking.FiniteRecovery.DivisionOperandRouting
public import Transformer.Grokking.FiniteRecovery.DivisionReadout

/-! The three scratch coordinates left by the ordinary zero-gate attention
residual share a positive real scale. The initial equals-sign embedding
has no component in the classifier or operand coordinates. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 FlatSpectrum
open scoped ComplexConjugate

def numerator (x : DivisionInput) : ZMod 97 := x.1.elim 0 Units.val

def quotient (x : DivisionInput) : ZMod 97 := x.1.elim 0 (fun a => (a/x.2).val)

theorem answer_token (x : DivisionInput) : divisionAnswer x = residueToken (quotient x) := by
  obtain ⟨a,b⟩ := x
  cases a <;> rfl

theorem phase_product (χ : Character) (x : DivisionInput) :
    χ (numerator x) * conj (χ x.2.val) = χ (quotient x) := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  obtain ⟨a,b⟩ := x
  cases a with
  | none => simp [numerator,quotient,χ.map_zero]
  | some a =>
    have h := congrFun (congrFun (FeatureCodes.powerKernel_eq_outer χ 1) a) b
    simpa [numerator,quotient,FeatureCodes.powerKernel,Matrix.vecMulVec_apply,Complex.star_def] using h.symm

theorem flag_product (x : DivisionInput) :
    (flag (residueToken (numerator x)))^2 = flag (residueToken (quotient x)) := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  obtain ⟨a,b⟩ := x
  cases a with
  | none => simp [numerator,quotient,flag]
  | some a =>
    have hn : residueToken a.val ≠ residueToken 0 := fun h => a.ne_zero (residueToken_injective h)
    have hq : residueToken (a/b).val ≠ residueToken 0 := fun h => (a/b).ne_zero (residueToken_injective h)
    change (flag (residueToken a.val))^2 = flag (residueToken (a/b).val)
    rw [flag,flag,ite_eq_right hn,ite_eq_right hq]
    norm_num

noncomputable def scratch (χ : Character) (x : DivisionInput) (i : Fin 64) : ℂ :=
  if i = 5 then χ (numerator x) else if i = 6 then conj (χ x.2.val)
    else if i = 7 then (flag (residueToken (numerator x)) : ℂ) else 0

private theorem decoded (χ : Character) (x : DivisionInput) (i : Fin 64) :
    attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5) i =
      (posGain 1 : ℂ) * scratch χ x i := by
  have hn : divisionPrompt x 1 = residueToken (numerator x) := rfl
  have hd : divisionPrompt x 3 = residueToken x.2.val := rfl
  have h1 : normalized χ x 1 1 = (posGain 1 : ℂ) * χ (numerator x) := by
    change (posGain 1 : ℂ) * embedding χ (divisionPrompt x 1) 1 = _
    rw [hn,(embedding_coords _ _).2.1,phase_residue]
  have h2 : normalized χ x 3 2 = (posGain 1 : ℂ) * conj (χ x.2.val) := by
    change (posGain 3 : ℂ) * embedding χ (divisionPrompt x 3) 2 = _
    rw [hd,(embedding_coords _ _).2.2.1,phase_residue]
    simp [posGain]
  have h3 : normalized χ x 1 3 = (posGain 1 : ℂ) * (flag (residueToken (numerator x)) : ℂ) := by
    change (posGain 1 : ℂ) * embedding χ (divisionPrompt x 1) 3 = _
    rw [hn,(embedding_coords _ _).2.2.2.1]
  rw [routed,h1,h2,h3,scratch]
  split <;> first | rfl | split <;> first | rfl | split <;> simp

noncomputable def promptInput (χ : Character) (x : DivisionInput) (j : Fin 5) : Stream :=
  argandRMS Parity.offset (embedding χ (divisionPrompt x j))

noncomputable def operandState (χ : Character) (x : DivisionInput) (v : Stream) : Prop :=
  ∃ a : ℝ, 0 < a ∧ ∀ i : Fin 64, i ≠ 0 → v i = (a:ℂ) * scratch χ x i

noncomputable def afterAttention (χ : Character) (x : DivisionInput) (scale : ℝ) : Fin 5 → Stream :=
  sublayer ⟨0,0⟩ Parity.offset scale (attentionWith (causalGeometry 0) attentionMap Parity.offset 1)
    (promptInput χ x)

theorem afterAttention_operands (χ : Character) (x : DivisionInput) {scale : ℝ} (hs : 0 < scale) :
    operandState χ x (afterAttention χ x scale 4) := by
  let out := attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5)
  have hin : (fun j => argandRMS Parity.offset (promptInput χ x j)) = normalized χ x :=
    funext (normalized_prompt χ x)
  have he : afterAttention χ x scale 4 = promptInput χ x 4 + (scale:ℂ) • argandRMS Parity.offset out := by
    simp only [afterAttention,sublayer_zero_gates,hin,out]
  have hsmall := small_features χ (t := 1) (by decide)
  refine ⟨scale * Parity.gain out * posGain 1,by positivity [Parity.gain_pos out,posGain_pos 1],?_⟩
  intro i hi
  rw [he]
  have hskip : promptInput χ x 4 i = 0 := by
    change (Parity.gain (embedding χ 1) : ℂ) * embedding χ 1 i = 0
    simp [embedding,hsmall.1,hsmall.2,EuclideanSpace.single,PiLp.single_apply,hi]
  change promptInput χ x 4 i + (scale:ℂ) * ((Parity.gain out : ℂ) * out i) = _
  rw [hskip,zero_add,show out i = _ from decoded χ x i]
  push_cast
  ring

end Transformer.Grokking.FiniteRecovery.Argand.Division

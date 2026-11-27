module

public import Transformer.Grokking.FiniteRecovery.DivisionProduct

/-! Embed one ordinary attention/product layer in the width-64,
six-layer, one-pass native architecture. The zero branches in the other
five layers are identity maps at their native gate and score settings. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2

def identityLayer : Layer 4 16 := ⟨⟨0,0,0,0⟩,⟨0,0,0⟩,1,⟨0,0⟩,⟨0,0⟩⟩

private theorem identityLayer_forward {P : Type*} [Fintype P] (scale : ℝ)
    (geometry : AttentionGeometry P 16) (v : P → Stream) :
    identityLayer.forwardWith Parity.offset scale geometry v = v := by
  funext i
  simp [identityLayer,Layer.forwardWith,sublayer,attentionWith,argandRMS]

noncomputable def sixLayers {A : Type*} (E : A → Stream) (first : Layer 4 16) : Model A 4 16 6 :=
  ⟨E,![first,identityLayer,identityLayer,identityLayer,identityLayer,identityLayer],0,Parity.offset⟩

noncomputable def layerScale : ℝ := 1 / Real.sqrt (2 * ((6:ℕ):ℝ) * ((1:ℕ):ℝ))

theorem layerScale_pos : 0 < layerScale := by dsimp [layerScale]; positivity

theorem sixLayers_hidden {A : Type*} {T : ℕ} (E : A → Stream) (first : Layer 4 16)
    (tokens : Fin T → A) (i : Fin T) : (sixLayers E first).hidden 1 tokens i =
    argandRMS Parity.offset (first.forwardWith Parity.offset layerScale (causalGeometry 0)
      (fun j => argandRMS Parity.offset (E (tokens j))) i) := by
  simp only [Model.hidden,Model.hiddenWith,sixLayers,Function.iterate_one,List.ofFn_succ,
    List.ofFn_zero,List.foldl_cons,List.foldl_nil,Matrix.cons_val_succ,Matrix.cons_val_zero,
    identityLayer_forward]
  rfl

theorem sixLayers_initial {A : Type*} (E : A → Stream) (attn : Attention 64 4 16) (ffn : FFN 64 256) :
    let m := sixLayers E ⟨attn,ffn,1,⟨0,0⟩,⟨0,0⟩⟩
    sgdInitialWeights true m.tensorEntry = m.toWeights := by
  funext i
  rcases i with ⟨p,b⟩ | j
  · cases p with
    | score l => fin_cases l <;> cases b <;> rfl
    | residualGate l s g j => fin_cases l <;> fin_cases s <;> fin_cases g <;> cases b <;> rfl
    | _ => cases b <;> rfl
  · rfl

end Transformer.Grokking.FiniteRecovery.Argand.Division

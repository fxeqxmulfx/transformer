module

public import Transformer.Grokking.FiniteRecovery.ArgandAdamW
public import Transformer.Landscape.Lojasiewicz.LojasiewiczInnerProduct

/-! A bijection of finite vocabularies permutes the input embedding and
the tied readout together. All other native coordinates keep their indices.
The complete labeled-position loss, its Euclidean gradient and the initial
state of `experiments/argand` commute with this change of alphabet. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2

variable {A B : Type*} {H d L T : ℕ}

/-- Only embedding rows carry a token index in the native parameter layout. -/
def ComplexParameter.mapTokens (g : A → B) : ComplexParameter A H d L → ComplexParameter B H d L
  | .embedding a i => .embedding (g a) i
  | .attentionIn l k i h j => .attentionIn l k i h j
  | .attentionOut l h j i => .attentionOut l h j i
  | .feedForwardIn l k i j => .feedForwardIn l k i j
  | .feedForwardOut l j i => .feedForwardOut l j i
  | .score l => .score l
  | .residualGate l s k i => .residualGate l s k i

def tensorTokenEquiv (e : A ≃ B) : ComplexParameter A H d L ≃ ComplexParameter B H d L where
  toFun := ComplexParameter.mapTokens e
  invFun := ComplexParameter.mapTokens e.symm
  left_inv p := by cases p <;> simp [ComplexParameter.mapTokens]
  right_inv p := by cases p <;> simp [ComplexParameter.mapTokens]

def parameterTokenEquiv (e : A ≃ B) : Parameter A H d L ≃ Parameter B H d L :=
  ((tensorTokenEquiv e).prodCongr (Equiv.refl Bool)).sumCongr (Equiv.refl (Fin d))

def relabelWeights (e : A ≃ B) (w : Weights A H d L) : Weights B H d L :=
  w ∘ (parameterTokenEquiv e).symm

def relabelDraw (e : A ≃ B) (z : ComplexParameter A H d L → ℂ) : ComplexParameter B H d L → ℂ :=
  z ∘ (tensorTokenEquiv e).symm

def Model.relabel (e : A ≃ B) (m : Model A H d L) : Model B H d L where
  embedding := m.embedding ∘ e.symm
  layer := m.layer
  alpha := m.alpha
  eps := m.eps

/-- Relabeling the actual storage realizes the corresponding change of model. -/
theorem fromWeights_relabel (e : A ≃ B) (w : Weights A H d L) :
    Model.fromWeights (relabelWeights e w) = (Model.fromWeights w).relabel e := rfl

theorem initialWeights_relabel (e : A ≃ B) (realScore : Bool) (z : ComplexParameter A H d L → ℂ) :
    sgdInitialWeights realScore (relabelDraw e z) = relabelWeights e (sgdInitialWeights realScore z) := by
  funext p
  rcases p with ⟨p,b⟩ | j
  · cases p <;> cases b <;> rfl
  · rfl

theorem Model.hidden_relabel (e : A ≃ B) (m : Model A H d L) (r : ℕ) (x : Word A T) :
    (m.relabel e).hidden r (e ∘ x) = m.hidden r x := by
  simp only [Model.hidden,Model.hiddenWith,Model.relabel,Function.comp_apply,Equiv.symm_apply_apply]

/-- Equality holds at each causal position and for every competing token. -/
theorem Model.logits_relabel (e : A ≃ B) (m : Model A H d L) (r : ℕ) (x : Word A T)
    (i : Fin T) (a : A) : (m.relabel e).logits r (e ∘ x) i (e a) = m.logits r x i a := by
  unfold Model.logits
  rw [Model.hidden_relabel]
  simp only [Model.readout,Model.relabel,Function.comp_apply,Equiv.symm_apply_apply]

def relabelPosition (e : A ≃ B) (p : LabeledPosition A) : LabeledPosition B :=
  ⟨p.1, e ∘ p.2.1, p.2.2.1, e p.2.2.2⟩

variable [Fintype A] [Fintype B]

theorem Model.probability_relabel (e : A ≃ B) (m : Model A H d L) (r : ℕ) (x : Word A T)
    (i : Fin T) (a : A) :
    (m.relabel e).probability r (e ∘ x) i (e a) = m.probability r x i a := by
  dsimp only [Model.probability]
  rw [← e.sum_comp (fun b => Real.exp ((m.relabel e).logits r (e ∘ x) i b))]
  simp only [Model.logits_relabel]

theorem batchLoss_relabel (e : A ≃ B) (r : ℕ) (u : List (LabeledPosition A))
    (w : Weights A H d L) :
    batchLoss r (u.map (relabelPosition e)) (relabelWeights e w) = batchLoss r u w := by
  simp only [batchLoss,List.map_map,List.length_map,fromWeights_relabel]
  congr 2
  apply List.map_congr_left
  intro p _
  change -Real.log (((Model.fromWeights w).relabel e).probability r (e ∘ p.2.1) p.2.2.1
    (e p.2.2.2)) = _
  rw [Model.probability_relabel]

/-- Reindexing preserves the actual L2 metric used for differentiation and
global gradient clipping, with all real and imaginary coordinates included. -/
theorem batchGradient_relabel (e : A ≃ B) (r : ℕ) (u : List (LabeledPosition A))
    (w : Weights A H d L) :
    batchGradient r (u.map (relabelPosition e)) (relabelWeights e w) =
      relabelWeights e (batchGradient r u w) := by
  let E := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (parameterTokenEquiv (H := H) (d := d) (L := L) e)
  let f := fun v : Transformer.AMSGradW.Kernite.Space (Parameter A H d L) => batchLoss r u (WithLp.ofLp v)
  have hfun : (fun v : Transformer.AMSGradW.Kernite.Space (Parameter B H d L) =>
      batchLoss r (u.map (relabelPosition e)) (WithLp.ofLp v)) =
      f ∘ E.symm := by
    funext v
    obtain ⟨x,rfl⟩ := E.surjective v
    simp only [Function.comp_apply,E.symm_apply_apply]
    exact batchLoss_relabel e r u (WithLp.ofLp x)
  unfold batchGradient
  rw [hfun,Transformer.Landscape.Lojasiewicz.gradient_comp_linearIsometryEquiv_symm]
  change WithLp.ofLp (E (gradient f (E.symm (E (WithLp.toLp 2 w))))) = _
  rw [E.symm_apply_apply]
  rfl

end Transformer.Grokking.FiniteRecovery.Argand

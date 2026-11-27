module

public import Transformer.Grokking.FiniteRecovery.ArgandParityDraw
public import Transformer.Grokking.FiniteRecovery.ArgandGaussianRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandPAGE
public import Transformer.Optimization.PAGE.FiniteHorizon

/-! The observed-loss PAGE path with independently Gaussian native tensor draws.
Both parity and its complement occur with positive probability on an entire
finite training horizon, at the same positive step and on the same labels. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.Optimization.PAGE MeasureTheory Set Metric
open scoped NNReal

variable {I : Type*} [Fintype I] {H d L T : ℕ}

/-- All native coordinates, including the initially zero residual gates, take
part in each PAGE weight update. The observation indices are the only labels. -/
noncomputable def observedPAGE (u : I → Word (ZMod 2) (T+1)) (f : Word (ZMod 2) (T+1) → ZMod 2)
    (r : ℕ) (η : ℝ) (draw : ComplexParameter (ZMod 2) H d L → ℂ) (path : ℕ → Option I)
    (t : ℕ) : Weights (ZMod 2) H d L := WithLp.ofLp
  (run (observedComponent r u f) η (WithLp.toLp 2 (sgdInitialWeights true draw)) path t).weights

omit [Fintype I] in
private theorem continuous_component (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (r : ℕ) (i : I) :
    Continuous (gradient (observedComponent (H := H) (d := d) (layers := L) r u f i)) :=
  (locallyLipschitz_fullGradient _ _ _).continuous

/-- A strict classifier in the native support persists with positive Gaussian
probability throughout any prescribed finite training horizon, even if its
predictions disagree with the labels used for every update. -/
theorem gaussian_page_prefix (variance : ComplexParameter (ZMod 2) H d L → ℝ≥0)
    (hv : ∀ i, 0 < variance i) (r K : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f g : Word (ZMod 2) (T+1) → ZMod 2) (witness : ComplexParameter (ZMod 2) H d L → ℂ)
    (hw : (Model.fromWeights (sgdInitialWeights true witness)).realizes r g) :
    ∃ δ > 0, ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ η : ℝ, |η| < δ → ∀ draw ∈ U, ∀ path : ℕ → Option I, ∀ t ≤ K,
        (Model.fromWeights (observedPAGE u f r η draw path t)).realizes r g := by
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter (ZMod 2) H d L => ℝ)
  let init := fun draw => C.symm (sgdInitialWeights true draw)
  have hi : Continuous init := C.symm.continuous.comp (continuous_sgdInitialWeights true)
  let G := {w : WeightSpace (ZMod 2) H d L | (Model.fromWeights (C w)).realizes r g}
  have hG : IsOpen G := (isOpen_realizingWeights r g).preimage C.continuous
  obtain ⟨δ,hδ,h⟩ := stays_open_finite (observedComponent (H := H) (d := d) (layers := L) r u f)
    (continuous_component u f r) K hG (x := init witness) hw
  let U := init ⁻¹' ball (init witness) δ
  have hU : IsOpen U := isOpen_ball.preimage hi
  let := independentGaussianDraws_openPos variance hv
  exact ⟨δ,hδ,U,hU,hU.measure_pos _ ⟨witness, mem_ball_self hδ⟩,
    fun η hη draw hd => h η hη (init draw) hd⟩

/-- At any specified constant step and finite horizon, all binary gaps can
remain arbitrarily small with positive Gaussian probability, on every path
and every word. The open event contains nonzero embedding draws. -/
theorem gaussian_page_small_gaps (variance : ComplexParameter (ZMod 2) H d L → ℝ≥0)
    (hv : ∀ i, 0 < variance i) (r K : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (η : ℝ) {γ : ℝ} (hγ : 0 < γ) :
    ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ draw ∈ U, ∀ path : ℕ → Option I, ∀ t ≤ K, ∀ x : Word (ZMod 2) (T+1),
        let m := Model.fromWeights (observedPAGE u f r η draw path t)
        |m.logits r x (Fin.last T) 0 - m.logits r x (Fin.last T) 1| < γ := by
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter (ZMod 2) H d L => ℝ)
  let G : Set (WeightSpace (ZMod 2) H d L) := {w | ∀ x : Word (ZMod 2) (T+1),
    let m := Model.fromWeights (C w)
    |m.logits r x (Fin.last T) 0 - m.logits r x (Fin.last T) 1| < γ}
  have hG : IsOpen G := by
    simp only [G, ofPred_forall]
    refine isOpen_iInter_of_finite fun x => isOpen_lt ?_ continuous_const
    exact (((continuous_logits_fromWeights r x _ 0).comp C.continuous).sub
      ((continuous_logits_fromWeights r x _ 1).comp C.continuous)).abs
  let init := fun draw => C.symm (sgdInitialWeights true draw)
  have hi : Continuous init := C.symm.continuous.comp (continuous_sgdInitialWeights true)
  let G' := {z : ℝ × WeightSpace (ZMod 2) H d L | ∀ path : ℕ → Option I, ∀ t ≤ K,
    (run (observedComponent r u f) z.1 z.2 path t).weights ∈ G}
  let U := (fun d => (η,init d)) ⁻¹' G'
  have hU : IsOpen U := (isOpen_all_prefix _ (continuous_component u f r) K hG).preimage
    (continuous_const.prodMk hi)
  let z : ComplexParameter (ZMod 2) H d L → ℂ := fun _ => 0
  have hz : ZeroEmbeddings (sgdInitialWeights true z) := by intro a j b; cases b <;> rfl
  have he (a : ZMod 2) : (Model.fromWeights (sgdInitialWeights true z)).embedding a = 0 := by
    ext j
    exact Complex.ext (hz a j false) (hz a j true)
  have hmem : z ∈ U := by
    intro path t _
    have hf := (page_zero_embeddings_fixed r u f _ hz η path t).1
    change (run (observedComponent r u f) η (WithLp.toLp 2 (sgdInitialWeights true z)) path t).weights ∈ G
    rw [hf]
    intro x
    change |(Model.fromWeights (sgdInitialWeights true z)).logits r x _ 0 -
      (Model.fromWeights (sgdInitialWeights true z)).logits r x _ 1| < γ
    simpa only [Model.logits, Model.readout, he, inner_zero_left, Complex.zero_re, zero_div,
      sub_zero, abs_zero] using hγ
  let := independentGaussianDraws_openPos variance hv
  exact ⟨U,hU,hU.measure_pos _ ⟨z,hmem⟩,fun _ h => h⟩

private def flipRows (draw : ComplexParameter (ZMod 2) H d L → ℂ) :
    ComplexParameter (ZMod 2) H d L → ℂ
  | .embedding a i => draw (.embedding (a+1) i)
  | p => draw p

private theorem flipped_logits (draw : ComplexParameter (ZMod 2) H d L → ℂ) (r : ℕ)
    (x : Word (ZMod 2) (T+1)) (i : Fin (T+1)) (a : ZMod 2) :
    (Model.fromWeights (sgdInitialWeights true (flipRows draw))).logits r x i a =
      (Model.fromWeights (sgdInitialWeights true draw)).logits r (fun j => x j+1) i (a+1) := by
  rfl

private theorem flipped_parity {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) k)
    (x : Word (ZMod 2) (k+k)) : modularSum e (fun j => x j+1) = modularSum e x := by
  have hc : ((k+k : ℕ) : ZMod 2) = 0 := by
    rw [← two_mul, Nat.cast_mul, show ((2 : ℕ) : ZMod 2) = 0 by decide, zero_mul]
  simp [Parity.modularSum_eq_ones, Parity.ones, Nat.cast_sum,
    Finset.sum_add_distrib, hc]

/-- For an even input length, permuting the two tied embedding rows flips all
output decisions while complementing every input leaves its parity unchanged. -/
theorem modularSum_initial_or_complement {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (b : ZMod 2) :
    ∃ d : ComplexParameter (ZMod 2) 4 16 6 → ℂ,
      (Model.fromWeights (sgdInitialWeights true d)).realizes (T := (k+1)+k) 1
        (fun x => modularSum e x+b) := by
  obtain ⟨d,hd⟩ := Parity.exists_initial_realizes (n := (k+1)+k) hk
  have h0 : (Model.fromWeights (sgdInitialWeights true d)).realizes (T := (k+1)+k) 1 (modularSum e) := by
    rwa [funext (Parity.modularSum_eq_ones e)]
  have h1 : (Model.fromWeights (sgdInitialWeights true (flipRows d))).realizes (T := (k+1)+k) 1
      (fun x => modularSum e x+1) := by
    intro x a ha
    dsimp only at ha ⊢
    have two : (1 : ZMod 2) + 1 = 0 := by decide
    have cancel (a : ZMod 2) : a+1+1 = a := by rw [add_assoc, two, add_zero]
    rw [flipped_logits, flipped_logits, cancel]
    have hn : a+1 ≠ modularSum e x := by
      intro h
      apply ha
      rw [← h, cancel]
    let y : Word (ZMod 2) ((k+1)+(k+1)) := fun j => x j+1
    have he : modularSum e y = modularSum e x := flipped_parity e x
    exact (h0 y (a+1) (fun h => hn (h.trans he))).trans_eq
      (congrArg (fun a => (Model.fromWeights (sgdInitialWeights true d)).logits 1 y
        (Fin.last ((k+1)+k)) a) he)
  have hb : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  rcases hb b with rfl | rfl
  · exact ⟨d, by simpa using h0⟩
  · exact ⟨flipRows d,h1⟩

/-- Identical observed labels and the same step size admit two open sets of
native starts. Every sampled path in the first classifies every word correctly
up to K. Every path in the second reverses all binary decisions up to K. -/
theorem modularSum_gaussian_page_prefix {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (variance : ComplexParameter (ZMod 2) 4 16 6 → ℝ≥0)
    (hv : ∀ i, 0 < variance i) (u : I → Word (ZMod 2) ((k+1)+(k+1))) (K : ℕ) :
    ∃ δ > 0, ∀ b : ZMod 2, ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ η : ℝ, |η| < δ → ∀ draw ∈ U, ∀ path : ℕ → Option I, ∀ t ≤ K,
        (Model.fromWeights (observedPAGE u (modularSum e) 1 η draw path t)).realizes
          (T := (k+1)+k) 1 (fun x => modularSum e x + b) := by
  obtain ⟨d,h0⟩ := modularSum_initial_or_complement e hk 0
  obtain ⟨v,h1⟩ := modularSum_initial_or_complement e hk 1
  obtain ⟨δ,hδ,U,hU,hp,good⟩ := gaussian_page_prefix variance hv 1 K u (modularSum e) _ d h0
  obtain ⟨ε,hε,V,hV,hq,bad⟩ := gaussian_page_prefix variance hv 1 K u (modularSum e) _ v h1
  refine ⟨min δ ε, lt_min hδ hε, fun b => ?_⟩
  have hb : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  rcases hb b with rfl | rfl
  · exact ⟨U,hU,hp,fun η hη draw hd path t ht => by
      simpa using good η (hη.trans_le (min_le_left _ _)) draw hd path t ht⟩
  · exact ⟨V,hV,hq,fun η hη => bad η (hη.trans_le (min_le_right _ _))⟩

end Transformer.Grokking.FiniteRecovery.Argand

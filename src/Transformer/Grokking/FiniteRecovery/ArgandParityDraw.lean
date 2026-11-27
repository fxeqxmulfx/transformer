module

public import Transformer.Grokking.FiniteRecovery.ArgandModularSum

/-! A parity witness inside the native initialization support. A small input
embedding leaves a small uncancelled skip. Rescaling the value projection
keeps the pooling branch exactly the same. The reference with a variable
skip is used only to prove continuity at the strict parity witness. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Parity

open Transformer.PhasorV2 Set Metric
open scoped InnerProductSpace Topology

private noncomputable def referencePool (n : ℕ) (a : ℝ) : Layer 4 16 :=
  { poolLayer n with attentionGates :=
      ⟨((1-a : ℝ) : ℂ) • (poolLayer n).attentionGates.skip, 0⟩ }

private noncomputable def reference (n : ℕ) (a : ℝ) : Model (ZMod 2) 4 16 6 :=
  { model n with layer := ![referencePool n a, productLayer 16, productLayer 8,
      productLayer 4, productLayer 2, productLayer 1] }

private theorem reference_zero (n : ℕ) : reference n 0 = model n := by
  simp [reference, referencePool, model, poolLayer]

set_option maxHeartbeats 1000000 in
private theorem continuous_reference (n : ℕ) : Continuous (fun a => (reference n a).toWeights) := by
  apply continuous_pi
  rintro (⟨p,b⟩ | j)
  · cases p with
    | residualGate l s g j =>
      fin_cases l <;> fin_cases s <;> fin_cases g <;> cases b <;>
        first | exact continuous_const
              | simp [Model.toWeights, Model.tensorEntry, reference, referencePool]; fun_prop
    | embedding a j => cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, model] <;> exact continuous_const
    | attentionIn l k i h j => fin_cases l <;> cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, referencePool, model] <;> exact continuous_const
    | attentionOut l h j i => fin_cases l <;> cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, referencePool, model] <;> exact continuous_const
    | feedForwardIn l k i j => fin_cases l <;> cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, referencePool, model] <;> exact continuous_const
    | feedForwardOut l j i => fin_cases l <;> cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, referencePool, model] <;> exact continuous_const
    | score l => fin_cases l <;> cases b <;>
        dsimp [Model.toWeights, Model.tensorEntry, reference, referencePool, model] <;> exact continuous_const
  · change Continuous (fun _ : ℝ => (0 : ℝ))
    exact continuous_const

private noncomputable def sampledPool (n : ℕ) (r : ℝ) : Layer 4 16 :=
  { poolLayer n with
    attentionProjections := { pool n with WV := ((inputGain / radial (radial r) : ℝ) : ℂ) • (pool n).WV }
    scoreMultiplier := 1
    attentionGates := ⟨0,0⟩ }

private noncomputable def sampledProduct (b : ℕ) : Layer 4 16 :=
  { productLayer b with scoreMultiplier := 1 }

private noncomputable def sampled (n : ℕ) (r : ℝ) : Model (ZMod 2) 4 16 6 :=
  { model n with embedding := fun a => (r : ℂ) • embedding a
                 layer := ![sampledPool n r, sampledProduct 16, sampledProduct 8,
                   sampledProduct 4, sampledProduct 2, sampledProduct 1] }

private theorem initial_sampled (n : ℕ) (r : ℝ) :
    sgdInitialWeights true (sampled n r).tensorEntry = (sampled n r).toWeights := by
  funext p
  rcases p with ⟨p,b⟩ | j
  · cases p with
    | score l => fin_cases l <;> cases b <;> rfl
    | residualGate l s g j => fin_cases l <;> fin_cases s <;> fin_cases g <;> cases b <;> rfl
    | _ => cases b <;> rfl
  · rfl

private theorem pool_match {n T : ℕ} {r : ℝ} (hr : 0 < r) (scale : ℝ)
    (x : Word (ZMod 2) T) (i : Fin T) :
    (sampledPool n r).forward offset scale 0
        (fun j => argandRMS offset ((r : ℂ) • embedding (x j))) i =
      (referencePool n (radial r / radial 1)).forward offset scale 0
        (fun j => argandRMS offset (embedding (x j))) i := by
  have hR := radial_pos hr
  have hD := radial_pos hR
  have hOne := radial_pos (r := 1) zero_lt_one
  have h1 (a : ZMod 2) : argandRMS offset (embedding a) = (radial 1 : ℂ) • embedding a := by
    simpa using rms_row (r := 1) zero_lt_one a
  let q := radial r / radial 1
  have hgate (a : ZMod 2) : (referencePool n q).attentionGates.skip
      ((inputGain : ℂ) • embedding a) = (q : ℂ) - 1 := by
    change (↑(1-q) : ℂ) * (-((inputGain : ℂ)⁻¹) *
      ((inputGain : ℂ) * embedding a 0 + (inputGain : ℂ) * embedding a 1)) = _
    rw [← mul_add, (row_coords a).1, mul_one, neg_mul, inv_mul_cancel₀ (by exact_mod_cast inputGain_pos.ne')]
    push_cast
    ring
  have hvalue (a : ZMod 2) : (sampledPool n r).attentionProjections.WV
      ((radial (radial r) : ℂ) • embedding a) = (pool n).WV ((inputGain : ℂ) • embedding a) := by
    change ((↑(inputGain / radial (radial r)) : ℂ) • (pool n).WV) _ = _
    simp only [LinearMap.smul_apply, map_smul, smul_smul, Complex.ofReal_div]
    rw [mul_comm, div_mul_cancel₀ _ (show (radial (radial r) : ℂ) ≠ 0 by exact_mod_cast hD.ne')]
  have hatt : attentionWith (causalGeometry 0) (sampledPool n r).attentionProjections offset 1
        (fun j => (radial (radial r) : ℂ) • embedding (x j)) i =
      attentionWith (causalGeometry 0) (pool n) offset 0
        (fun j => (inputGain : ℂ) • embedding (x j)) i := by
    unfold attentionWith
    simp only [hvalue]
    congr 1
    simp [sampledPool, pool, causalGeometry, normalizeHead, turn, sqLen]
  have hskip : (q : ℂ) • ((radial 1 : ℂ) • embedding (x i)) = (radial r : ℂ) • embedding (x i) := by
    rw [smul_smul, show (q : ℂ) * (radial 1 : ℂ) = (radial r : ℂ) by
      dsimp [q]; push_cast; field_simp [hOne.ne']]
  change (sampledPool n r).forward _ _ _ _ _ = (referencePool n q).forward _ _ _ _ _
  simp only [Layer.forward, Layer.forwardWith, sublayer, rms_row hr, rms_row hR,
    normalized_input, hgate, add_sub_cancel]
  simp only [h1, hskip]
  simp only [sampledPool, referencePool, poolLayer, LinearMap.zero_apply,
    smul_zero, map_zero, zero_mul, argandRMS, add_zero, one_smul, mul_one] at hatt ⊢
  rw [hatt]

private theorem product_match {T : ℕ} (b : ℕ) (scale : ℝ) (x : Word Stream T) :
    (sampledProduct b).forward offset scale 0 x = (productLayer b).forward offset scale 0 x := by
  simp [sampledProduct, productLayer, Layer.forward, Layer.forwardWith, sublayer, attentionWith]

set_option maxHeartbeats 1000000 in
private theorem logits_match {n : ℕ} {r : ℝ} (hr : 0 < r) (x : Word (ZMod 2) (n+1))
    (i : Fin (n+1)) (a : ZMod 2) :
    (sampled (n+1) r).logits 1 x i a =
      r * (reference (n+1) (radial r / radial 1)).logits 1 x i a := by
  let scale := 1 / Real.sqrt (2 * ((6 : ℕ) : ℝ) * ((1 : ℕ) : ℝ))
  have hpool := funext (pool_match (n := n+1) hr scale x)
  have hh : (sampled (n+1) r).hidden 1 x i = (reference (n+1) (radial r / radial 1)).hidden 1 x i := by
    simp only [Model.hidden, Model.hiddenWith, sampled, reference, model, Function.iterate_one,
      List.ofFn_succ, List.ofFn_zero, List.foldl_cons, List.foldl_nil, Matrix.cons_val_succ, Matrix.cons_val_zero]
    unfold Layer.forward at hpool
    have hp (b : ℕ) (z : Word Stream (n+1)) := product_match b scale z
    unfold Layer.forward at hp
    dsimp only [scale] at hpool hp
    rw [hpool]
    simp only [hp]
  unfold Model.logits
  rw [hh]
  simp only [Model.readout, sampled, reference, model, inner_smul_left,
    Complex.conj_ofReal, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
  ring

/-- At each nonempty length up to 32, some raw tensor draw has strictly correct
parity even after the native initializer sets every gate and frequency to zero
and every score to one. All of these coordinates are trainable afterwards. -/
theorem exists_initial_realizes {n : ℕ} (hn : n+1 ≤ 32) :
    ∃ draw, (Model.fromWeights (sgdInitialWeights (H := 4) (d := 16) (layers := 6) true draw)).realizes
      (T := n) 1 (fun x => (ones x : ZMod 2)) := by
  let f := fun x : Word (ZMod 2) (n+1) => (ones x : ZMod 2)
  let U := {a : ℝ | (Model.fromWeights (reference (n+1) a).toWeights).realizes 1 f}
  have hU : IsOpen U := (isOpen_realizingWeights 1 f).preimage (continuous_reference (n+1))
  have hz : 0 ∈ U := by simpa [U, f, reference_zero] using native_realizes_ones hn
  have hlim : ContinuousAt (fun r => radial r / radial 1) 0 := by
    unfold radial
    fun_prop (disch := norm_num [offset])
  have hm : ∀ᶠ r in nhds (0 : ℝ), radial r / radial 1 ∈ U :=
    hlim.eventually (by simpa [radial] using hU.mem_nhds hz)
  obtain ⟨r, hr, hu⟩ := ((Filter.Eventually.filter_mono nhdsWithin_le_nhds hm).and
    (self_mem_nhdsWithin : ∀ᶠ r in nhdsWithin (0 : ℝ) (Ioi 0), 0 < r)).exists
  refine ⟨(sampled (n+1) r).tensorEntry, ?_⟩
  rw [initial_sampled, Model.fromWeights_toWeights _ rfl]
  change (Model.fromWeights (reference (n+1) (radial r / radial 1)).toWeights).realizes 1 f at hr
  rw [Model.fromWeights_toWeights _ rfl] at hr
  intro x a ha
  rw [logits_match hu, logits_match hu]
  exact mul_lt_mul_of_pos_left (hr x a ha) hu

example := exists_initial_realizes (n := 31) (by decide)

end Transformer.Grokking.FiniteRecovery.Argand.Parity

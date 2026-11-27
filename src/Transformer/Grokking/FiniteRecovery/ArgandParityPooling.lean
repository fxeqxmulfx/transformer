module

public import Transformer.Grokking.FiniteRecovery.ArgandParityCircuit

/-! The first layer counts the one-tokens using the final causal attention row
of ARGAND.md. The split softmax contributes the same factor 1+i in every head.
The output projection removes it. Both tied input rows have the same norm. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Parity

open Transformer.PhasorV2 Finset

noncomputable def embedding (a : ZMod 2) : Stream :=
  EuclideanSpace.single (if a = 0 then 1 else 0) 1

noncomputable def radial (r : ℝ) : ℝ := Real.sqrt 64 / (r + offset) * r

theorem radial_pos {r : ℝ} (h : 0 < r) : 0 < radial r := by
  dsimp [radial, offset]
  positivity

/-- RMS maps a positive multiple of a token row to its scalar radial response. -/
theorem rms_row {r : ℝ} (hr : 0 < r) (a : ZMod 2) :
    argandRMS offset ((r : ℂ) • embedding a) = ((radial r : ℝ) : ℂ) • embedding a := by
  have he : ‖embedding a‖ = 1 := by simp [embedding, EuclideanSpace.single]
  rw [argandRMS, norm_smul, he, Complex.norm_real, Real.norm_of_nonneg hr.le, mul_one, smul_smul]
  norm_num [radial, Complex.ofReal_mul]

noncomputable def inputGain : ℝ := radial (radial 1)

theorem inputGain_pos : 0 < inputGain := radial_pos (radial_pos zero_lt_one)

theorem normalized_input (a : ZMod 2) :
    argandRMS offset (argandRMS offset (embedding a)) = (inputGain : ℂ) • embedding a := by
  have h := rms_row zero_lt_one a
  simp only [Complex.ofReal_one, one_smul] at h
  rw [h, rms_row (radial_pos zero_lt_one)]
  rfl

noncomputable def joint : ℂ := ⟨1,1⟩

private theorem joint_ne : joint ≠ 0 := by intro h; have := congrArg Complex.re h; norm_num [joint] at this

noncomputable def headCoord (j : Fin 16) : Heads 4 16 →ₗ[ℂ] ℂ :=
  (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 16 => ℂ) j).comp
    (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 4 => Fin 16 → ℂ) 0)

/-- Common mass in value slot zero and mass on token one in value slot one. -/
noncomputable def pool (n : ℕ) : Attention 64 4 16 where
  WQ := 0
  WK := 0
  WV := LinearMap.pi fun _ => LinearMap.pi fun j =>
    if j = 0 then coord 0 + coord 1 else coord 0
  WO := assemble fun i => if 32 ≤ i.val then
    joint⁻¹ • ((((i.val : ℂ) - 32 + 1 / 2) • headCoord 0) - (n : ℂ) • headCoord 1) else 0

/-- A common skip gate clears the two embedding rows before the count is written. -/
noncomputable def poolLayer (n : ℕ) : Layer 4 16 where
  attentionProjections := pool n
  feedForward := ⟨0,0,0⟩
  scoreMultiplier := 0
  attentionGates := ⟨(-((inputGain : ℂ)⁻¹)) • (coord 0 + coord 1), 0⟩
  feedForwardGates := ⟨0,0⟩

/-- The two input coordinates encode the constant one and the token indicator. -/
theorem row_coords (a : ZMod 2) :
    embedding a 0 + embedding a 1 = 1 ∧ embedding a 0 = (a.val : ℂ) := by
  have h : ∀ b : ZMod 2, b = 0 ∨ b = 1 := by decide
  rcases h a with rfl | rfl <;>
    norm_num [embedding, EuclideanSpace.single, PiLp.single_apply, ZMod.val_one_eq_one_mod]

private theorem pool_mask {n : ℕ} (j : Fin (n+1)) :
    maskedSoftmax ((causalGeometry (0 : Fin 16 → ℝ)).visible (Fin.last n))
      (fun _ => 0) j = 1 / ((n+1 : ℕ) : ℝ) := by
  simp [maskedSoftmax, causalGeometry, Fin.le_last]

private theorem pool_coords {n : ℕ} (x : Word (ZMod 2) (n+1)) (i : Fin 64) :
    attentionWith (causalGeometry 0) (pool (n+1)) offset 0
        (fun j => (inputGain : ℂ) • embedding (x j)) (Fin.last n) i =
      if 32 ≤ i.val then ((inputGain * value (ones x) 0 i.val : ℝ) : ℂ) else 0 := by
  let N := n+1
  let G : ℂ := inputGain
  let w : ℂ := joint / N
  let z (j : Fin (n+1)) := (inputGain : ℂ) • embedding (x j)
  have hn : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.succ_ne_zero n)
  have hw : (⟨1 / (N : ℝ), 1 / (N : ℝ)⟩ : ℂ) = w := by
    change _ = joint / ((N : ℝ) : ℂ)
    apply Complex.ext <;> simp [joint]
  have h0 (j : Fin (n+1)) : (pool N).WV (z j) 0 0 = G := by
    change G * embedding (x j) 0 + G * embedding (x j) 1 = G
    rw [← mul_add, (row_coords _).1, mul_one]
  have h1 (j : Fin (n+1)) : (pool N).WV (z j) 0 1 = G * (x j).val := by
    change G * embedding (x j) 0 = _
    rw [(row_coords _).2]
  have sum0 : (∑ j, w * (pool N).WV (z j) 0 0) = N * (w * G) := by simp [h0, N]
  have sum1 : (∑ j, w * (pool N).WV (z j) 0 1) = w * G * (ones x : ℕ) := by
    simp only [h1, ← mul_assoc, ← mul_sum, ones, Nat.cast_sum]
  unfold attentionWith
  simp only [zero_mul, Complex.zero_re, Complex.zero_im, pool_mask]
  change (pool N).WO (fun h k => ∑ j, (⟨1/(N:ℝ), 1/(N:ℝ)⟩ : ℂ) * (pool N).WV (z j) h k) i = _
  rw [hw]
  change (if 32 ≤ i.val then _ else (0 : Heads 4 16 →ₗ[ℂ] ℂ)) _ = _
  split
  · change joint⁻¹ * (_ * (∑ j, w * (pool N).WV (z j) 0 0) -
      (N:ℂ) * (∑ j, w * (pool N).WV (z j) 0 1)) = _
    rw [sum0, sum1]
    simp only [value, Complex.ofReal_mul, Complex.ofReal_sub, Complex.ofReal_add,
      Complex.ofReal_div, Complex.ofReal_natCast, Complex.ofReal_one, Complex.ofReal_ofNat]
    dsimp only [w, G]
    field_simp [hn, joint_ne]
  · rfl

/-- At the last position the first layer writes all 32 half-integer factors. -/
theorem poolLayer_band {n : ℕ} (x : Word (ZMod 2) (n+1)) {scale : ℝ} (hs : 0 < scale) :
    Band 32 0 (ones x) ((poolLayer (n+1)).forward offset scale 0
      (fun j => argandRMS offset (embedding (x j))) (Fin.last n)) := by
  let out := attentionWith (causalGeometry 0) (pool (n+1)) offset 0
    (fun j => (inputGain : ℂ) • embedding (x j)) (Fin.last n)
  have hc : (inputGain : ℂ) ≠ 0 := by exact_mod_cast inputGain_pos.ne'
  have sk (a : ZMod 2) : (poolLayer (n+1)).attentionGates.skip ((inputGain : ℂ) • embedding a) = -1 := by
    change -((inputGain : ℂ)⁻¹) * (↑inputGain * embedding a 0 + ↑inputGain * embedding a 1) = -1
    rw [← mul_add, (row_coords a).1, mul_one]
    field_simp
  have he : (poolLayer (n+1)).forward offset scale 0
      (fun j => argandRMS offset (embedding (x j))) (Fin.last n) = (scale : ℂ) • argandRMS offset out := by
    simp only [Layer.forward, Layer.forwardWith, sublayer, normalized_input, sk, add_neg_cancel,
      zero_smul, zero_add]
    simp [poolLayer, argandRMS, out]
  rw [he]
  constructor
  · intro j hj
    change (scale : ℂ) * ((gain out : ℂ) * out j) = 0
    rw [show out j = _ from pool_coords x j, ite_eq_right (by omega), mul_zero, mul_zero]
  · refine ⟨scale * gain out * inputGain, by positivity [gain_pos out, inputGain_pos], ?_⟩
    intro j hj _
    change (scale : ℂ) * ((gain out : ℂ) * out j) = _
    rw [show out j = _ from pool_coords x j, ite_eq_left hj]
    push_cast
    ring

end Transformer.Grokking.FiniteRecovery.Argand.Parity

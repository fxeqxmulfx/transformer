import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSigns

/-!
# A missing product circuit cannot be discovered from zero seeds

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
product logits and one-zero-factor initialization; native AdamW at
lab commit 6c335bd. Unlike the paper's positive second-factor Gen seed,
set both Gen factors and their actual native buffers to zero. Actual
CE derivatives in these coordinates are then zero, despite its negative
product-weight slope. Clipping, retained moments and uniform native
decay cannot create this completely absent fixed circuit.

The invariant is proved on the closed feedback trajectory. A zero
current gradient alone is insufficient with nonzero retained momentum;
zero actual initial buffers are essential here. With nonnegative Mem
factors, true held-out CE stays at least its uniform-initialization value
log q. This exact-real fixed-table obstruction neither models stochastic
feature discovery nor refutes the paper's positive-seed simulation.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Both actual Gen derivatives vanish at two zero factors even
while Mem may be active. Source: appendix C product chain rule;
the common training CE slope itself remains strictly negative. -/
theorem native_zero_gen_raw_gradient (remaining : ℕ) (state : NativeSubweightState)
    (h0 : (state 0).parameter = 0) (h1 : (state 1).parameter = 0) :
    rawNativeSubweightGradient remaining state 0 = 0 ∧
      rawNativeSubweightGradient remaining state 1 = 0 := by
  constructor
  · rw [native_raw_gradient_partner]
    change (state 1).parameter * _ = 0
    rw [h1, zero_mul]
  · rw [native_raw_gradient_partner]
    change (state 0).parameter * _ = 0
    rw [h0, zero_mul]

example : (seededNativeSubweights ((0, 0), (1, 1)) 0).parameter = 0 ∧
    (seededNativeSubweights ((0, 0), (1, 1)) 1).parameter = 0 := by
  exact ⟨rfl, rfl⟩

/-- The shared clipping wrapper cannot insert a gradient into the
absent pair. Source: clip_grad_norm_ at 6c335bd, scaling actual CE
partials rather than supplying an external circuit-activation signal. -/
theorem native_zero_gen_applied_gradient (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (h0 : (state 0).parameter = 0) (h1 : (state 1).parameter = 0) :
    appliedNativeSubweightGradient remaining bound state 0 = 0 ∧
      appliedNativeSubweightGradient remaining bound state 1 = 0 := by
  have hg := native_zero_gen_raw_gradient remaining state h0 h1
  constructor
  · change coordinateClipFactor bound _ * _ = 0
    rw [hg.1, mul_zero]
  · change coordinateClipFactor bound _ * _ = 0
    rw [hg.2, mul_zero]

example : (seededNativeSubweights ((0, 0), (1, 1)) 0).parameter = 0 ∧
    (seededNativeSubweights ((0, 0), (1, 1)) 1).parameter = 0 := by
  exact ⟨rfl, rfl⟩

/-- Zero parameters with zero retained buffers stay zero, and each
clock advances. Source: native AdamW at 6c335bd and appendix C's
actual product CE. Arbitrary nonzero stale first moments are excluded. -/
theorem native_zero_gen_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState)
    (h0 : state 0 = zeroScalarStateAt (state 0).clock)
    (h1 : state 1 = zeroScalarStateAt (state 1).clock) :
    nativeSubweightStep remaining bound b1 b2 eps decay rate state 0 =
        zeroScalarStateAt ((state 0).clock + 1) ∧
      nativeSubweightStep remaining bound b1 b2 eps decay rate state 1 =
        zeroScalarStateAt ((state 1).clock + 1) := by
  have hp0 : (state 0).parameter = 0 := by rw [h0]; rfl
  have hp1 : (state 1).parameter = 0 := by rw [h1]; rfl
  have hg := native_zero_gen_applied_gradient remaining bound state hp0 hp1
  constructor
  · change scalarNativeStep b1 b2 eps decay rate (state 0) _ = _
    rw [hg.1, h0]
    exact scalar_native_zero_step b1 b2 eps decay rate _
  · change scalarNativeStep b1 b2 eps decay rate (state 1) _ = _
    rw [hg.2, h1]
    exact scalar_native_zero_step b1 b2 eps decay rate _

example : seededNativeSubweights ((0, 0), (1, 1)) 0 = zeroScalarStateAt 0 ∧
    seededNativeSubweights ((0, 0), (1, 1)) 1 = zeroScalarStateAt 0 := by
  exact ⟨rfl, rfl⟩

/-- The absent pair remains absent on the actual closed trajectory,
including buffers and completed clocks. Sources: appendix C product
CE and retained native AdamW at 6c335bd; future signs are not assumed. -/
theorem native_zero_gen_path (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (clock0 clock1 n : ℕ)
    (h0 : initial 0 = zeroScalarStateAt clock0) (h1 : initial 1 = zeroScalarStateAt clock1) :
    nativeSubweightPath remaining bound b1 b2 eps decay rate initial n 0 =
        zeroScalarStateAt (clock0 + n) ∧
      nativeSubweightPath remaining bound b1 b2 eps decay rate initial n 1 =
        zeroScalarStateAt (clock1 + n) := by
  induction n with
  | zero => exact ⟨h0, h1⟩
  | succ n ih =>
    have hc0 : (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n 0).clock =
        clock0 + n := by rw [ih.1]; rfl
    have hc1 : (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n 1).clock =
        clock1 + n := by rw [ih.2]; rfl
    have hs := native_zero_gen_step remaining bound b1 b2 eps decay rate
      (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n)
      (by rw [hc0]; exact ih.1) (by rw [hc1]; exact ih.2)
    rw [hc0, hc1] at hs
    simpa only [nativeSubweightPath, Nat.add_assoc] using hs

example : seededNativeSubweights ((0, 0), (1, 1)) 0 = zeroScalarStateAt 0 ∧
    seededNativeSubweights ((0, 0), (1, 1)) 1 = zeroScalarStateAt 0 := by
  exact ⟨rfl, rfl⟩

/-- The actual zero-buffer initialization is an invariant absent
Gen pair at every finite clock, even with arbitrary Mem seeds.
Sources: appendix C's product parameterization and native AdamW. -/
theorem native_zero_gen_seed_path (remaining : ℕ) (bound b1 b2 eps decay rate a b : ℝ) (n : ℕ) :
    nativeSubweightPath remaining bound b1 b2 eps decay rate
        (seededNativeSubweights ((0, 0), (a, b))) n 0 = zeroScalarStateAt n ∧
      nativeSubweightPath remaining bound b1 b2 eps decay rate
        (seededNativeSubweights ((0, 0), (a, b))) n 1 = zeroScalarStateAt n := by
  have hz := native_zero_gen_path remaining bound b1 b2 eps decay rate
    (seededNativeSubweights ((0, 0), (a, b))) 0 0 n rfl rfl
  simpa only [Nat.zero_add] using hz

/-- The corresponding actual Gen logit is zero along that path.
Source: appendix C, sim-overall-logits; the invariant controls the
trainable factors and moments, rather than prescribing a zero logit. -/
theorem native_zero_gen_seed_product (remaining : ℕ) (bound b1 b2 eps decay rate a b : ℝ) (n : ℕ) :
    let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((0, 0), (a, b))) n
    (state 0).parameter * (state 1).parameter = 0 := by
  dsimp only
  rw [(native_zero_gen_seed_path remaining bound b1 b2 eps decay rate a b n).1]
  exact zero_mul _

/-- Without Gen, a nonnegative Mem score cannot improve true test
CE over the uniform initial prediction. Source: appendix C test-loss
formula, including all q=remaining+2 classes rather than only two. -/
theorem table_zero_gen_heldout_ce_floor (remaining : ℕ) (weight : ℝ) (hw : 0 ≤ weight) :
    Real.log ((remaining : ℝ) + 2) ≤ Transformer.Grokking.NaiveLoss.crossEntropy
      (heldoutTableLogits remaining 0 weight) 0 := by
  have he : 1 ≤ Real.exp weight := Real.one_le_exp_iff.mpr hw
  rw [table_heldout_ce_formula, Real.exp_zero, sub_zero]
  exact Real.log_le_log (by positivity) (by linarith)

example : (0 : ℝ) ≤ 1 := by norm_num

/-- Actual nonnegative Mem trajectories with a fully absent Gen
pair never improve held-out CE beyond chance initialization. Sources:
appendix C tables and native AdamW at 6c335bd; this is a closed-path
obstruction, not an assumed bad gradient history or argmax-tie claim. -/
theorem native_zero_gen_path_heldout_ce_floor (remaining : ℕ)
    (bound b1 b2 eps decay rate a b : ℝ) (n : ℕ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((0, 0), (a, b))) n
    Real.log ((remaining : ℝ) + 2) ≤ Transformer.Grokking.NaiveLoss.crossEntropy
      (heldoutTableLogits remaining ((state 0).parameter * (state 1).parameter)
        ((state 2).parameter * (state 3).parameter)) 0 := by
  dsimp only
  rw [native_zero_gen_seed_product]
  have hs := native_nonnegative_path remaining bound b1 b2 eps decay rate
    (seededNativeSubweights ((0, 0), (a, b))) n hclip hb1 h1 hb2 h2 he heta hd
    (native_seeded_nonnegative 0 0 a b le_rfl le_rfl ha hb)
  exact table_zero_gen_heldout_ce_floor remaining _ (mul_nonneg (hs 2).1 (hs 3).1)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency

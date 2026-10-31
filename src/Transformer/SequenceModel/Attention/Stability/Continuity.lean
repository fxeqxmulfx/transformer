module

public import Transformer.SequenceModel.Attention.Stability.Softmax

/-!
# Continuity with varying source phases and residuals

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A3;
`diff_gpt/domain/model/{rope,mha}.py` at `b98c665b`. Positive normalizer
epsilons give continuity even on XSA's clipping sphere. The earlier analytic
theory excluded that sphere; this approximation argument does not need to.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini
open scoped BigOperators

variable {P : Type*} [TopologicalSpace P]

/-- A3: the actual epsilon RMSNorm is continuous everywhere, including zero. -/
theorem continuous_rms {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps) :
    Continuous (rmsNormEps eps : EucSpace d → EucSpace d) := by
  unfold rmsNormEps
  have hn : Continuous (fun x : EucSpace d => Real.sqrt (‖x‖ ^ 2 + d * eps)) :=
    ((continuous_norm.fun_pow 2).fun_add continuous_const).sqrt
  exact (continuous_const.div hn (fun x => (Real.sqrt_pos.mpr
    (add_pos_of_nonneg_of_pos (sq_nonneg _) (mul_pos (Nat.cast_pos.mpr hd) heps))).ne')).fun_smul
      continuous_id

/-- A3: clipping in XSA is continuous at the clipping boundary itself. -/
theorem continuous_l2 {d : ℕ} {eps : ℝ} (heps : 0 < eps) :
    Continuous (normL2 eps : EucSpace d → EucSpace d) := by
  unfold normL2
  exact (continuous_const.div (continuous_norm.max continuous_const)
    (fun _ => (lt_max_of_lt_right heps).ne')).fun_smul continuous_id

/-- A3: arbitrary half-split phases and input vectors may vary jointly. -/
theorem continuous_rotation {d : ℕ} (a : P → Fin (d / 2) → ℝ) (v : P → EucSpace d)
    (ha : ∀ c, Continuous (fun w => a w c)) (hv : Continuous v) :
    Continuous (fun w => rotateHead d (a w) (v w)) := by
  apply (PiLp.continuous_toLp 2 _).comp
  apply continuous_pi
  intro i
  rcases ropeSplit d i with (k | k) | j
  · simp only [rotate, splitPairs, Fin.isValue, one_ne_zero, ↓reduceIte]
    fun_prop
  · simp only [rotate, splitPairs, Fin.isValue, one_ne_zero, ↓reduceIte]
    fun_prop
  · fun_prop

/-- A3: joint continuity of source logits in phases, queries and keys. -/
theorem continuous_headScore {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (a b : P → Fin (d / 2) → ℝ) (q k : P → EucSpace d)
    (ha : ∀ c, Continuous (fun w => a w c)) (hb : ∀ c, Continuous (fun w => b w c))
    (hq : Continuous q) (hk : Continuous k) :
    Continuous (fun w => headScore d eps (a w) (b w) (q w) (k w)) :=
  ((continuous_rotation a _ ha ((continuous_rms hd heps).comp hq)).inner
    (continuous_rotation b _ hb ((continuous_rms hd heps).comp hk))).div_const _

/-- A3: active softmax denominators remain positive while all scores vary. -/
theorem continuous_weight {T : ℕ} (s : P → Fin T → Fin T → ℝ)
    (hs : ∀ i j, Continuous (fun w => s w i j)) (i j : Fin T) :
    Continuous (fun w => causalWeight (s w) i j) := by
  unfold causalWeight
  split_ifs
  · exact (hs i j).rexp.div (continuous_finsetSum _ (fun k _ => (hs i k).rexp))
      (fun w => (denominator_pos (s w) i).ne')
  · exact continuous_const

/-- A3: all parts of the source head are continuous with varying phases
and residual-derived Q/K/V, including the explicit first-position zero. -/
theorem continuous_headOutput {T d : ℕ} (hd : 0 < d) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps)
    (a : P → Fin T → Fin (d / 2) → ℝ) (q k v : P → Fin T → EucSpace d)
    (ha : ∀ j c, Continuous (fun w => a w j c))
    (hq : ∀ j, Continuous (fun w => q w j)) (hk : ∀ j, Continuous (fun w => k w j))
    (hv : ∀ j, Continuous (fun w => v w j)) (i : Fin T) :
    Continuous (fun w => headOutput eps xeps (a w) (q w) (k w) (v w) i) := by
  let s := fun w i j => headScore d eps (a w i) (a w j) (q w i) (k w j)
  have hs : ∀ i j, Continuous (fun w => s w i j) := fun i j =>
    continuous_headScore hd heps _ _ _ _ (ha i) (ha j) (hq i) (hk j)
  have hy : Continuous (fun w => ∑ j, causalWeight (s w) i j • v w j) :=
    continuous_finsetSum _ (fun j _ => (continuous_weight s hs i j).fun_smul (hv j))
  have hu := (continuous_l2 hxeps).comp (hv i)
  unfold headOutput
  split_ifs
  · exact continuous_const
  · exact hy.sub ((hy.inner (𝕜 := ℝ) hu).fun_smul hu)

example := continuous_rms (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
example := continuous_l2 (d := 4) (eps := 1 / 1000000000000) (by norm_num)
example := continuous_rotation (d := 4) (fun t : ℝ => fun _ => t / 7)
  (fun t => WithLp.toLp 2 (fun _ => t)) (by intro c; fun_prop) (by fun_prop)
example := continuous_headScore (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
  (fun t : ℝ => fun _ => t / 7) (fun t => fun _ => t / 11)
  (fun t => WithLp.toLp 2 (fun _ => t)) (fun t => WithLp.toLp 2 (fun _ => t + 1))
  (by intro c; fun_prop) (by intro c; fun_prop) (by fun_prop) (by fun_prop)
example := continuous_weight (fun t : ℝ => fun i j : Fin 3 => t * (i.val + j.val))
  (by intro i j; fun_prop) 2 1
example := continuous_headOutput (T := 3) (d := 4) (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun t : ℝ => fun j _ => t * j.val) (fun t => fun _ => WithLp.toLp 2 (fun _ => t))
  (fun t => fun _ => WithLp.toLp 2 (fun _ => t + 1))
  (fun t => fun _ => WithLp.toLp 2 (fun _ => t - 1))
  (by intro j c; fun_prop) (by intro j; fun_prop) (by intro j; fun_prop)
  (by intro j; fun_prop) 2

end Transformer.SequenceModel.Attention.Stability

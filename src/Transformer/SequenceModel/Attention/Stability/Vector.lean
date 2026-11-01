module

public import Transformer.SequenceModel.Attention.Stability.Rounding
public import Transformer.GPTMini.ReshapeDist

/-!
# Vector and projection bounds for source error propagation

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B1;
diff_gpt `b98c665b`, `mha.py` and `block.py`. Pre-normalization bounds the
actual fused-QKV outputs independently of residual size. Fixed-point vector
rounding still requires a separate range check in each coordinate.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini
open scoped BigOperators

/-- The existing global RMSNorm Lipschitz bound, with the source epsilon. -/
def rmsLip (eps : ℝ) : ℝ := 2 / Real.sqrt eps

/-- Uniform norm bound on every source Q/K/V head after pre-normalization. -/
def valueBound (cfg : Config) (B : BlockParams cfg) : ℝ :=
  ‖B.qkv‖ * Real.sqrt (cfg.d_model : ℝ)

/-- Sensitivity of the source pre-normalized Q/K/V projection. -/
def componentLip (cfg : Config) (B : BlockParams cfg) (eps : ℝ) : ℝ := ‖B.qkv‖ * rmsLip eps

/-- B1: coordinate error bounds give a Euclidean error bound at any width. -/
theorem norm_le_coordinates {d : ℕ} (x : EucSpace d) (b : ℝ) (hb : 0 ≤ b)
    (hx : ∀ c, |x c| ≤ b) : ‖x‖ ≤ Real.sqrt (d : ℝ) * b := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) hb)).mp
  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg d), EuclideanSpace.real_norm_sq_eq]
  calc
    _ ≤ ∑ c : Fin d, b ^ 2 := Finset.sum_le_sum (fun c _ => by
      simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hb).mpr (hx c))
    _ = _ := by simp

/-- B1: rounding an entire residual contributes at most sqrt(width) quanta. -/
theorem vector_round_error {d : ℕ} (p s : ℕ) (x : EucSpace d)
    (h : ∀ c, Fx.NoSaturation p s (x c)) :
    ‖Normalized.vector (fun c => Fx.round p s (x c)) - x‖ ≤
      Real.sqrt (d : ℝ) * (1 / (2 : ℝ) ^ s) := by
  apply norm_le_coordinates _ _ (by positivity)
  intro c
  exact (Fx.round_error_lt p s (x c) (h c)).le

/-- B1: the source half-split rotations are linear in their vectors. -/
theorem rotation_sub (d : ℕ) (a : Fin (d / 2) → ℝ) (x y : EucSpace d) :
    rotateHead d a x - rotateHead d a y = rotateHead d a (x - y) := by
  ext c
  simp only [PiLp.sub_apply, rotateHead, PiLp.toLp_apply]
  rcases ropeSplit d c with (k | k) | j <;>
    simp [rotate, splitPairs] <;> ring

/-- B1: keeping the exact same phases adds no vector-error amplification. -/
theorem rotation_dist (d : ℕ) (a : Fin (d / 2) → ℝ) (x y : EucSpace d) :
    ‖rotateHead d a x - rotateHead d a y‖ = ‖x - y‖ := by
  rw [rotation_sub, rotateHead_isometry]

/-- B1: source Q/K/V norm bounds follow from the actual pre-RMSNorm and
fused projection, not a bounded-residual assumption. -/
theorem component_norm (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    (part : Fin cfg.d_model → Fin (3 * cfg.d_model)) (hpart : Function.Injective part)
    (x : EucSpace cfg.d_model) (h : Fin cfg.n_heads) :
    ‖Normalized.component cfg B eps part x h‖ ≤ valueBound cfg B := by
  exact (headSlice_norm_le cfg _ h).trans ((qkvSlice_norm_le cfg part hpart _).trans
    ((B.qkv.le_opNorm _).trans (mul_le_mul_of_nonneg_left
      (rmsNormEps_norm_le eps heps cfg.d_model_pos x) (norm_nonneg _))))

/-- B1: source Q/K/V changes are bounded before head RMSNorm and RoPE. -/
theorem component_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    (part : Fin cfg.d_model → Fin (3 * cfg.d_model)) (hpart : Function.Injective part)
    (x y : EucSpace cfg.d_model) : ∀ h,
    ‖Normalized.component cfg B eps part x h - Normalized.component cfg B eps part y h‖ ≤
      componentLip cfg B eps * ‖x - y‖ := by
  intro h
  refine (headSlice_dist_le cfg _ _ h).trans ((qkvSlice_dist_le cfg part hpart _ _).trans ?_)
  rw [← map_sub]
  refine (B.qkv.le_opNorm _).trans ?_
  have hb := mul_le_mul_of_nonneg_left
    (rmsNormEps_lipschitz eps heps cfg.d_model_pos x y) (norm_nonneg B.qkv)
  simpa only [componentLip, rmsLip, mul_assoc] using hb

example := norm_le_coordinates (WithLp.toLp 2 ![1, -2, 3]) 3 (by norm_num)
  (by intro c; fin_cases c <;> norm_num)
example := vector_round_error 8 4 (WithLp.toLp 2 ![(1 : ℝ) / 3, -(1 / 3)])
  (by intro c; fin_cases c <;> norm_num [Fx.NoSaturation])
example (B : BlockParams witnessConfig) := component_norm witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (qkvV witnessConfig) (qkvV_injective _)
  (WithLp.toLp 2 ![1, 2, 3, 4]) 0
example (B : BlockParams witnessConfig) := component_error witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (qkvQ witnessConfig) (qkvQ_injective _)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![1, 2, 3, 5]) 0

end Transformer.SequenceModel.Attention.Stability

module

public import Transformer.SequenceModel.Attention.Stability.Quotient
public import Transformer.CRASP.Rotary.Source

/-!
# Error certificate for the actual nonperiodic counting head

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A6.
The numerator/denominator tables and final rounding are exactly those of
`CRASP.Rotary.sourceLayer`. The reference is the real softmax average on
the same stored residual stream. This is a local error bound; it does not
identify that stream with the full unrounded source stack.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini
open scoped BigOperators

/-- A6: source Q/K RMSNorm supplies a strictly positive exponential floor,
uniformly over all positions and residual states. -/
theorem sourceWeight_lower (cfg : Config) (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    {p s : ℕ} (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (r q : Rotary.State cfg p s)
    (h : Fin cfg.n_heads) (i j : ℕ) :
    Real.exp (-Real.sqrt (cfg.head_dim : ℝ)) ≤ Rotary.sourceWeight cfg B eps a r q h i j := by
  apply Real.exp_le_exp.mpr
  exact (abs_le.mp (headScore_bound cfg.head_dim cfg.head_dim_pos eps heps _ _ _ _)).1

/-- A6: the counting model's integer mantissa quotient is precisely the
ratio of the interpreted fixed-point coefficients; the common scale cancels. -/
theorem source_aggregate_eq (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    (p s : ℕ) (a : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (r : Rotary.State cfg p s) (q : ℕ → Rotary.State cfg p s)
    (i : ℕ) (c : Normalized.Coordinate cfg) :
    (Rotary.sourceLayer cfg B eps xeps p s a).aggregate r q i c =
      Fx.round p s (roundedRatio (Finset.Icc 1 i) p s
        (fun j => Rotary.sourceWeight cfg B eps a r (q j) c.1 i j)
        (fun j => Normalized.component cfg B eps (qkvV cfg) (Normalized.vector (q j).2) c.1 c.2)) := by
  unfold Rotary.Layer.aggregate roundedRatio
  congr 1
  simp only [Rotary.sourceLayer, Fx.val, Int.cast_sum]
  rw [← Finset.sum_div, ← Finset.sum_div]
  field_simp

/-- A6: an actual rounded source attention coordinate differs from its
real average by at most u(1+B)/(exp(-sqrt(d))-u)+u, for u=2^-s.
The first term is coefficient error; the final u is quotient rounding.
The explicit range hypotheses cover every rounding site in this aggregate. -/
theorem source_aggregate_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ}
    (heps : 0 < eps) (xeps : ℝ) (p s : ℕ)
    (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (r : Rotary.State cfg p s)
    (q : ℕ → Rotary.State cfg p s) (i : ℕ) (hi : 0 < i)
    (c : Normalized.Coordinate cfg) (bound : ℝ)
    (hprecision : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) :
    let weights := fun j => Rotary.sourceWeight cfg B eps a r (q j) c.1 i j
    let values := fun j => Normalized.component cfg B eps (qkvV cfg) (Normalized.vector (q j).2) c.1 c.2
    (∀ j ∈ Finset.Icc 1 i, |values j| ≤ bound) →
    (∀ j ∈ Finset.Icc 1 i, Fx.NoSaturation p s (weights j)) →
    (∀ j ∈ Finset.Icc 1 i, Fx.NoSaturation p s (weights j * values j)) →
    Fx.NoSaturation p s (roundedRatio (Finset.Icc 1 i) p s weights values) →
    |((Rotary.sourceLayer cfg B eps xeps p s a).aggregate r q i c).val -
      (∑ j ∈ Finset.Icc 1 i, weights j * values j) / (∑ j ∈ Finset.Icc 1 i, weights j)| ≤
        (1 / (2 : ℝ) ^ s) * (1 + bound) /
          (Real.exp (-Real.sqrt (cfg.head_dim : ℝ)) - 1 / (2 : ℝ) ^ s) + 1 / (2 : ℝ) ^ s := by
  intro weights values hvalues hw hn hquot
  have hne : (Finset.Icc 1 i).Nonempty := ⟨i, Finset.mem_Icc.mpr ⟨hi, le_rfl⟩⟩
  have he := roundedRatio_error (Finset.Icc 1 i) hne p s weights values _ bound hprecision
    (fun j hj => sourceWeight_lower cfg B heps a r (q j) c.1 i j) hvalues hw hn
  have hround := (Fx.round_error_lt p s _ hquot).le
  rw [source_aggregate_eq]
  exact (abs_sub_le _ (roundedRatio (Finset.Icc 1 i) p s weights values) _).trans
    (by linarith)

/-- A6: the underflow safeguard is an actual positive denominator, not a
convention for division by zero. It holds independently of prefix length. -/
theorem source_denominator_pos (cfg : Config) (B : BlockParams cfg) {eps : ℝ}
    (heps : 0 < eps) (p s : ℕ) (a : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (r : Rotary.State cfg p s) (q : ℕ → Rotary.State cfg p s) (i : ℕ) (hi : 0 < i)
    (c : Normalized.Coordinate cfg)
    (hprecision : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)))
    (hrange : ∀ j ∈ Finset.Icc 1 i,
      Fx.NoSaturation p s (Rotary.sourceWeight cfg B eps a r (q j) c.1 i j)) :
    0 < ∑ j ∈ Finset.Icc 1 i, (Fx.round p s (Rotary.sourceWeight cfg B eps a r (q j) c.1 i j)).val := by
  have hp j hj := Fx.round_pos p s _ _ (hrange j hj)
    (sourceWeight_lower cfg B heps a r (q j) c.1 i j) hprecision
  exact Finset.sum_pos' (fun j hj => (hp j hj).le)
    ⟨i, Finset.mem_Icc.mpr ⟨hi, le_rfl⟩, hp i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩)⟩

example := sourceWeight_lower witnessConfig ((decode witnessConfig 0).blocks 0)
  (eps := 1 / 8388608) (by norm_num) (p := 8) (s := 4)
  (fun i c => (i + c.val : ℝ)) (false, fun _ => 0) (true, fun _ => 0) 0 3 2

private def witnessBlock : BlockParams witnessConfig := ⟨0, 0, 0, 0, 0⟩
private def witnessState : Rotary.State witnessConfig 8 4 := (false, fun _ => 0)
private def witnessAngles (i : ℕ) (c : Fin (witnessConfig.head_dim / 2)) : ℝ := i + c.val
local instance : NeZero witnessConfig.head_dim := ⟨Nat.ne_of_gt witnessConfig.head_dim_pos⟩

private theorem witness_precision :
    1 / (2 : ℝ) ^ 4 < Real.exp (-Real.sqrt (witnessConfig.head_dim : ℝ)) := by
  have he : Real.exp 2 < (16 : ℝ) := by
    rw [show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add]
    nlinarith [Real.exp_one_lt_three, Real.exp_pos 1]
  have hi := one_div_lt_one_div_of_lt (Real.exp_pos 2) he
  convert hi using 1 <;> norm_num [witnessConfig, Config.head_dim, Real.exp_neg]

private theorem witness_weight (i j : ℕ) :
    Rotary.sourceWeight witnessConfig witnessBlock (1 / 8388608) witnessAngles
      witnessState witnessState 0 i j = 1 := by
  have hz (a : Fin (witnessConfig.head_dim / 2) → ℝ) :
      rotateHead witnessConfig.head_dim a 0 = 0 :=
    norm_eq_zero.mp (by rw [rotateHead_isometry, norm_zero])
  have hc (part : Fin witnessConfig.d_model → Fin (3 * witnessConfig.d_model))
      (x : EucSpace witnessConfig.d_model) :
      Normalized.component witnessConfig witnessBlock (1 / 8388608) part x 0 = 0 := by
    ext c
    simp [Normalized.component, witnessBlock]
  simp only [Rotary.sourceWeight, hc]
  simp [headScore, rmsNormEps, hz]

private theorem witness_value : Normalized.component witnessConfig witnessBlock (1 / 8388608)
    (qkvV witnessConfig) (Normalized.vector witnessState.2) 0 0 = 0 := by
  simp [Normalized.component, witnessBlock, qkvSlice, headSlice]

example := source_denominator_pos witnessConfig witnessBlock (eps := 1 / 8388608)
  (by norm_num) 8 4 witnessAngles witnessState (fun _ => witnessState) 3 (by decide) (0, 0)
  witness_precision (by intro j hj; rw [witness_weight]; norm_num [Fx.NoSaturation])

example :
    |((Rotary.sourceLayer witnessConfig witnessBlock (1 / 8388608) (1 / 1000000000000)
      8 4 witnessAngles).aggregate witnessState (fun _ => witnessState) 3 (0, 0)).val| ≤
      (1 / (2 : ℝ) ^ 4) * (1 + 1) /
        (Real.exp (-Real.sqrt (witnessConfig.head_dim : ℝ)) - 1 / (2 : ℝ) ^ 4) + 1 / (2 : ℝ) ^ 4 := by
  have h := source_aggregate_error witnessConfig witnessBlock (eps := 1 / 8388608)
    (by norm_num) (1 / 1000000000000) 8 4 witnessAngles witnessState (fun _ => witnessState)
    3 (by decide) (0, 0) 1 witness_precision
  dsimp only at h
  simp only [witness_weight, witness_value, mul_zero, Finset.sum_const_zero,
    zero_div, sub_zero] at h
  apply h (by intro j hj; norm_num) (by intro j hj; norm_num [Fx.NoSaturation])
    (by intro j hj; norm_num [Fx.NoSaturation])
  simp [roundedRatio, Fx.round_zero, Fx.NoSaturation]

end Transformer.SequenceModel.Attention.Stability

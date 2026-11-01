module

public import Transformer.AttentionResiduals.Section3_Attention
public import Transformer.GPTMini.RMSNorm

/-!
# Query/key normalized depth attention

Source: `review/residual_connections.md`, *Normalized depth attention*.
This is the user's proposed modification of arXiv:2603.15031v1 §3.1,
not a claim that the paper normalizes its pseudo-query. Both query and
keys use parameter-free RMSNorm with positive epsilon; scores divide by
sqrt(width). There is no unconstrained gain or learned temperature.
Values remain unnormalized. Bounds concern scores and aggregation, not
optimizer convergence. Small epsilon still permits steep derivatives.
-/

@[expose] public section
noncomputable section

namespace Transformer.AttentionResiduals

open scoped BigOperators
open GPTMini

/-- ND1: the original key-only formula has no query-independent score cap,
even at width one with unit gain and a fixed nonzero key. This corrects
any transfer of the source transformer's Q/K bound to AttnRes §3.1. -/
theorem source_score_unbounded (B : ℝ) :
    ∃ query : Fin 1 → ℝ,
      B < depthScore 1 (fun _ => 1) query (fun _ => 1) := by
  refine ⟨fun _ => (B + 1) * Real.sqrt 2, ?_⟩
  simp only [depthScore, keyRMS, Fin.sum_univ_one]
  norm_num

/-- ND1: normalizing the query is insufficient if an unconstrained gain
is applied after key normalization. Even the RMS-normalized scalar query
1/sqrt(2) and a fixed unit key permit arbitrarily large scores. -/
theorem source_gain_unbounded (B : ℝ) :
    ∃ gain : Fin 1 → ℝ,
      B < depthScore 1 gain (fun _ => 1 / Real.sqrt 2) (fun _ => 1) := by
  refine ⟨fun _ => 2 * (B + 1), ?_⟩
  simp only [depthScore, keyRMS, Fin.sum_univ_one]
  norm_num
  have hs : Real.sqrt 2 ≠ 0 := by positivity
  have hsq := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  field_simp
  rw [hsq]
  nlinarith

/-- ND2: normalize both operands, without affine gains, and use the same
1/sqrt(width) scaling as the source transformer's token attention. -/
def qkDepthScore {d : ℕ} (eps : ℝ) (query key : EucSpace d) : ℝ :=
  inner ℝ (rmsNormEps eps query) (rmsNormEps eps key) / Real.sqrt (d : ℝ)

/-- ND2: the new score has a uniform cap independent of query/key norms.
Positive epsilon includes zero operands; no lower norm assumption is hidden. -/
theorem qkDepthScore_bound {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (query key : EucSpace d) : |qkDepthScore eps query key| ≤ Real.sqrt (d : ℝ) := by
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  rw [qkDepthScore, abs_div, abs_of_pos hs, div_le_iff₀ hs]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul (rmsNormEps_norm_le eps heps hd query)
      (rmsNormEps_norm_le eps heps hd key) (norm_nonneg _) hs.le)

/-- ND2: an explicit bounded temperature scales the cap. An unrestricted
learned temperature would require its own bound instead. -/
theorem scaled_qkDepthScore_bound {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (query key : EucSpace d) (temperature cap : ℝ) (ht : |temperature| ≤ cap) :
    |temperature * qkDepthScore eps query key| ≤ cap * Real.sqrt (d : ℝ) := by
  rw [abs_mul]
  exact mul_le_mul ht (qkDepthScore_bound hd eps heps query key)
    (abs_nonneg _) (le_trans (abs_nonneg _) ht)

/-- ND3: the finite-epsilon score is globally Lipschitz, with a constant
that grows when epsilon shrinks. A score cap alone is not a small-gradient
guarantee. The existing RMSNorm estimate supplies the factor 2/sqrt(eps). -/
theorem qkDepthScore_lipschitz {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (q k q' k' : EucSpace d) :
    |qkDepthScore eps q k - qkDepthScore eps q' k'| ≤
      (2 / Real.sqrt eps) * (‖q - q'‖ + ‖k - k'‖) := by
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  have hq := rmsNormEps_lipschitz eps heps hd q q'
  have hk := rmsNormEps_lipschitz eps heps hd k k'
  have hkn := rmsNormEps_norm_le eps heps hd k
  have hqn := rmsNormEps_norm_le eps heps hd q'
  have hinner : inner ℝ (rmsNormEps eps q) (rmsNormEps eps k) -
      inner ℝ (rmsNormEps eps q') (rmsNormEps eps k') =
      inner ℝ (rmsNormEps eps q - rmsNormEps eps q') (rmsNormEps eps k) +
      inner ℝ (rmsNormEps eps q') (rmsNormEps eps k - rmsNormEps eps k') := by
    rw [inner_sub_left, inner_sub_right]
    ring
  rw [qkDepthScore, qkDepthScore, ← sub_div, abs_div, abs_of_pos hs, div_le_iff₀ hs]
  rw [hinner]
  calc
    _ ≤ |inner ℝ (rmsNormEps eps q - rmsNormEps eps q') (rmsNormEps eps k)| +
        |inner ℝ (rmsNormEps eps q') (rmsNormEps eps k - rmsNormEps eps k')| := abs_add_le _ _
    _ ≤ ‖rmsNormEps eps q - rmsNormEps eps q'‖ * ‖rmsNormEps eps k‖ +
        ‖rmsNormEps eps q'‖ * ‖rmsNormEps eps k - rmsNormEps eps k'‖ :=
      add_le_add (abs_real_inner_le_norm _ _) (abs_real_inner_le_norm _ _)
    _ ≤ (2 / Real.sqrt eps * ‖q - q'‖) * Real.sqrt (d : ℝ) +
        Real.sqrt (d : ℝ) * (2 / Real.sqrt eps * ‖k - k'‖) := by
      gcongr
    _ = _ := by ring

/-- ND4: the modified operator retains the original unnormalized values
and applies depth softmax to the corrected query/key scores. -/
def qkAttentionResidual {ι : Type*} [Fintype ι] {d : ℕ}
    (eps : ℝ) (query : EucSpace d) (values : ι → EucSpace d) : EucSpace d :=
  attend (fun i => qkDepthScore eps query (values i)) values

/-- ND4: normalized depth attention stays inside a common source-norm ball.
It does not bound the branch functions that produced those sources. -/
theorem qkAttentionResidual_norm_le {ι : Type*} [Fintype ι] [Nonempty ι] {d : ℕ}
    (eps : ℝ) (query : EucSpace d) (values : ι → EucSpace d) (R : ℝ)
    (hv : ∀ i, ‖values i‖ ≤ R) : ‖qkAttentionResidual eps query values‖ ≤ R := by
  unfold qkAttentionResidual
  exact attend_norm_le _ values R hv

/-- ND5: bounded depth scores impose a finite mass budget per source.
Normalization prevents unlimited sharpening by query scale; it does not
guarantee arbitrarily concentrated retrieval at fixed width and depth. -/
theorem depthWeight_le_of_abs_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (scores : ι → ℝ) (B : ℝ) (hb : ∀ j, |scores j| ≤ B) (i : ι) :
    depthWeight scores i ≤ Real.exp (2 * B) / (Fintype.card ι : ℝ) := by
  have hn : (0 : ℝ) < (Fintype.card ι : ℝ) := by exact_mod_cast Fintype.card_pos
  have hlo : (Fintype.card ι : ℝ) * Real.exp (-B) ≤ depthPartition scores := by
    have h := Finset.card_nsmul_le_sum Finset.univ
      (fun j => Real.exp (scores j)) (Real.exp (-B))
      (fun j _ => Real.exp_le_exp.mpr (neg_le_of_abs_le (hb j)))
    simpa only [Finset.card_univ, nsmul_eq_mul, depthPartition] using h
  rw [depthWeight, div_le_div_iff₀ (depthPartition_pos scores) hn]
  calc
    _ ≤ Real.exp B * (Fintype.card ι : ℝ) :=
      mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr (le_of_abs_le (hb i))) hn.le
    _ = Real.exp (2 * B) * ((Fintype.card ι : ℝ) * Real.exp (-B)) := by
      have he : Real.exp B = Real.exp (2 * B) * Real.exp (-B) := by
        rw [← Real.exp_add]
        congr 1
        ring
      rw [he]
      ring
    _ ≤ _ := mul_le_mul_of_nonneg_left hlo (Real.exp_pos _).le

/-- ND5 specialized to the corrected operator, for any nonempty finite
history of depth sources. Increasing depth also increases this source count. -/
theorem qkDepthWeight_le {ι : Type*} [Fintype ι] [Nonempty ι] {d : ℕ}
    (hd : 0 < d) (eps : ℝ) (heps : 0 < eps) (query : EucSpace d)
    (values : ι → EucSpace d) (i : ι) :
    depthWeight (fun j => qkDepthScore eps query (values j)) i ≤
      Real.exp (2 * Real.sqrt (d : ℝ)) / (Fintype.card ι : ℝ) := by
  exact depthWeight_le_of_abs_le _ _ (fun j => qkDepthScore_bound hd eps heps query (values j)) i

example := qkDepthScore_bound (d := 4) (by decide) (1 / 1000000) (by norm_num)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![0, 0, 0, 0])

example := scaled_qkDepthScore_bound (d := 4) (by decide) 1 (by norm_num)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![4, 3, 2, 1]) 2 3 (by norm_num)

example := qkDepthScore_lipschitz (d := 4) (by decide) 1 (by norm_num)
  (WithLp.toLp 2 ![0, 0, 0, 0]) (WithLp.toLp 2 ![4, 3, 2, 1])
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![2, 1, 4, 3])

example := qkAttentionResidual_norm_le 1 (WithLp.toLp 2 ![1, 2])
  (fun i : Fin 3 => WithLp.toLp 2 ![(i.val : ℝ), 1])
  (∑ i : Fin 3, ‖WithLp.toLp 2 ![(i.val : ℝ), 1]‖)
  (fun i => Finset.single_le_sum
    (fun j _ => norm_nonneg (WithLp.toLp 2 ![(j.val : ℝ), 1])) (Finset.mem_univ i))

example := depthWeight_le_of_abs_le (fun j : Fin 3 => if j = 0 then 1 else -1) 1
  (by intro j; split_ifs <;> norm_num) 0

example := qkDepthWeight_le (d := 2) (by decide) 1 (by norm_num)
  (WithLp.toLp 2 ![1, 2]) (fun i : Fin 3 => WithLp.toLp 2 ![(i.val : ℝ), 1]) 0

end Transformer.AttentionResiduals

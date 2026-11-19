module

public import Transformer.KERPLE.Section4_Attention
public import Transformer.KERPLE.Section4_T5
public import Transformer.Sandwich.Section6_Sandwich

/-!
# §6.2–6.3 The smoothed Sandwich, KERPLE and T5

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §6.2: "we … approximate (least-squared) the
original temporal bias pattern using a log curve, which gives a snug fit:
`y = −0.825 · log(1 + |m − n|) − 0.8`"; "the log curve decays relatively fast when two tokens are
nearby and plateaus when the distance between them increases." §6.3: "KERPLE has the formulation
of `c − r₁ · log(1 + r₂|m − n|)`. The `−0.8` in our fitted log curve term can be absorbed by `c`,
as Softmax is shift-invariant, and if we set `r₁ = 0.825` and `r₂ = 1`, Sandwich becomes a special
case of KERPLE. T5 adopts the log-binning strategy that assigns distinct bins to nearby tokens
whereas distant tokens all share the same bin."

The fit itself, least squares over the 50 most recent points of Figure 5, is numerical.

* `smoothed_eq_biasKernel`: the smoothed Sandwich is KERPLE's logarithmic bias kernel with
  `c = −0.8`, `r₁ = 0.825`, `r₂ = 1`.
* `attnWeight_smoothed_eq_kerple`: KERPLE adds its kernel before the scaling by `1/√k` (KERPLE's
  Eq. 1), Sandwich after it (§6.1). So the smoothed Sandwich's attention is KERPLE's with
  `r₁ = 0.825 √k`, not `0.825`, and any `c`.
* `sandwich_ne_kerple`: the exact Sandwich is no logarithmic KERPLE bias: it is bounded below
  (`neg_le_sandwich`), and `c − r₁ log(1 + r₂ t)` is not for `r₁, r₂ > 0`. Only the smoothed curve
  is a special case.
* `smoothed_sub_le`: the plateau, `0 ≤ y(t) − y(t + g) ≤ 0.825 g/(1 + t)` for `t, g ≥ 0`.
* `t5Bucket_injOn`, `t5Bucket_of_le`: T5's buckets (with the offset of
  `Transformer.KERPLE.t5Bucket`) are distinct for the distances `0, …, 15` and all `31` from `128`
  on.
-/

@[expose] public section

namespace Transformer.Sandwich

open Matrix Transformer.BAM

variable {d k D : ℕ}

/-- The smoothed Sandwich (§6.2, Figure 5): the log fit `−0.825 log(1 + |t|) − 0.8` at the
distance `t = m − n`. -/
noncomputable def smoothed (t : ℝ) : ℝ := -0.825 * Real.log (1 + |t|) - 0.8

/-- **The smoothed Sandwich is KERPLE's logarithmic bias kernel** (§6.3) with `c = −0.8`,
`r₁ = 0.825` and `r₂ = 1`. -/
theorem smoothed_eq_biasKernel (m n : ℝ) :
    smoothed (m - n) = KERPLE.biasKernel (-0.8) (KERPLE.logKernel 0.825 1) m n := by
  simp only [smoothed, KERPLE.biasKernel, KERPLE.logKernel, one_mul]
  ring

/-- The smoothed Sandwich as a bias of the distance. -/
noncomputable def smoothedBias (t : ℕ) : EReal := ((smoothed t : ℝ) : EReal)

/-- **The smoothed Sandwich's attention is KERPLE's** (§6.3, corrected): Sandwich adds its bias
after the scaling by `1/√k`, KERPLE inside it, so the weights of the query `m` over the keys
`0, …, m` are KERPLE's with the kernel `c − 0.825 √k log(1 + |m − n|)`, for every `c`. -/
theorem attnWeight_smoothed_eq_kerple (A : Head d k) (hk : 0 < k) (x : Embs d) (m : ℕ)
    (n : Fin (m + 1)) (c : ℝ) :
    attnWeight A smoothedBias x m n =
      KERPLE.attention (fun i : Fin (m + 1) => WithLp.toLp 2 (A.Wq *ᵥ x i))
        (fun i => WithLp.toLp 2 (A.Wk *ᵥ x i))
        (KERPLE.biasKernel c (KERPLE.logKernel (0.825 * Real.sqrt k) 1)) (Fin.last m) n := by
  have hs : Real.sqrt k ≠ 0 := (Real.sqrt_pos.2 (by exact_mod_cast hk)).ne'
  set z : Fin (m + 1) → ℝ := fun i =>
    (A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x i) / Real.sqrt k + smoothed ((m - i : ℕ) : ℝ) with hz
  have hw : ∀ i : Fin (m + 1), weight (score A smoothedBias x m i) = Real.exp (z i) := fun i => by
    simp only [score, smoothedBias, ← EReal.coe_add, weight_coe, z]
  have hL : attnWeight A smoothedBias x m n = KERPLE.softmax z n := by
    rw [attnWeight, ← Fin.sum_univ_eq_sum_range (fun i => weight (score A smoothedBias x m i)),
      KERPLE.softmax, hw]
    simp only [hw]
  rw [hL, KERPLE.attention, ← KERPLE.softmax_add_const z (c / Real.sqrt k + 0.8)]
  congr 1
  funext i
  have hi : (i : ℕ) ≤ m := Nat.lt_succ_iff.1 i.2
  simp only [z, EuclideanSpace.inner_eq_star_dotProduct, star_trivial,
    Fin.val_last, KERPLE.biasKernel, KERPLE.logKernel, smoothed, Nat.cast_sub hi, one_mul,
    dotProduct_comm (A.Wk *ᵥ x i)]
  rw [abs_of_nonneg (sub_nonneg.2 (Nat.cast_le.2 hi))]
  field_simp
  ring

/-- The hypothesis of `attnWeight_smoothed_eq_kerple` holds for heads of dimension `1`. -/
example (A : Head d 1) (x : Embs d) (m : ℕ) (n : Fin (m + 1)) :
    attnWeight A smoothedBias x m n =
      KERPLE.attention (fun i : Fin (m + 1) => WithLp.toLp 2 (A.Wq *ᵥ x i))
        (fun i => WithLp.toLp 2 (A.Wk *ᵥ x i))
        (KERPLE.biasKernel 0 (KERPLE.logKernel (0.825 * Real.sqrt ((1 : ℕ) : ℝ)) 1))
        (Fin.last m) n :=
  attnWeight_smoothed_eq_kerple A one_pos x m n 0

/-- **The exact Sandwich is no logarithmic KERPLE bias** (§6.3): for every `c` and `r₁, r₂ > 0`,
some distance `t` has `sandwich(t) ≠ c − r₁ log(1 + r₂ t)`, since Sandwich is bounded below and
the logarithm is not. -/
theorem sandwich_ne_kerple {h : ℝ} (hh : 0 < h) (c : ℝ) {r₁ r₂ : ℝ} (hr₁ : 0 < r₁)
    (hr₂ : 0 < r₂) :
    ∃ t : ℕ, sandwich D h t ≠ KERPLE.biasKernel c (KERPLE.logKernel r₁ r₂) t 0 := by
  obtain ⟨t, ht⟩ := exists_nat_gt ((Real.exp ((c + 2 * D / h) / r₁) - 1) / r₂)
  refine ⟨t, fun heq => ?_⟩
  have h1 := neg_le_sandwich (D := D) hh t
  rw [heq] at h1
  simp only [KERPLE.biasKernel, KERPLE.logKernel, sub_zero, Nat.abs_cast] at h1
  have hpos : 0 < 1 + r₂ * t := by positivity
  have h2 : Real.exp ((c + 2 * D / h) / r₁) < 1 + r₂ * t := by
    have := (div_lt_iff₀ hr₂).1 ht
    linarith
  have h3 : (c + 2 * D / h) / r₁ < Real.log (1 + r₂ * t) := (Real.lt_log_iff_exp_lt hpos).2 h2
  rw [div_lt_iff₀ hr₁] at h3
  have h5 : -(2 * (D : ℝ)) / h = -(2 * D / h) := by ring
  linarith

/-- The hypotheses of `sandwich_ne_kerple` hold for `r₁ = r₂ = 1`. -/
example : ∃ t : ℕ, sandwich 64 8 t ≠ KERPLE.biasKernel 0 (KERPLE.logKernel 1 1) t 0 :=
  sandwich_ne_kerple (by norm_num) 0 one_pos one_pos

/-- **The smoothed Sandwich plateaus** (§6.2): from the distance `t ≥ 0` on, `g` more tokens lower
the bias by at most `0.825 g/(1 + t)`. -/
theorem smoothed_sub_le {t g : ℝ} (ht : 0 ≤ t) (hg : 0 ≤ g) :
    0 ≤ smoothed t - smoothed (t + g) ∧ smoothed t - smoothed (t + g) ≤ 0.825 * g / (1 + t) := by
  have h1 : 0 < 1 + t := by linarith
  have key : smoothed t - smoothed (t + g) = 0.825 * Real.log ((1 + (t + g)) / (1 + t)) := by
    simp only [smoothed, abs_of_nonneg ht, abs_of_nonneg (add_nonneg ht hg)]
    rw [Real.log_div (by linarith) h1.ne']
    ring
  have hratio : 1 ≤ (1 + (t + g)) / (1 + t) := by
    rw [le_div_iff₀ h1]
    linarith
  have hlog := Real.log_le_sub_one_of_pos (lt_of_lt_of_le one_pos hratio)
  have hsub : (1 + (t + g)) / (1 + t) - 1 = g / (1 + t) := by
    field_simp
    ring
  rw [key, mul_div_assoc]
  exact ⟨by positivity [Real.log_nonneg hratio],
    mul_le_mul_of_nonneg_left (hsub ▸ hlog) (by norm_num)⟩

/-- The hypotheses of `smoothed_sub_le` hold at the distance `0`. -/
example : 0 ≤ smoothed 0 - smoothed (0 + 1) ∧ smoothed 0 - smoothed (0 + 1) ≤ 0.825 * 1 / (1 + 0) :=
  smoothed_sub_le le_rfl zero_le_one

/-- **T5 gives nearby tokens distinct bins** (§6.3): the distances `0, …, 15`. -/
theorem t5Bucket_injOn : Set.InjOn KERPLE.t5Bucket (Set.Ico 0 16) := by
  intro r hr s hs h
  simp only [Set.mem_Ico] at hr hs
  simp only [KERPLE.t5Bucket, ite_eq_right (not_lt.2 hr.1), ite_eq_left hr.2,
    ite_eq_right (not_lt.2 hs.1), ite_eq_left hs.2] at h
  omega

/-- **T5 gives distant tokens one bin** (§6.3): every distance `≥ 128` has the bucket `31`. -/
theorem t5Bucket_of_le {r : ℤ} (hr : 128 ≤ r) : KERPLE.t5Bucket r = 31 := by
  have h128 : KERPLE.t5Bucket 128 = 31 := by
    have hl : Real.log (128 / 16 : ℝ) ≠ 0 := (Real.log_pos (by norm_num)).ne'
    have hq : Real.log (((128 : ℤ) : ℝ) / 16) / Real.log (128 / 16) * 16 = 16 := by
      push_cast
      rw [div_self hl, one_mul]
    simp only [KERPLE.t5Bucket, hq]
    norm_num
  have := KERPLE.monotone_t5Bucket hr
  have := KERPLE.t5Bucket_lt r
  omega

/-- The hypothesis of `t5Bucket_of_le` holds for `r = 128`. -/
example : KERPLE.t5Bucket 128 = 31 := t5Bucket_of_le le_rfl

end Transformer.Sandwich

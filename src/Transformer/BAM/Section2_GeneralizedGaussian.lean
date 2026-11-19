module

public import Transformer.BAM.Section2_Priors

/-!
# §2.3 GGD-BAM: a generalized Gaussian prior, and Theorem 2

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.3; Theorem 2 is proved in
Appendix B.5.

GGD-BAM adds the bias `b_ij = −|(j − i − μ)/α|^β` to the scores (`ggdBias`). The power is taken
in `[0, ∞]`, so `|0|^β = ∞` for `β < 0` and the bias is `−∞` at `j − i = μ`; for `β ≥ 0`, or
off `j − i = μ`, it is the real number of the paper (`ggdBias_eq`).

* "This makes `p(g_pos) = GGD(μ, α, β, j − i)`, for `β > 0` and `α > 0`": with the causal mask's
  `Uniform(1, i, j)` as in Lemma 2 and up to normalization (`softmax_add_ggd`).
* "When `μ = 0`, `β = 1`, and `α = 1/m`, we have an instance of ALiBi" (`ggdBias_alibi`).
* Theorem 2: `B_ij / A_ij = −|(j − i − μ)/α|^β / (−m|j − i|) < 1` for some `μ` and `α`, for
  `0 < β < 1` (`exists_ggd_div_alibi_lt_one`). The proof's witness `μ = 0`, `α = m^{−1/β}` gives
  the ratio `|j − i|^{β − 1}`, which is `1` at `|j − i| = 1` (`ggd_div_alibi_paper_witness`);
  `α = (2/m)^{1/β}` works. The theorem says "for `β < 1`" under §2.3's standing `β > 0`: at
  `β = 0` it fails for `m ≤ 1` (`ggd_div_alibi_shape_zero`). For `β < 1` and every `α`, the ratio
  tends to `0` with the distance (`tendsto_ggd_div_alibi`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset Filter Topology

variable {L : ℕ}

/-- The GGD-BAM bias `b = −|(x − μ)/α|^β` at the relative position `x = j − i` (§2.3), with the
power in `[0, ∞]`: at `x = μ` it is `−∞` for `β < 0`, the limit there (§2.4, Theorem 3). -/
noncomputable def ggdBias (μ α β x : ℝ) : EReal :=
  -((ENNReal.ofReal |(x - μ) / α| ^ β : ENNReal) : EReal)

/-- Off `x = μ`, or for `β ≥ 0`, the bias is the real number `−|(x − μ)/α|^β`. -/
theorem ggdBias_eq {μ α β x : ℝ} (h : (x - μ) / α ≠ 0 ∨ 0 ≤ β) :
    ggdBias μ α β x = ((-(|(x - μ) / α| ^ β) : ℝ) : EReal) := by
  have hpow : ENNReal.ofReal |(x - μ) / α| ^ β = ENNReal.ofReal (|(x - μ) / α| ^ β) := by
    rcases h with h | h
    · exact ENNReal.ofReal_rpow_of_pos (abs_pos.2 h)
    · exact ENNReal.ofReal_rpow_of_nonneg (abs_nonneg _) h
  rw [ggdBias, hpow, EReal.coe_ennreal_ofReal,
    max_eq_left (Real.rpow_nonneg (abs_nonneg _) _), EReal.coe_neg]

theorem weight_ggdBias {μ α β x : ℝ} (h : (x - μ) / α ≠ 0 ∨ 0 ≤ β) :
    weight (ggdBias μ α β x) = Real.exp (-(|(x - μ) / α| ^ β)) := by
  rw [ggdBias_eq h, weight_coe]

/-- For `β < 0` the bias at `x = μ` is `−∞`. -/
theorem ggdBias_self_of_neg {μ α β : ℝ} (hβ : β < 0) : ggdBias μ α β μ = ⊥ := by
  simp [ggdBias, ENNReal.zero_rpow_of_neg hβ]

/-- The bias is a penalty: it is at most `0`. -/
theorem ggdBias_nonpos (μ α β x : ℝ) : ggdBias μ α β x ≤ 0 :=
  EReal.neg_le_zero.2 (EReal.coe_ennreal_nonneg _)

/-- GGD-BAM's prior: `Uniform(1, i, z) · GGD(μ, α, β, z − i) = c · e^{M_iz + b_iz}` with
`c = β / (2α Γ(1/β)) / (i + 1)`. -/
theorem uniform_mul_ggd {μ α β : ℝ} (hβ : 0 < β) (i z : Fin L) :
    uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * ggd μ α β (relPos i z) =
      β / (2 * α * Real.Gamma (1 / β)) / ((i : ℕ) + 1) *
        weight (causalMask i z + ggdBias μ α β (relPos i z)) := by
  rw [weight_add, weight_causalMask, uniform_causal, weight_ggdBias (Or.inr hβ.le), ggd]
  split_ifs
  · field_simp
  · simp

/-- **§2.3: GGD-BAM's position prior is `GGD(μ, α, β, j − i)`** for `β > 0` and `α > 0`:
`softmax(qK^⊤ + M_i• + B_i•)_j = p(f)_j · Uniform(1, i, j) · GGD(μ, α, β, j − i) / Z`, with
`Z = Σ_z p(f)_z · Uniform(1, i, z) · GGD(μ, α, β, z − i)`. The paper writes the prior without the
causal mask's `Uniform(1, i, j)` and without the normalizer. -/
theorem softmax_add_ggd {μ α β : ℝ} (hα : 0 < α) (hβ : 0 < β) (f : Fin L → EReal) (i j : Fin L) :
    softmax (fun z => f z + causalMask i z + ggdBias μ α β (relPos i z)) j =
      softmax f j * (uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * ggd μ α β (relPos i j)) /
        ∑ z, softmax f z * (uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * ggd μ α β (relPos i z)) := by
  have hΓ : 0 < Real.Gamma (1 / β) := Real.Gamma_pos_of_pos (one_div_pos.2 hβ)
  have hfun : (fun z => f z + causalMask i z + ggdBias μ α β (relPos i z)) =
      f + fun z => causalMask i z + ggdBias μ α β (relPos i z) := by
    funext z
    simp [add_assoc]
  rw [hfun]
  exact softmax_add_eq_of_proportional (by positivity) (uniform_mul_ggd hβ i) j

/-- The hypotheses of `softmax_add_ggd` hold for `α = 2`, `β = 1/2`. -/
example (μ : ℝ) (f : Fin 3 → EReal) (i j : Fin 3) :
    softmax (fun z => f z + causalMask i z + ggdBias μ 2 (1 / 2) (relPos i z)) j =
      softmax f j * (uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * ggd μ 2 (1 / 2) (relPos i j)) /
        ∑ z, softmax f z *
          (uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * ggd μ 2 (1 / 2) (relPos i z)) :=
  softmax_add_ggd two_pos (by norm_num) f i j

/-- **§2.3: ALiBi is GGD-BAM with `μ = 0`, `β = 1` and `α = 1/m`**, the Laplace prior. -/
theorem ggdBias_alibi {m : ℝ} (hm : 0 < m) (x : ℝ) : ggdBias 0 (1 / m) 1 x = alibi m x := by
  rw [ggdBias_eq (Or.inr zero_le_one), alibi, Real.rpow_one, sub_zero, abs_div, abs_one_div,
    abs_of_pos hm, div_div_eq_mul_div, div_one, mul_comm]

/-- The hypothesis of `ggdBias_alibi` holds for the slope `m = 1/4`. -/
example (x : ℝ) : ggdBias 0 (1 / (1 / 4)) 1 x = alibi (1 / 4) x := ggdBias_alibi (by norm_num) x

/-- Distinct positions are at least `1` apart. -/
theorem one_le_abs_relPos {i j : Fin L} (h : j ≠ i) : 1 ≤ |relPos i j| := by
  have hz : ((j : ℕ) : ℤ) - (i : ℕ) ≠ 0 := sub_ne_zero.2 (by exact_mod_cast Fin.val_ne_of_ne h)
  have h2 : relPos i j = ((((j : ℕ) : ℤ) - (i : ℕ) : ℤ) : ℝ) := by simp [relPos]
  rw [h2, ← Int.cast_abs]
  exact_mod_cast Int.one_le_abs hz

/-- The hypothesis of `one_le_abs_relPos` holds for the positions `0 ≠ 2`. -/
example : 1 ≤ |relPos (2 : Fin 3) 0| := one_le_abs_relPos (by decide)

/-- **Theorem 2.** For every ALiBi slope `m > 0` and shape `0 < β < 1` there are `μ` and `α > 0`
with `B_ij / A_ij = −|(j − i − μ)/α|^β / (−m|j − i|) < 1` at every pair of distinct positions
(§2.3, Appendix B.5). The witness is `μ = 0`, `α = (2/m)^{1/β}`, not the proof's
`α = m^{−1/β}` (`ggd_div_alibi_paper_witness`). -/
theorem exists_ggd_div_alibi_lt_one {m β : ℝ} (hm : 0 < m) (hβ₀ : 0 < β) (hβ₁ : β < 1) :
    ∃ μ α : ℝ, 0 < α ∧ ∀ (L : ℕ) (i j : Fin L), j ≠ i →
      -(|(relPos i j - μ) / α| ^ β) / (-(m * |relPos i j|)) < 1 := by
  refine ⟨0, (2 / m) ^ β⁻¹, by positivity, fun L i j hij => ?_⟩
  have hd := one_le_abs_relPos hij
  have hα : ((2 / m) ^ β⁻¹) ^ β = 2 / m := Real.rpow_inv_rpow (by positivity) hβ₀.ne'
  have hdβ : |relPos i j| ^ β ≤ |relPos i j| := by
    simpa using Real.rpow_le_rpow_of_exponent_le hd hβ₁.le
  rw [sub_zero, abs_div, abs_of_pos (by positivity : 0 < (2 / m) ^ β⁻¹),
    Real.div_rpow (abs_nonneg _) (by positivity), hα, neg_div_neg_eq,
    div_lt_one (by positivity)]
  calc |relPos i j| ^ β / (2 / m) ≤ |relPos i j| / (2 / m) := by gcongr
    _ < m * |relPos i j| := by
      rw [div_div_eq_mul_div]
      linarith [mul_pos hm (by linarith : (0 : ℝ) < |relPos i j|)]

/-- The hypotheses of `exists_ggd_div_alibi_lt_one` hold for `m = 1`, `β = 1/2`. -/
example : ∃ μ α : ℝ, 0 < α ∧ ∀ (L : ℕ) (i j : Fin L), j ≠ i →
    -(|(relPos i j - μ) / α| ^ (1 / 2 : ℝ)) / (-(1 * |relPos i j|)) < 1 :=
  exists_ggd_div_alibi_lt_one one_pos (by norm_num) (by norm_num)

/-- **The witness of the proof of Theorem 2 fails at distance `1`.** With `μ = 0` and
`α = m^{−1/β}`, the ratio `B_ij / A_ij = |j − i|^{β − 1}` is `1`, not `< 1`, when
`|j − i| = 1` (Appendix B.5). -/
theorem ggd_div_alibi_paper_witness {m β x : ℝ} (hm : 0 < m) (hβ : β ≠ 0) (hx : |x| = 1) :
    -(|(x - 0) / (m ^ β⁻¹)⁻¹| ^ β) / (-(m * |x|)) = 1 := by
  rw [sub_zero, div_inv_eq_mul, abs_mul, hx, one_mul, abs_of_pos (by positivity),
    Real.rpow_inv_rpow hm.le hβ, mul_one, neg_div_neg_eq, div_self hm.ne']

/-- The hypotheses of `ggd_div_alibi_paper_witness` hold for `m = 2`, `β = 1/2`, `x = −1`. -/
example : -(|((-1 : ℝ) - 0) / ((2 : ℝ) ^ (1 / 2 : ℝ)⁻¹)⁻¹| ^ (1 / 2 : ℝ)) /
    (-(2 * |(-1 : ℝ)|)) = 1 :=
  ggd_div_alibi_paper_witness two_pos (by norm_num) (by norm_num)

/-- **Theorem 2 needs `β > 0`.** At `β = 0`, every `μ` and `α` give the ratio `1/m ≥ 1` at
distance `1` when `m ≤ 1` (§2.3). -/
theorem ggd_div_alibi_shape_zero {m x : ℝ} (hm : 0 < m) (hm1 : m ≤ 1) (hx : |x| = 1)
    (μ α : ℝ) : 1 ≤ -(|(x - μ) / α| ^ (0 : ℝ)) / (-(m * |x|)) := by
  rw [Real.rpow_zero, hx, mul_one, neg_div_neg_eq, one_le_div hm]
  exact hm1

/-- The hypotheses of `ggd_div_alibi_shape_zero` hold for `m = 1` and `x = 1`. -/
example (μ α : ℝ) : 1 ≤ -(|((1 : ℝ) - μ) / α| ^ (0 : ℝ)) / (-(1 * |(1 : ℝ)|)) :=
  ggd_div_alibi_shape_zero one_pos le_rfl abs_one μ α

/-- **For `β < 1`, GGD-BAM ends up penalizing distance less than ALiBi at every scale.** With
`μ = 0`, for every `α > 0` and slope `m > 0`, `B / A = |d/α|^β / (m d)` tends to `0` as the
distance `d = |j − i|` grows (§2.3, Theorem 2: "BAM shrinks at a slower rate than ALiBi"). -/
theorem tendsto_ggd_div_alibi {m α β : ℝ} (hm : 0 < m) (hα : 0 < α) (hβ : β < 1) :
    Tendsto (fun d : ℝ => |d / α| ^ β / (m * d)) atTop (𝓝 0) := by
  have h := (tendsto_rpow_neg_atTop (sub_pos.2 hβ)).const_mul (α ^ (-β) / m)
  rw [mul_zero] at h
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with d hd
  rw [abs_of_pos (div_pos hd hα), Real.div_rpow hd.le hα.le, Real.rpow_neg hα.le, neg_sub,
    Real.rpow_sub hd, Real.rpow_one]
  have : 0 < α ^ β := Real.rpow_pos_of_pos hα β
  field_simp

/-- The hypotheses of `tendsto_ggd_div_alibi` hold for `m = 1`, `α = 1`, `β = 1/2`. -/
example : Tendsto (fun d : ℝ => |d / 1| ^ (1 / 2 : ℝ) / (1 * d)) atTop (𝓝 0) :=
  tendsto_ggd_div_alibi one_pos one_pos (by norm_num)

end Transformer.BAM

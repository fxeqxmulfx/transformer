module

public import Mathlib.Analysis.SpecialFunctions.Arsinh
public import Transformer.BAM.Section2_GeneralizedGaussian

/-!
# Appendix C: the learnable parametrization of GGD-BAM

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, Appendix C.

Each attention head learns `θ_μ`, `θ_α`, `θ_β`, and its bias is
`b_ij = −e^{θ_α} (|(j − i) − (e^{θ_μ} − e^{−θ_μ})| + ε)^{θ_β}` (`implBias`). "So the GGD is
parametrized in the following way: `μ = e^{θ_μ} − e^{−θ_μ}`, `β = θ_β` and `α = e^{−θ_α/θ_β}`,
that ensures `α ≥ 0`. `ε = 10^{−5}` avoids division by zero when `β < 0`."

* With `ε = 0` and `θ_β ≠ 0` the bias is `−|(x − μ)/α|^β` (`implBias_zero`).
* The parameters reached are every `μ`, every `α > 0` and every `β ≠ 0`, and only those
  (`implParams_iff`). At `θ_β = 0` the formula for `α` divides by zero.
* For `ε > 0` and `β ≤ 0` the bias is at least `−e^{θ_α} ε^{θ_β}` (`neg_le_implBias`): it is
  finite, so the implementation gives a positive weight to the key at `j − i = μ`, which
  Theorem 3 ignores.
* As `ε → 0`, for `β < 0`, the bias tends to `ggdBias μ α β`, its value `−∞` at `x = μ` included
  (`tendsto_implBias`).
-/

@[expose] public section

namespace Transformer.BAM

open Filter Topology

/-- The bias of the implementation at the relative position `x = j − i` (Appendix C):
`−e^{θ_α} (|x − (e^{θ_μ} − e^{−θ_μ})| + ε)^{θ_β}`. -/
noncomputable def implBias (θμ θα θβ ε x : ℝ) : ℝ :=
  -(Real.exp θα * (|x - (Real.exp θμ - Real.exp (-θμ))| + ε) ^ θβ)

/-- **The implementation's bias is GGD-BAM's** (Appendix C). With `ε = 0` and `θ_β ≠ 0` it is
`−|(x − μ)/α|^β` for `μ = e^{θ_μ} − e^{−θ_μ}`, `α = e^{−θ_α/θ_β}` and `β = θ_β`. -/
theorem implBias_zero {θμ θα θβ : ℝ} (hβ : θβ ≠ 0) (x : ℝ) :
    implBias θμ θα θβ 0 x =
      -(|(x - (Real.exp θμ - Real.exp (-θμ))) / Real.exp (-θα / θβ)| ^ θβ) := by
  have hα : Real.exp (-θα / θβ) ^ θβ = Real.exp (-θα) := by
    rw [← Real.exp_mul, div_mul_cancel₀ _ hβ]
  rw [implBias, add_zero, abs_div, abs_of_pos (Real.exp_pos _),
    Real.div_rpow (abs_nonneg _) (Real.exp_pos _).le, hα, Real.exp_neg θα, div_inv_eq_mul,
    mul_comm]

/-- The hypothesis of `implBias_zero` holds for `θ_β = 2`. -/
example (x : ℝ) : implBias 0 0 2 0 x =
    -(|(x - (Real.exp 0 - Real.exp (-0))) / Real.exp (-0 / 2)| ^ (2 : ℝ)) :=
  implBias_zero two_ne_zero x

/-- **The parametrization reaches every GGD with `α > 0` and `β ≠ 0`, and only those**
(Appendix C: "`α = e^{−θ_α/θ_β}`, that ensures `α ≥ 0`"). -/
theorem implParams_iff {μ α β : ℝ} :
    (∃ θμ θα θβ : ℝ, θβ ≠ 0 ∧ μ = Real.exp θμ - Real.exp (-θμ) ∧ α = Real.exp (-θα / θβ) ∧
      β = θβ) ↔ 0 < α ∧ β ≠ 0 := by
  constructor
  · rintro ⟨θμ, θα, θβ, hβ, -, rfl, rfl⟩
    exact ⟨Real.exp_pos _, hβ⟩
  · rintro ⟨hα, hβ⟩
    refine ⟨Real.arsinh (μ / 2), -(β * Real.log α), β, hβ, ?_, ?_, rfl⟩
    · have h := Real.sinh_arsinh (μ / 2)
      rw [Real.sinh_eq] at h
      linarith
    · rw [neg_neg, mul_div_cancel_left₀ _ hβ, Real.exp_log hα]

/-- **`ε > 0` keeps the bias finite for `β ≤ 0`** (Appendix C: "`ε = 10^{−5}` avoids division by
zero when `β < 0`"): it is at least `−e^{θ_α} ε^{θ_β}`, its value at `x = μ`. -/
theorem neg_le_implBias {θμ θα θβ ε : ℝ} (hε : 0 < ε) (hβ : θβ ≤ 0) (x : ℝ) :
    -(Real.exp θα * ε ^ θβ) ≤ implBias θμ θα θβ ε x := by
  rw [implBias, neg_le_neg_iff]
  exact mul_le_mul_of_nonneg_left
    (Real.rpow_le_rpow_of_nonpos hε (le_add_of_nonneg_left (abs_nonneg _)) hβ)
    (Real.exp_pos θα).le

/-- The hypotheses of `neg_le_implBias` hold for the paper's `ε = 10^{−5}` and `θ_β = −1`. -/
example (x : ℝ) : -(Real.exp 0 * (1 / 100000 : ℝ) ^ (-1 : ℝ)) ≤ implBias 0 0 (-1) (1 / 100000) x :=
  neg_le_implBias (by norm_num) (by norm_num) x

/-- **As `ε → 0` the implementation's bias tends to GGD-BAM's** for `β < 0`, to `−∞` at
`x = μ` (Appendix C). -/
theorem tendsto_implBias {θμ θα θβ : ℝ} (hβ : θβ < 0) (x : ℝ) :
    Tendsto (fun ε => (implBias θμ θα θβ ε x : EReal)) (𝓝[>] 0)
      (𝓝 (ggdBias (Real.exp θμ - Real.exp (-θμ)) (Real.exp (-θα / θβ)) θβ x)) := by
  rcases eq_or_ne x (Real.exp θμ - Real.exp (-θμ)) with rfl | hx
  · rw [ggdBias_self_of_neg hβ]
    have h1 : Tendsto (fun ε : ℝ => ε ^ θβ) (𝓝[>] 0) atTop := tendsto_rpow_neg_nhdsGT_zero hβ
    refine (EReal.tendsto_coe_atBot.comp
      (tendsto_neg_atTop_atBot.comp (h1.const_mul_atTop (Real.exp_pos θα)))).congr' ?_
    filter_upwards with ε
    simp [implBias]
  · have hd : |x - (Real.exp θμ - Real.exp (-θμ))| + 0 ≠ 0 := by
      rw [add_zero]
      exact abs_ne_zero.2 (sub_ne_zero.2 hx)
    have hlin : Tendsto (fun ε : ℝ => |x - (Real.exp θμ - Real.exp (-θμ))| + ε) (𝓝[>] 0)
        (𝓝 (|x - (Real.exp θμ - Real.exp (-θμ))| + 0)) :=
      (tendsto_const_nhds.add tendsto_id).mono_left nhdsWithin_le_nhds
    have hcont : Tendsto (fun ε => implBias θμ θα θβ ε x) (𝓝[>] 0)
        (𝓝 (implBias θμ θα θβ 0 x)) :=
      ((hlin.rpow_const (Or.inl hd)).const_mul (Real.exp θα)).neg
    rw [ggdBias_eq (Or.inl (div_ne_zero (sub_ne_zero.2 hx) (Real.exp_pos _).ne')),
      ← implBias_zero hβ.ne]
    exact (continuous_coe_real_ereal.tendsto _).comp hcont

/-- The hypothesis of `tendsto_implBias` holds for `θ_β = −1`. -/
example (x : ℝ) : Tendsto (fun ε => (implBias 0 0 (-1) ε x : EReal)) (𝓝[>] 0)
    (𝓝 (ggdBias (Real.exp 0 - Real.exp (-0)) (Real.exp (-0 / -1)) (-1) x)) :=
  tendsto_implBias (by norm_num) x

end Transformer.BAM

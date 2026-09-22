/-
# The number of modes of a Gaussian KDE — the quotients are bounded on `T`

The five quotients of `Section2_PhiTQuot.lean` lie in `(1/2, 2)` for every
`t ∈ T`, for all large `k`, along a regime sequence `(n_k, β_k)`.  This is the
uniform form of `lem:moments-p` that `lem:phi-t` reads its `≍ over t ∈ T` from.

The uniformity comes from the sequential statement
`tendsto_sq_div_of_mem_intervalT`: since `0 ∈ T` always, a `t_k ∈ T_k` at
which a bound fails, whenever there is one, is itself a sequence in `T`.

Source: arXiv:2412.09080v3, `lem:moments-p`, `lem:phi-t`; §5.2.
-/

import Transformer.Modes.Section2_PhiTQuot
import Transformer.Modes.Section2_MomentsPCov

open Real Filter
open scoped Topology

namespace Transformer
namespace Modes

/-- `0 ∈ T`, for every `n`, `β`, `ω`: the interval is centred and, when the
radicand is negative, reduces to `{0}`. -/
theorem zero_mem_intervalT (n : ℕ) (β w : ℝ) : (0 : ℝ) ∈ intervalT n β w :=
  ⟨by simp, Real.sqrt_nonneg _⟩

/-- **`(1/β, t²/β) → 0` uniformly on `T`.**

Source: arXiv:2412.09080v3, §5.2, first paragraph. -/
theorem eventually_forall_mem_intervalT {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {U : Set (ℝ × ℝ)}
    (hU : U ∈ 𝓝 ((0 : ℝ), (0 : ℝ))) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)), ((B k)⁻¹, t ^ 2 / B k) ∈ U := by
  classical
  by_contra h
  rw [not_eventually] at h
  let s : ℕ → ℝ := fun k =>
    if hk : ∃ t ∈ intervalT (N k) (B k) (ω (B k)), ((B k)⁻¹, t ^ 2 / B k) ∉ U
    then hk.choose else 0
  have hs : ∀ k, s k ∈ intervalT (N k) (B k) (ω (B k)) := by
    intro k
    simp only [s]
    split_ifs with hk
    · exact hk.choose_spec.1
    · exact zero_mem_intervalT _ _ _
  have hlim := (tendsto_inv_atTop_zero.comp hreg.tendsto_B).prodMk_nhds
    (tendsto_sq_div_of_mem_intervalT hreg hω (Eventually.of_forall hs))
  obtain ⟨k, hk, hkU⟩ := (h.and_eventually (hlim hU)).exists
  push Not at hk
  simp only [Function.comp_apply, s, dite_eq_left hk] at hkU
  exact hk.choose_spec.2 hkU

/-- The hypotheses of `eventually_forall_mem_intervalT` are satisfiable. -/
example : ∀ᶠ k in atTop, ∀ t ∈ intervalT (k + 1) ((k + 1 : ℕ) : ℝ)
    (Real.sqrt (Real.log (Real.log ((k + 1 : ℕ) : ℝ)))),
      ((((k + 1 : ℕ) : ℝ))⁻¹, t ^ 2 / ((k + 1 : ℕ) : ℝ)) ∈ (Set.univ : Set (ℝ × ℝ)) :=
  eventually_forall_mem_intervalT isRegime_succ isSlowGrowth_sqrt_log_log Filter.univ_mem

/-- A function continuous at the origin stays in `(a, b)` on `T`, for all
large `k`, if its value at the origin does. -/
theorem eventually_forall_intervalT_Ioo {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) {G : ℝ × ℝ → ℝ}
    (hG : ContinuousAt G (0, 0)) {v a b : ℝ} (hv : G (0, 0) = v) (ha : a < v) (hb : v < b) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      a < G ((B k)⁻¹, t ^ 2 / B k) ∧ G ((B k)⁻¹, t ^ 2 / B k) < b :=
  eventually_forall_mem_intervalT hreg hω
    (hG.preimage_mem_nhds (Ioo_mem_nhds (hv ▸ ha) (hv ▸ hb)))

/-- The hypotheses of `eventually_forall_intervalT_Ioo` are satisfiable. -/
example : ∀ᶠ k in atTop, ∀ t ∈ intervalT (k + 1) ((k + 1 : ℕ) : ℝ)
    (Real.sqrt (Real.log (Real.log ((k + 1 : ℕ) : ℝ)))),
      (-1 : ℝ) < (fun p : ℝ × ℝ => p.1) ((((k + 1 : ℕ) : ℝ))⁻¹, t ^ 2 / ((k + 1 : ℕ) : ℝ)) ∧
        (fun p : ℝ × ℝ => p.1) ((((k + 1 : ℕ) : ℝ))⁻¹, t ^ 2 / ((k + 1 : ℕ) : ℝ)) < 1 :=
  eventually_forall_intervalT_Ioo isRegime_succ isSlowGrowth_sqrt_log_log continuousAt_fst rfl
    (by norm_num) (by norm_num)

/-- `A + bΨ ∈ (1/2, 2)` when `A ∈ (3/4, 5/4)` and `|bΨ| ≤ 1/4`. -/
theorem add_mul_mem_Ioo {A b Ψ C : ℝ} (hC : 0 < C) (hA : 3 / 4 < A ∧ A < 5 / 4)
    (hb : |b| ≤ C) (hΨ : -(1 / (4 * C)) < Ψ ∧ Ψ < 1 / (4 * C)) :
    1 / 2 < A + b * Ψ ∧ A + b * Ψ < 2 := by
  have h1 : |b * Ψ| ≤ 1 / 4 := by
    rw [abs_mul]
    calc |b| * |Ψ| ≤ C * (1 / (4 * C)) :=
          mul_le_mul hb (abs_lt.2 hΨ).le (abs_nonneg _) hC.le
      _ = 1 / 4 := by field_simp
  have := abs_le.1 h1
  constructor <;> linarith

/-- The hypotheses of `add_mul_mem_Ioo` are satisfiable. -/
example : 1 / 2 < (1 : ℝ) + 0 * 0 ∧ (1 : ℝ) + 0 * 0 < 2 :=
  add_mul_mem_Ioo one_pos (by norm_num) (by norm_num) (by norm_num)

/-- **The five quotients lie in `(1/2, 2)` on `T`**, for all large `k`: the
uniform form of `lem:moments-p`.

Source: arXiv:2412.09080v3, `lem:moments-p`; §5.2. -/
theorem eventually_quot_mem_Ioo {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      (1 / 2 < gMean ((B k)⁻¹, t ^ 2 / B k) ∧ gMean ((B k)⁻¹, t ^ 2 / B k) < 2) ∧
      (1 / 2 < gMean' ((B k)⁻¹, t ^ 2 / B k) ∧ gMean' ((B k)⁻¹, t ^ 2 / B k) < 2) ∧
      (1 / 2 < qVar (B k) t ∧ qVar (B k) t < 2) ∧
      (1 / 2 < qCov (B k) t ∧ qCov (B k) t < 2) ∧
      (1 / 2 < qVar' (B k) t ∧ qVar' (B k) t < 2) := by
  filter_upwards [
    eventually_forall_intervalT_Ioo hreg hω (G := gMean)
      (by unfold gMean; fun_prop (disch := norm_num)) (v := 1) (by norm_num [gMean])
      (a := 1 / 2) (b := 2) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gMean')
      (by unfold gMean'; fun_prop (disch := norm_num)) (v := 1) (by norm_num [gMean'])
      (a := 1 / 2) (b := 2) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gVarA)
      (by unfold gVarA; fun_prop (disch := norm_num)) (v := 1) (by norm_num [gVarA]; field_simp)
      (a := 3 / 4) (b := 5 / 4) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gVarΨ)
      (by unfold gVarΨ; fun_prop (disch := norm_num)) (v := 0) (by simp [gVarΨ])
      (a := -(1 / (4 * 2))) (b := 1 / (4 * 2)) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gCovA)
      (by unfold gCovA; fun_prop (disch := norm_num)) (v := 1) (by norm_num [gCovA]; field_simp)
      (a := 3 / 4) (b := 5 / 4) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gCovΨ)
      (by unfold gCovΨ; fun_prop (disch := norm_num)) (v := 0) (by simp [gCovΨ])
      (a := -(1 / (4 * 4))) (b := 1 / (4 * 4)) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gVar'A)
      (by unfold gVar'A; fun_prop (disch := norm_num)) (v := 1)
      (by norm_num [gVar'A]; field_simp) (a := 3 / 4) (b := 5 / 4) (by norm_num) (by norm_num),
    eventually_forall_intervalT_Ioo hreg hω (G := gVar'Ψ)
      (by unfold gVar'Ψ; fun_prop (disch := norm_num)) (v := 0) (by simp [gVar'Ψ])
      (a := -(1 / (4 * 24))) (b := 1 / (4 * 24)) (by norm_num) (by norm_num),
    hreg.tendsto_B.eventually_ge_atTop 1] with k h1 h2 h3 h4 h5 h6 h7 h8 hB t ht
  have hε0 : 0 ≤ (B k)⁻¹ := inv_nonneg.2 (by linarith)
  have hε1 : (B k)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hB
  refine ⟨h1 t ht, h2 t ht, add_mul_mem_Ioo two_pos (h3 t ht) ?_ (h4 t ht),
    add_mul_mem_Ioo (by norm_num) (h5 t ht) (abs_exp_mul_one_sub_sq_add_le hε0 hε1 t) (h6 t ht),
    add_mul_mem_Ioo (by norm_num) (h7 t ht) (abs_exp_mul_one_add_sub_sq_sq_le hε0 hε1 t)
      (h8 t ht)⟩
  rw [abs_of_nonneg (by positivity)]
  exact sq_mul_exp_neg_half_sq_le t

/-- The hypotheses of `eventually_quot_mem_Ioo` are satisfiable. -/
example := eventually_quot_mem_Ioo isRegime_succ isSlowGrowth_sqrt_log_log

/-- The hypotheses of `phiAlpha_eq` are satisfiable: along the regime `β = n`,
`t = 0 ∈ T`, all three quotients exceed `1/2`. -/
example : ∃ β t : ℝ, 0 < β ∧ qVar β t ≠ 0 ∧
    6 * qVar β t * qVar' β t - t ^ 2 / β * qCov β t ^ 2 ≠ 0 := by
  obtain ⟨k, hk⟩ := (eventually_quot_mem_Ioo isRegime_succ isSlowGrowth_sqrt_log_log).exists
  obtain ⟨-, -, ⟨a3, -⟩, -, ⟨a5, -⟩⟩ := hk 0 (zero_mem_intervalT _ _ _)
  refine ⟨((k + 1 : ℕ) : ℝ), 0, by positivity, by linarith, ?_⟩
  rw [show (0 : ℝ) ^ 2 / ((k + 1 : ℕ) : ℝ) * qCov ((k + 1 : ℕ) : ℝ) 0 ^ 2 = 0 by simp, sub_zero]
  exact (mul_pos (mul_pos (by norm_num) (by linarith)) (by linarith)).ne'

end Modes
end Transformer

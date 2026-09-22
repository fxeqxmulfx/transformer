/-
# Attention's forward pass and Frank-Wolfe — one softmax step contracts

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, the one-step estimates behind
`prop: origin`.

A step of `eq: softmax.ODE` moves each particle to a convex combination of the
configuration whose weights are the attention scores, so it stays in the convex
hull (`softmaxStep_mem_configHull`), and on a bounded configuration every score
is at least one constant `c > 0` (`attWeight_ge`).  In any direction `u` the
spread `max_j ⟨u, x_j⟩ - min_j ⟨u, x_j⟩` then shrinks by the factor `1 - γc`;
taking for `u` the difference of two particles, so does every pairwise distance
(`norm_sub_softmaxStep_le`).  A step moves a particle by at most the diameter
(`norm_softmaxStep_sub_le`).
-/

import Transformer.FrankWolfe.Section5_Process

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- `(SA_β)` as a convex combination: `x_i ↦ (1 - γ) x_i + γ Σ_j a_ij x_j`, with
`a_ij` the attention scores. -/
theorem softmaxStep_eq (β γ : ℝ) (X : Idx n → EucSpace d) (i : Idx n) :
    softmaxStep β γ X i = (1 - γ) • X i + γ • ∑ j, attWeight β X i j • X j := by
  simp only [softmaxStep, attWeight, Finset.smul_sum, smul_smul, div_eq_inv_mul]

theorem inner_softmaxStep (β γ : ℝ) (X : Idx n → EucSpace d) (i : Idx n) (u : EucSpace d) :
    inner (𝕜 := ℝ) u (softmaxStep β γ X i) = (1 - γ) * inner (𝕜 := ℝ) u (X i) +
      γ * ∑ j, attWeight β X i j * inner (𝕜 := ℝ) u (X j) := by
  rw [softmaxStep_eq, inner_add_right, real_inner_smul_right, real_inner_smul_right, inner_sum]
  simp only [real_inner_smul_right]

/-- After one step `⟨u, x_i⟩` lies below `M ≥ max_j ⟨u, x_j⟩` by at least
`γ c (M - ⟨u, x_{j₀}⟩)`, where `c` bounds the score `a_{ij₀}` from below. -/
theorem inner_softmaxStep_le (β : ℝ) {γ c M : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (X : Idx n → EucSpace d) (i j₀ : Idx n) (u : EucSpace d)
    (hM : ∀ j, inner (𝕜 := ℝ) u (X j) ≤ M) (hc : c ≤ attWeight β X i j₀) :
    inner (𝕜 := ℝ) u (softmaxStep β γ X i) ≤ M - γ * c * (M - inner (𝕜 := ℝ) u (X j₀)) := by
  have hsum : attWeight β X i j₀ * (M - inner (𝕜 := ℝ) u (X j₀)) ≤
      ∑ j, attWeight β X i j * (M - inner (𝕜 := ℝ) u (X j)) :=
    Finset.single_le_sum (f := fun j => attWeight β X i j * (M - inner (𝕜 := ℝ) u (X j)))
      (fun j _ => mul_nonneg (attWeight_pos β X i j).le (sub_nonneg.mpr (hM j)))
      (Finset.mem_univ j₀)
  have hexp : ∑ j, attWeight β X i j * (M - inner (𝕜 := ℝ) u (X j)) =
      M - ∑ j, attWeight β X i j * inner (𝕜 := ℝ) u (X j) := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, sum_attWeight, one_mul]
  have hc' := mul_le_mul_of_nonneg_right hc (sub_nonneg.mpr (hM j₀))
  have e1 := mul_le_mul_of_nonneg_left (hM i) (sub_nonneg.mpr hγ1)
  have e2 := mul_le_mul_of_nonneg_left (show ∑ j, attWeight β X i j * inner (𝕜 := ℝ) u (X j) ≤
    M - c * (M - inner (𝕜 := ℝ) u (X j₀)) by linarith) hγ0
  rw [inner_softmaxStep]
  linarith

/-- The hypotheses of `inner_softmaxStep_le` are satisfiable: one particle at
the origin, `γ = 1`, `c = 0`, `M = 0`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ j : Idx 1, inner (𝕜 := ℝ) (0 : EucSpace 1) ((fun _ : Idx 1 => (0 : EucSpace 1)) j) ≤ 0) ∧
    (0 : ℝ) ≤ attWeight 1 (fun _ : Idx 1 => (0 : EucSpace 1)) 0 0 :=
  ⟨zero_le_one, le_rfl, fun _ => by simp, (attWeight_pos _ _ _ _).le⟩

/-- For `0 ≤ γ ≤ 1` a step stays in the convex hull of the configuration. -/
theorem softmaxStep_mem_configHull (β : ℝ) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (X : Idx n → EucSpace d) (i : Idx n) : softmaxStep β γ X i ∈ configHull X := by
  rw [softmaxStep_eq]
  exact convex_convexHull ℝ _ (subset_convexHull ℝ _ (Set.mem_range_self i))
    ((convex_convexHull ℝ _).sum_mem (fun j _ => (attWeight_pos β X i j).le)
      (sum_attWeight β X i) fun j _ => subset_convexHull ℝ _ (Set.mem_range_self j))
    (sub_nonneg.mpr hγ1) hγ0 (sub_add_cancel 1 γ)

/-- The hypotheses of `softmaxStep_mem_configHull` are satisfiable: `γ = 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 := ⟨zero_le_one, le_rfl⟩

/-- On a configuration in the ball of radius `R` every score is at least
`e^{-|β|R²} / (n e^{|β|R²})`. -/
theorem attWeight_ge (β : ℝ) {R : ℝ} (X : Idx n → EucSpace d) (hR : ∀ j, ‖X j‖ ≤ R)
    (i j : Idx n) :
    exp (-(|β| * (R * R))) / (n * exp (|β| * (R * R))) ≤ attWeight β X i j := by
  have hb : ∀ k, |β * inner (𝕜 := ℝ) (X i) (X k)| ≤ |β| * (R * R) := fun k => by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans
      (mul_le_mul (hR i) (hR k) (norm_nonneg _) ((norm_nonneg _).trans (hR i)))) (abs_nonneg β)
  have hZ : ∑ k, exp (β * inner (𝕜 := ℝ) (X i) (X k)) ≤ n * exp (|β| * (R * R)) :=
    calc ∑ k, exp (β * inner (𝕜 := ℝ) (X i) (X k)) ≤ ∑ _k : Idx n, exp (|β| * (R * R)) :=
          Finset.sum_le_sum fun k _ => exp_le_exp.mpr (abs_le.mp (hb k)).2
      _ = n * exp (|β| * (R * R)) := by simp
  exact div_le_div₀ (exp_pos _).le (exp_le_exp.mpr (abs_le.mp (hb j)).1)
    (Finset.sum_pos (fun k _ => exp_pos _) ⟨i, Finset.mem_univ i⟩) hZ

/-- The hypothesis of `attWeight_ge` is satisfiable: the origin, `R = 0`. -/
example : ∀ j : Idx 1, ‖(fun _ : Idx 1 => (0 : EucSpace 1)) j‖ ≤ 0 := fun _ => by simp

private theorem le_of_sq_le_mul {a b : ℝ} (hb : 0 ≤ b) (h : a ^ 2 ≤ a * b) :
    a ≤ b := by
  nlinarith

/-- The hypotheses of `le_of_sq_le_mul` are satisfiable: `a = b = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ^ 2 ≤ 0 * 0 := ⟨le_rfl, by norm_num⟩

/-- **One step contracts the configuration.**  If every score is at least
`c ≥ 0` and all pairwise distances are at most `D`, then after one step they are
at most `(1 - γc) D`. -/
theorem norm_sub_softmaxStep_le (β : ℝ) {γ c D : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hc0 : 0 ≤ c) (X : Idx n → EucSpace d) (hc : ∀ i j, c ≤ attWeight β X i j)
    (hD : ∀ j j', ‖X j - X j'‖ ≤ D) (i i' : Idx n) :
    ‖softmaxStep β γ X i - softmaxStep β γ X i'‖ ≤ (1 - γ * c) * D := by
  have : Nonempty (Idx n) := ⟨i⟩
  obtain ⟨u, hu⟩ : ∃ u, softmaxStep β γ X i - softmaxStep β γ X i' = u := ⟨_, rfl⟩
  obtain ⟨j₀, hj₀⟩ := Finite.exists_min fun j => inner (𝕜 := ℝ) u (X j)
  obtain ⟨j₁, hj₁⟩ := Finite.exists_max fun j => inner (𝕜 := ℝ) u (X j)
  have hup := inner_softmaxStep_le β hγ0 hγ1 X i j₀ u hj₁ (hc i j₀)
  have hlo := inner_softmaxStep_le β hγ0 hγ1 X i' j₁ (-u) (M := -inner (𝕜 := ℝ) u (X j₀))
    (fun j => by rw [inner_neg_left]; exact neg_le_neg (hj₀ j)) (hc i' j₁)
  rw [inner_neg_left, inner_neg_left] at hlo
  have hsq : ‖u‖ ^ 2 = inner (𝕜 := ℝ) u (softmaxStep β γ X i) -
      inner (𝕜 := ℝ) u (softmaxStep β γ X i') := by
    rw [← inner_sub_right, hu, real_inner_self_eq_norm_sq]
  have hspread : inner (𝕜 := ℝ) u (X j₁) - inner (𝕜 := ℝ) u (X j₀) ≤ ‖u‖ * D := by
    rw [← inner_sub_right]
    exact (real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left (hD j₁ j₀) (norm_nonneg u))
  have hc1 : γ * c ≤ 1 := by
    nlinarith [mul_le_mul hγ1 ((hc i i).trans (attWeight_le_one β X i i)) hc0 zero_le_one]
  have h1 : 0 ≤ γ * c * (inner (𝕜 := ℝ) u (X j₁) - inner (𝕜 := ℝ) u (X j₀)) :=
    mul_nonneg (mul_nonneg hγ0 hc0) (sub_nonneg.mpr (hj₀ j₁))
  have h2 := mul_le_mul_of_nonneg_left hspread (sub_nonneg.mpr hc1)
  rw [hu]
  exact le_of_sq_le_mul
    (mul_nonneg (sub_nonneg.mpr hc1) ((norm_nonneg _).trans (hD i i))) (by linarith)

/-- The hypotheses of `norm_sub_softmaxStep_le` are satisfiable: one particle,
`γ = 1`, `c = 0`, `D = 0`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ i j : Idx 1, (0 : ℝ) ≤ attWeight 1 (fun _ : Idx 1 => (0 : EucSpace 1)) i j) ∧
    ∀ j j' : Idx 1, ‖(fun _ : Idx 1 => (0 : EucSpace 1)) j - (fun _ => 0) j'‖ ≤ 0 :=
  ⟨zero_le_one, le_rfl, le_rfl, fun _ _ => (attWeight_pos _ _ _ _).le, fun _ _ => by simp⟩

/-- **One step moves a particle by at most the diameter.** -/
theorem norm_softmaxStep_sub_le (β : ℝ) {γ D : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (X : Idx n → EucSpace d) (hD : ∀ j j', ‖X j - X j'‖ ≤ D) (i : Idx n) :
    ‖softmaxStep β γ X i - X i‖ ≤ D := by
  have : Nonempty (Idx n) := ⟨i⟩
  obtain ⟨u, hu⟩ : ∃ u, softmaxStep β γ X i - X i = u := ⟨_, rfl⟩
  obtain ⟨j₀, hj₀⟩ := Finite.exists_min fun j => inner (𝕜 := ℝ) u (X j)
  obtain ⟨j₁, hj₁⟩ := Finite.exists_max fun j => inner (𝕜 := ℝ) u (X j)
  have hup := inner_softmaxStep_le β hγ0 hγ1 X i j₀ u hj₁ (attWeight_pos β X i j₀).le
  have hsq : ‖u‖ ^ 2 = inner (𝕜 := ℝ) u (softmaxStep β γ X i) - inner (𝕜 := ℝ) u (X i) := by
    rw [← inner_sub_right, hu, real_inner_self_eq_norm_sq]
  have hspread : inner (𝕜 := ℝ) u (X j₁) - inner (𝕜 := ℝ) u (X j₀) ≤ ‖u‖ * D := by
    rw [← inner_sub_right]
    exact (real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left (hD j₁ j₀) (norm_nonneg u))
  rw [hu]
  exact le_of_sq_le_mul ((norm_nonneg _).trans (hD i i))
    (by linarith [hj₀ i])

/-- The hypotheses of `norm_softmaxStep_sub_le` are satisfiable: one particle,
`γ = 1`, `D = 0`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    ∀ j j' : Idx 1, ‖(fun _ : Idx 1 => (0 : EucSpace 1)) j - (fun _ => 0) j'‖ ≤ 0 :=
  ⟨zero_le_one, le_rfl, fun _ _ => by simp⟩

end FrankWolfe
end Transformer

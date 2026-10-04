/-
# The power iteration

arXiv:1211.5063, supplementary "Analytical analysis of the exploding and
vanishing gradients problem".  For the linear model `∂x_t/∂x_k = W_rec^l`,
`l = t - k` (`factor_id`), where eq. (`prod_wk`) writes `(W_recᵀ)^l`.  The
paper writes the error `∂L_t/∂x_t = Σ_i c_i q_iᵀ` in eigenvectors `q_i` of
`W_rec`, by `q_iᵀ (W_recᵀ)^l = λ_i^l q_iᵀ`; for the corrected factor the `q_iᵀ`
are left eigenvectors `p_i`, `p_i W_rec = λ_i p_i`, so `p_i W_rec^l = λ_i^l p_i`
(`comp_pow_eq_smul`) and `∂L_t/∂x_t ∂x_t/∂x_k = Σ_i c_i λ_i^l p_i`
(`comp_factor_id`).  With `|λ_1| > ⋯ > |λ_n|` and `j` the first index with
`c_j ≠ 0`, this over `λ_j^l` tends to `c_j p_j`, eq. (`approx_comp`)
(`tendsto_div_pow_comp_factor_id`), for `λ_j ≠ 0`, which the ratios
`λ_i/λ_j` presuppose.  Neither a basis of eigenvectors nor the order of all the
`|λ_i|` is needed: only `|λ_i| < |λ_j|` for the other `i` with `c_i ≠ 0`
(`tendsto_div_pow_sum`), which also bounds the sum below by `C |λ_j|^l`
(`eventually_le_norm_sum`); `SectionA_Growth` draws the growth claims.
-/

import Transformer.RecurrentGradients.Section2_Linear
import Mathlib.Analysis.SpecificLimits.Normed

open Filter Topology

namespace Transformer.RecurrentGradients

section Dominant

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {n : ℕ} {x : Fin n → V}
  {c μ : Fin n → ℝ} {j : Fin n}

/-- **Eq. (`approx_comp`), abstractly**: `Σ_i c_i λ_i^l x_i` over `λ_j^l` tends to
`c_j x_j` when `|λ_i| < |λ_j|` for every other `i` with `c_i ≠ 0`. -/
theorem tendsto_div_pow_sum (hμ : μ j ≠ 0) (hc : ∀ i ≠ j, c i ≠ 0 → |μ i| < |μ j|) :
    Tendsto (fun l : ℕ => (μ j ^ l)⁻¹ • ∑ i, (c i * μ i ^ l) • x i) atTop (𝓝 (c j • x j)) := by
  have e : (fun l : ℕ => (μ j ^ l)⁻¹ • ∑ i, (c i * μ i ^ l) • x i) =
      fun l => ∑ i, (c i * (μ i / μ j) ^ l) • x i := by
    funext l
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [smul_smul, div_pow]
    congr 1
    field_simp
  rw [e]
  have h := tendsto_finsetSum (f := fun i (l : ℕ) => (c i * (μ i / μ j) ^ l) • x i)
    (a := fun i => if i = j then c j • x j else 0) (x := atTop) Finset.univ fun i _ => by
      split_ifs with hij
      · subst hij
        simp [div_self hμ]
      · rcases eq_or_ne (c i) 0 with hci | hci
        · simp [hci]
        · have := tendsto_pow_atTop_nhds_zero_of_abs_lt_one (r := μ i / μ j) (by
            rw [abs_div, div_lt_one (abs_pos.2 hμ)]
            exact hc i hij hci)
          simpa using (this.const_mul (c i)).smul_const (x i)
  simpa using h

/-- Hence `‖Σ_i c_i λ_i^l x_i‖ ≥ C |λ_j|^l` eventually, `C > 0`, when `c_j x_j ≠ 0`. -/
theorem eventually_le_norm_sum (hμ : μ j ≠ 0) (hc : ∀ i ≠ j, c i ≠ 0 → |μ i| < |μ j|)
    (hx : c j • x j ≠ 0) :
    ∃ C > 0, ∀ᶠ l : ℕ in atTop, C * |μ j| ^ l ≤ ‖∑ i, (c i * μ i ^ l) • x i‖ := by
  have h0 := norm_pos_iff.2 hx
  refine ⟨‖c j • x j‖ / 2, half_pos h0, ?_⟩
  filter_upwards [(tendsto_div_pow_sum hμ hc).norm.eventually (lt_mem_nhds (half_lt_self h0))]
    with l hl
  rw [norm_smul (μ j ^ l)⁻¹, norm_inv, norm_pow, Real.norm_eq_abs, inv_mul_eq_div,
    lt_div_iff₀ (pow_pos (abs_pos.2 hμ) l)] at hl
  exact hl.le

/-- The hypotheses of `tendsto_div_pow_sum` are satisfiable: `λ = (2, 1)`,
`c = (1, 1)`, `x = (1, 1)`, `j = 0`. -/
example : Tendsto (fun l : ℕ => (![(2 : ℝ), 1] 0 ^ l)⁻¹ •
    ∑ i, ((fun _ => (1 : ℝ)) i * ![(2 : ℝ), 1] i ^ l) • (1 : ℝ)) atTop
    (𝓝 ((fun _ => (1 : ℝ)) 0 • (1 : ℝ))) :=
  tendsto_div_pow_sum (by norm_num) fun i hi _ => by
    fin_cases i
    · exact absurd rfl hi
    · norm_num

/-- The hypotheses of `eventually_le_norm_sum` are satisfiable: as above. -/
example : ∃ C > 0, ∀ᶠ l : ℕ in atTop,
    C * |![(2 : ℝ), 1] 0| ^ l ≤ ‖∑ i, ((fun _ => (1 : ℝ)) i * ![(2 : ℝ), 1] i ^ l) • (1 : ℝ)‖ :=
  eventually_le_norm_sum (by norm_num) (fun i hi _ => by
    fin_cases i
    · exact absurd rfl hi
    · norm_num) (by norm_num)

end Dominant

/-- `|λ_1| > ⋯ > |λ_n|` and `c_i = 0` before `j` leave `|λ_i| < |λ_j|` for the
other `i` with `c_i ≠ 0`. -/
theorem abs_lt_of_strictAnti {n : ℕ} {c μ : Fin n → ℝ} (hμ : StrictAnti fun i => |μ i|)
    {j : Fin n} (hj : ∀ i < j, c i = 0) : ∀ i ≠ j, c i ≠ 0 → |μ i| < |μ j| := fun i hij hci =>
  hμ ((lt_or_gt_of_ne hij).resolve_left fun h => hci (hj i h))

/-- The coordinates are the left eigenvectors of a diagonal matrix. -/
theorem proj_comp_diagonal {ι : Type*} [Fintype ι] [DecidableEq ι] (d : ι → ℝ) (i : ι) :
    EuclideanSpace.proj i ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal d) =
      d i • EuclideanSpace.proj i := by
  ext v
  simp [Matrix.mulVec_diagonal]

/-- `|2| > |1|`. -/
theorem strictAnti_two_one : StrictAnti fun i => |![(2 : ℝ), 1] i| := by
  intro a b hab
  fin_cases a <;> fin_cases b <;> simp at hab ⊢

/-- The hypothesis of `abs_lt_of_strictAnti` is satisfiable: `λ = (2, 1)`, `j = 0`. -/
example : ∀ i ≠ 0, (1 : ℝ) ≠ 0 → |![(2 : ℝ), 1] i| < |![(2 : ℝ), 1] 0| :=
  abs_lt_of_strictAnti (c := fun _ => 1) strictAnti_two_one fun i h => absurd h (Fin.not_lt_zero i)

/-- `p A = λ p` gives `p A^l = λ^l p`: the paper's `q_iᵀ (W_recᵀ)^l = λ_i^l q_iᵀ`,
for the corrected factor `W_rec^l` and left eigenvectors `p`. -/
theorem comp_pow_eq_smul {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {A : E →L[ℝ] E} {p : E →L[ℝ] ℝ} {μ : ℝ} (hp : p ∘L A = μ • p) (l : ℕ) :
    p ∘L A ^ l = μ ^ l • p := by
  induction l with
  | zero => ext v; simp
  | succ l ih =>
    rw [pow_succ A, ContinuousLinearMap.mul_def, ← ContinuousLinearMap.comp_assoc, ih,
      ContinuousLinearMap.smul_comp, hp, smul_smul, ← pow_succ]

/-- The hypothesis of `comp_pow_eq_smul` is satisfiable: `diag(2, 1)`. -/
example (l : ℕ) : EuclideanSpace.proj 0 ∘L Matrix.toEuclideanCLM (𝕜 := ℝ)
    (Matrix.diagonal ![(2 : ℝ), 1]) ^ l = ![(2 : ℝ), 1] 0 ^ l • EuclideanSpace.proj 0 :=
  comp_pow_eq_smul (proj_comp_diagonal _ 0) l

variable {n : ℕ} {κ : Type*} [Fintype κ] {W : Matrix (Fin n) (Fin n) ℝ}
  {Win : Matrix (Fin n) κ ℝ} {b : EuclideanSpace ℝ (Fin n)}

/-- The first equality of eq. (`approx_comp`), corrected:
`(Σ_i c_i p_i) ∂x_{k+l}/∂x_k = Σ_i c_i λ_i^l p_i`. -/
theorem comp_factor_id {p : Fin n → EuclideanSpace ℝ (Fin n) →L[ℝ] ℝ} {μ : Fin n → ℝ}
    (hp : ∀ i, p i ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) W = μ i • p i) (c : Fin n → ℝ)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ (Fin n)) (k l : ℕ) :
    (∑ i, c i • p i) ∘L factor id W Win b u x₀ k l = ∑ i, (c i * μ i ^ l) • p i := by
  rw [factor_id, map_pow, ContinuousLinearMap.finsetSum_comp]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [ContinuousLinearMap.smul_comp, comp_pow_eq_smul (hp i), smul_smul]

/-- The hypothesis of `comp_factor_id` is satisfiable: `W_rec = diag(2, 1)` and the
coordinates. -/
example (Win : Matrix (Fin 2) κ ℝ) (b : EuclideanSpace ℝ (Fin 2)) (c : Fin 2 → ℝ)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ (Fin 2)) (k l : ℕ) :
    (∑ i, c i • EuclideanSpace.proj i) ∘L factor id (Matrix.diagonal ![2, 1]) Win b u x₀ k l =
      ∑ i, (c i * ![(2 : ℝ), 1] i ^ l) • EuclideanSpace.proj i :=
  comp_factor_id (proj_comp_diagonal _) c u x₀ k l

/-- **Eq. (`approx_comp`), corrected**: for `|λ_1| > ⋯ > |λ_n|`, `j` the first
index with `c_j ≠ 0` and `λ_j ≠ 0`, `(Σ_i c_i p_i) ∂x_{k+l}/∂x_k` over `λ_j^l`
tends to `c_j p_j`. -/
theorem tendsto_div_pow_comp_factor_id {p : Fin n → EuclideanSpace ℝ (Fin n) →L[ℝ] ℝ}
    {μ : Fin n → ℝ} (hp : ∀ i, p i ∘L Matrix.toEuclideanCLM (𝕜 := ℝ) W = μ i • p i)
    (hμ : StrictAnti fun i => |μ i|) {c : Fin n → ℝ} {j : Fin n} (hj : ∀ i < j, c i = 0)
    (hμj : μ j ≠ 0) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ (Fin n)) (k : ℕ) :
    Tendsto (fun l => (μ j ^ l)⁻¹ • ((∑ i, c i • p i) ∘L factor id W Win b u x₀ k l)) atTop
      (𝓝 (c j • p j)) := by
  simp only [comp_factor_id hp]
  exact tendsto_div_pow_sum hμj (abs_lt_of_strictAnti hμ hj)

/-- The hypotheses of `tendsto_div_pow_comp_factor_id` are satisfiable:
`W_rec = diag(2, 1)`, the coordinates, the error `(1, 1)`. -/
example (Win : Matrix (Fin 2) κ ℝ) (b : EuclideanSpace ℝ (Fin 2)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 2)) (k : ℕ) :
    Tendsto (fun l => (![(2 : ℝ), 1] 0 ^ l)⁻¹ • ((∑ i, (fun _ => (1 : ℝ)) i •
      EuclideanSpace.proj i) ∘L factor id (Matrix.diagonal ![2, 1]) Win b u x₀ k l)) atTop
      (𝓝 ((fun _ => (1 : ℝ)) 0 • EuclideanSpace.proj 0)) :=
  tendsto_div_pow_comp_factor_id (proj_comp_diagonal _) strictAnti_two_one
    (fun i h => absurd h (Fin.not_lt_zero i)) (by norm_num) u x₀ k

end Transformer.RecurrentGradients

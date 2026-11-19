module

public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
public import Mathlib.Topology.MetricSpace.ProperSpace
public import Mathlib.Topology.Order.Compact
public import Transformer.KERPLE.Section3_Powers

/-!
# Lemma 1 and Appendix A.1: the CPD shift lemma

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.2, Lemma 1 (CPD Shift Lemma), proved in
Appendix A.1: "Let `k̃ : X × X → ℝ` be a CPD kernel. There exists `c ≥ 0` such that `c + k̃` is a
PD kernel."

* Lemma 1 is false (`not_cpd_shift_lemma`): `k̃(x, y) = x + y` is CPD on `ℝ`, its form vanishing
  on coefficients that sum to `0` (`isCPDKernel_add`), and no real `c` makes `c + k̃` PD: at the
  points `1/2, −1/2` the form of `c + k̃` takes the value `−1` (`not_isPDKernel_const_add_add`).
* It fails for a kernel of Corollary 1 as well: `−(x − x')²` is CPD (Fact 4), and no real `c`
  makes `c − (x − x')²` PD, already at the points `0, 1, 2` (`not_isPDKernel_const_sub_sq`).
* Step (iii) of the proof, "since `K` has no zero eigenvalue, we cannot have `vᵀ K v = 0` when
  `vᵀ 1 = 0`", fails (`step_iii_counterexample`): `K = diag(1, −1)`, the matrix of `x + y` at
  `1/2, −1/2`, has determinant `−1`, while `v = (1, −1)` has `vᵀ 1 = 0` and `vᵀ K v = 0`.
* What steps (iv)–(vi) show holds (`exists_isPDKernel_const_add`): on a finite set, a symmetric
  kernel whose form is positive at every nonzero `v` with `Σ v = 0` (strictly CPD) becomes PD
  after adding a large enough `c ≥ 0`; the proof is theirs, a minimum over a compact set.
* A larger constant keeps a kernel PD (`IsPDKernel.add_const_of_le`), so the geometric search for
  `c` in §3.2 stops once it passes some `c` that works.
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset

variable {X : Type*}

/-- `k̃(x, y) = x + y` is CPD on `ℝ`: on coefficients summing to `0` its form is
`2 (Σ_i c_i x_i)(Σ_j c_j) = 0`. -/
theorem isCPDKernel_add : IsCPDKernel fun x y : ℝ => x + y := by
  refine ⟨fun x y => add_comm x y, fun N x c hc => ?_⟩
  have h : ∑ i, ∑ j, c i * c j * (x i + x j) = 0 := by
    simp only [mul_add, sum_add_distrib]
    rw [sum_sum_mul_left x c, sum_sum_mul_right x c, hc]
    ring
  rw [h]

/-- No real `c` makes `c + (x + y)` PD on `ℝ`: at the points `1/2, −1/2` with the coefficients
`(−c/2, 1 + c/2)` the form is `−1`. -/
theorem not_isPDKernel_const_add_add (c : ℝ) : ¬IsPDKernel fun x y : ℝ => c + (x + y) := by
  intro h
  have := h.2 2 ![1 / 2, -1 / 2] ![-c / 2, 1 + c / 2]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one] at this
  ring_nf at this
  linarith

/-- **Lemma 1 (CPD Shift Lemma) is false**: some CPD kernel `k̃` has no `c ≥ 0`, indeed no real
`c`, with `c + k̃` PD. -/
theorem not_cpd_shift_lemma :
    ¬∀ k : ℝ → ℝ → ℝ, IsCPDKernel k → ∃ c ≥ 0, IsPDKernel fun x y => c + k x y := fun h => by
  obtain ⟨c, -, hc⟩ := h _ isCPDKernel_add
  exact not_isPDKernel_const_add_add c hc

/-- **Lemma 1 fails for `−(x − x')²`**, a CPD kernel of Corollary 1 (`p = 2`): no real `c` makes
`c − (x − y)²` PD on `ℝ`. At the points `0, 1, 2` with the coefficients `(1 + d, −2d, d)` the
form is `c − 4d`, negative for `d = |c| + 1`. -/
theorem not_isPDKernel_const_sub_sq (c : ℝ) : ¬IsPDKernel fun x y : ℝ => c + -(x - y) ^ 2 := by
  intro h
  have := h.2 3 ![0, 1, 2] ![1 + (|c| + 1), -2 * (|c| + 1), |c| + 1]
  simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons] at this
  ring_nf at this
  linarith [le_abs_self c, abs_nonneg c]

/-- **Step (iii) of the proof of Lemma 1 fails.** `K = diag(1, −1)` is the matrix of the CPD
kernel `x + y` at the points `1/2, −1/2` and has no zero eigenvalue (determinant `−1`), yet
`v = (1, −1)` has `vᵀ 1 = 0` and `vᵀ K v = 0`. -/
theorem step_iii_counterexample :
    (Matrix.of fun i j : Fin 2 => ![(1 / 2 : ℝ), -1 / 2] i + ![(1 / 2 : ℝ), -1 / 2] j) =
        !![1, 0; 0, -1] ∧
      (!![1, 0; 0, -1] : Matrix (Fin 2) (Fin 2) ℝ).det ≠ 0 ∧ ∑ i, ![(1 : ℝ), -1] i = 0 ∧
      ![(1 : ℝ), -1] ⬝ᵥ (!![1, 0; 0, -1] : Matrix (Fin 2) (Fin 2) ℝ).mulVec ![1, -1] = 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num
  · rw [Matrix.det_fin_two_of]
    norm_num
  · simp
  · simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two]

/-- A larger constant keeps a kernel PD: `c' + k = (c' − c) + (c + k)`. -/
theorem IsPDKernel.add_const_of_le {k : X → X → ℝ} {c c' : ℝ}
    (hk : IsPDKernel fun x y => c + k x y) (hc : c ≤ c') : IsPDKernel fun x y => c' + k x y :=
  ((isPDKernel_const (sub_nonneg.mpr hc)).add hk).congr fun x y => by ring

/-- The hypotheses of `IsPDKernel.add_const_of_le` hold for the zero kernel and `0 ≤ 1`. -/
example : IsPDKernel fun _ _ : ℕ => (1 : ℝ) + 0 :=
  IsPDKernel.add_const_of_le (c := 0) ((isPDKernel_const le_rfl).congr fun _ _ => by ring)
    zero_le_one

/-- Scaling the coefficients by `r` scales a form by `r²`. -/
theorem sum_sum_smul_coeff [Fintype X] (k : X → X → ℝ) (r : ℝ) (u : X → ℝ) :
    ∑ x, ∑ y, (r • u) x * (r • u) y * k x y = r ^ 2 * ∑ x, ∑ y, u x * u y * k x y := by
  simp only [Pi.smul_apply, smul_eq_mul, mul_sum]
  exact sum_congr rfl fun x _ => sum_congr rfl fun y _ => by ring

/-- On the unit sphere of `X → ℝ`, `X` finite, some `c ≥ 0` has `c (Σ u)² + Q(u) ≥ 0`, when the
form `Q` of `k` is positive at the nonzero `v` with `Σ v = 0` (steps (iv)–(vi) of Appendix A.1). -/
theorem exists_const_on_sphere [Fintype X] {k : X → X → ℝ}
    (hk : ∀ v : X → ℝ, v ≠ 0 → ∑ x, v x = 0 → 0 < ∑ x, ∑ y, v x * v y * k x y) :
    ∃ c ≥ 0, ∀ u ∈ Metric.sphere (0 : X → ℝ) 1,
      0 ≤ c * (∑ x, u x) ^ 2 + ∑ x, ∑ y, u x * u y * k x y := by
  have hQc : Continuous fun v : X → ℝ => ∑ x, ∑ y, v x * v y * k x y := by fun_prop
  have hhc : Continuous fun v : X → ℝ => (∑ x, v x) ^ 2 := by fun_prop
  have hS : IsCompact (Metric.sphere (0 : X → ℝ) 1) := isCompact_sphere 0 1
  obtain ⟨M, hM⟩ := (hS.image_of_continuousOn hQc.continuousOn).bddBelow
  set T := Metric.sphere (0 : X → ℝ) 1 ∩ {v | ∑ x, ∑ y, v x * v y * k x y ≤ 0}
  have hT : IsCompact T := hS.inter_right (isClosed_le hQc continuous_const)
  rcases T.eq_empty_or_nonempty with hTe | hTne
  · refine ⟨0, le_rfl, fun u hu => ?_⟩
    have hu' : u ∉ T := hTe ▸ Set.notMem_empty u
    have : ¬∑ x, ∑ y, u x * u y * k x y ≤ 0 := fun h => hu' ⟨hu, h⟩
    linarith [not_le.mp this]
  obtain ⟨w, hwT, hw⟩ := hT.exists_isMinOn hTne hhc.continuousOn
  have hpos : 0 < (∑ x, w x) ^ 2 := by
    refine (sq_nonneg _).lt_of_ne fun h => ?_
    have hw0 : w ≠ 0 := by
      rintro rfl
      have := hwT.1
      simp at this
    exact absurd hwT.2 (not_le.mpr (hk w hw0 (pow_eq_zero_iff two_ne_zero |>.mp h.symm)))
  refine ⟨|M| / (∑ x, w x) ^ 2, by positivity, fun u hu => ?_⟩
  have hMu : M ≤ ∑ x, ∑ y, u x * u y * k x y := hM ⟨u, hu, rfl⟩
  by_cases hQu : ∑ x, ∑ y, u x * u y * k x y ≤ 0
  · have hwu : (∑ x, w x) ^ 2 ≤ (∑ x, u x) ^ 2 := isMinOn_iff.mp hw u ⟨hu, hQu⟩
    have : |M| ≤ |M| / (∑ x, w x) ^ 2 * (∑ x, u x) ^ 2 := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hpos]
      exact mul_le_mul_of_nonneg_left hwu (abs_nonneg M)
    linarith [neg_abs_le M]
  · have : 0 ≤ |M| / (∑ x, w x) ^ 2 * (∑ x, u x) ^ 2 := by positivity
    linarith [not_le.mp hQu]

/-- **Lemma 1, corrected**: what steps (iv)–(vi) of Appendix A.1 show. On a finite set, a
symmetric kernel whose form is positive at every nonzero `v` with `Σ_x v_x = 0` (strictly CPD)
becomes PD after adding a large enough constant `c ≥ 0`. -/
theorem exists_isPDKernel_const_add [Fintype X] {k : X → X → ℝ} (hsymm : ∀ x y, k x y = k y x)
    (hk : ∀ v : X → ℝ, v ≠ 0 → ∑ x, v x = 0 → 0 < ∑ x, ∑ y, v x * v y * k x y) :
    ∃ c ≥ 0, IsPDKernel fun x y => c + k x y := by
  obtain ⟨c, hc0, hc⟩ := exists_const_on_sphere hk
  refine ⟨c, hc0, ?_⟩
  rw [isPDKernel_iff_posSemidef]
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun v => ?_
  · ext i j
    simp [Matrix.conjTranspose_apply, hsymm i j]
  have hform := dotProduct_mulVec_submatrix (fun x y => c + k x y) id v
  rw [Matrix.submatrix_id_id] at hform
  have h2 := sum_sum_add_const k c id v
  simp only [id] at hform h2
  rw [hform, h2]
  rcases eq_or_ne v 0 with rfl | hv
  · simp
  have hr : 0 < ‖v‖ := norm_pos_iff.mpr hv
  have hu : ‖v‖⁻¹ • v ∈ Metric.sphere (0 : X → ℝ) 1 := by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hr.ne']
  have h := hc _ hu
  have hsum : ∑ x, (‖v‖⁻¹ • v) x = ‖v‖⁻¹ * ∑ x, v x := by
    simp only [Pi.smul_apply, smul_eq_mul, mul_sum]
  rw [sum_sum_smul_coeff, hsum] at h
  have hr2 : 0 < (‖v‖⁻¹) ^ 2 := by positivity
  refine le_of_mul_le_mul_left ?_ hr2
  calc (‖v‖⁻¹) ^ 2 * 0 = 0 := mul_zero _
    _ ≤ c * (‖v‖⁻¹ * ∑ x, v x) ^ 2 + (‖v‖⁻¹) ^ 2 * ∑ x, ∑ y, v x * v y * k x y := h
    _ = _ := by ring

/-- The hypotheses of `exists_isPDKernel_const_add` hold for `−|i − j|` on two points: its form at
`v` with `v₀ + v₁ = 0` is `2 v₀²`. -/
example : ∃ c ≥ 0, IsPDKernel fun i j : Fin 2 => c + (if i = j then (0 : ℝ) else -1) := by
  refine exists_isPDKernel_const_add (fun i j => ?_) fun v hv hs => ?_
  · by_cases h : i = j
    · rw [h]
    · simp [h, Ne.symm h]
  · simp only [Fin.sum_univ_two, Fin.isValue] at hs ⊢
    have h1 : v 1 = -v 0 := by linarith
    have h0 : v 0 ≠ 0 := fun h => hv (funext fun i => by fin_cases i <;> simp [h, h1])
    simp [h1]
    positivity

/-- The hypothesis of `exists_const_on_sphere` holds for the same kernel. -/
example : ∃ c ≥ 0, ∀ u ∈ Metric.sphere (0 : Fin 2 → ℝ) 1, 0 ≤ c * (∑ x, u x) ^ 2 +
    ∑ x, ∑ y, u x * u y * (if x = y then (0 : ℝ) else -1) := by
  refine exists_const_on_sphere fun v hv hs => ?_
  simp only [Fin.sum_univ_two, Fin.isValue] at hs ⊢
  have h1 : v 1 = -v 0 := by linarith
  have h0 : v 0 ≠ 0 := fun h => hv (funext fun i => by fin_cases i <;> simp [h, h1])
  simp [h1]
  positivity

end Transformer.KERPLE

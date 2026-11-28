module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling

/-! A priori bounds along the N-particle system. The moment cone bounds the
bias-corrected direction by 12 for every floor eps ≥ 0. With decoupled
weight decay 1/10 and rates in [0,10], each weight stays within
max(|w₀|, 120) of the origin, and one update moves it by at most
rate · (max(|w₀|, 120)/10 + 12). -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Transformer.AMSGradW.Kernite

variable {P K : Type*}

/-- The direction bound of `AdamW.direction_bound` for every floor eps ≥ 0. -/
theorem abs_direction_le {eps : ℝ} (he : 0 ≤ eps) (t : ℕ) {m v : ℝ} (h : 14 * m^2 ≤ 19 * v) :
    |direction eps t m v| ≤ 12 := by
  have hv : 0 ≤ v := by nlinarith [sq_nonneg m]
  have hs := Real.sq_sqrt hv
  have hx : |m| ≤ 6/5 * Real.sqrt v := by
    nlinarith [sq_abs m, abs_nonneg m, Real.sqrt_nonneg v, sq_nonneg (|m| - (6/5 : ℝ) * Real.sqrt v)]
  have hc : 0 < Real.sqrt (1 - (95/100 : ℝ)^(t+1)) :=
    Real.sqrt_pos.2 (sub_pos.2 (pow_lt_one₀ (by norm_num) (by norm_num) (Nat.succ_ne_zero t)))
  have hc1 : Real.sqrt (1 - (95/100 : ℝ)^(t+1)) ≤ 1 := Real.sqrt_le_one.2 (sub_le_self _ (by positivity))
  have hscale : Real.sqrt v ≤ Real.sqrt v / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) + eps :=
    (le_div_self (Real.sqrt_nonneg v) hc hc1).trans (le_add_of_nonneg_right he)
  have hb := firstBias_lower t
  rcases (Real.sqrt_nonneg v).lt_or_eq with hpos | hzero
  · have hd : 0 < denominator eps t v := mul_pos (by linarith) (lt_of_lt_of_le hpos hscale)
    rw [direction, abs_div, abs_of_pos hd, div_le_iff₀ hd, denominator]
    have := mul_le_mul hb hscale (Real.sqrt_nonneg v) (by linarith)
    linarith
  · have hm : m = 0 := abs_nonpos_iff.1 (by rw [← hzero] at hx; linarith)
    simp [direction, hm]

theorem step_weights (eps η : ℝ) (t : ℕ) (g : P → ℝ) (s : AdamW.State P) (p : P) :
    (step eps η t g s).weights p = (1 - η/10) * s.weights p -
      η * direction eps t ((step eps η t g s).first p) ((step eps η t g s).second p) := rfl

/-- A step keeps a weight inside any radius M ≥ 120. -/
theorem abs_step_weights_le {eps η M : ℝ} (he : 0 ≤ eps) (hη0 : 0 ≤ η) (hη : η ≤ 10)
    (hM : 120 ≤ M) (t : ℕ) (g : P → ℝ) {s : AdamW.State P} (hs : Cone s) (p : P)
    (hw : |s.weights p| ≤ M) : |(step eps η t g s).weights p| ≤ M := by
  have hd := abs_direction_le he t (cone_step eps η t g hs p)
  rw [step_weights]
  generalize direction eps t ((step eps η t g s).first p) ((step eps η t g s).second p) = d at hd ⊢
  have h1 : |(1 - η/10) * s.weights p| ≤ (1 - η/10) * M := by
    rw [abs_mul, abs_of_nonneg (by linarith)]
    exact mul_le_mul_of_nonneg_left hw (by linarith)
  have h2 : |η * d| ≤ η * 12 := by
    rw [abs_mul, abs_of_nonneg hη0]
    exact mul_le_mul_of_nonneg_left hd hη0
  have h3 := mul_nonneg hη0 (sub_nonneg.2 hM)
  calc _ ≤ |(1 - η/10) * s.weights p| + |η * d| := abs_sub _ _
    _ ≤ _ := by linarith

/-- A step moves a weight by at most η · (|w|/10 + 12). -/
theorem abs_step_sub_le {eps η : ℝ} (he : 0 ≤ eps) (hη0 : 0 ≤ η) (t : ℕ) (g : P → ℝ)
    {s : AdamW.State P} (hs : Cone s) (p : P) :
    |(step eps η t g s).weights p - s.weights p| ≤ η * (|s.weights p| / 10 + 12) := by
  have hd := abs_direction_le he t (cone_step eps η t g hs p)
  rw [step_weights]
  generalize direction eps t ((step eps η t g s).first p) ((step eps η t g s).second p) = d at hd ⊢
  rw [show (1 - η/10) * s.weights p - η * d - s.weights p = -(η * (s.weights p / 10 + d)) by ring,
    abs_neg, abs_mul, abs_of_nonneg hη0]
  refine mul_le_mul_of_nonneg_left ((abs_add_le _ _).trans ?_) hη0
  rw [abs_div, abs_of_pos (by norm_num : (0:ℝ) < 10)]
  linarith

theorem abs_weights_le (A : Training P K) (he : 0 ≤ A.eps) (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ 10)
    {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (i : Fin N) (p : P) :
    |(system A init t i).weights p| ≤ max |init i p| 120 := by
  induction t with
  | zero => exact le_max_left _ _
  | succ t ih =>
    rw [system_succ]
    exact abs_step_weights_le he (hr t).1 (hr t).2 (le_max_right _ _) t _ (cone_system A init t i) p ih

/-- One update moves a weight by at most rate · (max(|w₀|, 120)/10 + 12). -/
theorem abs_displacement_le (A : Training P K) (he : 0 ≤ A.eps)
    (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ 10) {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (i : Fin N)
    (p : P) :
    |(system A init (t+1) i).weights p - (system A init t i).weights p| ≤
      A.rate t * (max |init i p| 120 / 10 + 12) := by
  have hw := abs_weights_le A he hr init t i p
  rw [system_succ]
  refine (abs_step_sub_le he (hr t).1 t _ (cone_system A init t i) p).trans ?_
  gcongr
  exact (hr t).1

section Examples

example := abs_direction_le (eps := 1) zero_le_one 0 (m := 0) (v := 0) (by norm_num)
example := abs_step_weights_le (P := Fin 1) (eps := 1) (η := 1) (M := 120) zero_le_one zero_le_one
  (by norm_num) le_rfl 0 0 (s := ⟨0, 0, 0⟩) (fun _ => by simp) 0 (by simp)
example := abs_step_sub_le (P := Fin 1) (eps := 1) (η := 1) zero_le_one zero_le_one 0 0
  (s := ⟨0, 0, 0⟩) (fun _ => by simp) 0
example := abs_weights_le identityTraining zero_le_one (fun _ => ⟨zero_le_one, by norm_num [identityTraining]⟩)
  (N := 2) (fun _ => 0) 3 0 0
example := abs_displacement_le identityTraining zero_le_one (fun _ => ⟨zero_le_one, by norm_num [identityTraining]⟩)
  (N := 2) (fun _ => 0) 3 0 0
example := cone_step (P := Fin 1) 1 1 0 0 (s := ⟨0, 0, 0⟩) (fun _ => by simp)
example := second_nonneg_of_cone (P := Fin 1) (s := ⟨0, 0, 0⟩) (fun _ => by simp) 0
example := coordinate_step (eps := 1) (η := 1) one_pos zero_le_one (by norm_num) 0 (w := 0) (m := 0)
  (v := 0) (w' := 0) (m' := 0) (v' := 0) (g := 0) (g' := 0) (by norm_num) le_rfl
example := gap_step (P := Fin 1) (eps := 1) (η := 1) one_pos zero_le_one (by norm_num) 0
  (s := ⟨0, 0, 0⟩) (s' := ⟨0, 0, 0⟩) (fun _ => by simp) (fun _ => by simp) 0 0

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth

module

public import Transformer.ScheduleFree.AdamW.Basic

/-!
# Well-defined updates and averaging in revised AdamW

Source: facebookresearch/schedule_free, commit `59ed9b662d5ec5799aded7390b2482758423653c`,
`adamw_schedulefree.py`, `step`. These results establish positive bias corrections,
positive epsilon-regularized denominators, nonnegative variances and weights,
and convex averaging along actual runs. They do not assert optimization convergence
or a floating-point error bound. A zero current rate freezes `z` but need not
freeze `x`: the code's running maximum can retain earlier positive rates.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- The code's second-moment bias correction belongs to `(0,1]` for beta2∈[0,1). -/
theorem correction_bounds (β : ℝ) (t : ℕ) (hβ : 0 ≤ β) (hβ1 : β < 1) :
    0 < correction β t ∧ correction β t ≤ 1 := by
  have hlt : β ^ (t + 1) < 1 := pow_lt_one₀ hβ hβ1 (Nat.succ_ne_zero t)
  have hnonneg := pow_nonneg hβ (t + 1)
  dsimp [correction]
  constructor <;> linarith

example (t : ℕ) : 0 < correction (999/1000) t ∧ correction (999/1000) t ≤ 1 :=
  correction_bounds _ _ (by norm_num) (by norm_num)

/-- The actual raw-variance update preserves nonnegativity under beta2∈[0,1]. -/
theorem second_nonneg (p : Parameters) (h : History ι) (g : ι → ℝ)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo ≤ 1)
    (hv : ∀ i, 0 ≤ h.second i) (i : ι) : 0 ≤ nextSecond p h g i := by
  exact add_nonneg (mul_nonneg hβ (hv i))
    (mul_nonneg (sub_nonneg.mpr hβ1) (sq_nonneg _))

example : 0 ≤ nextSecond ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
    (initial (fun _ : Fin 1 => 2)).history (fun _ => 3) 0 :=
  second_nonneg _ _ _ (by norm_num) (by norm_num) (fun _ => le_rfl) _

/-- Positive epsilon makes the revised, bias-corrected denominator positive. -/
theorem denominator_pos (p : Parameters) (t : ℕ) (h : History ι) (g : ι → ℝ)
    (hε : 0 < p.adam.epsilon) (i : ι) : 0 < denominator p t h g i :=
  adam_denominator_pos _ _ hε

example : 0 < denominator ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
    (initial (fun _ : Fin 1 => 2)).history (fun _ => 0) 0 :=
  denominator_pos _ _ _ _ zero_lt_one _

/-- The new rate, rather than the `-1` sentinel, controls the first maximum. -/
theorem maximum_nonneg (p : Parameters) (t : ℕ) (h : History ι)
    (hlr : 0 ≤ p.adam.learningRate) : 0 ≤ nextMaximum p t h :=
  le_trans (rate_nonneg p t hlr) (le_max_left _ _)

example (t : ℕ) : 0 ≤ nextMaximum ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
    (initial (fun _ : Fin 1 => 2)).history := maximum_nonneg _ _ _ zero_le_one

/-- The code's group maximum never decreases, including before initialization. -/
theorem maximum_monotone (p : Parameters) (t : ℕ) (h : History ι) :
    h.maximumRate ≤ nextMaximum p t h := le_max_right _ _

/-- General real averaging powers give nonnegative weights on nonnegative rates. -/
theorem weight_nonneg (p : Parameters) (t : ℕ) (h : History ι)
    (hlr : 0 ≤ p.adam.learningRate) : 0 ≤ weight p t h :=
  mul_nonneg (Real.rpow_nonneg (by positivity) _)
    (Real.rpow_nonneg (maximum_nonneg p t h hlr) _)

example (t : ℕ) : 0 ≤ weight ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 1/2, 2⟩ t
    (initial (fun _ : Fin 1 => 2)).history := weight_nonneg _ _ _ zero_le_one

/-- Revised cumulative weights increase even when the running rate has plateaued. -/
theorem total_monotone (p : Parameters) (t : ℕ) (s : State ι) (g : ι → ℝ)
    (hlr : 0 ≤ p.adam.learningRate) : s.history.total ≤ (step p t s g).history.total := by
  change s.history.total ≤ s.history.total + weight p t s.history
  linarith [weight_nonneg p t s.history hlr]

example : (initial (fun _ : Fin 1 => 2)).history.total ≤
    (step ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
      (initial (fun _ : Fin 1 => 2)) (fun _ => 3)).history.total :=
  total_monotone _ _ _ _ zero_le_one

/-- Convex weights include the zero-total prefix, where the code selects zero. -/
theorem coefficient_bounds (p : Parameters) (t : ℕ) (h : History ι)
    (hlr : 0 ≤ p.adam.learningRate) (htotal : 0 ≤ h.total) :
    0 ≤ coefficient p t h ∧ coefficient p t h ≤ 1 := by
  have hw := weight_nonneg p t h hlr
  have hsum := add_nonneg htotal hw
  constructor
  · exact div_nonneg hw hsum
  · exact div_le_one_of_le₀ (by linarith) hsum

example (t : ℕ) :
    0 ≤ coefficient ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
        (initial (fun _ : Fin 1 => 2)).history ∧
      coefficient ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
        (initial (fun _ : Fin 1 => 2)).history ≤ 1 :=
  coefficient_bounds _ _ _ zero_le_one le_rfl

/-- Every reachable total is nonnegative without an assumption on oracle outputs. -/
theorem run_total_nonneg (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hlr : 0 ≤ p.adam.learningRate) (t : ℕ) :
    0 ≤ (run p oracle point t).history.total := by
  induction t with
  | zero => exact le_rfl
  | succ t ih => exact le_trans ih (total_monotone p t _ _ hlr)

example (t : ℕ) : 0 ≤ (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
    (fun _ y => y) (fun _ : Fin 1 => 2) t).history.total :=
  run_total_nonneg _ _ _ zero_le_one t

/-- The revised query-driven run preserves its actual raw variance buffers. -/
theorem run_second_nonneg (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo ≤ 1)
    (t : ℕ) (i : ι) : 0 ≤ (run p oracle point t).history.second i := by
  induction t generalizing i with
  | zero => exact le_rfl
  | succ t ih => exact second_nonneg p _ _ hβ hβ1 ih i

example (t : ℕ) : 0 ≤ (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
    (fun _ y => y) (fun _ : Fin 1 => 2) t).history.second 0 :=
  run_second_nonneg _ _ _ (by norm_num) (by norm_num) t 0

/-- On every actual oracle-driven update, valid beta2 and positive epsilon
make the bias division, square-root input, and final gradient division valid. -/
theorem run_normalization_well_defined (p : Parameters)
    (oracle : ℕ → (ι → ℝ) → (ι → ℝ)) (point : ι → ℝ)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo < 1)
    (hε : 0 < p.adam.epsilon) (t : ℕ) (i : ι) :
    let s := run p oracle point t
    let g := oracle t (query p s)
    0 < correction p.adam.betaTwo t ∧
      0 ≤ nextSecond p s.history g i / correction p.adam.betaTwo t ∧
        0 < denominator p t s.history g i := by
  have hb := (correction_bounds p.adam.betaTwo t hβ hβ1).1
  have hv := second_nonneg p (run p oracle point t).history
    (oracle t (query p (run p oracle point t))) hβ hβ1.le
    (run_second_nonneg p oracle point hβ hβ1.le t) i
  exact ⟨hb, div_nonneg hv hb.le, denominator_pos p t _ _ hε i⟩

example (t : ℕ) :
    let p : Parameters := ⟨⟨1, 0, 10, 9/10, 999/1000, 1⟩, 0, 2⟩
    let s := run p (fun _ y => y) (fun _ : Fin 1 => 2) t
    let g := query p s
    0 < correction p.adam.betaTwo t ∧
      0 ≤ nextSecond p s.history g 0 / correction p.adam.betaTwo t ∧
        0 < denominator p t s.history g 0 :=
  run_normalization_well_defined _ _ _ (by norm_num) (by norm_num) zero_lt_one t 0

/-- The oracle-driven implementation always averages with a coefficient in `[0,1]`. -/
theorem run_coefficient_bounds (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hlr : 0 ≤ p.adam.learningRate) (t : ℕ) :
    0 ≤ coefficient p t (run p oracle point t).history ∧
      coefficient p t (run p oracle point t).history ≤ 1 :=
  coefficient_bounds p t _ hlr (run_total_nonneg p oracle point hlr t)

example (t : ℕ) :
    0 ≤ coefficient ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
        (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
          (fun _ y => y) (fun _ : Fin 1 => 2) t).history ∧
      coefficient ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
        (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
          (fun _ y => y) (fun _ : Fin 1 => 2) t).history ≤ 1 :=
  run_coefficient_bounds _ _ _ zero_le_one t

/-- A zero current rate freezes the base, irrespective of the retained maximum. -/
theorem zero_rate_base (p : Parameters) (t : ℕ) (s : State ι) (g : ι → ℝ)
    (hlr : p.adam.learningRate = 0) : (step p t s g).base = s.base := by
  funext i
  simp [step, rate, hlr]

example : (step ⟨⟨0, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
    (initial (fun _ : Fin 1 => 2)) (fun _ => 3)).base = (fun _ => 2) :=
  zero_rate_base _ _ _ _ rfl

/-- Retaining a positive maximum can keep averaging at zero current rate. -/
theorem zero_rate_can_average :
    (step ⟨⟨0, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
      ⟨fun _ : Fin 1 => 2, fun _ => 0, ⟨fun _ => 0, 1, 1⟩⟩
      (fun _ => 0)).evaluation 0 = 1 := by
  norm_num [step, coefficient, weight, nextMaximum, rate, warmupFactor]

end Transformer.ScheduleFree.AdamW

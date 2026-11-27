module

public import Transformer.AMSGradW.Basic

/-! Exact-real equations of the clipped, bias-corrected Kernite update in
experiments/phasor_value_transport/src/argand/{weights,kernels}.rs.
The input can be the gradient of any minibatch at the current weights.
The constants are the documented Argand AMSGradW settings, with square-root
scale one and the second moment coefficient 0.98. -/

@[expose] public section

namespace Transformer.AMSGradW.Kernite

variable {ι : Type*} [Fintype ι]

abbrev Space (ι : Type*) := WithLp 2 (ι → ℝ)

structure State (ι : Type*) where
  weights : ι → ℝ
  first : ι → ℝ
  second : ι → ℝ
  maximum : ι → ℝ

noncomputable def clip (g : ι → ℝ) (i : ι) : ℝ :=
  g i / max ‖(WithLp.toLp 2 g : Space ι)‖ 1

noncomputable def firstBias (completed : ℕ) : ℝ := 1 - (9 / 10 : ℝ) ^ (completed + 1)

noncomputable def denominator (completed : ℕ) (maximum : ℝ) : ℝ :=
  Real.sqrt maximum / Real.sqrt (1 - (98 / 100 : ℝ) ^ (completed + 1)) + 1 / 100000000

noncomputable def direction (completed : ℕ) (first maximum : ℝ) : ℝ :=
  first / (firstBias completed * denominator completed maximum)

noncomputable def step (rate : ℝ) (completed : ℕ) (g : ι → ℝ) (s : State ι) : State ι :=
  let first i := (9 / 10 : ℝ) * s.first i + (1 / 10 : ℝ) * clip g i
  let second i := (98 / 100 : ℝ) * s.second i + (2 / 100 : ℝ) * (clip g i) ^ 2
  let maximum i := max (s.maximum i) (second i)
  ⟨fun i => (1 - rate / 10) * s.weights i - rate * direction completed (first i) (maximum i),
    first, second, maximum⟩

/-- The minibatch gradient can change at each step. The histories start
at zero and every coordinate has its own first, second and maximum buffers. -/
noncomputable def run (rate : ℕ → ℝ) (batchGradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) : ℕ → State ι
  | 0 => ⟨initial, 0, 0, 0⟩
  | t + 1 => let s := run rate batchGradient initial t
    step (rate t) t (batchGradient t s.weights) s

theorem abs_clip_le (g : ι → ℝ) (i : ι) : |clip g i| ≤ 1 := by
  have hden : 0 < max ‖(WithLp.toLp 2 g : Space ι)‖ 1 :=
    zero_lt_one.trans_le (le_max_right _ _)
  rw [clip, abs_div, abs_of_pos hden, div_le_one hden]
  have h := PiLp.norm_apply_le (WithLp.toLp 2 g : Space ι) i
  rw [Real.norm_eq_abs] at h
  exact h.trans (le_max_left _ _)

theorem firstBias_lower (t : ℕ) : (1 / 10 : ℝ) ≤ firstBias t := by
  have hp : (9 / 10 : ℝ) ^ t ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  rw [firstBias, pow_succ]
  linarith [mul_le_mul_of_nonneg_right hp (by norm_num : (0 : ℝ) ≤ 9 / 10)]

theorem denominator_lower (t : ℕ) (v : ℝ) : (1 / 100000000 : ℝ) ≤ denominator t v :=
  le_add_of_nonneg_left (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))

/-- The buffers stay in a common bounded set, and the maximum is monotone
in each coordinate, even for a changing minibatch. -/
theorem run_buffer_bounds (rate : ℕ → ℝ) (batchGradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (i : ι) (t : ℕ) :
    let s := run rate batchGradient initial t
    |s.first i| ≤ 1 ∧ 0 ≤ s.second i ∧ s.second i ≤ 1 ∧ 0 ≤ s.maximum i ∧ s.maximum i ≤ 1 ∧
      s.maximum i ≤ (run rate batchGradient initial (t + 1)).maximum i := by
  induction t with
  | zero =>
    refine ⟨by simp [run], le_rfl, zero_le_one, le_rfl, zero_le_one, ?_⟩
    exact le_max_left _ _
  | succ t ih =>
    let s := run rate batchGradient initial t
    let g := clip (batchGradient t s.weights) i
    have hg := abs_clip_le (batchGradient t s.weights) i
    have hgs : g ^ 2 ≤ 1 := by
      have h := sq_le_sq.mpr (hg.trans_eq (abs_one : |(1 : ℝ)| = 1).symm)
      simpa only [one_pow] using h
    have hfirst : |(9 / 10 : ℝ) * s.first i + (1 / 10 : ℝ) * g| ≤ 1 := by
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10),
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 10)]
      linarith [ih.1]
    have hlow : 0 ≤ (98 / 100 : ℝ) * s.second i + (2 / 100 : ℝ) * g ^ 2 :=
      add_nonneg (mul_nonneg (by norm_num) ih.2.1) (by positivity)
    have hupp : (98 / 100 : ℝ) * s.second i + (2 / 100 : ℝ) * g ^ 2 ≤ 1 := by
      linarith [ih.2.2.1]
    exact ⟨hfirst, hlow, hupp, ih.2.2.2.1.trans (le_max_left _ _),
      max_le ih.2.2.2.2.1 hupp, le_max_left _ _⟩

theorem abs_direction_le (t : ℕ) {first : ℝ} (hfirst : |first| ≤ 1) (maximum : ℝ) :
    |direction t first maximum| ≤ 1000000000 := by
  have hb := firstBias_lower t
  have hd := denominator_lower t maximum
  have h : (1 / 1000000000 : ℝ) ≤ firstBias t * denominator t maximum := by
    calc
      _ = (1 / 10 : ℝ) * (1 / 100000000) := by norm_num
      _ ≤ _ := mul_le_mul hb hd (by norm_num) (by linarith)
  have hpos : 0 < firstBias t * denominator t maximum := (by norm_num :
    (0 : ℝ) < 1 / 1000000000).trans_le h
  rw [direction, abs_div, abs_of_pos hpos]
  apply (div_le_iff₀ hpos).mpr
  linarith

/-- Positive decoupled decay and clipping bound every weight from arbitrary
initial coordinates, with no assumption on the examples or minibatch order. -/
theorem run_abs_weight_le (rate : ℕ → ℝ) (batchGradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (hrate : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (i : ι) (t : ℕ) :
    |(run rate batchGradient initial t).weights i| ≤ max |initial i| 10000000000 := by
  let B := max |initial i| (10000000000 : ℝ)
  induction t with
  | zero => exact le_max_left _ _
  | succ t ih =>
    let s := run rate batchGradient initial t
    let u := run rate batchGradient initial (t + 1)
    let v := direction t (u.first i) (u.maximum i)
    have hdec : 0 ≤ 1 - rate t / 10 := by linarith [(hrate t).2]
    have hv := abs_direction_le t (run_buffer_bounds rate batchGradient initial i (t + 1)).1
      (u.maximum i)
    have hB : (10000000000 : ℝ) ≤ B := le_max_right _ _
    change |(1 - rate t / 10) * s.weights i - rate t * v| ≤ B
    calc
      _ ≤ |(1 - rate t / 10) * s.weights i| + |rate t * v| := abs_sub _ _
      _ = (1 - rate t / 10) * |s.weights i| + rate t * |v| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hdec, abs_of_nonneg (hrate t).1]
      _ ≤ (1 - rate t / 10) * B + rate t * (B / 10) :=
        add_le_add (mul_le_mul_of_nonneg_left ih hdec)
          (mul_le_mul_of_nonneg_left (by linarith) (hrate t).1)
      _ = B := by ring

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  run_buffer_bounds (fun _ => 1 / 1000) g initial 0 1000

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  run_abs_weight_le (fun _ => 1 / 1000) g initial (fun _ => by norm_num) 0 1000

example := abs_direction_le 0 (first := 1) (by norm_num) 0

example (g : Fin 2 → ℝ) := abs_clip_le g 0

end Transformer.AMSGradW.Kernite

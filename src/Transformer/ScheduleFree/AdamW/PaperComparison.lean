module

public import Transformer.ScheduleFree.AdamW.BiasCorrection

/-!
# Revised AdamW versus the manuscript's Algorithm 1

Source: facebookresearch/schedule_free, commit `59ed9b662d5ec5799aded7390b2482758423653c`,
`adamw_schedulefree.py` and `adamw_schedulefree_paper.py`, compared with
arXiv:2405.15682v4, §4.4, Algorithm 1. The old paper model is imported,
not rewritten to mean the revised algorithm. With epsilon zero, a common
old state and gradient give exactly the decay difference below. A first-step
example with positive epsilon also proves that the complete base updates differ.

The revised default weights use the maximum warmup-only rate. With no warmup
and a positive fixed rate, they give exactly `c=1/(t+1)` from the first step,
independently of beta2. The old squared bias-corrected rates generally do not:
at beta2=3/4 the second averaging coefficient is 7/11 rather than 1/2.
The upstream paper implementation additionally takes a running maximum;
the comparison to `adamStep` here uses the manuscript's own squared rates.
The base comparison fixes a common old state and gradient; the coefficient
example executes each algorithm's own initialized history.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- View a common three-sequence state in the existing manuscript model.
The old model has no maximum-rate field because Algorithm 1 has none. -/
def paperState {d : ℕ} (s : State (Fin d)) : AdamState d :=
  ⟨s.evaluation, s.base, s.history.second, s.history.total⟩

/-- For the same old state and gradient, epsilon zero leaves exactly the
additional query-point decay `-eta*(1-sqrt(b))*lambda*y` in the revised base. -/
theorem base_difference_without_epsilon {d : ℕ} (p : Parameters) (t : ℕ)
    (s : State (Fin d)) (g : Fin d → ℝ) (hε : p.adam.epsilon = 0)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo < 1)
    (i : Fin d) (hv : 0 < nextSecond p s.history g i) :
    (step p t s g).base i - (adamStep p.adam t (paperState s) g).base i =
      -(rate p t * (1 - Real.sqrt (correction p.adam.betaTwo t))) *
        p.adam.decay * query p s i := by
  have hg : rate p t * (g i / Real.sqrt
      (nextSecond p s.history g i / correction p.adam.betaTwo t)) =
      adamLearningRate p.adam t * g i / Real.sqrt (nextSecond p s.history g i) := by
    rw [← mul_div_assoc, gradient_without_epsilon _ _ _ _ hv
      (correction_bounds _ _ hβ hβ1).1, paper_rate]
  change (s.base i - rate p t *
      (g i / (Real.sqrt (nextSecond p s.history g i / correction p.adam.betaTwo t) +
        p.adam.epsilon) + p.adam.decay * query p s i)) -
      (s.base i - adamLearningRate p.adam t * g i /
        (Real.sqrt (nextSecond p s.history g i) + p.adam.epsilon) -
        adamLearningRate p.adam t * p.adam.decay * query p s i) = _
  rw [hε, add_zero, add_zero, mul_add, hg, paper_rate]
  ring

example :
    let p : Parameters := ⟨⟨1, 1/2, 0, 9/10, 3/4, 0⟩, 0, 2⟩
    let s := initial (fun _ : Fin 1 => 1)
    (step p 0 s (fun _ => 2)).base 0 -
        (adamStep p.adam 0 (paperState s) (fun _ => 2)).base 0 =
      -(rate p 0 * (1 - Real.sqrt (correction p.adam.betaTwo 0))) *
        p.adam.decay * query p s 0 :=
  base_difference_without_epsilon _ _ _ _ rfl (by norm_num) (by norm_num) _
    (by norm_num [nextSecond, initial])

/-- The revised and manuscript algorithms really differ, even with positive epsilon,
valid beta coefficients, zero gradients, and the same initialized parameters. -/
theorem first_base_differs :
    let p : Parameters := ⟨⟨1, 1/2, 0, 9/10, 3/4, 1⟩, 0, 2⟩
    let s := initial (fun _ : Fin 1 => 1)
    (step p 0 s (fun _ => 0)).base 0 = 1/2 ∧
      (adamStep p.adam 0 (paperState s) (fun _ => 0)).base 0 = 3/4 := by
  norm_num [step, direction, query, rate, warmupFactor, denominator, nextSecond,
    correction, initial, adamStep, adamQuery, adamLearningRate, paperState, Real.sqrt_div]

/-- At constant nonnegative rate without warmup, the running maximum equals
that rate at every update, including the first replacement of the sentinel. -/
theorem constant_maximum (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hlr : 0 ≤ p.learningRate) (hw : p.warmup = 0) (t : ℕ) :
    nextMaximum ⟨p, 0, 2⟩ t (run ⟨p, 0, 2⟩ oracle point t).history = p.learningRate := by
  induction t with
  | zero =>
    have hsentinel : (-1 : ℝ) ≤ p.learningRate := by linarith
    simp [nextMaximum, rate, hw, warmupFactor, run, initial, max_eq_left hsentinel]
  | succ t ih =>
    change max (rate ⟨p, 0, 2⟩ (t + 1))
      (nextMaximum ⟨p, 0, 2⟩ t (run ⟨p, 0, 2⟩ oracle point t).history) = _
    rw [ih]
    simp [rate, hw, warmupFactor]

example (t : ℕ) : nextMaximum ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
    (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
      (fun _ y => y) (fun _ : Fin 1 => 2) t).history = 1 :=
  constant_maximum _ _ _ zero_le_one rfl t

/-- No warmup and the default averaging powers give a total of `t*lr^2`. -/
theorem constant_total (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hlr : 0 ≤ p.learningRate) (hw : p.warmup = 0) (t : ℕ) :
    (run ⟨p, 0, 2⟩ oracle point t).history.total = (t : ℝ) * p.learningRate ^ 2 := by
  induction t with
  | zero => simp [run, initial]
  | succ t ih =>
    change (run ⟨p, 0, 2⟩ oracle point t).history.total +
        weight ⟨p, 0, 2⟩ t (run ⟨p, 0, 2⟩ oracle point t).history = _
    rw [ih, default_weight, constant_maximum p oracle point hlr hw]
    push_cast
    ring

example (t : ℕ) : (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
    (fun _ y => y) (fun _ : Fin 1 => 2) t).history.total = (t : ℝ) * 1 ^ 2 :=
  constant_total _ _ _ zero_le_one rfl t

/-- Without warmup, revised default averaging is exactly uniform from step one.
The positive-rate hypothesis excludes the code's zero-weight fallback. -/
theorem constant_coefficient (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (hlr : 0 < p.learningRate) (hw : p.warmup = 0) (t : ℕ) :
    coefficient ⟨p, 0, 2⟩ t (run ⟨p, 0, 2⟩ oracle point t).history = 1 / ((t : ℝ) + 1) := by
  unfold coefficient
  rw [default_weight, constant_maximum p oracle point hlr.le hw,
    constant_total p oracle point hlr.le hw]
  have hsq := pow_ne_zero 2 hlr.ne'
  have ht : (t : ℝ) + 1 ≠ 0 := by positivity
  have htotal : (t : ℝ) * p.learningRate ^ 2 + p.learningRate ^ 2 ≠ 0 := by
    nlinarith [sq_nonneg p.learningRate, Nat.cast_nonneg (α := ℝ) t]
  field_simp

example (t : ℕ) : coefficient ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ t
    (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩
      (fun _ y => y) (fun _ : Fin 1 => 2) t).history = 1 / ((t : ℝ) + 1) :=
  constant_coefficient _ _ _ zero_lt_one rfl t

/-- Removing bias correction from the weights changes averaging too: this
second-update example compares the actual initialized revised and paper runs. -/
theorem second_coefficient_differs :
    let p : AdamParameters := ⟨1, 0, 0, 9/10, 3/4, 1⟩
    let point := fun _ : Fin 1 => (1 : ℝ)
    let oracle := fun (_ : ℕ) (_ : Fin 1 → ℝ) (_ : Fin 1) => (0 : ℝ)
    coefficient ⟨p, 0, 2⟩ 1 (run ⟨p, 0, 2⟩ oracle point 1).history = 1/2 ∧
      adamLearningRate p 1 ^ 2 /
        ((adamRun p oracle point 1).total + adamLearningRate p 1 ^ 2) = 7/11 := by
  norm_num [coefficient, weight, nextMaximum, rate, warmupFactor, run, step,
    nextHistory, initial, adamLearningRate, adamRun, adamStep, Real.sq_sqrt, mul_pow]
  norm_num [div_pow, Real.sq_sqrt]

end Transformer.ScheduleFree.AdamW

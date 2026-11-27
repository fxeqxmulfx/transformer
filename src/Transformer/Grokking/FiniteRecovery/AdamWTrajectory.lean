module

public import Transformer.AMSGradW.Kernite
public import Mathlib.Topology.Algebra.InfiniteSum.Real

/-! The ordinary AdamW kernel used for the Argand control in
`experiments/argand/src/engine/adamw.rs`, in real arithmetic. The two moment
decays are 0.9 and 0.95. The second moment is used directly. A quadratic
invariant bounds the normalized update independently of the epsilon floor. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

open Filter Transformer.AMSGradW.Kernite
open scoped Topology

variable {ι : Type*} [Fintype ι]

structure State (ι : Type*) where
  weights : ι → ℝ
  first : ι → ℝ
  second : ι → ℝ

noncomputable def scale (t : ℕ) (v : ℝ) : ℝ :=
  Real.sqrt v / Real.sqrt (1 - (95 / 100 : ℝ) ^ (t + 1)) + 1 / 100000000

noncomputable def direction (t : ℕ) (m v : ℝ) : ℝ := m / (firstBias t * scale t v)

/-- Zero moment initialization and one independently adapted update per real coordinate.
The gradient callback may use any minibatch or depend arbitrarily on the current weights. -/
noncomputable def run (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) :
    ℕ → State ι
  | 0 => ⟨initial, 0, 0⟩
  | t + 1 =>
    let s := run rate grad initial t
    let g := clip (grad t s.weights)
    let m i := (9 / 10 : ℝ) * s.first i + (1 / 10 : ℝ) * g i
    let v i := (95 / 100 : ℝ) * s.second i + (5 / 100 : ℝ) * (g i)^2
    ⟨fun i => (1 - rate t / 10) * s.weights i - rate t * direction t (m i) (v i), m, v⟩

variable (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)

/-- The same clipped gradient drives both moments. The slack obeys a
linear recurrence with a square as forcing, for every coordinate and time. -/
theorem moment_cone (i : ι) (t : ℕ) :
    let s := run rate grad initial t
    14 * (s.first i)^2 ≤ 19 * s.second i := by
  induction t with
  | zero => norm_num [run]
  | succ t ih =>
    let s := run rate grad initial t
    let g := clip (grad t s.weights) i
    change 14 * ((9/10 : ℝ) * s.first i + 1/10 * g)^2 ≤
      19 * ((95/100 : ℝ) * s.second i + 5/100 * g^2)
    change 14 * (s.first i)^2 ≤ 19 * s.second i at ih
    nlinarith [sq_nonneg ((14/10 : ℝ) * s.first i - (9/10 : ℝ) * g)]

/-- A common bound on the bias-corrected direction, independent of epsilon,
the batch size, dimension and gradient history. -/
theorem direction_bound (t : ℕ) {m v : ℝ} (h : 14 * m^2 ≤ 19 * v) :
    |direction t m v| ≤ 12 := by
  have hv : 0 ≤ v := by nlinarith [sq_nonneg m]
  have hs := Real.sq_sqrt hv
  have hx : |m| ≤ 6/5 * Real.sqrt v := by
    nlinarith [sq_abs m, abs_nonneg m, Real.sqrt_nonneg v,
      sq_nonneg (|m| - (6/5 : ℝ) * Real.sqrt v)]
  let c := Real.sqrt (1 - (95 / 100 : ℝ) ^ (t + 1))
  have hc : 0 < c := Real.sqrt_pos.2 (sub_pos.2
    (pow_lt_one₀ (by norm_num) (by norm_num) (Nat.succ_ne_zero t)))
  have hc1 : c ≤ 1 := Real.sqrt_le_one.2 (sub_le_self _ (by positivity))
  have hscale : Real.sqrt v ≤ scale t v := by
    unfold scale
    exact (le_div_self (Real.sqrt_nonneg v) hc hc1).trans (le_add_of_nonneg_right (by norm_num))
  have hpos : 0 < firstBias t * scale t v := by
    have hb := firstBias_lower t
    have hd : 0 < scale t v := by unfold scale; positivity
    exact mul_pos (by linarith) hd
  rw [direction, abs_div, abs_of_pos hpos, div_le_iff₀ hpos]
  nlinarith [mul_le_mul (firstBias_lower t) hscale (Real.sqrt_nonneg v)
    ((by norm_num : (0:ℝ) ≤ 1/10).trans (firstBias_lower t))]

/-- Decoupled decay makes this box forward invariant for all minibatches. -/
theorem weight_bound (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (i : ι) (t : ℕ) :
    |(run rate grad initial t).weights i| ≤ max ‖initial‖ 120 := by
  let B := max ‖initial‖ (120 : ℝ)
  induction t with
  | zero =>
    exact (norm_le_pi_norm initial i).trans (le_max_left _ _)
  | succ t ih =>
    let s := run rate grad initial
    let u := direction t ((s (t+1)).first i) ((s (t+1)).second i)
    have hu : |u| ≤ 12 := direction_bound t (moment_cone rate grad initial i (t+1))
    have hdec : 0 ≤ 1 - rate t / 10 := by linarith [(hr t).2]
    change |(1-rate t/10) * (s t).weights i - rate t * u| ≤ B
    calc
      _ ≤ |(1-rate t/10) * (s t).weights i| + |rate t*u| := abs_sub _ _
      _ = (1-rate t/10) * |(s t).weights i| + rate t * |u| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hdec, abs_of_nonneg (hr t).1]
      _ ≤ (1-rate t/10) * B + rate t * (B/10) :=
        add_le_add (mul_le_mul_of_nonneg_left ih hdec)
          (mul_le_mul_of_nonneg_left (by linarith [show (120 : ℝ) ≤ B from le_max_right _ _]) (hr t).1)
      _ = B := by ring

/-- A dimension-independent sup-norm step bound for the complete weight vector. -/
theorem step_dist (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (t : ℕ) :
    dist (run rate grad initial t).weights (run rate grad initial (t+1)).weights ≤
      (max ‖initial‖ 120 / 10 + 12) * rate t := by
  have hB : 0 ≤ max ‖initial‖ (120:ℝ) := (norm_nonneg _).trans (le_max_left _ _)
  apply (dist_pi_le_iff (mul_nonneg (by positivity) (hr t).1)).2
  intro i
  let s := run rate grad initial
  let u := direction t ((s (t+1)).first i) ((s (t+1)).second i)
  have he : (s t).weights i - (s (t+1)).weights i = ((s t).weights i/10 + u) * rate t := by
    change (s t).weights i - ((1-rate t/10)*(s t).weights i-rate t*u) = _
    ring
  change dist ((s t).weights i) ((s (t+1)).weights i) ≤ _
  rw [Real.dist_eq, he, abs_mul, abs_of_nonneg (hr t).1]
  apply mul_le_mul_of_nonneg_right _ (hr t).1
  refine (abs_add_le _ _).trans (add_le_add ?_ (direction_bound t (moment_cone rate grad initial i (t+1))))
  rw [abs_div, abs_of_pos (by norm_num : (0:ℝ) < 10)]
  exact div_le_div_of_nonneg_right (weight_bound rate grad initial hr i t) (by norm_num)

/-- A summable schedule gives an actual limit and a quantitative distance
from every finite checkpoint to that limit, for arbitrary gradient streams. -/
theorem limit_and_tail (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (hs : Summable rate) :
    ∃ w, Tendsto (fun t => (run rate grad initial t).weights) atTop (𝓝 w) ∧
      ∀ n, dist (run rate grad initial n).weights w ≤
        (max ‖initial‖ 120 / 10 + 12) * ∑' k, rate (n+k) := by
  let C := max ‖initial‖ 120 / 10 + 12
  have hd := step_dist rate grad initial hr
  obtain ⟨w, hw⟩ := cauchySeq_tendsto_of_complete
    (cauchySeq_of_dist_le_of_summable _ hd (hs.mul_left C))
  refine ⟨w, hw, fun n => ?_⟩
  simpa only [C, ← tsum_mul_left] using dist_le_tsum_of_dist_le_of_tendsto _ hd (hs.mul_left C) hw n

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (w : Fin 2 → ℝ) :=
  limit_and_tail (fun t => (1/2 : ℝ)^t) g w
    (fun t => ⟨by positivity, (pow_le_one₀ (by norm_num) (by norm_num)).trans (by norm_num)⟩)
    (summable_geometric_of_lt_one (by norm_num) (by norm_num))

end Transformer.Grokking.FiniteRecovery.AdamW

module

public import Transformer.Grokking.Geometry.ProbabilityTransfer

/-!
# Prospective softmax ceilings from contrast motion

Source: `review/flat_spectrum.md`, *Prospective probability control*, PM1–PM2.
All rivals remain in the denominator. Bounds concern the same input before
and after motion; a future minibatch needs this premise on every possible
training input. No future probability is assumed to be small.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.Geometry

open Finset
variable {C : Type*} [Fintype C] [DecidableEq C]

/-- PM1: lower bounds on rival logits give an upper bound on a class's
actual softmax probability. This includes the class's own denominator term. -/
theorem probability_le_of_lower_contrasts (z L : C → ℝ) (u : C)
    (h : ∀ c, c ≠ u → L c ≤ z c - z u) :
    probability z u ≤ 1 / (1 + oddsEnvelope L u) := by
  rw [probability_eq_one_div_odds]
  apply one_div_le_one_div_of_le (by have := oddsEnvelope_nonneg L u; linarith)
  have ho : oddsEnvelope L u ≤ oddsEnvelope (fun c => z c-z u) u := by
    apply sum_le_sum
    intro c hc
    exact Real.exp_le_exp.mpr (h c (mem_erase.mp hc).1)
  linarith

/-- PM1: a separate deterioration budget for each rival transports a
checkpoint's odds to an unknown later state. Common logit shifts cancel. -/
theorem probability_le_motion_envelope (z z' D : C → ℝ) (u : C)
    (h : ∀ c, c ≠ u → (z' u-z' c)-(z u-z c) ≤ D c) :
    probability z' u ≤ 1 / (1 + oddsEnvelope (fun c => z c-z u-D c) u) := by
  apply probability_le_of_lower_contrasts
  intro c hc
  linarith [h c hc]

/-- PM1: the sharp scalar ceiling for a uniform contrast-motion budget. -/
def probabilityCeiling (q D : ℝ) : ℝ := q*Real.exp D/(1-q+q*Real.exp D)

/-- PM1: the ceiling denominator is positive throughout the probability
simplex, including q=0 and q=1 and either sign of the motion budget. -/
theorem probabilityCeiling_den_pos (q D : ℝ) (hq : 0 ≤ q) (hq1 : q ≤ 1) :
    0 < 1-q+q*Real.exp D := by
  have he := Real.exp_pos D
  by_cases h : q = 1
  · subst q
    simpa using he
  · have hlt : q < 1 := lt_of_le_of_ne hq1 h
    have hp := mul_nonneg hq he.le
    linarith

/-- PM1: increasing the checkpoint probability bound increases the
transported ceiling. No sign condition on D is required. -/
theorem probabilityCeiling_mono (p q D : ℝ)
    (hp : 0 ≤ p) (hpq : p ≤ q) (hq1 : q ≤ 1) :
    probabilityCeiling p D ≤ probabilityCeiling q D := by
  have hq : 0 ≤ q := hp.trans hpq
  have hp1 : p ≤ 1 := hpq.trans hq1
  unfold probabilityCeiling
  rw [div_le_div_iff₀ (probabilityCeiling_den_pos p D hp hp1)
    (probabilityCeiling_den_pos q D hq hq1)]
  have he := mul_le_mul_of_nonneg_right hpq (Real.exp_pos D).le
  nlinarith

/-- PM1: a propagated probability ceiling stays in the probability simplex. -/
theorem probabilityCeiling_bounds (q D : ℝ) (hq : 0 ≤ q) (hq1 : q ≤ 1) :
    0 ≤ probabilityCeiling q D ∧ probabilityCeiling q D ≤ 1 := by
  have hd := probabilityCeiling_den_pos q D hq hq1
  constructor
  · exact div_nonneg (mul_nonneg hq (Real.exp_pos D).le) hd.le
  · unfold probabilityCeiling
    apply (div_le_one hd).mpr
    linarith

/-- PM1: all rival contrast changes bounded by D yield the sharp
probability ceiling from the old probability, rather than a future q. -/
theorem probability_le_uniform_motion (z z' : C → ℝ) (u : C) (D : ℝ)
    (h : ∀ c, c ≠ u → (z' u-z' c)-(z u-z c) ≤ D) :
    probability z' u ≤ probabilityCeiling (probability z u) D := by
  have hb := probability_le_motion_envelope z z' (fun _ => D) u h
  have ho : oddsEnvelope (fun c => z c-z u-D) u =
      oddsEnvelope (fun c => z c-z u) u / Real.exp D := by
    simp only [oddsEnvelope, Real.exp_sub, sum_div]
  rw [ho] at hb
  have hO := oddsEnvelope_nonneg (fun c => z c-z u) u
  have he := Real.exp_pos D
  have hid : 1 / (1 + oddsEnvelope (fun c => z c-z u) u / Real.exp D) =
      probabilityCeiling (probability z u) D := by
    unfold probabilityCeiling
    rw [probability_eq_one_div_odds]
    generalize oddsEnvelope (fun c => z c-z u) u = O at hO ⊢
    have h1 : 1+O ≠ 0 := ne_of_gt (by linarith : 0 < 1+O)
    have h2 : Real.exp D+O ≠ 0 := ne_of_gt (by linarith : 0 < Real.exp D+O)
    field_simp [h1, h2]
    have hs : 1+O-1+Real.exp D = Real.exp D+O := by ring
    rw [hs, div_self h2]
  rwa [hid] at hb

/-- PM1: a checkpoint ceiling q applies uniformly after motion. -/
theorem probability_le_checkpoint_ceiling (z z' : C → ℝ) (u : C) (q D : ℝ)
    (hq : probability z u ≤ q) (hq1 : q ≤ 1)
    (h : ∀ c, c ≠ u → (z' u-z' c)-(z u-z c) ≤ D) :
    probability z' u ≤ probabilityCeiling q D := by
  exact (probability_le_uniform_motion z z' u D h).trans
    (probabilityCeiling_mono _ q D (probability_pos z u).le hq hq1)

omit [Fintype C] [DecidableEq C] in
/-- PM2: individual contrast increments telescope over every finite prefix.
The budget may be signed; no monotonicity of the actual logits is assumed. -/
theorem contrast_motion_prefix (z : ℕ → C → ℝ) (u c : C) (D : ℕ → ℝ)
    (h : ∀ t, (z (t+1) u-z (t+1) c)-(z t u-z t c) ≤ D t) (n : ℕ) :
    (z n u-z n c)-(z 0 u-z 0 c) ≤ ∑ t ∈ range n, D t := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ]
    linarith [h n]

/-- PM2: cumulative contrast control supplies the future probability premise
at any horizon using only an initial ceiling and declared motion budgets. -/
theorem probability_le_cumulative_motion (z : ℕ → C → ℝ) (u : C) (q : ℝ)
    (D : ℕ → ℝ) (hq : probability (z 0) u ≤ q) (hq1 : q ≤ 1)
    (h : ∀ t c, c ≠ u → (z (t+1) u-z (t+1) c)-(z t u-z t c) ≤ D t) (n : ℕ) :
    probability (z n) u ≤ probabilityCeiling q (∑ t ∈ range n, D t) := by
  apply probability_le_checkpoint_ceiling (z 0) (z n) u q _ hq hq1
  intro c hc
  exact contrast_motion_prefix z u c D (fun t => h t c hc) n

/-- PM1: nonzero motion attains the ceiling. The old probability is 1/5,
the new probability is 1/3, and the contrast deteriorates by log 2. -/
theorem probability_motion_sharp_witness :
    let z : Bool → ℝ := fun c => if c then 0 else Real.log 4
    let z' : Bool → ℝ := fun c => if c then Real.log 2 else Real.log 4
    probability z true = 1/5 ∧ probability z' true = 1/3 ∧
      probabilityCeiling (1/5) (Real.log 2) = 1/3 ∧
      (z' true-z' false)-(z true-z false) = Real.log 2 := by
  norm_num [probability, probabilityCeiling, Real.exp_log]

example := probability_le_of_lower_contrasts (fun c : Bool => if c then 1 else 0)
  (fun c => if c then 0 else -1) true (by intro c hc; cases c <;> simp_all)
example := probability_le_motion_envelope (fun _ : Bool => (0 : ℝ))
  (fun c => if c then 1 else 0) (fun _ => 1) true
  (by intro c hc; cases c <;> simp_all)
example := probabilityCeiling_den_pos (1/3) (-1) (by norm_num) (by norm_num)
example := probabilityCeiling_mono (1/3) (1/2) 2 (by norm_num) (by norm_num) (by norm_num)
example := probabilityCeiling_bounds (1/3) 2 (by norm_num) (by norm_num)
example := probability_le_uniform_motion (fun _ : Bool => (0 : ℝ))
  (fun c => if c then 1 else 0) true 1 (by intro c hc; cases c <;> simp_all)
example := probability_le_checkpoint_ceiling (fun _ : Bool => (0 : ℝ))
  (fun c => if c then 1 else 0) true (1/2) 1
  (by norm_num [probability]) (by norm_num) (by intro c hc; cases c <;> simp_all)
example (n : ℕ) := contrast_motion_prefix
  (fun t (c : Bool) => if c then (t : ℝ) else 0) true false (fun _ => 1)
  (by intro t; simp) n
example (n : ℕ) := probability_le_cumulative_motion
  (fun t (c : Bool) => if c then (t : ℝ) else 0) true (1/2) (fun _ => 1)
  (by norm_num [probability]) (by norm_num)
  (by intro t c hc; cases c <;> simp_all) n

end Transformer.Grokking.Geometry

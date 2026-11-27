module

public import Transformer.Optimization.PAGE.Descent
public import Transformer.Optimization.StrongConvexity
public import Mathlib.Topology.Algebra.InfiniteSum.Real
public import Mathlib.Analysis.SpecificLimits.Normed

/-! Finite-sum PAGE (Li et al., 2021, Theorems 1 and 5). We prove an
explicit gradient budget and last-iterate mean-square stationarity. Under
gradient domination the expected objective gap decays geometrically.
Expectations are over the actual refresh/correction decisions in Algorithm 1. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Finset Filter
open scoped Topology

variable {I E : Type*} [Fintype I] [Nonempty I]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  (loss : I → E → ℝ) {η p L V lower : ℝ}

private theorem expected_nonneg (hp : 0 ≤ p) (hp' : p ≤ 1) (t : ℕ) (s : State E)
    (h : State E → ℝ) (hh : ∀ z, 0 ≤ h z) : 0 ≤ expected loss η p t h s := by
  simpa only [expected_const] using expected_mono loss η hp hp' (u := fun _ => 0) hh t s

omit [Nonempty I] in
private theorem energy_nonneg (hη : 0 < η) (hp : 0 < p) (hb : ∀ x, lower ≤ objective loss x)
    (s : State E) : 0 ≤ energy loss η p lower s :=
  add_nonneg (sub_nonneg.mpr (hb _)) (mul_nonneg (div_pos hη hp).le (sq_nonneg _))

omit [Nonempty I] in
private theorem energy_start (x : E) :
    energy loss η p lower (start loss x) = objective loss x - lower := by
  simp [energy, error, start]

/-- Sum over the actual first N iterates, for any initialization.
Dividing by N gives the finite-time squared-stationarity rate. -/
theorem gradient_budget (hd : ∀ i, Differentiable ℝ (loss i))
    (hη : 0 < η) (hp : 0 < p) (hp' : p ≤ 1) (hl : SmoothObjective (objective loss) L)
    (hv : AverageSmooth loss V) (hs : StableStep η p L V) (hb : ∀ x, lower ≤ objective loss x)
    (x : E) (N : ℕ) :
    η/2 * ∑ t ∈ range N, expected loss η p t (fun s => ‖gradient (objective loss) s.weights‖^2)
      (start loss x) ≤ objective loss x - lower := by
  let H := energy loss η p lower
  let G := fun s : State E => ‖gradient (objective loss) s.weights‖^2
  let s₀ := start loss x
  have one (s : State E) : next loss η p H s ≤ H s - η/2 * G s := by
    have h := next_energy loss hd hη hp hp' hl hv hs lower s
    have hn : 0 ≤ η/2 * error loss s := mul_nonneg (by positivity) (sq_nonneg _)
    dsimp only [H, G]
    rw [gradient_objective loss hd]
    linarith
  have budget (n : ℕ) : η/2 * ∑ t ∈ range n, expected loss η p t G s₀ +
      expected loss η p n H s₀ ≤ objective loss x - lower := by
    induction n with
    | zero => simp [expected, s₀, H, energy_start]
    | succ n ih =>
      have h := expected_mono loss η hp.le hp' one n s₀
      rw [expected_sub_mul] at h
      rw [sum_range_succ]
      change _ + expected loss η p n (next loss η p H) s₀ ≤ _
      linarith
  exact (le_add_of_nonneg_right (expected_nonneg loss hp.le hp' N s₀ H
    (energy_nonneg loss hη hp hb))).trans (budget N)

/-- The expected squared gradient of the last iterate tends to zero. -/
theorem gradient_tendsto_zero (hd : ∀ i, Differentiable ℝ (loss i))
    (hη : 0 < η) (hp : 0 < p) (hp' : p ≤ 1) (hl : SmoothObjective (objective loss) L)
    (hv : AverageSmooth loss V) (hs : StableStep η p L V) (hb : ∀ x, lower ≤ objective loss x)
    (x : E) : Tendsto (fun t => expected loss η p t (fun s => ‖gradient (objective loss) s.weights‖^2)
      (start loss x)) atTop (𝓝 0) := by
  apply Summable.tendsto_atTop_zero
  apply summable_of_sum_range_le (fun n => expected_nonneg loss hp.le hp' n _ _ fun _ => sq_nonneg _)
    (c := (objective loss x - lower)/(η/2))
  intro N
  apply (le_div_iff₀ (by positivity : 0 < η/2)).2
  simpa only [mul_comm (η/2)] using gradient_budget loss hd hη hp hp' hl hv hs hb x N

/-- Polyak–Łojasiewicz gradient domination at a specified objective floor. -/
def Dominated (f : E → ℝ) (μ floor : ℝ) : Prop :=
  ∀ x, 2*μ*(f x - floor) ≤ ‖gradient f x‖^2

theorem expected_scale (c : ℝ) (t : ℕ) (v : State E → ℝ) (s : State E) :
    expected loss η p t (fun z => c * v z) s = c * expected loss η p t v s := by
  simpa only [neg_mul, zero_sub, neg_neg, expected_const] using
    expected_sub_mul loss η p (-c) t (fun _ => 0) v s

/-- Last-iterate geometric rate under PL. The factor is 1 - min(ημ,p/2). -/
theorem pl_rate (hd : ∀ i, Differentiable ℝ (loss i)) (hη : 0 < η)
    (hp : 0 < p) (hp' : p ≤ 1) (hl : SmoothObjective (objective loss) L)
    (hv : AverageSmooth loss V) (hs : StableStep η p L V) (hb : ∀ x, lower ≤ objective loss x)
    {μ : ℝ} (hμ : 0 < μ) (hpl : Dominated (objective loss) μ lower) (x : E) (t : ℕ) :
    expected loss η p t (fun s => objective loss s.weights - lower) (start loss x) ≤
      (1 - min (η*μ) (p/2))^t * (objective loss x - lower) := by
  let r := min (η*μ) (p/2)
  let H := energy loss η p lower
  have hr : 0 < r := lt_min (mul_pos hη hμ) (by positivity)
  have hsmall : r ≤ p/2 := min_le_right _ _
  have hq : 0 ≤ 1-r := by linarith
  have one (s : State E) : next loss η p H s ≤ (1-r)*H s := by
    have h := next_energy loss hd hη hp hp' hl hv hs lower s
    have hg := hpl s.weights
    rw [gradient_objective loss hd] at hg
    have h1 := mul_le_mul_of_nonneg_left hg (by positivity : 0 ≤ η/2)
    have h2 := mul_le_mul_of_nonneg_right (min_le_left (η*μ) (p/2)) (sub_nonneg.mpr (hb s.weights))
    have h3 : r*(η/p) ≤ η/2 := by
      calc
        _ ≤ p/2 * (η/p) := mul_le_mul_of_nonneg_right hsmall (div_pos hη hp).le
        _ = _ := by field_simp
    have h4 := mul_le_mul_of_nonneg_right h3 (show 0 ≤ error loss s from sq_nonneg _)
    refine h.trans ?_
    dsimp only [H, energy]
    change r * _ ≤ _ at h2
    linarith
  have bound (n : ℕ) : expected loss η p n H (start loss x) ≤
      (1-r)^n * (objective loss x - lower) := by
    induction n with
    | zero => simp [H, expected, energy_start]
    | succ n ih =>
      have h := expected_mono loss η hp.le hp' one n (start loss x)
      rw [expected_scale] at h
      exact h.trans ((mul_le_mul_of_nonneg_left ih hq).trans_eq (by rw [pow_succ]; ring))
  refine (expected_mono loss η hp.le hp' (v := H) (fun s => ?_) t _).trans (bound t)
  exact le_add_of_nonneg_right (mul_nonneg (div_pos hη hp).le (sq_nonneg _))

/-- Mean objective gap tends to zero at the stated floor. -/
theorem pl_loss_tendsto (hd : ∀ i, Differentiable ℝ (loss i))
    (hη : 0 < η) (hp : 0 < p) (hp' : p ≤ 1) (hl : SmoothObjective (objective loss) L)
    (hv : AverageSmooth loss V) (hs : StableStep η p L V) (hb : ∀ x, lower ≤ objective loss x)
    {μ : ℝ} (hμ : 0 < μ) (hpl : Dominated (objective loss) μ lower) (x : E) :
    Tendsto (fun t => expected loss η p t (fun s => objective loss s.weights - lower)
      (start loss x)) atTop (𝓝 0) := by
  have hr : 0 < min (η*μ) (p/2) := lt_min (mul_pos hη hμ) (by positivity)
  have hs' := min_le_right (η*μ) (p/2)
  have hq : |1-min (η*μ) (p/2)| < 1 := abs_lt.mpr ⟨by linarith, by linarith⟩
  apply squeeze_zero (fun n => expected_nonneg loss hp.le hp' n _ _ fun s => sub_nonneg.mpr (hb _))
    (pl_rate loss hd hη hp hp' hl hv hs hb hμ hpl x)
  simpa using (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hq).mul_const (objective loss x - lower)

/-- Strong convexity turns the preceding loss result into mean-square
convergence of the sampled weight vectors to the stationary minimizer. -/
theorem strongly_convex_weights_tendsto (hd : ∀ i, Differentiable ℝ (loss i))
    (hη : 0 < η) (hp : 0 < p) (hp' : p ≤ 1) (hl : SmoothObjective (objective loss) L)
    (hv : AverageSmooth loss V) (hs : StableStep η p L V) {μ : ℝ} (hμ : 0 < μ)
    (sc : StrongLowerModel (objective loss) μ) (star : E) (hstar : gradient (objective loss) star = 0)
    (x : E) : Tendsto (fun t => expected loss η p t (fun s => ‖s.weights-star‖^2) (start loss x))
      atTop (𝓝 0) := by
  let f := objective loss
  have hgap := fun y => strong_gap_bounds f μ star y sc hμ hstar
  have hb (y : E) : f star ≤ f y := by
    have h := (hgap y).1
    have h0 : 0 ≤ μ/2 * ‖y-star‖^2 := by positivity
    linarith
  have hdom : Dominated f μ (f star) := fun y => (hgap y).2
  have hconv := pl_loss_tendsto loss hd hη hp hp' hl hv hs hb hμ hdom x
  have hbound (s : State E) : ‖s.weights-star‖^2 ≤ (2/μ)*(f s.weights - f star) := by
    have h := mul_le_mul_of_nonneg_left (hgap s.weights).1 (by positivity : 0 ≤ 2/μ)
    rwa [← mul_assoc, show 2/μ*(μ/2) = 1 by field_simp, one_mul] at h
  apply squeeze_zero (fun t => expected_nonneg loss hp.le hp' t _ _ (fun _ => sq_nonneg _))
    (fun t => (expected_mono loss η hp.le hp' hbound t _).trans_eq (expected_scale loss _ _ _ _))
  simpa only [mul_zero] using hconv.const_mul (2/μ)

end Transformer.Optimization.PAGE

import Transformer.Clusters.Section12_ReluFlow
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# A genuine ReLU scalar trajectory with a separated limit

For the test of §12, `eq: mlp.transfo` in arXiv:2305.05465v6,
start the moving coordinate at minus eight. The auxiliary clock
construction has constant scale one and clock equal to real time.
Its moving derivative lies between zero and `exp(-2t)`.
Consequently `y(t) + exp(-2t)/2` is nonincreasing. At nonnegative
times this bounds `y(t)` above by minus seven; at negative times,
monotonicity bounds it above by its initial value minus eight.

Thus the moving coordinate always lies below minus six, the
explicit existence truncation is inactive, and the trajectory
solves the original scalar attention drift. It is nondecreasing
and bounded above, so it has a limit `q ≤ -7` as time tends to
infinity; monotonicity also gives the lower limit bound `-8 ≤ q`.
Together with the fixed coordinates `-1`, `0`, and `1`,
this yields four separated limiting tokens. The following module
will verify the full vector feed-forward equation and its good
triple hypotheses before drawing the hyperplane conclusion.
-/

open Real Filter Topology
namespace Transformer.Clusters

/-- The auxiliary scale remains one and its clock equals real time.
Source: arXiv:2305.05465v6, §12, the global existence
construction for the four-token test of `eq: mlp.transfo`. -/
theorem reluClock_solution_time_scale (γ : ℝ → ReluClockState)
    (h0 : γ 0 = (0, (-8, 1)))
    (hγ : ∀ t : ℝ, HasDerivAt γ (reluClockField (γ t)) t) :
    (∀ t : ℝ, (γ t).2.2 = 1) ∧ (∀ t : ℝ, (γ t).1 = t) := by
  let Lw := (ContinuousLinearMap.snd ℝ ℝ ℝ).comp
    (ContinuousLinearMap.snd ℝ ℝ (ℝ × ℝ))
  have hwD (t : ℝ) : HasDerivAt (fun s => (γ s).2.2) 0 t :=
    Lw.hasFDerivAt.comp_hasDerivAt t (hγ t)
  have hw (t : ℝ) : (γ t).2.2 = 1 := by
    have hconst := is_const_of_deriv_eq_zero (fun s => (hwD s).differentiableAt)
      (fun s => (hwD s).deriv) t 0
    simpa only [h0] using hconst
  have htD (t : ℝ) : HasDerivAt (fun s => (γ s).1) 1 t := by
    have h := (ContinuousLinearMap.fst ℝ ℝ (ℝ × ℝ)).hasFDerivAt.comp_hasDerivAt t (hγ t)
    change HasDerivAt (fun s => (γ s).1) (γ t).2.2 t at h
    simpa only [hw t] using h
  have hgD (t : ℝ) : HasDerivAt (fun s => (γ s).1 - s) 0 t := by
    simpa only [sub_self] using (htD t).fun_sub (hasDerivAt_id' (x := t))
  refine ⟨hw, fun t => ?_⟩
  have hconst := is_const_of_deriv_eq_zero (fun s => (hgD s).differentiableAt)
    (fun s => (hgD s).deriv) t 0
  simp only [h0, sub_zero] at hconst
  linarith

/-- The clock and scale hypotheses hold for a proved global solution. -/
example : ∃ γ : ℝ → ReluClockState, γ 0 = (0, (-8, 1)) ∧
    ∀ t : ℝ, HasDerivAt γ (reluClockField (γ t)) t := exists_reluClock_solution

/-- A genuine moving trajectory exists, and the existence truncation
is inactive at every real time. Source: arXiv:2305.05465v6, §12,
`eq: mlp.transfo`, at the four-token configuration `(y, -1, 0, 1)`. -/
theorem exists_reluFour_scalar :
    ∃ y : ℝ → ℝ, y 0 = -8 ∧ (∀ t : ℝ, y t ≤ -7) ∧
      ∀ t : ℝ, HasDerivAt y (reluFourDrift (Real.exp (2 * t)) (y t) 1 0) t := by
  obtain ⟨γ, h0, hγ⟩ := exists_reluClock_solution
  obtain ⟨hw, ht⟩ := reluClock_solution_time_scale γ h0 hγ
  let y := fun t : ℝ => (γ t).2.1
  have hy0 : y 0 = -8 := by simp only [y, h0]
  let F := fun t => max (reluFourDrift (Real.exp (2 * t)) (min (y t) (-6)) 1 0) 0
  let Ly := (ContinuousLinearMap.fst ℝ ℝ ℝ).comp
    (ContinuousLinearMap.snd ℝ ℝ (ℝ × ℝ))
  have hD (t : ℝ) : HasDerivAt y (F t) t := by
    have h := Ly.hasFDerivAt.comp_hasDerivAt t (hγ t)
    change HasDerivAt y (max (reluFourDrift (Real.exp (2 * (γ t).1))
      (min (y t) (-6 * (γ t).2.2)) (γ t).2.2 0) 0) t at h
    simpa only [hw t, ht t, mul_one] using h
  have hbound (t : ℝ) : F t ≤ Real.exp (-(2 * t)) :=
    max_le (reluFourDrift_zero_le_exp t (min_le_right _ _)) (Real.exp_pos _).le
  have hmono : Monotone y := monotone_of_hasDerivAt_nonneg hD (fun t => le_max_right _ _)
  let g := fun t => y t + Real.exp (-(2 * t)) / 2
  have hgD (t : ℝ) : HasDerivAt g (F t - Real.exp (-(2 * t))) t := by
    have he : HasDerivAt (fun s : ℝ => Real.exp (-(2 * s)) / 2)
        (-Real.exp (-(2 * t))) t := by
      convert (((hasDerivAt_id' (x := t)).const_mul 2).neg.exp.div_const 2) using 1
      simp only [Pi.neg_apply]
      ring
    simpa only [sub_eq_add_neg] using (hD t).fun_add he
  have hanti : Antitone g := antitone_of_hasDerivAt_nonpos hgD
    (fun t => sub_nonpos.mpr (hbound t))
  have hy (t : ℝ) : y t ≤ -7 := by
    rcases le_total 0 t with hpos | hneg
    · have h := hanti hpos
      dsimp [g] at h
      simp only [hy0, mul_zero, neg_zero, Real.exp_zero] at h
      linarith [(Real.exp_pos (-(2 * t))).le]
    · have h := hmono hneg
      rw [hy0] at h
      linarith
  refine ⟨y, hy0, hy, fun t => ?_⟩
  have hcut : min (y t) (-6) = y t := min_eq_left (by linarith [hy t])
  have hnonneg := reluFourDrift_zero_nonneg (Real.exp (2 * t)) (by linarith [hy t])
  have h := hD t
  dsimp only [F] at h
  rw [hcut, max_eq_left hnonneg] at h
  exact h

/-- The genuine moving trajectory is monotone and converges to a limit
separated from the three frozen coordinates. Source: arXiv:2305.05465v6,
§12, `eq: mlp.transfo`; this proves the limit in its four-token test. -/
theorem exists_reluFour_scalar_limit :
    ∃ (y : ℝ → ℝ) (q : ℝ), y 0 = -8 ∧ (∀ t : ℝ, y t ≤ -7) ∧
      (∀ t : ℝ, HasDerivAt y (reluFourDrift (Real.exp (2 * t)) (y t) 1 0) t) ∧
        -8 ≤ q ∧ q ≤ -7 ∧ Tendsto y atTop (nhds q) := by
  obtain ⟨y, h0, hy, hD⟩ := exists_reluFour_scalar
  have hmono : Monotone y := monotone_of_hasDerivAt_nonneg hD
    (fun t => reluFourDrift_zero_nonneg _ (by linarith [hy t]))
  have hbounded : BddAbove (Set.range y) := by
    refine ⟨-7, ?_⟩
    rintro r ⟨t, rfl⟩
    exact hy t
  have hlower : -8 ≤ ⨆ t, y t := by
    rw [← h0]
    exact le_ciSup hbounded 0
  exact ⟨y, ⨆ t, y t, h0, hy, hD, hlower, ciSup_le hy,
    tendsto_atTop_ciSup hmono hbounded⟩

/-- The limiting scalar coordinates are pairwise distinct.
Source: arXiv:2305.05465v6, §12, the four-token test of its
proposed three-hyperplane extension; the moving limit stays below minus seven. -/
theorem reluFourCoords_limit_injective {q : ℝ} (hq : q ≤ -7) :
    Function.Injective (reluFourCoords q 1) := by
  intro i j hij
  fin_cases i <;> fin_cases j <;> simp [reluFourCoords] at hij ⊢ <;> linarith

example : (-8 : ℝ) ≤ -7 := by norm_num

/-- The constructed trajectory's initial coordinate is genuinely distinct
from the three reference coordinates. Source: arXiv:2305.05465v6,
§12, the four-token test of its proposed three-hyperplane extension. -/
theorem reluFourCoords_initial_injective : Function.Injective (reluFourCoords (-8) 1) := by
  exact reluFourCoords_limit_injective (by norm_num)

/-- The auxiliary initial values agree with the four scalar coordinates.
Source: arXiv:2305.05465v6, §12, the concrete test configuration. -/
theorem reluFourCoords_initial :
    reluFourCoords (-8) 1 = (![-8, -1, 0, 1] : Idx 4 → ℝ) := rfl

end Transformer.Clusters

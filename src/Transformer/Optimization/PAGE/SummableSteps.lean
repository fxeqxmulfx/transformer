module

public import Transformer.Optimization.PAGE.FiniteSum
public import Mathlib.Topology.EMetricSpace.Lipschitz
public import Mathlib.Topology.Algebra.InfiniteSum.Real
public import Mathlib.Analysis.Normed.Group.Bounded

/-! A pathwise result for PAGE with a summable learning-rate schedule. On a
ball the correction to the estimate is bounded by the component Lipschitz
constant times the distance travelled. A small total rate then keeps the
entire run in an open set and gives a finite limit of the actual weights. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Filter Finset Metric Set
open scoped Topology

variable {I E : Type*} [Fintype I] [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- PAGE with the learning rate supplied on each step and the ordinary initial
full gradient. All other state updates are those of Algorithm 1. -/
noncomputable def scheduledRun (loss : I → E → ℝ) (rate : ℕ → ℝ) (x : E)
    (d : ℕ → Option I) : ℕ → State E
  | 0 => start loss x
  | t+1 => step loss (rate t) (scheduledRun loss rate x d t) (d t)

theorem scheduledRun_const (loss : I → E → ℝ) (η : ℝ) (x : E) (d : ℕ → Option I) (t : ℕ) :
    scheduledRun loss (fun _ => η) x d t = run loss η x d t := by
  induction t with
  | zero => rfl
  | succ t ih => rw [scheduledRun, run, ih]

/-- Uniform over starts in a smaller ball and every sequence of sampled
components. The actual limit and every trained weight remain in G. -/
theorem stays_open_summable [ProperSpace E] (loss : I → E → ℝ)
    (hl : ∀ i, LocallyLipschitz (gradient (loss i))) {G : Set E} (hG : IsOpen G)
    {c : E} (hc : c ∈ G) :
    ∃ r > 0, ∃ δ > 0, ∀ y ∈ ball c r, ∀ rate : ℕ → ℝ,
      Summable (fun t => |rate t|) → (∑' t, |rate t|) ≤ δ → ∀ path : ℕ → Option I,
      let w := fun t => (scheduledRun loss rate y path t).weights
      (∀ t, w t ∈ G) ∧ ∃ star ∈ G, Tendsto w atTop (𝓝 star) := by
  obtain ⟨R,hR,hball⟩ := Metric.nhds_basis_closedBall.mem_iff.mp (hG.mem_nhds hc)
  let B := closedBall c R
  have hcompact : IsCompact B := isCompact_closedBall c R
  choose l lip using fun i => (hl i).locallyLipschitzOn.exists_lipschitzOnWith_of_compact hcompact
  let L : ℝ := ∑ i, (l i : ℝ)
  have hL : 0 ≤ L := sum_nonneg fun _ _ => by positivity
  have hfull : Continuous (full loss) :=
    (continuous_finsetSum _ (fun i _ => (hl i).continuous)).const_smul _
  obtain ⟨M,hm⟩ := hcompact.exists_bound_of_continuousOn hfull.continuousOn
  have hM : 0 ≤ M := (norm_nonneg _).trans (hm c (mem_closedBall_self hR.le))
  let C := M + L*R + 1
  have hC : 0 < C := by dsimp [C]; positivity
  refine ⟨R/2,half_pos hR,R/(2*C),by positivity,?_⟩
  intro y hy rate hs hb path
  let fit := scheduledRun loss rate y path
  let w (t : ℕ) := (fit t).weights
  let g (t : ℕ) := (fit t).estimate
  let travel (t : ℕ) := ∑ j ∈ range t, dist (w j) (w (j+1))
  let budget (t : ℕ) := ∑ j ∈ range t, |rate j|
  have hcap (t : ℕ) : C * budget t ≤ R/2 := by
    have ht := (hs.sum_le_tsum (range t) (fun i _ => abs_nonneg (rate i))).trans hb
    have h := (le_div_iff₀ (by positivity : 0 < 2*C)).mp ht
    dsimp only [budget]
    linarith only [h]
  have inside (t : ℕ) (h : travel t ≤ C * budget t) : w t ∈ B := by
    have hd : dist (w 0) (w t) ≤ travel t := dist_le_range_sum_dist w t
    have hy' : dist (w 0) c < R / 2 := hy
    have tri := dist_triangle_left (w t) c (w 0)
    exact (by linarith [hcap t] : dist (w t) c ≤ R)
  have diff (t : ℕ) : dist (w t) (w (t+1)) = |rate t| * ‖g t‖ := by
    change dist (w t) (w t - rate t • g t) = _
    rw [dist_eq_norm, sub_sub_cancel, norm_smul, Real.norm_eq_abs]
  have bounds (t : ℕ) : travel t ≤ C*budget t ∧ ‖g t‖ ≤ M + L*travel t := by
    induction t with
    | zero =>
      have hyB : y ∈ B := (mem_ball.mp hy).le.trans (by linarith)
      simpa [travel, budget, g, fit, scheduledRun, start] using hm y hyB
    | succ t ih =>
      have hyB := inside t ih.1
      have hg : ‖g t‖ ≤ C := by
        have hh := mul_le_mul_of_nonneg_left (ih.1.trans (hcap t)) hL
        dsimp [C]; linarith [ih.2, mul_nonneg hL hR.le]
      have hd := mul_le_mul_of_nonneg_left hg (abs_nonneg (rate t))
      have hstep : travel (t+1) ≤ C*budget (t+1) := by
        simp only [travel, budget, sum_range_succ, diff] at ih ⊢
        linarith [ih.1]
      refine ⟨hstep,?_⟩
      have hn := inside (t+1) hstep
      cases h : path t with
      | none =>
        have he : g (t+1) = full loss (w (t+1)) := by simp [g,w,fit,scheduledRun,step,h]
        rw [he]
        exact (hm _ hn).trans (le_add_of_nonneg_right (mul_nonneg hL (sum_nonneg (fun _ _ => dist_nonneg))))
      | some i =>
        have he : g (t+1) = g t + (gradient (loss i) (w (t+1)) - gradient (loss i) (w t)) := by
          simp [g,w,fit,scheduledRun,step,h]
        have hi : (l i : ℝ) ≤ L := single_le_sum (f := fun j => (l j : ℝ)) (fun _ _ => by positivity) (mem_univ i)
        have hd := (lip i).dist_le_mul (x := w (t+1)) hn (y := w t) hyB
        rw [dist_eq_norm, dist_comm (w (t+1)) (w t)] at hd
        have hinc := hd.trans (mul_le_mul_of_nonneg_right hi dist_nonneg)
        rw [he]
        refine (norm_add_le _ _).trans ?_
        change _ ≤ M + L * (∑ j ∈ range (t+1), _)
        rw [sum_range_succ, mul_add]
        linarith only [ih.2,hinc]
  have increments (t : ℕ) : dist (w t) (w (t+1)) ≤ |rate t| * C := by
    rw [diff]
    apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
    have hh := mul_le_mul_of_nonneg_left ((bounds t).1.trans (hcap t)) hL
    dsimp [C]; linarith [(bounds t).2, mul_nonneg hL hR.le]
  obtain ⟨star,hstar⟩ := cauchySeq_tendsto_of_complete
    (cauchySeq_of_dist_le_of_summable _ increments (hs.mul_right C))
  have hall (t : ℕ) := inside t (bounds t).1
  exact ⟨fun t => hball (hall t),star,hball (isClosed_closedBall.mem_of_tendsto hstar
    (Eventually.of_forall hall)),hstar⟩

end Transformer.Optimization.PAGE

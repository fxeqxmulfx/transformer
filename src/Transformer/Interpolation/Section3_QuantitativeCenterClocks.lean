module

public import Transformer.Interpolation.Section1_BalancedRidgePhases
public import Transformer.Interpolation.Section3_PositiveBarycenters

/-!
# Quantitative original center contraction clocks and parameter costs

ArXiv:2411.04551v3, §3.2, Step 2 of selective contraction,
and §5, the logarithmic concentration cost for `thm: targets.atoms`.
The proved original ridge estimate is exponential with rate (a-c)/2
on a common initial cap with height at least a greater than c.
Choose its actual elapsed clock explicitly using log(2)+log(1/δ).

For 0<δ≤1, this clock is positive and bounded by a data-dependent
constant times 1+log(1/δ). The same time contracts every original
probability with that initial cap condition. Balanced actual parameters
realize this clock at any positive physical horizon with square-root
parameter growth, retaining original global weak existence and the
support conclusion for every original weak solution.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A nonnegative geometric mean is bounded by the sum of its two
factors. Source auxiliary: arXiv:2411.04551v3, §5, balancing the
actual clock cost against time and logarithmic concentration. -/
theorem sqrt_mul_le_add_nonneg (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    sqrt (x * y) ≤ x + y := by
  apply sqrt_le_iff.mpr
  refine ⟨add_nonneg hx hy, ?_⟩
  nlinarith [sq_nonneg x, sq_nonneg y, mul_nonneg hx hy]

/-- The balanced clock cost is bounded additively by inverse physical
time and one plus the logarithmic radius term. Source auxiliary:
arXiv:2411.04551v3, §5, the form of the final parameter estimate. -/
theorem sqrt_clock_le_additive (C H L : ℝ) (hC : 0 ≤ C) (hH : 0 < H) (hL : 0 ≤ L) :
    sqrt (C * (1 + L) / H) ≤ (C + 1) * (1 / H + 1 + L) := by
  have he : C * (1 + L) / H = (C / H) * (1 + L) := by ring
  rw [he]
  apply (sqrt_mul_le_add_nonneg (C / H) (1 + L) (div_nonneg hC hH.le) (by linarith)).trans
  have hq : 0 ≤ (1 : ℝ) / H := div_nonneg zero_le_one hH.le
  have hp : 0 ≤ C * (1 + L) := mul_nonneg hC (by linarith)
  have hid : (C + 1) * (1 / H + 1 + L) - (C / H + (1 + L)) = 1 / H + C * (1 + L) := by ring
  have hd : 0 ≤ (C + 1) * (1 / H + 1 + L) - (C / H + (1 + L)) := by
    rw [hid]
    exact add_nonneg hq hp
  linarith

/-- The actual original center-contraction clock can be chosen
uniformly over all cap-supported laws with logarithmic radius growth.
Source: arXiv:2411.04551v3, §3.2 and §5, quantitative concentration. -/
theorem exists_logarithmic_center_clock (a c : ℝ) (hc : 0 ≤ c) (hca : c < a) :
    ∃ C : ℝ, 0 < C ∧ ∀ δ : ℝ, 0 < δ → δ ≤ 1 →
      ∃ τ : ℝ, 0 < τ ∧ τ ≤ C * (1 + log (1 / δ)) ∧
        ∀ (p : SSphere d) (ν : Perspective.ProbSphere d),
          (∀ x ∈ (ν : Measure (SSphere d)).support,
            a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) →
          (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν τ :
            Measure (SSphere d)).support ⊆ Metric.closedBall p δ := by
  let rate := (a - c) / 2
  have hrate : 0 < rate := div_pos (sub_pos.mpr hca) (by norm_num)
  let C := rate⁻¹
  refine ⟨C, inv_pos.mpr hrate, ?_⟩
  intro δ hδ hδ1
  have hinv : 1 ≤ 1 / δ := (le_div_iff₀ hδ).mpr (by simpa using hδ1)
  have hlog : 0 ≤ log (1 / δ) := log_nonneg hinv
  have hlog2 : 0 < log (2 : ℝ) := log_pos (by norm_num)
  have hlog2le : log (2 : ℝ) ≤ 1 := by
    have hh := log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
    linarith
  let τ := (log (2 : ℝ) + log (1 / δ)) / rate
  have hτ : 0 < τ := div_pos (by linarith) hrate
  have hτle : τ ≤ C * (1 + log (1 / δ)) := by
    calc
      τ ≤ (1 + log (1 / δ)) / rate :=
        div_le_div_of_nonneg_right (by linarith) hrate.le
      _ = C * (1 + log (1 / δ)) := by dsimp only [C]; ring
  have hrateτ : rate * τ = log (2 : ℝ) + log (1 / δ) := by
    dsimp only [τ]
    field_simp
  have herror : 2 * exp (-(rate * τ)) = δ := by
    rw [hrateτ, exp_neg, exp_add, exp_log (by norm_num : (0 : ℝ) < 2), exp_log (div_pos one_pos hδ)]
    field_simp
  refine ⟨τ, hτ, hτle, ?_⟩
  intro p ν hν
  have hs := support_ridgeLaw_center_radius p ν a c hc hca hν τ hτ.le
  rw [herror] at hs
  exact hs

/-- One explicit logarithmic clock constant gives balanced actual
original controls with a quantitative square-root cost, uniformly
over every original law satisfying the cap-height hypothesis.
Source auxiliary: arXiv:2411.04551v3, §3.2 and §5, concentration at a prescribed time. -/
theorem exists_quantitative_center_controls (i : Fin d) (p : SSphere d)
    (a c : ℝ) (hc : 0 ≤ c) (hca : c < a) :
    ∃ C : ℝ, 0 < C ∧ ∀ δ : ℝ, 0 < δ → δ ≤ 1 → ∀ H : ℝ, 0 < H →
      ∃ (θ : TimeParams d) (τ : ℝ), 0 < τ ∧ τ ≤ C * (1 + log (1 / δ)) ∧
        PiecewiseConstant d θ H 1 ∧ (∀ s, (θ s).V = 0 ∧ (θ s).B = 0) ∧
        (∀ s, ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
          ≤ (2 + c) * sqrt (C * (1 + log (1 / δ)) / H)) ∧
        (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = ρ ∧ cauchyPB d θ μ) ∧
        ∀ ν : Perspective.ProbSphere d,
          (∀ x ∈ (ν : Measure (SSphere d)).support,
            a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) →
          ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d θ μ →
            (μ H : Measure (SSphere d)).support ⊆ Metric.closedBall p δ := by
  obtain ⟨C, hC, hclock⟩ := exists_logarithmic_center_clock (d := d) a c hc hca
  refine ⟨C, hC, ?_⟩
  intro δ hδ hδ1 H hH
  obtain ⟨τ, hτ, hτle, hcontract⟩ := hclock δ hδ hδ1
  obtain ⟨θ, hpc, hVB, hnorm, hex, hend⟩ := exists_balanced_ridge_controls i p
    (-(p : EucSpace d)) (-c) τ H hτ.le hH
  refine ⟨θ, τ, hτ, hτle, hpc, hVB, ?_, hex, ?_⟩
  · intro s
    have hcoef : 1 + ‖-(p : EucSpace d)‖ + |-c| = 2 + c := by
      rw [norm_neg, abs_neg, mem_sphere_zero_iff_norm.mp p.2, abs_of_nonneg hc]
      ring
    have hn := hnorm s
    rw [hcoef] at hn
    calc
      _ ≤ sqrt (τ / H) * (2 + c) := hn
      _ ≤ sqrt (C * (1 + log (1 / δ)) / H) * (2 + c) :=
        mul_le_mul_of_nonneg_right (sqrt_le_sqrt (div_le_div_of_nonneg_right hτle hH.le)) (by linarith)
      _ = _ := mul_comm _ _
  · intro ν hν μ h0 hμ
    rw [hend μ hμ, h0]
    exact hcontract p ν hν

/-- Actual center contraction has an additive inverse-time and
logarithmic parameter bound, with its constant chosen before the
radius or physical horizon. Source auxiliary: arXiv:2411.04551v3,
§3.2 and §5, quantitative concentration controls. -/
theorem exists_additive_center_controls (i : Fin d) (p : SSphere d)
    (a c : ℝ) (hc : 0 ≤ c) (hca : c < a) :
    ∃ D : ℝ, 0 < D ∧ ∀ δ : ℝ, 0 < δ → δ ≤ 1 → ∀ H : ℝ, 0 < H →
      ∃ θ : TimeParams d, PiecewiseConstant d θ H 1 ∧
        (∀ s, (θ s).V = 0 ∧ (θ s).B = 0) ∧
        (∀ s, ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
          ≤ D * (1 / H + 1 + log (1 / δ))) ∧
        (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = ρ ∧ cauchyPB d θ μ) ∧
        ∀ ν : Perspective.ProbSphere d,
          (∀ x ∈ (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) →
          ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d θ μ →
            (μ H : Measure (SSphere d)).support ⊆ Metric.closedBall p δ := by
  obtain ⟨C, hC, hcontrols⟩ := exists_quantitative_center_controls i p a c hc hca
  let D := (2 + c) * (C + 1)
  refine ⟨D, by dsimp only [D]; positivity, ?_⟩
  intro δ hδ hδ1 H hH
  obtain ⟨θ, _, _, _, hpc, hVB, hnorm, hex, hfinal⟩ := hcontrols δ hδ hδ1 H hH
  refine ⟨θ, hpc, hVB, ?_, hex, hfinal⟩
  intro s
  have hlog : 0 ≤ log (1 / δ) := log_nonneg ((le_div_iff₀ hδ).mpr (by simpa using hδ1))
  have hb := mul_le_mul_of_nonneg_left (sqrt_clock_le_additive C H (log (1 / δ)) hC.le hH hlog)
    (by linarith : 0 ≤ 2 + c)
  exact (hnorm s).trans (by simpa only [D, mul_assoc] using hb)

/-- A genuine off-axis original atom lies in a nontrivial active
cap with height floor 3/4 and threshold 1/2. Positive radius and
physical horizon meet all logarithmic and balanced-control premises. -/
example : ∃ (p : SSphere 2) (ν : Perspective.ProbSphere 2) (a c δ H : ℝ),
    0 ≤ c ∧ c < a ∧ 0 < δ ∧ δ ≤ 1 ∧ 0 < H ∧
      ∀ x ∈ (ν : Measure (SSphere 2)).support,
        a ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
  let p := basePoint 1
  let x : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let ν := Perspective.diracProb 2 x
  refine ⟨p, ν, 3 / 4, 1 / 2, 1 / 4, 1, by norm_num, by norm_num,
    by norm_num, by norm_num, one_pos, ?_⟩
  intro y hy
  rw [eq_of_mem_support_dirac hy]
  norm_num [p, x, circlePt, basePoint, EuclideanSpace.inner_single_right]

/-- Positive concrete scalar costs and time meet all additive
balancing hypotheses simultaneously. -/
example : (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Interpolation

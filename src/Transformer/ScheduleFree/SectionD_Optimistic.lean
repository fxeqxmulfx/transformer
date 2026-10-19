module

public import Transformer.ScheduleFree.SectionD_OptimisticScale

/-!
# The concrete optimistic algorithm

arXiv:2405.15682v4, Appendix D.1, Lemma 1. We follow the step size in
the displayed recursion; the paragraph's preceding, different formula
for `η_t` is not used. Real division supplies the zero-denominator
convention, so the all-zero oracle leaves every iterate zero.

The printed lemma omits `‖x_star‖≤D`, although its proof explicitly uses
that inequality. `optimistic_comparator_counterexample` refutes the
unrestricted statement. The corrected bounded-comparator lemma is proved
for the complete algorithm, including zero radius and zero gradient
prefixes. Its scale-dependent potential gives the stronger intermediate
constant `6`, which implies the printed constant `7`.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1: both the squared maximum and clipped energy are bounded
by the total observed squared gradient differences. -/
theorem optimisticRun_variation_bounds (D : ℝ) (g : ℕ → E) (T : ℕ) :
    (optimisticRun D g T).peak ^ 2 ≤
      ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2 ∧
    (optimisticRun D g T).energy ≤
      ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2 := by
  induction T with
  | zero => simp [optimisticRun]
  | succ t ih =>
    let s := optimisticRun D g t
    have hm := optimisticRun_peak_nonneg D g t
    have hc := clipGradient_variation_le hm s.previous (g t)
    have hcsq := (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr hc
    simp only [Finset.sum_range_succ, ← optimisticRun_previous D g t]
    constructor
    · change max s.peak ‖g t - s.previous‖ ^ 2 ≤ _
      by_cases h : s.peak ≤ ‖g t - s.previous‖
      · rw [max_eq_right h]
        nlinarith [ih.1, sq_nonneg s.peak]
      · rw [max_eq_left (le_of_not_ge h)]
        nlinarith [ih.1, sq_nonneg ‖g t - s.previous‖]
    · change s.energy + ‖clipGradient s.peak s.previous (g t) - s.previous‖ ^ 2 ≤ _
      linarith [ih.2]

/-- Appendix D.1: the final denominator is bounded by the observed variation. -/
theorem optimisticRun_scale_bound (D : ℝ) (g : ℕ → E) (T : ℕ) :
    optimisticScale (optimisticRun D g T) ≤
      Real.sqrt (2 * ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2) := by
  have h := optimisticRun_variation_bounds D g T
  apply Real.sqrt_le_sqrt
  linarith [h.1, h.2]


/-- Counterexample to Lemma 1 without the comparator-radius condition.
At `T=1`, `D=1`, `g_1=1`, `z_1=0`, and `x_star=-11`, regret is 11,
larger than `7√2`; all algorithm recursions are the actual ones above. -/
theorem optimistic_comparator_counterexample :
    7 * (1 : ℝ) * Real.sqrt (2 * (∑ t ∈ Finset.range 1,
      ‖(1 : ℝ) - previousOracle (fun _ => (1 : ℝ)) t‖ ^ 2)) <
        regret (fun t => (optimisticRun 1 (fun _ => (1 : ℝ)) t).z) (fun _ => 1) (-11) 1 := by
  norm_num [regret, optimisticRun, previousOracle, Finset.sum_range_succ]
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_nonneg 2]

/-- Appendix D.1: the telescoped regret, retaining its final nonnegative potential. -/
theorem optimistic_regret_potential {D : ℝ} (hD : 0 < D) (g : ℕ → E) (u : E)
    (hu : ‖u‖ ≤ D) (T : ℕ) :
    regret (fun t => (optimisticRun D g t).z) g u T ≤
      -(optimisticScale (optimisticRun D g T) / (2 * D) *
        ‖(optimisticRun D g T).auxiliary - u‖ ^ 2) +
      4 * D * optimisticScale (optimisticRun D g T) +
      2 * D * (optimisticRun D g T).peak := by
  induction T with
  | zero => simp [regret, optimisticRun, optimisticScale]
  | succ t ih =>
    have hs := optimistic_run_step hD g t u hu
    simp only [regret, Finset.sum_range_succ] at ih ⊢
    linarith

example (g : ℕ → ℝ) (T : ℕ) :
    regret (fun t => (optimisticRun 1 g t).z) g 0 T ≤
      -(optimisticScale (optimisticRun 1 g T) / (2 * 1) *
        ‖(optimisticRun 1 g T).auxiliary - 0‖ ^ 2) +
      4 * 1 * optimisticScale (optimisticRun 1 g T) +
      2 * 1 * (optimisticRun 1 g T).peak :=
  optimistic_regret_potential zero_lt_one g 0 (by norm_num) T

/-- Appendix D.1, Lemma 1: a stronger intermediate bound `6Dq_T` for
positive radius. The potential costs `4Dq_T` and clipping at most `2Dq_T`.
The actual iterates and the displayed adaptive step are unchanged. -/
theorem optimistic_regret_scale {D : ℝ} (hD : 0 < D) (g : ℕ → E) (u : E)
    (hu : ‖u‖ ≤ D) (T : ℕ) :
    regret (fun t => (optimisticRun D g t).z) g u T ≤
      6 * D * optimisticScale (optimisticRun D g T) := by
  let s := optimisticRun D g T
  have hreg := optimistic_regret_potential hD g u hu T
  have hm := optimisticRun_peak_nonneg D g T
  have he := optimisticRun_energy_nonneg D g T
  have hq := optimisticScale_nonneg s
  have hs := optimisticScale_sq s he
  have hmq : s.peak ≤ optimisticScale s := by
    apply (sq_le_sq₀ hm hq).mp
    linarith
  have hpeak := mul_le_mul_of_nonneg_left hmq (show 0 ≤ 2 * D by positivity)
  have hpotential : 0 ≤ optimisticScale s / (2 * D) * ‖s.auxiliary - u‖ ^ 2 := by
    positivity
  change regret (fun t => (optimisticRun D g t).z) g u T ≤
    -(optimisticScale s / (2 * D) * ‖s.auxiliary - u‖ ^ 2) +
      4 * D * optimisticScale s + 2 * D * s.peak at hreg
  change regret (fun t => (optimisticRun D g t).z) g u T ≤ 6 * D * optimisticScale s
  nlinarith only [hreg, hpeak, hpotential]

example (g : ℕ → ℝ) (T : ℕ) :
    regret (fun t => (optimisticRun 1 g t).z) g 0 T ≤
      6 * 1 * optimisticScale (optimisticRun 1 g T) :=
  optimistic_regret_scale zero_lt_one g 0 (by norm_num) T

/-- Appendix D.1, Lemma 1, corrected by the missing comparator condition
`‖u‖≤D` used in its proof. Both zero radius and zero gradient prefixes are included. -/
theorem optimistic_regret {D : ℝ} (hD : 0 ≤ D) (g : ℕ → E) (u : E) (hu : ‖u‖ ≤ D) (T : ℕ) :
    regret (fun t => (optimisticRun D g t).z) g u T ≤
      7 * D * Real.sqrt (2 * ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2) := by
  by_cases hz : D = 0
  · subst D
    have hu0 : u = 0 := norm_le_zero_iff.mp hu
    have hzero (t : ℕ) : (optimisticRun 0 g t).z = 0 :=
      norm_le_zero_iff.mp (optimisticRun_z_ball le_rfl g t)
    simp [regret, hu0, hzero]
  · have hpos : 0 < D := lt_of_le_of_ne hD (Ne.symm hz)
    have hreg := optimistic_regret_scale hpos g u hu T
    have hscale := optimisticRun_scale_bound D g T
    have hscaled := mul_le_mul_of_nonneg_left hscale (show 0 ≤ 6 * D by positivity)
    have hrest := mul_nonneg hD
      (Real.sqrt_nonneg (2 * ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2))
    nlinarith only [hreg, hscaled, hrest]

example (T : ℕ) :
    regret (fun t => (optimisticRun 1 (fun _ => (1 : ℝ)) t).z) (fun _ => 1) (-1) T ≤
      7 * 1 * Real.sqrt (2 * ∑ t ∈ Finset.range T,
        ‖(1 : ℝ) - previousOracle (fun _ => (1 : ℝ)) t‖ ^ 2) :=
  optimistic_regret zero_le_one (fun _ => 1) (-1) (by norm_num) T

example (g : ℕ → ℝ) (T : ℕ) :
    regret (fun t => (optimisticRun 0 g t).z) g 0 T ≤
      7 * 0 * Real.sqrt (2 * ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2) :=
  optimistic_regret le_rfl g 0 (by norm_num) T

end Transformer.ScheduleFree

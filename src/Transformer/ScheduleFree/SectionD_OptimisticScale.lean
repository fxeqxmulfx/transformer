module

public import Transformer.ScheduleFree.SectionD_OptimisticStep

/-!
# The optimistic algorithm's regularization scale

arXiv:2405.15682v4, Appendix D.1, the initial observations in the proof
of Lemma 1. The maximum and accumulated clipped energy control the
scale. A zero scale forces the stored gradient to be zero, which is the
case omitted when the source proof divides by the preceding step.

The one-step estimate retains the terminal quadratic potential. Its
change of coefficient and clipped variation each cost at most `2D`
times the scale increment. The remaining clipping error telescopes
with the running maximum, so no bound on individual gradient magnitudes
is required. The final sum is performed in `SectionD_Optimistic`.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1: discarded gradient differences are charged to the maximum increment. -/
theorem optimistic_clipping_error {D m : ℝ} (hm : 0 ≤ m) (previous g z u : E)
    (hz : ‖z‖ ≤ D) (hu : ‖u‖ ≤ D) :
    inner ℝ (g - clipGradient m previous g) (z - u) ≤
      2 * D * (max m ‖g - previous‖ - m) := by
  have hdist : ‖z - u‖ ≤ 2 * D := by linarith [norm_sub_le z u]
  calc
    _ ≤ ‖g - clipGradient m previous g‖ * ‖z - u‖ := real_inner_le_norm _ _
    _ ≤ ‖g - clipGradient m previous g‖ * (2 * D) :=
      mul_le_mul_of_nonneg_left hdist (norm_nonneg _)
    _ = _ := by rw [clipGradient_error hm]; ring

example : inner ℝ ((0 : ℝ) - clipGradient 1 0 0) (0 - 0) ≤
    2 * 1 * (max 1 ‖(0 : ℝ) - 0‖ - 1) :=
  optimistic_clipping_error zero_le_one 0 0 0 0 (by norm_num) (by norm_num)

/-- Appendix D.1: moving both quadratic charges into a telescoping potential.
The regularizer increment pays at most `4D(q-p)`, including its change
of coefficient on the old comparator distance. -/
theorem optimistic_potential_step {D p q : ℝ} (hD : 0 < D) (hp : 0 < p) (hpq : p ≤ q)
    (y previous clipped u : E) (hy : ‖y‖ ≤ D) (hu : ‖u‖ ≤ D)
    (hsmall : ‖clipped - previous‖ ^ 2 ≤ p ^ 2)
    (hincrement : ‖clipped - previous‖ ^ 2 ≤ q ^ 2 - p ^ 2) :
    let z := projectBall D (y - (D / p) • previous)
    let next := projectBall D (y - (D / q) • clipped)
    inner ℝ clipped (z - u) ≤ p / (2 * D) * ‖y - u‖ ^ 2 -
      q / (2 * D) * ‖next - u‖ ^ 2 + 4 * D * (q - p) := by
  let z := projectBall D (y - (D / p) • previous)
  let next := projectBall D (y - (D / q) • clipped)
  have hbase := optimistic_projection_step hD hp hpq y previous clipped u hu
  have hcharge := mul_le_mul_of_nonneg_left (clipped_charge hp.le hpq hsmall hincrement) hD.le
  have hdistance := mul_le_mul_of_nonneg_left (ball_distance_sq hD.le y u hy hu)
    (show 0 ≤ (q - p) / (2 * D) by positivity)
  have he : (q - p) / (2 * D) * (4 * D ^ 2) = 2 * D * (q - p) := by
    field_simp
    ring
  rw [he] at hdistance
  change inner ℝ clipped (z - u) ≤ _
  change inner ℝ clipped (z - u) ≤ _ at hbase
  simp only [div_eq_mul_inv] at hbase hcharge hdistance ⊢
  nlinarith only [hbase, hcharge, hdistance]

example (previous clipped : ℝ) (hsmall : ‖clipped - previous‖ ^ 2 ≤ (1 : ℝ) ^ 2)
    (hincrement : ‖clipped - previous‖ ^ 2 ≤ (2 : ℝ) ^ 2 - 1 ^ 2) :
    let z := projectBall 1 (0 - ((1 : ℝ) / 1) • previous)
    let next := projectBall 1 (0 - ((1 : ℝ) / 2) • clipped)
    inner ℝ clipped (z - 0) ≤ (1 : ℝ) / (2 * 1) * ‖(0 : ℝ) - 0‖ ^ 2 -
      (2 : ℝ) / (2 * 1) * ‖next - 0‖ ^ 2 + 4 * 1 * (2 - 1) :=
  optimistic_potential_step zero_lt_one zero_lt_one (by norm_num) 0 previous clipped 0
    (by simp) (by simp) hsmall hincrement

example : ‖(0 : ℝ) - 0‖ ^ 2 ≤ (1 : ℝ) ^ 2 ∧
    ‖(0 : ℝ) - 0‖ ^ 2 ≤ (2 : ℝ) ^ 2 - 1 ^ 2 := by norm_num

/-- Appendix D.1: the actual algorithm's one-step potential, valid at zero prefixes too. -/
theorem optimistic_run_step {D : ℝ} (hD : 0 < D) (g : ℕ → E) (t : ℕ)
    (u : E) (hu : ‖u‖ ≤ D) :
    inner ℝ (g t) ((optimisticRun D g t).z - u) ≤
      optimisticScale (optimisticRun D g t) / (2 * D) *
        ‖(optimisticRun D g t).auxiliary - u‖ ^ 2 -
      optimisticScale (optimisticRun D g (t + 1)) / (2 * D) *
        ‖(optimisticRun D g (t + 1)).auxiliary - u‖ ^ 2 +
      4 * D * (optimisticScale (optimisticRun D g (t + 1)) -
        optimisticScale (optimisticRun D g t)) +
      2 * D * ((optimisticRun D g (t + 1)).peak - (optimisticRun D g t).peak) := by
  let s := optimisticRun D g t
  let next := optimisticRun D g (t + 1)
  let c := clipGradient s.peak s.previous (g t)
  have hm := optimisticRun_peak_nonneg D g t
  have he := optimisticRun_energy_nonneg D g t
  have hy := optimisticRun_auxiliary_ball hD.le g t
  have hz := optimisticRun_z_ball hD.le g t
  change ‖s.auxiliary‖ ≤ D at hy
  have hquery := optimisticRun_query hD.le g t
  have hstats := optimisticStep_scale_properties D s (g t) hm he
  have herror := optimistic_clipping_error hm s.previous (g t) s.z u hz hu
  have hnext : next.auxiliary = projectBall D (s.auxiliary -
      (D / optimisticScale next) • c) := rfl
  have hsplit : inner ℝ (g t) (s.z - u) = inner ℝ c (s.z - u) +
      inner ℝ (g t - c) (s.z - u) := by rw [inner_sub_left]; ring
  have hbase : inner ℝ c (s.z - u) ≤
      optimisticScale s / (2 * D) * ‖s.auxiliary - u‖ ^ 2 -
      optimisticScale next / (2 * D) * ‖next.auxiliary - u‖ ^ 2 +
      4 * D * (optimisticScale next - optimisticScale s) := by
    by_cases hp : optimisticScale s = 0
    · have hzero := optimisticRun_scale_zero D g t hp
      have hc : c = 0 := by dsimp [c]; rw [hzero.1, clipGradient_zero, hzero.2]
      have hn : next.auxiliary = s.auxiliary := by
        rw [hnext, hc, smul_zero, sub_zero, projectBall, ite_eq_left hy]
      have hqn := optimisticScale_nonneg next
      have hd := mul_le_mul_of_nonneg_left (ball_distance_sq hD.le s.auxiliary u hy hu)
        (show 0 ≤ optimisticScale next / (2 * D) by positivity)
      have hcancel : optimisticScale next / (2 * D) * (4 * D ^ 2) =
          2 * D * optimisticScale next := by field_simp; ring
      rw [hcancel] at hd
      rw [hc, hp, hn]
      simp only [inner_zero_left, zero_div, zero_mul, zero_sub, sub_zero]
      nlinarith [mul_nonneg hD.le (optimisticScale_nonneg next)]
    · have hp' : 0 < optimisticScale s :=
        lt_of_le_of_ne (optimisticScale_nonneg _) (Ne.symm hp)
      have hb := optimistic_potential_step (q := optimisticScale next) hD hp' hstats.1
        s.auxiliary s.previous c u hy hu hstats.2.1 hstats.2.2
      change inner ℝ c (projectBall D (s.auxiliary -
        (D / optimisticScale s) • s.previous) - u) ≤ _ at hb
      rw [← hquery, ← hnext] at hb
      exact hb
  change inner ℝ (g t) (s.z - u) ≤ _
  rw [hsplit]
  change inner ℝ (g t - c) (s.z - u) ≤ 2 * D * (next.peak - s.peak) at herror
  linarith

example (g : ℕ → ℝ) (t : ℕ) :
    inner ℝ (g t) ((optimisticRun 1 g t).z - 0) ≤
      optimisticScale (optimisticRun 1 g t) / (2 * 1) *
        ‖(optimisticRun 1 g t).auxiliary - 0‖ ^ 2 -
      optimisticScale (optimisticRun 1 g (t + 1)) / (2 * 1) *
        ‖(optimisticRun 1 g (t + 1)).auxiliary - 0‖ ^ 2 +
      4 * 1 * (optimisticScale (optimisticRun 1 g (t + 1)) -
        optimisticScale (optimisticRun 1 g t)) +
      2 * 1 * ((optimisticRun 1 g (t + 1)).peak - (optimisticRun 1 g t).peak) :=
  optimistic_run_step zero_lt_one g t 0 (by norm_num)


end Transformer.ScheduleFree

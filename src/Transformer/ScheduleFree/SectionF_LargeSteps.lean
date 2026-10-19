module

public import Transformer.ScheduleFree.Section2_GradientDescent

/-!
# Large constant steps

arXiv:2405.15682v4, Theorem 3 and Appendix F. We make the hidden
big-O constant explicit: average regret is at most `4D√(Σ‖g‖²)/T`.
The quantitative proof works whenever `0<G≤√(Σ‖g‖²)`; choosing the
maximum observed gradient norm gives exactly the source's step size.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix F: the cumulative observed gradient norm bound. -/
def maxGradient (g : ℕ → E) : ℕ → ℝ
  | 0 => 0
  | t + 1 => max (maxGradient g t) ‖g t‖

omit [InnerProductSpace ℝ E] in
theorem maxGradient_nonneg (g : ℕ → E) (T : ℕ) : 0 ≤ maxGradient g T := by
  cases T with
  | zero => exact le_rfl
  | succ T => exact (norm_nonneg _).trans (le_max_right _ _)

omit [InnerProductSpace ℝ E] in
/-- The maximum observed norm is bounded by the root sum of squares. -/
theorem maxGradient_le_energy (g : ℕ → E) (T : ℕ) :
    maxGradient g T ≤ Real.sqrt (∑ t ∈ Finset.range T, ‖g t‖ ^ 2) := by
  apply (Real.le_sqrt (maxGradient_nonneg g T) (Finset.sum_nonneg fun _ _ => sq_nonneg _)).mpr
  induction T with
  | zero => simp [maxGradient]
  | succ T ih =>
    rw [maxGradient, Finset.sum_range_succ]
    rcases le_total (maxGradient g T) ‖g T‖ with h | h
    · rw [max_eq_right h]
      exact le_add_of_nonneg_left (Finset.sum_nonneg fun _ _ => sq_nonneg _)
    · rw [max_eq_left h]
      exact ih.trans (le_add_of_nonneg_right (sq_nonneg _))

/-- Appendix F: the sum of the actual base displacements. -/
theorem sum_base_updates (η : ℝ) (z g : ℕ → E)
    (hstep : ∀ t, z (t + 1) = z t - η • g t) (T : ℕ) :
    η • (∑ t ∈ Finset.range T, g t) = z 0 - z T := by
  induction T with
  | zero => simp
  | succ T ih => rw [Finset.sum_range_succ, smul_add, ih, hstep]; module

example (η : ℝ) (g : ℕ → E) (initial : E) (T : ℕ) :
    η • (∑ t ∈ Finset.range T, g t) =
      baseRun η g initial 0 - baseRun η g initial T :=
  sum_base_updates η _ g (fun _ => rfl) T

/-- Appendix F, eq. (16): Cauchy--Schwarz for regret at the initial point. -/
theorem regret_initial_bound (η : ℝ) (z g : ℕ → E) (u : E)
    (hstep : ∀ t, z (t + 1) = z t - η • g t) (T : ℕ) :
    η * regret z g u T ≤ ‖z 0 - z T‖ * ‖z 0 - u‖ + η * regret z g (z 0) T := by
  have hsplit : regret z g u T =
      inner ℝ (∑ t ∈ Finset.range T, g t) (z 0 - u) + regret z g (z 0) T := by
    rw [regret, regret, sum_inner, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t ht
    simp only [inner_sub_right]
    ring
  have hi := real_inner_le_norm (z 0 - z T) (z 0 - u)
  have heinner : η * inner ℝ (∑ t ∈ Finset.range T, g t) (z 0 - u) =
      inner ℝ (z 0 - z T) (z 0 - u) := by
    rw [← real_inner_smul_left, sum_base_updates η z g hstep T]
  rw [← heinner] at hi
  rw [hsplit]
  nlinarith

example (g : ℕ → E) (initial u : E) (T : ℕ) :
    1 * regret (baseRun 1 g initial) g u T ≤
      ‖baseRun 1 g initial 0 - baseRun 1 g initial T‖ * ‖baseRun 1 g initial 0 - u‖ +
        1 * regret (baseRun 1 g initial) g (baseRun 1 g initial 0) T :=
  regret_initial_bound 1 _ g u (fun _ => rfl) T

/-- Theorem 3 / Appendix F: the monitored condition implies regret at most
`4D√(Σ‖g‖²)`. Nonpositive regret needs no displacement bound. -/
theorem large_step_regret {D G : ℝ} (hD : 0 < D) (hG : 0 < G)
    (z g : ℕ → E) (u : E) (T : ℕ) (hinit : ‖z 0 - u‖ = D)
    (hstep : ∀ t, z (t + 1) = z t - (D / G) • g t)
    (hGenergy : G ≤ Real.sqrt (∑ t ∈ Finset.range T, ‖g t‖ ^ 2))
    (hmonitor : regret z g (z 0) T ≤ D * Real.sqrt (∑ t ∈ Finset.range T, ‖g t‖ ^ 2)) :
    regret z g u T ≤ 4 * D * Real.sqrt (∑ t ∈ Finset.range T, ‖g t‖ ^ 2) := by
  let η := D / G
  let S := Real.sqrt (∑ t ∈ Finset.range T, ‖g t‖ ^ 2)
  let a := ‖z 0 - z T‖
  have hη : 0 < η := div_pos hD hG
  have hS : 0 ≤ S := Real.sqrt_nonneg _
  have heS : S ^ 2 = ∑ t ∈ Finset.range T, ‖g t‖ ^ 2 :=
    Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hb := regret_initial_bound η z g u hstep T
  rw [hinit] at hb
  have he := regret_energy η z g (z 0) T hstep
  rw [sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0), norm_sub_rev, ← heS] at he
  change 2 * η * regret z g (z 0) T = 0 - a ^ 2 + η ^ 2 * S ^ 2 at he
  by_cases hR : regret z g u T ≤ 0
  · exact hR.trans (by positivity)
  have ha : a ≤ 2 * D + η * S := by
    have hq : a ^ 2 ≤ 2 * a * D + η ^ 2 * S ^ 2 := by
      change η * regret z g u T ≤ a * D + η * regret z g (z 0) T at hb
      nlinarith [mul_pos hη (lt_of_not_ge hR)]
    by_contra h
    have hgt : 0 < a - (2 * D + η * S) := by linarith
    have ha0 : 0 < a := by nlinarith [mul_nonneg hη.le hS]
    have hprod := mul_pos hgt (add_pos_of_pos_of_nonneg ha0 (mul_nonneg hη.le hS))
    nlinarith [mul_nonneg hD.le (mul_nonneg hη.le hS)]
  have hηG : η * G = D := by dsimp [η]; field_simp
  have hmul := mul_le_mul_of_nonneg_right ha hD.le
  have hm := mul_le_mul_of_nonneg_left hmonitor hη.le
  have hDG := mul_le_mul_of_nonneg_left hGenergy (by positivity : 0 ≤ 2 * D)
  have hbase : η * regret z g u T ≤ 2 * D ^ 2 + 2 * η * D * S := by
    change η * regret z g u T ≤ a * D + η * regret z g (z 0) T at hb
    nlinarith [hmul, hm]
  have hscaled : regret z g u T ≤ 2 * D * G + 2 * D * S := by
    apply (mul_le_mul_iff_right₀ hη).mp
    calc
      _ ≤ 2 * D ^ 2 + 2 * η * D * S := hbase
      _ = η * (2 * D * G + 2 * D * S) := by nlinarith [hηG]
  nlinarith

example : regret (baseRun 1 (fun _ => (1 : ℝ)) 1) (fun _ => 1) 0 1 ≤
    4 * 1 * Real.sqrt (∑ _ ∈ Finset.range 1, ‖(1 : ℝ)‖ ^ 2) := by
  apply large_step_regret (D := 1) (G := 1) zero_lt_one zero_lt_one
  · simp [baseRun]
  · intro t
    simp [baseRun]
  · norm_num [Finset.sum_range_succ]
  · norm_num [regret, baseRun, Finset.sum_range_succ]

end Transformer.ScheduleFree

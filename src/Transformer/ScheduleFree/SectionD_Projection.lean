module

public import Transformer.ScheduleFree.SectionE_Regret

/-!
# Projection inequalities for optimistic mirror descent

arXiv:2405.15682v4, Appendix D.1, proof of Lemma 1. The explicit radial
projection satisfies the variational inequality used at both updates.
The energy inequality is multiplied by the step, so it also covers zero
steps without introducing an inverse of zero.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

section Geometry

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1: first-order optimality of the actual ball projection. -/
theorem projectBall_variational {D : ℝ} (hD : 0 ≤ D) (v u : E) (hu : ‖u‖ ≤ D) :
    0 ≤ inner ℝ (v - projectBall D v) (projectBall D v - u) := by
  unfold projectBall
  split_ifs with h
  · simp
  · have hv : 0 < ‖v‖ := lt_of_le_of_lt hD (lt_of_not_ge h)
    have ha : D / ‖v‖ ≤ 1 := (div_le_one hv).mpr (le_of_not_ge h)
    have he : D / ‖v‖ * ‖v‖ = D := div_mul_cancel₀ _ hv.ne'
    have he2 : D / ‖v‖ * ‖v‖ ^ 2 = ‖v‖ * D := by
      calc
        _ = (D / ‖v‖ * ‖v‖) * ‖v‖ := by ring
        _ = _ := by rw [he]; ring
    have hi : inner ℝ v u ≤ ‖v‖ * D :=
      (real_inner_le_norm v u).trans (mul_le_mul_of_nonneg_left hu hv.le)
    calc
      0 ≤ (1 - D / ‖v‖) * (‖v‖ * D - inner ℝ v u) :=
        mul_nonneg (sub_nonneg.mpr ha) (sub_nonneg.mpr hi)
      _ = inner ℝ (v - (D / ‖v‖) • v) ((D / ‖v‖) • v - u) := by
        symm
        calc
          _ = (1 - D / ‖v‖) * (D / ‖v‖ * ‖v‖ ^ 2 - inner ℝ v u) := by
            simp only [inner_sub_left, inner_sub_right, real_inner_smul_left,
              real_inner_smul_right]
            rw [real_inner_self_eq_norm_sq]
            ring
          _ = _ := by rw [he2]

example (v : E) : 0 ≤ inner ℝ (v - projectBall 1 v) (projectBall 1 v - 0) :=
  projectBall_variational zero_le_one v 0 (by simp)

/-- Appendix D.1: the three-point energy inequality for a projected step. -/
theorem projected_three_point {D : ℝ} (hD : 0 ≤ D) (η : ℝ) (y g u : E)
    (hu : ‖u‖ ≤ D) :
    2 * η * inner ℝ g (projectBall D (y - η • g) - u) ≤
      ‖y - u‖ ^ 2 - ‖projectBall D (y - η • g) - u‖ ^ 2 -
        ‖projectBall D (y - η • g) - y‖ ^ 2 := by
  let p := projectBall D (y - η • g)
  have hp := projectBall_variational hD (y - η • g) u hu
  change 0 ≤ inner ℝ (y - η • g - p) (p - u) at hp
  have hid : ‖y - u‖ ^ 2 - ‖p - u‖ ^ 2 - ‖p - y‖ ^ 2 =
      2 * inner ℝ (y - p) (p - u) := by
    simp only [norm_sub_sq_real, inner_sub_left, inner_sub_right,
      real_inner_self_eq_norm_sq]
    rw [real_inner_comm p y]
    ring
  change 2 * η * inner ℝ g (p - u) ≤ _
  rw [hid]
  simp only [inner_sub_left, real_inner_smul_left] at hp ⊢
  linarith

example (η : ℝ) (y g : E) :
    2 * η * inner ℝ g (projectBall 1 (y - η • g) - 0) ≤
      ‖y - 0‖ ^ 2 - ‖projectBall 1 (y - η • g) - 0‖ ^ 2 -
        ‖projectBall 1 (y - η • g) - y‖ ^ 2 :=
  projected_three_point zero_le_one η y g 0 (by simp)

omit [InnerProductSpace ℝ E] in
/-- Appendix D.1: every squared comparator distance inside the ball is at most `4D²`. -/
theorem ball_distance_sq {D : ℝ} (hD : 0 ≤ D) (y u : E)
    (hy : ‖y‖ ≤ D) (hu : ‖u‖ ≤ D) : ‖y - u‖ ^ 2 ≤ 4 * D ^ 2 := by
  have h := (norm_sub_le y u).trans (add_le_add hy hu)
  nlinarith [norm_nonneg (y - u)]

example : ‖(0 : ℝ) - 0‖ ^ 2 ≤ 4 * (1 : ℝ) ^ 2 :=
  ball_distance_sq zero_le_one 0 0 (by norm_num) (by norm_num)

end Geometry

section Clipping

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Appendix D.1: clip the new gradient difference at the previous maximum. -/
def clipGradient (peak : ℝ) (previous g : E) : E :=
  previous + (min peak ‖g - previous‖ / ‖g - previous‖) • (g - previous)

/-- Appendix D.1: the clipped variation norm, including a zero difference. -/
theorem clipGradient_variation {m : ℝ} (hm : 0 ≤ m) (previous g : E) :
    ‖clipGradient m previous g - previous‖ = min m ‖g - previous‖ := by
  simp only [clipGradient, add_sub_cancel_left, norm_smul, Real.norm_eq_abs]
  rw [abs_of_nonneg (div_nonneg (le_min hm (norm_nonneg _)) (norm_nonneg _))]
  by_cases h : ‖g - previous‖ = 0
  · simp [h, hm]
  · exact div_mul_cancel₀ _ h

example : ‖clipGradient 1 (0 : ℝ) 0 - 0‖ = min 1 ‖(0 : ℝ) - 0‖ :=
  clipGradient_variation zero_le_one 0 0

/-- Appendix D.1: the clipping error equals the running-maximum increment. -/
theorem clipGradient_error {m : ℝ} (hm : 0 ≤ m) (previous g : E) :
    ‖g - clipGradient m previous g‖ = max m ‖g - previous‖ - m := by
  by_cases hz : ‖g - previous‖ = 0
  · have hg : g = previous := sub_eq_zero.mp (norm_eq_zero.mp hz)
    subst g
    simp [clipGradient, hm]
  · have hn : 0 < ‖g - previous‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hz)
    have ha1 : min m ‖g - previous‖ / ‖g - previous‖ ≤ 1 :=
      (div_le_one hn).mpr (min_le_right _ _)
    have he : g - clipGradient m previous g =
        (1 - min m ‖g - previous‖ / ‖g - previous‖) • (g - previous) := by
      dsimp [clipGradient]
      module
    rw [he, norm_smul, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr ha1)]
    have hc := div_mul_cancel₀ (min m ‖g - previous‖) hz
    linarith [min_add_max m ‖g - previous‖]

example : ‖(0 : ℝ) - clipGradient 1 0 0‖ = max 1 ‖(0 : ℝ) - 0‖ - 1 :=
  clipGradient_error zero_le_one 0 0

/-- Appendix D.1: clipping never increases the observed variation norm. -/
theorem clipGradient_variation_le {m : ℝ} (hm : 0 ≤ m) (previous g : E) :
    ‖clipGradient m previous g - previous‖ ≤ ‖g - previous‖ := by
  rw [clipGradient_variation hm]
  exact min_le_right _ _

example : ‖clipGradient 1 (0 : ℝ) 0 - 0‖ ≤ ‖(0 : ℝ) - 0‖ :=
  clipGradient_variation_le zero_le_one 0 0

/-- Appendix D.1: a zero previous maximum discards the entire new variation. -/
theorem clipGradient_zero (previous g : E) : clipGradient 0 previous g = previous := by
  simp [clipGradient]


end Clipping

end Transformer.ScheduleFree

module

public import Transformer.ScheduleFree.SectionD_Projection

/-!
# The concrete clipped optimistic algorithm and its ball invariants

arXiv:2405.15682v4, Appendix D.1, Lemma 1. Source time `t=1` is time
`0` here. We use the step in the displayed recursion. Real division
supplies the zero-denominator convention, including an all-zero prefix.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

/-- Appendix D.1: the two iterates and the past oracle statistics. -/
structure OptimisticState (E : Type*) where
  z : E
  auxiliary : E
  previous : E
  peak : ℝ
  energy : ℝ

variable {E : Type*}

/-- Appendix D.1: the learning-rate denominator, before division by `D`. -/
def optimisticScale (s : OptimisticState E) : ℝ := Real.sqrt (s.peak ^ 2 + s.energy)

/-- Appendix D.1: the regularization scale is nonnegative. -/
theorem optimisticScale_nonneg (s : OptimisticState E) : 0 ≤ optimisticScale s :=
  Real.sqrt_nonneg _

/-- Appendix D.1: the exact squared regularization scale. -/
theorem optimisticScale_sq (s : OptimisticState E) (he : 0 ≤ s.energy) :
    optimisticScale s ^ 2 = s.peak ^ 2 + s.energy :=
  Real.sq_sqrt (add_nonneg (sq_nonneg _) he)

example : optimisticScale (⟨(0 : ℝ), 0, 0, 0, 0⟩ : OptimisticState ℝ) ^ 2 = 0 ^ 2 + 0 :=
  optimisticScale_sq _ le_rfl

variable [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1: one clipped optimistic step, including both projections. -/
def optimisticStep (D : ℝ) (s : OptimisticState E) (g : E) : OptimisticState E :=
  let clipped := clipGradient s.peak s.previous g
  let peak := max s.peak ‖g - s.previous‖
  let energy := s.energy + ‖clipped - s.previous‖ ^ 2
  let η := D / Real.sqrt (peak ^ 2 + energy)
  let auxiliary := projectBall D (s.auxiliary - η • clipped)
  ⟨projectBall D (auxiliary - η • g), auxiliary, g, peak, energy⟩

/-- Appendix D.1: all buffers start at zero. -/
def optimisticRun (D : ℝ) (g : ℕ → E) : ℕ → OptimisticState E
  | 0 => ⟨0, 0, 0, 0, 0⟩
  | t + 1 => optimisticStep D (optimisticRun D g t) (g t)

/-- `g_0=0` in the paper's one-based notation. -/
def previousOracle (g : ℕ → E) : ℕ → E
  | 0 => 0
  | t + 1 => g t

/-- Appendix D.1: the stored gradient is exactly the previous oracle value. -/
theorem optimisticRun_previous (D : ℝ) (g : ℕ → E) (t : ℕ) :
    (optimisticRun D g t).previous = previousOracle g t := by
  cases t <;> rfl

/-- Appendix D.1: the running maximum is always nonnegative. -/
theorem optimisticRun_peak_nonneg (D : ℝ) (g : ℕ → E) (t : ℕ) :
    0 ≤ (optimisticRun D g t).peak := by
  cases t with
  | zero => exact le_rfl
  | succ t =>
    exact (norm_nonneg _).trans (le_max_right _ _)

/-- Appendix D.1: accumulated squared clipped differences are nonnegative. -/
theorem optimisticRun_energy_nonneg (D : ℝ) (g : ℕ → E) (t : ℕ) :
    0 ≤ (optimisticRun D g t).energy := by
  induction t with
  | zero => exact le_rfl
  | succ t ih => exact add_nonneg ih (sq_nonneg _)

/-- Appendix D.1: every query stays inside the comparator ball. -/
theorem optimisticRun_z_ball {D : ℝ} (hD : 0 ≤ D) (g : ℕ → E) (t : ℕ) :
    ‖(optimisticRun D g t).z‖ ≤ D := by
  cases t with
  | zero => simpa [optimisticRun] using hD
  | succ t => exact norm_projectBall hD _

example (g : ℕ → ℝ) (t : ℕ) : ‖(optimisticRun 1 g t).z‖ ≤ 1 :=
  optimisticRun_z_ball zero_le_one g t

/-- Appendix D.1: the auxiliary iterate also stays inside the ball. -/
theorem optimisticRun_auxiliary_ball {D : ℝ} (hD : 0 ≤ D) (g : ℕ → E) (t : ℕ) :
    ‖(optimisticRun D g t).auxiliary‖ ≤ D := by
  cases t with
  | zero => simpa [optimisticRun] using hD
  | succ t => exact norm_projectBall hD _

example (g : ℕ → ℝ) (t : ℕ) : ‖(optimisticRun 1 g t).auxiliary‖ ≤ 1 :=
  optimisticRun_auxiliary_ball zero_le_one g t

/-- Appendix D.1: the query uses the stored previous gradient and its own scale. -/
theorem optimisticRun_query {D : ℝ} (hD : 0 ≤ D) (g : ℕ → E) (t : ℕ) :
    (optimisticRun D g t).z = projectBall D ((optimisticRun D g t).auxiliary -
      (D / optimisticScale (optimisticRun D g t)) • (optimisticRun D g t).previous) := by
  cases t with
  | zero => simp [optimisticRun, projectBall, hD]
  | succ t => rfl

example (g : ℕ → ℝ) (t : ℕ) :
    (optimisticRun 1 g t).z = projectBall 1 ((optimisticRun 1 g t).auxiliary -
      (1 / optimisticScale (optimisticRun 1 g t)) • (optimisticRun 1 g t).previous) :=
  optimisticRun_query zero_le_one g t

/-- Appendix D.1: a zero running maximum implies that the previous gradient is zero. -/
theorem optimisticRun_peak_zero (D : ℝ) (g : ℕ → E) (t : ℕ) :
    (optimisticRun D g t).peak = 0 → (optimisticRun D g t).previous = 0 := by
  induction t with
  | zero => simp [optimisticRun]
  | succ t ih =>
    intro hz
    let s := optimisticRun D g t
    change max s.peak ‖g t - s.previous‖ = 0 at hz
    have hm : s.peak = 0 := le_antisymm
      ((le_max_left _ _).trans_eq hz) (optimisticRun_peak_nonneg D g t)
    have hn : ‖g t - s.previous‖ = 0 := le_antisymm
      ((le_max_right _ _).trans_eq hz) (norm_nonneg _)
    change g t = 0
    exact (sub_eq_zero.mp (norm_eq_zero.mp hn)).trans (ih hm)

example : (optimisticRun 1 (fun _ => (0 : ℝ)) 0).previous = 0 :=
  optimisticRun_peak_zero 1 _ 0 rfl

/-- Appendix D.1: the zero-scale case has both a zero maximum and a zero past gradient. -/
theorem optimisticRun_scale_zero (D : ℝ) (g : ℕ → E) (t : ℕ)
    (hz : optimisticScale (optimisticRun D g t) = 0) :
    (optimisticRun D g t).peak = 0 ∧ (optimisticRun D g t).previous = 0 := by
  have hs := optimisticScale_sq _ (optimisticRun_energy_nonneg D g t)
  rw [hz] at hs
  have hm : (optimisticRun D g t).peak = 0 := by
    nlinarith [optimisticRun_peak_nonneg D g t, optimisticRun_energy_nonneg D g t]
  exact ⟨hm, optimisticRun_peak_zero D g t hm⟩

example : (optimisticRun 1 (fun _ => (0 : ℝ)) 0).peak = 0 ∧
    (optimisticRun 1 (fun _ => (0 : ℝ)) 0).previous = 0 :=
  optimisticRun_scale_zero 1 _ 0 (by simp [optimisticRun, optimisticScale])


end Transformer.ScheduleFree

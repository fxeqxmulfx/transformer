module

public import Transformer.ScheduleFree.AMSGrad.Defs

/-!
# Invariants of the Schedule-Free AMSGrad recurrence

Properties of the new composition defined in `Defs`, based on
arXiv:2405.15682v4, §4.4 and arXiv:1904.03590v4, Algorithm 1 and §6.
These concern the actual update buffers and its averaging coefficient.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AMSGrad

variable {ι : Type*}

/-- The initial Schedule-Free query is the supplied point for every interpolation. -/
@[simp] theorem query_initial (p : AdamParameters) (point : ι → ℝ) :
    query p (initial point) = point := by
  funext i
  dsimp [query, initial]
  ring

/-- The local extension's actual raw running maximum never decreases. -/
theorem maximum_mono (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) (i : ι) :
    s.maximum i ≤ (stepWith project p t s g).maximum i :=
  le_max_left _ _

/-- Every updated raw variance is bounded by the actual updated maximum. -/
theorem second_le_maximum (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) (i : ι) :
    (stepWith project p t s g).second i ≤ (stepWith project p t s g).maximum i :=
  le_max_right _ _

/-- §4.4's beta-two range preserves nonnegative raw second moments here too. -/
theorem second_nonneg (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) (hβ : 0 ≤ p.betaTwo) (hβ1 : p.betaTwo ≤ 1)
    (hv : ∀ i, 0 ≤ s.second i) (i : ι) :
    0 ≤ (stepWith project p t s g).second i := by
  exact add_nonneg (mul_nonneg hβ (hv i))
    (mul_nonneg (sub_nonneg.mpr hβ1) (sq_nonneg _))

example : 0 ≤ (stepWith id (ι := Fin 1) ⟨1, 0, 0, 1/2, 1/2, 1⟩ 0
    (initial (fun _ => 2)) (fun _ => 3)).second 0 := by
  apply second_nonneg
  · norm_num
  · norm_num
  · intro i
    norm_num [initial]

/-- The local extension preserves nonnegative maxima independently of projection. -/
theorem maximum_nonneg (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) (i : ι) (h : 0 ≤ s.maximum i) :
    0 ≤ (stepWith project p t s g).maximum i :=
  le_trans h (maximum_mono project p t s g i)

example : 0 ≤ (stepWith id (ι := Fin 1) ⟨1, 0, 0, 1/2, 1/2, 1⟩ 0
    (initial (fun _ => 2)) (fun _ => 3)).maximum 0 :=
  maximum_nonneg _ _ _ _ _ _ le_rfl

/-- Squared effective rates make the actual cumulative weight nondecreasing. -/
theorem total_mono (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) :
    s.total ≤ (stepWith project p t s g).total := by
  change s.total ≤ s.total + adamLearningRate p t ^ 2
  nlinarith [sq_nonneg (adamLearningRate p t)]

/-- Cumulative averaging weights remain nonnegative after the actual step. -/
theorem total_nonneg (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) (h : 0 ≤ s.total) :
    0 ≤ (stepWith project p t s g).total :=
  le_trans h (total_mono project p t s g)

example : 0 ≤ (stepWith id (ι := Fin 1) ⟨1, 0, 0, 1/2, 1/2, 1⟩ 0
    (initial (fun _ => 2)) (fun _ => 3)).total :=
  total_nonneg _ _ _ _ _ le_rfl

/-- The local extension's averaging coefficient lies in the closed unit interval,
including the all-zero-rate prefix, for which the coefficient is zero. -/
theorem coefficient_bounds (p : AdamParameters) (t : ℕ) (s : State ι) (h : 0 ≤ s.total) :
    0 ≤ coefficient p t s ∧ coefficient p t s ≤ 1 := by
  have hη := sq_nonneg (adamLearningRate p t)
  constructor
  · exact div_nonneg hη (add_nonneg h hη)
  · apply div_le_one_of_le₀
    · linarith
    · exact add_nonneg h hη

example : 0 ≤ coefficient (ι := Fin 1) ⟨1, 0, 0, 1/2, 0, 1⟩ 0
    (initial (fun _ => 2)) ∧
    coefficient (ι := Fin 1) ⟨1, 0, 0, 1/2, 0, 1⟩ 0 (initial (fun _ => 2)) ≤ 1 :=
  coefficient_bounds _ _ _ le_rfl

/-- §6's positive epsilon keeps every denominator of the local extension positive. -/
theorem denominator_pos (p : AdamParameters) (s : State ι) (g : ι → ℝ)
    (hε : 0 < p.epsilon) (i : ι) :
    0 < Real.sqrt (nextMaximum p s g i) + p.epsilon :=
  adam_denominator_pos _ _ hε

example : 0 < Real.sqrt (nextMaximum (ι := Fin 1) ⟨1, 0, 0, 1/2, 0, 1⟩
    (initial (fun _ => 2)) (fun _ => 3) 0) + 1 :=
  denominator_pos _ _ _ zero_lt_one _

/-- Zero learning rate freezes both iterates while the gradient histories still
advance. This also fixes the MD component-freezing convention. -/
theorem zero_rate_iterates (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters)
    (t : ℕ) (s : State ι) (g : ι → ℝ) (h : p.learningRate = 0) :
    (stepWith project p t s g).base = s.base ∧
      (stepWith project p t s g).evaluation = s.evaluation := by
  constructor
  · simp [stepWith, adamLearningRate, h]
  · funext i
    simp [stepWith, coefficient, adamLearningRate, h]

example : (stepWith id (ι := Fin 1) ⟨0, 0, 0, 1/2, 0, 1⟩ 0
    (initial (fun _ => 2)) (fun _ => 3)).base = (fun _ => 2) ∧
    (stepWith id (ι := Fin 1) ⟨0, 0, 0, 1/2, 0, 1⟩ 0
    (initial (fun _ => 2)) (fun _ => 3)).evaluation = (fun _ => 2) :=
  zero_rate_iterates _ _ _ _ _ rfl

/-- Every reachable cumulative averaging weight is nonnegative for every oracle. -/
theorem run_total_nonneg (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (t : ℕ) : 0 ≤ (run p oracle point t).total := by
  induction t with
  | zero => simp [run, initial]
  | succ t ih => exact total_nonneg id p t _ _ ih

/-- Consequently every coefficient used on an actual run is a convex weight,
including prefixes that contain only zero effective rates. -/
theorem run_coefficient_bounds (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (t : ℕ) :
    0 ≤ coefficient p t (run p oracle point t) ∧
      coefficient p t (run p oracle point t) ≤ 1 :=
  coefficient_bounds _ _ _ (run_total_nonneg p oracle point t)

/-- Nonnegative initial maxima remain nonnegative for every actual update. -/
theorem run_maximum_nonneg (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (t : ℕ) (i : ι) : 0 ≤ (run p oracle point t).maximum i := by
  induction t with
  | zero => simp [run, initial]
  | succ t ih => exact maximum_nonneg id p t _ _ i ih

end Transformer.ScheduleFree.AMSGrad

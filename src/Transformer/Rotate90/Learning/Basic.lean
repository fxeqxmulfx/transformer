module

public import Transformer.Rotate90.Learning.Defs
public import Transformer.Landscape.SaddleAvoidance.Sublevel

/-!
# Source update, softmax calibration and finite-step curvature

Source: `review/flat_spectrum.md`, *Learning transfer*, T2--T5.
The complete Schedule-Free evaluation displacement includes averaging and
query decay. Full-softmax loss bounds wrong winners and ties. C² regularity
provides finite segment-curvature constants, but not useful numerical values.
The conservative Taylor coefficient is L, as in the proved mean-value lemma.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Learning

open Landscape.Neural Landscape.SaddleAvoidance ScheduleFree
open scoped RealInnerProductSpace BigOperators

section Optimizer

variable {I : Type*}

/-- T2: evaluation movement is averaging drift minus the scaled Adam direction
and query decay. The only extra condition is the source eval() division. -/
theorem evaluation_displacement (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ) :
    evaluated p (AdamW.trainingStep p t s g) - evaluated p s =
      AdamW.coefficient p t s.history • (WithLp.toLp 2 s.base - evaluated p s) -
      (AdamW.coefficient p t s.history * AdamW.rate p t) • normalized p t s g -
      (AdamW.coefficient p t s.history * AdamW.rate p t * p.adam.decay) •
        WithLp.toLp 2 s.point := by
  ext i
  change AdamW.evaluation p (AdamW.trainingStep p t s g) i - AdamW.evaluation p s i = _
  simp only [evaluated, normalized, AdamW.evaluation, AdamW.trainingStep, AdamW.direction,
    PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
  field_simp
  ring

variable [Fintype I]

/-- T3: the complete first-order effect on any evaluation loss. The gradient
of that loss is at x; the supplied training gradient may be at the query y. -/
theorem directional_transfer (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ) (F : EuclideanSpace ℝ I → ℝ) :
    ⟪gradient F (evaluated p s),
      evaluated p (AdamW.trainingStep p t s g) - evaluated p s⟫ =
      AdamW.coefficient p t s.history *
        ⟪gradient F (evaluated p s), WithLp.toLp 2 s.base - evaluated p s⟫ -
      (AdamW.coefficient p t s.history * AdamW.rate p t) *
        ⟪gradient F (evaluated p s), normalized p t s g⟫ -
      (AdamW.coefficient p t s.history * AdamW.rate p t * p.adam.decay) *
        ⟪gradient F (evaluated p s), WithLp.toLp 2 s.point⟫ := by
  rw [evaluation_displacement p hβ, inner_sub_right, inner_sub_right]
  simp only [inner_smul_right]

example := evaluation_displacement (I := Fin 2)
  ⟨⟨3 / 1000, 1, 0, 9 / 10, 98 / 100, 1 / 100000000⟩, 0, 2⟩ (by norm_num)
  4 ⟨fun i => i.val + 1, fun i => 2 * i.val + 1,
    ⟨fun _ => 1 / 10, 4 * (3 / 1000) ^ 2, 3 / 1000⟩⟩ (fun i => i.val - 1)
example := directional_transfer (I := Fin 2)
  ⟨⟨3 / 1000, 1, 0, 9 / 10, 98 / 100, 1 / 100000000⟩, 0, 2⟩ (by norm_num)
  4 ⟨fun i => i.val + 1, fun i => 2 * i.val + 1,
    ⟨fun _ => 1 / 10, 4 * (3 / 1000) ^ 2, 3 / 1000⟩⟩ (fun i => i.val - 1)
  (fun x => ‖x‖ ^ 2)

end Optimizer

section Calibration

variable {V X : Type*} [Fintype V]

/-- T5, one row: an incorrect winner or tie costs at least log 2. -/
theorem log_two_le_crossEntropy (logits : V → ℝ) (target other : V)
    (hne : target ≠ other) (hscore : logits target ≤ logits other) :
    Real.log 2 ≤ crossEntropy logits target := by
  classical
  have hsum : 2 * Real.exp (logits target) ≤ ∑ q, Real.exp (logits q) := by
    have hpair := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.subset_univ ({target, other} : Finset V))
      (fun q _ _ => (Real.exp_pos (logits q)).le)
    rw [Finset.sum_pair hne] at hpair
    have hexp := Real.exp_le_exp.mpr hscore
    linarith
  have hlog := Real.log_le_log (mul_pos (by norm_num : (0 : ℝ) < 2)
    (Real.exp_pos (logits target))) hsum
  rw [Real.log_mul (by norm_num) (Real.exp_pos _).ne', Real.log_exp] at hlog
  unfold crossEntropy
  linarith

/-- Failed strict-winner rows in a given finite evaluation split. -/
noncomputable def failures (rows : Finset X) (logits : X → V → ℝ) (target : X → V) :
    Finset X := by
  classical
  exact rows.filter (fun r => ∃ q, q ≠ target r ∧ logits r (target r) ≤ logits r q)

/-- T5, finite split: count times log 2 is bounded by total full-softmax loss. -/
theorem failures_mul_log_two_le (rows : Finset X) (logits : X → V → ℝ) (target : X → V) :
    (failures rows logits target).card * Real.log 2 ≤
      ∑ r ∈ rows, crossEntropy (logits r) (target r) := by
  classical
  have h := Finset.sum_le_sum (s := failures rows logits target)
    (f := fun _ => Real.log 2) (g := fun r => crossEntropy (logits r) (target r))
    (fun r hr => by
      obtain ⟨q, hq, hscore⟩ := (Finset.mem_filter.mp hr).2
      exact log_two_le_crossEntropy (logits r) (target r) q hq.symm hscore)
  simp only [Finset.sum_const, nsmul_eq_mul] at h
  exact h.trans (Finset.sum_le_sum_of_subset_of_nonneg
    (Finset.filter_subset _ _) (fun r _ _ => crossEntropy_nonneg (logits r) (target r)))

/-- T5: failure fraction at most mean CE / log 2, also valid for an empty
split by total division. Experiments use nonempty heldout splits. -/
theorem failure_fraction_le_loss (rows : Finset X) (logits : X → V → ℝ) (target : X → V) :
    (failures rows logits target).card / (rows.card : ℝ) ≤
      meanCrossEntropy rows logits target / Real.log 2 := by
  have h := div_le_div_of_nonneg_right (failures_mul_log_two_le rows logits target)
    (Nat.cast_nonneg rows.card : (0 : ℝ) ≤ rows.card)
  rw [le_div_iff₀ (Real.log_pos (by norm_num : (1 : ℝ) < 2))]
  unfold meanCrossEntropy
  calc
    (↑(failures rows logits target).card / ↑rows.card) * Real.log 2 =
        (↑(failures rows logits target).card * Real.log 2) / ↑rows.card := by ring
    _ ≤ _ := h

example := log_two_le_crossEntropy (fun i : Fin 3 => (i.val : ℝ)) 0 2 (by decide) (by norm_num)
example := failures_mul_log_two_le (Finset.univ : Finset (Fin 4))
  (fun r q : Fin 4 => if r = q then 0 else 1) (fun r => r)
example := failure_fraction_le_loss (Finset.univ : Finset (Fin 4))
  (fun r q : Fin 4 => if r = q then 0 else 1) (fun r => r)

end Calibration

section Curvature

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- Every finite segment of a C² finite-dimensional model admits a finite
curvature bound. Existence does not supply a numerically certified constant. -/
theorem exists_segment_curvature {F : E → ℝ} (hF : ContDiff ℝ 2 F) (x y : E) :
    ∃ L : ℝ, 0 ≤ L ∧ ∀ w ∈ segment ℝ x y, ‖fderiv ℝ (gradient F) w‖ ≤ L := by
  have hc := (contDiff_gradient hF).continuous_fderiv (by norm_num)
  obtain ⟨L, hL⟩ := (isCompact_closedBall x (dist x y)).exists_bound_of_continuousOn
    hc.continuousOn
  refine ⟨max L 0, le_max_right _ _, fun w hw => ?_⟩
  exact (hL w (segment_subset_closedBall_left x y hw)).trans (le_max_left _ _)

/-- T4: a genuine segment-curvature bound controls the finite step remainder. -/
theorem segment_loss_le {F : E → ℝ} (hF : ContDiff ℝ 2 F) (x y : E) (L : ℝ)
    (hL : ∀ w ∈ segment ℝ x y, ‖fderiv ℝ (gradient F) w‖ ≤ L) :
    F y ≤ F x + ⟪gradient F x, y - x⟫ + L * ‖y - x‖ ^ 2 := by
  exact le_add_inner_add_mul_norm_sq_of_convex hF (convex_segment x y) hL
    (left_mem_segment ℝ x y) (right_mem_segment ℝ x y)

example := exists_segment_curvature (F := fun u : ℝ => u ^ 2) (by fun_prop) 1 2
example := segment_loss_le (F := fun u : ℝ => u) (by fun_prop) 1 (1 / 2) 0
  (by
    intro w hw
    have hg : gradient (fun u : ℝ => u) = fun _ => 1 := by funext u; simp [gradient_eq_deriv]
    rw [hg]; simp)

end Curvature

end Transformer.Rotate90.Learning

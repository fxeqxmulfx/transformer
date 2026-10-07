import Transformer.Grokking.GradientEvidence.BinaryObjectives
import Mathlib.Analysis.Real.Sqrt

/-!
# Almost parallel full gradients with opposed answer gradients

Source objective and measurement: Power et al.'s mean answer/EOS CE,
ported as ModularTask.loss at 43d4d66; fixed-batch component diagnostics
at d5bf4b4. This is a mathematical counterexample for interpreting that
measurement, using BinaryObjectives' actual softmax loss derivatives.

Deviation: binary logits and two independent readout coordinates, with a
shared input feature of scale `R`. Opposite answer labels have gradients
of opposite sign, yet the full mean-loss cosine is `(R² - 1)/(R² + 1)`.
For every positive tolerance it can exceed one minus that tolerance.
Both initialized losses remain `log 2` for every shared feature scale.

The signed energy and dot-product decompositions match the experimental
reader. Shared supervision can conceal conflict; this is not a theorem
that a particular EOS token explains every observed transformer gradient.
-/

namespace Transformer.Grokking.GradientEvidence

/-- Euclidean dot product used by the two-coordinate counterexample.
Source: fixed-batch gradient cosine protocol at d5bf4b4, dimension two. -/
def gradientDot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

/-- Squared Euclidean gradient norm. Source: component-gradient protocol
at d5bf4b4, specialized to the two actual readout coordinates. -/
def gradientNormSq (a : ℝ × ℝ) : ℝ := a.1 ^ 2 + a.2 ^ 2

/-- Mean of two supervised-position gradients. Source: ModularTask.loss
at 43d4d66; the mean weighting is exactly one half per position. -/
noncomputable def averageGradient (a e : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + e.1) / 2, (a.2 + e.2) / 2)

/-- Actual dot-over-norm-product cosine; its initial denominator is
proved positive rather than assumed. Source: gradient protocol at d5bf4b4. -/
noncomputable def gradientCosine (a b : ℝ × ℝ) : ℝ :=
  gradientDot a b / (Real.sqrt (gradientNormSq a) * Real.sqrt (gradientNormSq b))

/-- Averaging does not make component gradients orthogonal. Source:
the signed energy decomposition in objective_components.py at d5bf4b4;
the cross term must be retained. -/
theorem average_energy_identity (a e : ℝ × ℝ) :
    gradientNormSq (averageGradient a e) =
      gradientNormSq a / 4 + gradientNormSq e / 4 + gradientDot a e / 2 := by
  unfold gradientNormSq averageGradient gradientDot
  dsimp
  ring

/-- Split-mean alignment has two diagonal and two cross-component terms.
Source: objective_components.split_dot_terms at d5bf4b4; these terms
can be signed and are not individually cosines or energy fractions. -/
theorem average_dot_identity (a e b f : ℝ × ℝ) :
    gradientDot (averageGradient a e) (averageGradient b f) =
      (gradientDot a b + gradientDot a f + gradientDot e b + gradientDot e f) / 4 := by
  unfold gradientDot averageGradient
  dsimp
  ring

/-- Concrete cancellation rejects nonnegative attribution of both
component energies to the full gradient. Source: the signed component
protocol and cancellation test at d5bf4b4. -/
theorem negative_cross_term_counterexample :
    gradientNormSq (averageGradient (1, 0) (-1, 0)) = 0 ∧
      gradientDot (1, 0) (-1, 0) = -1 ∧
      0 < gradientNormSq (1, 0) + gradientNormSq (-1, 0) := by
  unfold gradientNormSq averageGradient gradientDot
  norm_num

/-- Exact full-gradient cosine from the actual binary softmax objective.
Source: the shared-target counterexample developed from ModularTask's
CE mean at 43d4d66; isolated answers remain opposed by BinaryObjectives. -/
theorem initial_softmax_cosine (R : ℝ) :
    gradientCosine (objectiveGradient true R 0 0) (objectiveGradient false R 0 0) =
      (R ^ 2 - 1) / (R ^ 2 + 1) := by
  have ha : gradientNormSq (objectiveGradient true R 0 0) = (1 + R ^ 2) / 16 :=
    (initial_gradient_energy true R).1
  have hb : gradientNormSq (objectiveGradient false R 0 0) = (1 + R ^ 2) / 16 :=
    (initial_gradient_energy false R).1
  unfold gradientCosine
  rw [ha, hb, Real.mul_self_sqrt (initial_gradient_energy true R).2.le,
    (objectiveGradient_initial R).1, (objectiveGradient_initial R).2]
  unfold gradientDot
  dsimp
  have hn : R ^ 2 + 1 ≠ 0 := by positivity
  field_simp
  ring

/-- The deficit from perfect alignment is exactly inverse quadratic in
the common feature scale. Source: the actual-CE counterexample above;
no normalization-dependent criterion is put into the loss definition. -/
theorem initial_cosine_deficit (R : ℝ) :
    1 - gradientCosine (objectiveGradient true R 0 0) (objectiveGradient false R 0 0) =
      2 / (R ^ 2 + 1) := by
  rw [initial_softmax_cosine]
  have hn : R ^ 2 + 1 ≠ 0 := by positivity
  field_simp
  ring

/-- Removing the shared feature exposes exactly opposite full gradients
without changing either initial loss. Source: the actual-CE shared-target
counterexample; it contrasts the same initialized logits at two scales. -/
theorem no_shared_target_retains_conflict :
    gradientCosine (objectiveGradient true 0 0 0) (objectiveGradient false 0 0 0) = -1 ∧
      pairedObjective true 0 0 0 = Real.log 2 ∧ pairedObjective false 0 0 0 = Real.log 2 := by
  rw [initial_softmax_cosine, initial_loss_independent_of_shared_scale,
    initial_loss_independent_of_shared_scale]
  norm_num

/-- An explicit scale suffices to make full gradients almost parallel.
Source: the actual-CE shared-target counterexample; the tolerance and
scale condition are stated, not fitted to transformer results. -/
theorem initial_cosine_gt_threshold (eps R : ℝ) (he : 0 < eps)
    (hR : 2 / eps ≤ R ^ 2) :
    1 - eps < gradientCosine (objectiveGradient true R 0 0) (objectiveGradient false R 0 0) := by
  rw [initial_softmax_cosine]
  apply (lt_div_iff₀ (by positivity : 0 < R ^ 2 + 1)).mpr
  have hp := (div_le_iff₀ he).mp hR
  nlinarith

example : 0 < (1 / 1000 : ℝ) ∧ 2 / (1 / 1000 : ℝ) ≤ (100 : ℝ) ^ 2 := by norm_num

/-- Arbitrarily strong full alignment is compatible with opposed
answer-only gradients. Source: the binary softmax specialization of the
answer/EOS protocol at 43d4d66; this refutes a task-gradient interpretation. -/
theorem arbitrarily_high_alignment_with_opposed_answers (eps : ℝ) (he : 0 < eps) :
    ∃ R : ℝ, 0 < R ∧
      deriv (binaryCE true) 0 * deriv (binaryCE false) 0 = -1 / 4 ∧
      1 - eps < gradientCosine (objectiveGradient true R 0 0) (objectiveGradient false R 0 0) := by
  have hp : 0 < 2 / eps := div_pos (by norm_num) he
  refine ⟨Real.sqrt (2 / eps), Real.sqrt_pos.mpr hp, isolated_answers_opposed.2.2, ?_⟩
  apply initial_cosine_gt_threshold eps _ he
  rw [Real.sq_sqrt hp.le]

example : 0 < (1 / 1000 : ℝ) := by norm_num

/-- The same example retains both initial losses while changing the
cosine arbitrarily close to one. Source: the actual-CE counterexample;
a shared target supplies alignment without resolving the answer conflict. -/
theorem high_alignment_at_the_same_loss (eps : ℝ) (he : 0 < eps) :
    ∃ R : ℝ, 0 < R ∧ pairedObjective true R 0 0 = Real.log 2 ∧
      pairedObjective false R 0 0 = Real.log 2 ∧
      deriv (binaryCE true) 0 * deriv (binaryCE false) 0 = -1 / 4 ∧
      1 - eps < gradientCosine (objectiveGradient true R 0 0) (objectiveGradient false R 0 0) := by
  obtain ⟨R, hR, hc, ha⟩ := arbitrarily_high_alignment_with_opposed_answers eps he
  exact ⟨R, hR, initial_loss_independent_of_shared_scale true R,
    initial_loss_independent_of_shared_scale false R, hc, ha⟩

example : 0 < (1 / 1000 : ℝ) := by norm_num

end Transformer.Grokking.GradientEvidence

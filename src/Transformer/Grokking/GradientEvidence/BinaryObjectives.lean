import Transformer.Grokking.NaiveLoss.Section4_LogitScaling
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# Actual softmax objectives with a shared supervised target

Source objective: Power et al.'s `openai/grok` at commit
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, as ported to
`lab.infrastructure.benchmarks.modular.ModularTask` at 43d4d66. Training
averages answer and EOS cross-entropies. The separate observations in
`experiments/grokking_internals/objective_components.py` retain that loss.

Deviation for this mathematical counterexample: two classes and two
independent real readout parameters. The answer logit is `u`; the common
target logit is `R * v`. Opposite answer labels share the same second
target. The loss uses the actual finite-class softmax cross-entropy, not
a quadratic surrogate or a definition of desirable gradient alignment.

All partial derivatives are proved. Initial loss is independent of the
shared feature scale, but its gradient need not be. This model does not
assert that the full transformer has these two independent coordinates.
-/

namespace Transformer.Grokking.GradientEvidence

/-- Binary softmax CE with score logits `(score, 0)`. A true label selects
class zero. Source: Power et al.'s supervised CE, binary specialization;
the finite-class definition is checked in the NaiveLoss formalization. -/
noncomputable def binaryCE (label : Bool) (score : ℝ) : ℝ :=
  NaiveLoss.crossEntropy (fun k : Fin 2 => if k = 0 then score else 0)
    (if label then 0 else 1)

/-- Answer and common-target CE averaged in the original protocol's
proportions. Source: ModularTask.loss at 43d4d66; deviation: two scalar
readout parameters and a fixed shared input feature `R`. -/
noncomputable def pairedObjective (label : Bool) (R u v : ℝ) : ℝ :=
  (binaryCE label u + binaryCE true (R * v)) / 2

/-- Actual coordinate derivatives of the paired softmax objective.
Source: the offline component-gradient protocol at d5bf4b4, specialized
to the binary readout above; no desired derivative is inserted here. -/
noncomputable def objectiveGradient (label : Bool) (R u v : ℝ) : ℝ × ℝ :=
  (deriv (fun s => pairedObjective label R s v) u,
    deriv (fun s => pairedObjective label R u s) v)

/-- The binary specialization is exactly ordinary log-sum-exp CE.
Source: standard supervised softmax loss in ModularTask.loss at 43d4d66;
the target subtraction is retained for each label. -/
theorem binaryCE_eq_standard (label : Bool) (score : ℝ) :
    binaryCE label score = Real.log (Real.exp score + 1) - (if label then score else 0) := by
  unfold binaryCE
  rw [NaiveLoss.crossEntropy_eq_standard, Fin.sum_univ_two]
  cases label <;> norm_num

/-- Differentiate the actual binary CE rather than assuming a gradient.
Source: ModularTask.loss at 43d4d66, two-class specialization. -/
theorem binaryCE_deriv (label : Bool) (score : ℝ) :
    HasDerivAt (binaryCE label)
      (Real.exp score / (Real.exp score + 1) - (if label then 1 else 0)) score := by
  have hf : HasDerivAt (fun s : ℝ => Real.log (Real.exp s + 1))
      (Real.exp score / (Real.exp score + 1)) score :=
    ((Real.hasDerivAt_exp score).add_const 1).log
      (by have hp := Real.exp_pos score; positivity)
  have ht : HasDerivAt (fun s : ℝ => (if label then 1 else 0) * s)
      (if label then 1 else 0) score := by
    simpa using (hasDerivAt_id score).const_mul (if label then (1 : ℝ) else 0)
  convert hf.sub ht using 1
  funext s
  rw [binaryCE_eq_standard]
  cases label <;> simp

/-- Answer-coordinate derivative under the unchanged mean weighting.
Source: ModularTask's answer/EOS CE mean at 43d4d66; binary readout. -/
theorem pairedObjective_deriv_answer (label : Bool) (R u v : ℝ) :
    HasDerivAt (fun s => pairedObjective label R s v)
      ((Real.exp u / (Real.exp u + 1) - (if label then 1 else 0)) / 2) u := by
  simpa only [pairedObjective] using
    ((binaryCE_deriv label u).add_const (binaryCE true (R * v))).div_const 2

/-- Common-target derivative carries the shared feature scale. Source:
ModularTask's mean weighting at 43d4d66; the `R` factor comes from this
counterexample's actual readout map, through the proved chain rule. -/
theorem pairedObjective_deriv_shared (label : Bool) (R u v : ℝ) :
    HasDerivAt (fun s => pairedObjective label R u s)
      (R * (Real.exp (R * v) / (Real.exp (R * v) + 1) - 1) / 2) v := by
  have hf := ((binaryCE_deriv true (R * v)).comp v
    ((hasDerivAt_id v).const_mul R)).const_add (binaryCE label u)
  convert hf.div_const 2 using 1
  · funext s
    rfl
  · dsimp
    ring

/-- The partial derivatives reconstruct the exact gradient. Source:
component-gradient protocol at d5bf4b4; the two coordinates are disjoint
in this readout, unlike potentially overlapping transformer parameters. -/
theorem objectiveGradient_eq (label : Bool) (R u v : ℝ) :
    objectiveGradient label R u v =
      ((Real.exp u / (Real.exp u + 1) - (if label then 1 else 0)) / 2,
        R * (Real.exp (R * v) / (Real.exp (R * v) + 1) - 1) / 2) := by
  unfold objectiveGradient
  rw [(pairedObjective_deriv_answer label R u v).deriv,
    (pairedObjective_deriv_shared label R u v).deriv]

/-- At initialization the opposite answers differ only in the first
gradient coordinate; the shared target supplies the same second one.
Source: the softmax shared-target counterexample derived above. -/
theorem objectiveGradient_initial (R : ℝ) :
    objectiveGradient true R 0 0 = (-1 / 4, -R / 4) ∧
      objectiveGradient false R 0 0 = (1 / 4, -R / 4) := by
  rw [objectiveGradient_eq, objectiveGradient_eq]
  norm_num
  ring

/-- Both initial gradients are nonzero and have equal squared norm.
Source: the two-class CE specialization of the component protocol at
d5bf4b4. This proves the cosine denominator is positive for every scale. -/
theorem initial_gradient_energy (label : Bool) (R : ℝ) :
    (objectiveGradient label R 0 0).1 ^ 2 + (objectiveGradient label R 0 0).2 ^ 2 =
      (1 + R ^ 2) / 16 ∧ 0 < (1 + R ^ 2) / 16 := by
  rw [objectiveGradient_eq]
  dsimp
  constructor
  · cases label <;> norm_num <;> ring
  · positivity

/-- The isolated answer gradients are opposed at the same point.
Source: the binary softmax specialization of answer-only CE used by the
component measurement at d5bf4b4; it excludes the common target. -/
theorem isolated_answers_opposed :
    deriv (binaryCE true) 0 = -1 / 2 ∧ deriv (binaryCE false) 0 = 1 / 2 ∧
      deriv (binaryCE true) 0 * deriv (binaryCE false) 0 = -1 / 4 := by
  rw [(binaryCE_deriv true 0).deriv, (binaryCE_deriv false 0).deriv]
  norm_num

/-- The same initialized CE loss can have any shared feature scale.
Source: the counterexample's actual softmax objective above, following
ModularTask's mean weighting. A scalar loss alone cannot reveal this scale. -/
theorem initial_loss_independent_of_shared_scale (label : Bool) (R : ℝ) :
    pairedObjective label R 0 0 = Real.log 2 := by
  unfold pairedObjective
  rw [binaryCE_eq_standard, binaryCE_eq_standard]
  cases label <;> norm_num

example : pairedObjective true 100 0 0 = pairedObjective true 1 0 0 := by
  rw [initial_loss_independent_of_shared_scale, initial_loss_independent_of_shared_scale]

end Transformer.Grokking.GradientEvidence

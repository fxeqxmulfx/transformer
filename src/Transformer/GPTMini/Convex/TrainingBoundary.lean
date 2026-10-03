/-
# A training obstruction that a convex attention row does not remove

The remaining `ReLU2_FFN.forward` in
`experiments/archive/gpt_mini/gpt_mini.py` has two trainable linear maps. An
exact one-neuron instance below violates Jensen's
inequality for binary cross-entropy, even with fixed input and positive
weights. This is a counterexample in the original weight coordinates,
not an impossibility result for a different parameterization or penalty.

Source context: arXiv:2211.11052v1, §3.4's gated-ReLU relaxation and §4's
stacking experiments do not establish convexity for this ReLU² FFN.
-/

import Transformer.GPTMini.Convex.Model
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section

namespace Transformer.GPTMini.Convex

/-- One neuron with trainable input and output weights and input `1`,
defined using the original `relu2FFN`, not a surrogate network.
Source: `ReLU2_FFN.forward` in `experiments/archive/gpt_mini/gpt_mini.py`. -/
def oneNeuron (weights : ℝ × ℝ) : ℝ :=
  (relu2FFN
    (weights.1 • ContinuousLinearMap.id ℝ (EucSpace 1))
    (weights.2 • ContinuousLinearMap.id ℝ (EucSpace 1))
    (EuclideanSpace.single (0 : Fin 1) 1)) 0

/-- The actual scalar FFN output is `w_out * max(0,w_in)²`.
Source: `ReLU2_FFN.forward` in `experiments/archive/gpt_mini/gpt_mini.py`. -/
theorem oneNeuron_formula (weights : ℝ × ℝ) :
    oneNeuron weights = weights.2 * relu2 weights.1 := by
  simp [oneNeuron, relu2FFN, relu2Vec_apply, PiLp.smul_apply, smul_eq_mul]

/-- Binary cross-entropy for logits `(oneNeuron weights, 0)` and target
class `1`. The second logit is zero, so the target-logit subtraction is
zero. Source: new training test of `ReLU2_FFN.forward`; §3.4 context. -/
def oneNeuronCELoss (weights : ℝ × ℝ) : ℝ :=
  Real.log (1 + Real.exp (oneNeuron weights))

/-- Both positive parameter assignments give logit one. Source: exact
counterexample for `ReLU2_FFN.forward`, beyond §3.4's gated-ReLU setting. -/
theorem oneNeuron_endpoints :
    oneNeuron (1, 1) = 1 ∧ oneNeuron (2, 1 / 4) = 1 := by
  simp only [oneNeuron_formula, relu2]
  norm_num

/-- The parameter midpoint has logit `45/32`, larger than both endpoints.
Source: exact counterexample for `ReLU2_FFN.forward`, §3.4 context. -/
theorem oneNeuron_midpoint :
    oneNeuron ((1 / 2 : ℝ) • (1, 1) + (1 / 2 : ℝ) • (2, 1 / 4)) = 45 / 32 := by
  simp only [oneNeuron_formula, Prod.smul_fst, Prod.smul_snd,
    Prod.fst_add, Prod.snd_add, smul_eq_mul, relu2]
  norm_num

/-- Exact Jensen violation for a trainable FFN under cross-entropy.
Source: `ReLU2_FFN.forward`; this failure survives replacing the attention
row and is outside the single-block gated-ReLU result of §3.4. -/
theorem oneNeuronCE_jensen_violation :
    (oneNeuronCELoss (1, 1) + oneNeuronCELoss (2, 1 / 4)) / 2 <
      oneNeuronCELoss ((1 / 2 : ℝ) • (1, 1) + (1 / 2 : ℝ) • (2, 1 / 4)) := by
  have he := oneNeuron_endpoints
  rw [oneNeuronCELoss, oneNeuronCELoss, oneNeuronCELoss,
    he.1, he.2, oneNeuron_midpoint]
  have hlog : Real.log (1 + Real.exp 1) < Real.log (1 + Real.exp (45 / 32)) := by
    apply Real.log_lt_log (by positivity)
    simpa only [add_comm] using add_lt_add_left
      (Real.exp_lt_exp.mpr (by norm_num : (1 : ℝ) < 45 / 32)) 1
  linarith

/-- The original FFN weights do not make this training objective convex,
even though each new attention inference problem is convex.
This does not rule out a lifted formulation or a different regularizer.
Source: `ReLU2_FFN.forward`; arXiv:2211.11052v1, §3.4/§4 scope boundary. -/
theorem oneNeuronCE_not_convex :
    ¬ ConvexOn ℝ (Set.univ : Set (ℝ × ℝ)) oneNeuronCELoss := by
  intro h
  have hj := h.2 (Set.mem_univ (1, 1)) (Set.mem_univ (2, 1 / 4))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hv := oneNeuronCE_jensen_violation
  simp only [smul_eq_mul] at hj
  linarith

end Transformer.GPTMini.Convex

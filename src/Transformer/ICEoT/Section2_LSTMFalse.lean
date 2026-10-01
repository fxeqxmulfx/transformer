/-
# IC-EoT: refutation of the cited IC-LSTM sequence-convexity claim

arXiv:2603.22095v2, §2.2.3, the conditions following Eqs. (4)–(11).
Those conditions do not guarantee joint convexity in the input sequence.
The shared-latent product argument within one time step does not apply
to the product of a new forget gate with the preceding cell state.
This does not refute §3.3's IC-EoT Theorem 1 or the conditional Corollary 2.
-/

import Transformer.ICEoT.Section2_LSTMModel

noncomputable section

namespace Transformer.ICEoT

/-- First three-step history for the §2.2.3 counterexample: `[0,2,4]`. -/
def lstmHistoryA : (Fin 3 × Fin 1) → ℝ :=
  fun ir => if ir.1 = 0 then 0 else if ir.1 = 1 then 2 else 4

/-- Second history for the §2.2.3 counterexample: `[0,4,2]`. -/
def lstmHistoryB : (Fin 3 × Fin 1) → ℝ :=
  fun ir => if ir.1 = 0 then 0 else if ir.1 = 1 then 4 else 2

/-- Exact evaluations of all three cell updates, the dense skip and readout;
§2.2.3, Eqs. (4)–(11). The midpoint gives `16 > (17+13)/2`. -/
theorem counterLSTM_values :
    lstmThreePredict counterLSTM lstmHistoryA 0 = 17 ∧
    lstmThreePredict counterLSTM lstmHistoryB 0 = 13 ∧
    lstmThreePredict counterLSTM
      ((1 / 2 : ℝ) • lstmHistoryA + (1 / 2 : ℝ) • lstmHistoryB) 0 = 16 := by
  norm_num [lstmThreePredict, lstmStep, lstmReadout, counterLSTM, affine,
    lstmExpand, positiveRelu, lstmHistoryA, lstmHistoryB,
    Fintype.sum_prod_type, Fintype.sum_bool]

/-- Refutes §2.2.3's claim that the listed shared-weight, non-negative
scaling and convex-monotone-positive activation conditions suffice for
input-sequence convexity. A one-layer, three-step model satisfies every
listed condition, with zero initial states, but violates Jensen at the
midpoint of `[0,2,4]` and `[0,4,2]`. All internal activations are ≥ 1,
so the refutation also covers a strictly positive reading of the source.
The conditional MPC result remains valid when its convexity premise is met.
Source: arXiv:2603.22095v2, §2.2.3, Eqs. (4)–(11). -/
theorem iclstm_paper_convexity_false : ICLSTMConditions counterLSTM ∧
    ¬ ComponentwiseConvex (lstmThreePredict counterLSTM) := by
  refine ⟨counterLSTM_conditions, ?_⟩
  intro h
  have hj := h.2 (Set.mem_univ lstmHistoryA) (Set.mem_univ lstmHistoryB)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1) 0
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    counterLSTM_values.1, counterLSTM_values.2.1, counterLSTM_values.2.2] at hj
  norm_num at hj

end Transformer.ICEoT

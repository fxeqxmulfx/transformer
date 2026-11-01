module

public import Transformer.SequenceModel.Attention.Backend.Average

/-!
# Connect checked positive masses to the source causal SDPA contract

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K4.
The original causal mask, including the current token, is retained. The
conclusion is exactly the average field of Execution.AttentionAccurate,
not a separate softmax model or a presumed complete attention-branch error.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open scoped BigOperators

/-- Exact exponential masses with the actual causal mask. -/
def causalMass {T : ℕ} (scores : Fin T → Fin T → ℝ) (i j : Fin T) : ℝ :=
  if j ≤ i then Real.exp (scores i j) else 0

/-- K4: the masked mass sum is exactly the source softmax denominator. -/
theorem causalMass_sum {T : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T) :
    ∑ j, causalMass scores i j = ∑ j ∈ Finset.Iic i, Real.exp (scores i j) := by
  classical
  have hs : (Finset.univ.filter (fun j : Fin T => j ≤ i)) = Finset.Iic i := by ext j; simp
  simp only [causalMass, ← Finset.sum_filter, hs]

/-- K4: the causal denominator is positive because the current token is visible. -/
theorem causalMass_sum_pos {T : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T) :
    0 < ∑ j, causalMass scores i j := by
  have hn (j : Fin T) : 0 ≤ causalMass scores i j := by unfold causalMass; split_ifs <;> positivity
  have hle := Finset.single_le_sum (fun j _ => hn j) (Finset.mem_univ i)
  have hp : 0 < causalMass scores i i := by simp [causalMass, Real.exp_pos]
  exact hp.trans_le hle

/-- K4: normalizing the masked masses recovers every source causal weight. -/
theorem causalMass_weight {T : ℕ} (scores : Fin T → Fin T → ℝ) (i j : Fin T) :
    causalMass scores i j / (∑ k, causalMass scores i k) = causalWeight scores i j := by
  rw [causalMass_sum]
  by_cases h : j ≤ i <;> simp [causalMass, causalWeight, h]

/-- K4: exponential certificates and a rational reduction residual fill
the source SDPA operation contract directly, at the recorded scores and
values. No claim is made about a fused kernel's internal summation order. -/
theorem causal_average_residual {T d : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T)
    (a : Fin T → ℝ) (v : Fin T → EucSpace d) (z : EucSpace d)
    (hA : 0 < ∑ j, a j) (E M β : ℝ) (hM : 0 ≤ M)
    (he : ∑ j, |a j - causalMass scores i j| ≤ E) (hv : ∀ j, ‖v j‖ ≤ M)
    (hz : ‖z - ∑ j, (a j / (∑ k, a k)) • v j‖ ≤ β) :
    ‖z - ∑ j, causalWeight scores i j • v j‖ ≤ β + (2 * E / (∑ j, a j)) * M := by
  have h := average_residual_error a (causalMass scores i) v z hA (causalMass_sum_pos scores i)
    (by intro j; unfold causalMass; split_ifs <;> positivity) E M β hM he hv hz
  simpa only [causalMass_weight] using h

example := causal_average_residual (!![(0 : ℝ), 1; 0, 0]) 1 ![1, 1]
  (fun i : Fin 2 => EuclideanSpace.single i 1)
  ((1 / 2 : ℝ) • EuclideanSpace.single 0 1 + (1 / 2 : ℝ) • EuclideanSpace.single 1 1 : EucSpace 2)
  (by norm_num [Fin.sum_univ_succ]) 0 1 0 (by norm_num)
  (by norm_num [causalMass, Fin.sum_univ_succ]) (by intro j; simp)
  (by norm_num [Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Backend

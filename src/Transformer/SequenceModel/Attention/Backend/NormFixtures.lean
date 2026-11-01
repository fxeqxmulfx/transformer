module

public import Transformer.SequenceModel.Attention.Backend.Normalization

/-!
# Normalization certificates for the captured CPU and CUDA fixtures

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K6;
`review/backend_contract_check.py`. The input and epsilons match the capture.
The output boxes are checked with exact rational arithmetic in Python. Lean
proves every output in each box satisfies the stated local operation bound.
This statement does not assert that all future kernel executions enter it.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open GPTMini Stability

/-- The shared four-coordinate input of the captured normalization operations. -/
def normInput : EucSpace 4 := WithLp.toLp 2 ![1, 2, 3, 4]

/-- K6: exact squared norm used in the rational scale-residual checks. -/
theorem normInput_sq : ‖normInput‖ ^ 2 = 30 := by
  norm_num [EuclideanSpace.real_norm_sq_eq, normInput, Fin.sum_univ_succ]

/-- K6: every captured RMS output in the checked coordinate box has
Euclidean error at most 4e-6, at the source float32 RMS epsilon. -/
theorem rms_fixture (z : EucSpace 4)
    (hz : ∀ c, |z c - (36514837 / 100000000 : ℝ) * normInput c| ≤ 1 / 1000000) :
    ‖z - rmsNormEps (1 / 8388608) normInput‖ ≤ 4 / 1000000 := by
  have hm : ‖z - (36514837 / 100000000 : ℝ) • normInput‖ ≤ 2 / 1000000 := by
    have h := norm_le_coordinates (z - (36514837 / 100000000 : ℝ) • normInput)
      (1 / 1000000) (by norm_num) hz
    norm_num at h ⊢
    exact h
  have h := rms_residual_error (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    normInput z (36514837 / 100000000) (2 / 1000000) (1 / 1000000) (by norm_num)
    (by rw [normInput_sq]; norm_num) hm
  norm_num at h ⊢
  exact h

/-- K6: every captured L2 output in its checked box has Euclidean error
at most 3e-6, with the source clipping epsilon retained. -/
theorem l2_fixture (z : EucSpace 4)
    (hz : ∀ c, |z c - (18257419 / 100000000 : ℝ) * normInput c| ≤ 1 / 1000000) :
    ‖z - normL2 (1 / 1000000000000) normInput‖ ≤ 3 / 1000000 := by
  have hn : (1 / 1000000000000 : ℝ) ≤ ‖normInput‖ := by
    have hs := normInput_sq
    have hp := norm_nonneg normInput
    nlinarith
  have hm : ‖z - (18257419 / 100000000 : ℝ) • normInput‖ ≤ 2 / 1000000 := by
    have h := norm_le_coordinates (z - (18257419 / 100000000 : ℝ) • normInput)
      (1 / 1000000) (by norm_num) hz
    norm_num at h ⊢
    exact h
  have h := l2_residual_error (eps := 1 / 1000000000000) (by norm_num)
    normInput z (18257419 / 100000000) (2 / 1000000) (1 / 1000000) (by norm_num)
    (by rw [max_eq_left hn, normInput_sq]; norm_num) hm
  norm_num at h ⊢
  exact h

example := rms_fixture ((36514837 / 100000000 : ℝ) • normInput) (by intro c; simp)
example := l2_fixture ((18257419 / 100000000 : ℝ) • normInput) (by intro c; simp)

end Transformer.SequenceModel.Attention.Backend

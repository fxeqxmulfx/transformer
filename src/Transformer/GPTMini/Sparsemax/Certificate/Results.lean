import Transformer.GPTMini.Sparsemax.Certificate.Counts

/-!
# Exact accuracy of the actual final prediction table

Source context: arXiv:2211.11052v1, §4, modular division, adapted to
prime 193 and the frozen 25% split. Counts have already been checked by
the kernel on every exported record. These statements report rational
accuracy; the PyTorch export remains a separately measured trust boundary.
-/

namespace Transformer.GPTMini.Sparsemax.Certificate

/-- The supplied final train table has exactly 100% complete-RHS
accuracy. Source: §4's train accuracy, prime-193 experiment adaptation. -/
theorem train_accuracy_exact : accuracy trainPredictions = 1 := by
  have hc := congrArg (fun result : ℕ × ℕ × Bool × Bool => result.1) train_certificate
  have hn := congrArg (fun result : ℕ × ℕ × Bool × Bool => result.2.1) train_certificate
  change rowCount trainPredictions = 9264 at hc
  change correctCount trainPredictions = 9264 at hn
  rw [accuracy, hn, hc]
  norm_num

/-- The supplied final held-out table has exact rational accuracy
9465/27792. Source: §4's test accuracy, prime-193 experiment adaptation. -/
theorem heldout_accuracy_exact : accuracy heldoutPredictions = 9465 / 27792 := by
  have hc := congrArg (fun result : ℕ × ℕ × Bool × Bool => result.1) heldout_certificate
  have hn := congrArg (fun result : ℕ × ℕ × Bool × Bool => result.2.1) heldout_certificate
  change rowCount heldoutPredictions = 27792 at hc
  change correctCount heldoutPredictions = 9465 at hn
  rw [accuracy, hn, hc]
  norm_num

/-- The exact held-out percentage lies strictly between 34.055% and
34.065%, hence rounds to 34.06% at two decimal places. Source: §4's
test accuracy, numerical reporting for the prime-193 final experiment. -/
theorem heldout_percentage_rounding_interval :
    (6811 : ℚ) / 200 < 100 * accuracy heldoutPredictions ∧
      100 * accuracy heldoutPredictions < 6813 / 200 := by
  rw [heldout_accuracy_exact]
  norm_num

/-- The supplied final held-out accuracy fails the frozen 99% target.
Source: §4's 99% test-accuracy criterion, prime-193 experiment adaptation. -/
theorem heldout_accuracy_below_target : accuracy heldoutPredictions < 99 / 100 := by
  rw [heldout_accuracy_exact]
  norm_num

end Transformer.GPTMini.Sparsemax.Certificate

import Transformer.GPTMini.Sparsemax.Certificate.Train
import Transformer.GPTMini.Sparsemax.Certificate.HeldoutFirst
import Transformer.GPTMini.Sparsemax.Certificate.HeldoutLast

/-!
# Kernel-checked counts of the actual final prediction table

Source context: arXiv:2211.11052v1, §4, modular division, adapted to
prime 193, the frozen 25% split, and the sparsemax 300000-step checkpoint.
These theorems check supplied predictions, not floating-point execution
of PyTorch. Plain `decide` is kernel checked; no native evaluation axiom
or stipulated correct-count premise is used.
-/

namespace Transformer.GPTMini.Sparsemax.Certificate

set_option maxRecDepth 200000
set_option maxHeartbeats 0

/-- All held-out predictions, in frozen corpus order. Source: §4's task,
prime-193 experiment adaptation and the actual final CPU export. -/
def heldoutPredictions : List (List ℕ) :=
  heldoutFirstPredictions ++ heldoutLastPredictions

/-- Kernel-checked train size, correct complete answers, operand bounds,
and EOS flags. Source: §4's division accuracy, experiment adaptation;
all actual train records are inspected rather than assuming the count. -/
theorem train_certificate :
    (rowCount trainPredictions, correctCount trainPredictions,
      allInputsValid trainPredictions, allEOSCorrect trainPredictions) =
    (9264, 9264, true, true) := by
  decide

/-- Kernel-checked held-out size, correct complete answers, operand
bounds, and EOS flags. Source: §4's test accuracy, experiment adaptation;
all actual held-out records are inspected rather than assuming the count. -/
theorem heldout_certificate :
    (rowCount heldoutPredictions, correctCount heldoutPredictions,
      allInputsValid heldoutPredictions, allEOSCorrect heldoutPredictions) =
    (27792, 9465, true, true) := by
  decide

end Transformer.GPTMini.Sparsemax.Certificate

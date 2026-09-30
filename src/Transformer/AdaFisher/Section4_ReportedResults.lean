/-
# AdaFisher: exact comparisons of the reported benchmark means

arXiv:2405.16397v3, §4, Tables 2–5, Appendix A.4, Table 6.
Scores are integer hundredths; `none` means the source reports no result.
These theorems check the tables' arithmetic, not the validity of the runs
or a universal claim about optimizer performance.
-/

import Mathlib.Tactic

namespace Transformer.AdaFisher

/-- Table 2, §4.1, reported CIFAR accuracy means in hundredths.
Dataset order: CIFAR10, CIFAR100. Model order: ResNet18, ResNet50,
ResNet101, DenseNet121, MobileNetV3, Tiny Swin, FocalNet, CCT-2/3×2.
Optimizer order: SGD, Adam(W), AdaHessian, K-FAC, Shampoo, AdaFisher(W). -/
def reportedCifar : Fin 2 → Fin 8 → Fin 6 → Option ℕ :=
  ![![![some 9564, some 9485, some 9544, some 9517, some 9408, some 9625],
       ![some 9571, some 9445, some 9554, some 9566, some 9459, some 9634],
       ![some 9598, some 9457, some 9529, some 9601, some 9463, some 9639],
       ![some 9609, some 9486, some 9611, some 9612, some 9566, some 9672],
       ![some 9443, some 9332, some 9286, some 9434, some 9381, some 9528],
       ![some 8234, some 8737, some 8415, some 6479, some 6391, some 8874],
       ![some 8203, some 8623, some 6418, some 3894, some 3796, some 8790],
       ![some 7876, some 8389, none, some 3308, some 3516, some 8494]],
    ![![some 7656, some 7574, some 7179, some 7603, some 7678, some 7728],
       ![some 7801, some 7465, some 7581, some 7740, some 7807, some 7977],
       ![some 7889, some 7556, some 7338, some 7701, some 7883, some 8065],
       ![some 8013, some 7587, some 7480, some 7979, some 8024, some 8136],
       ![some 7389, some 7062, some 5658, some 7375, some 7085, some 7756],
       ![some 5489, some 6021, some 5686, some 3445, some 3039, some 6605],
       ![some 4776, some 5271, some 3233, some 998, some 918, some 5369],
       ![some 5405, some 5978, none, some 717, some 860, some 6291]]]

/-- AdaFisher has the highest reported mean in every available comparison
of Table 2, §4.1. Missing AdaHessian entries remain missing. -/
theorem reportedCifar_best (dataset : Fin 2) (model : Fin 8) (baseline : Fin 5) :
    ∀ value, reportedCifar dataset model baseline.castSucc = some value →
      value < (reportedCifar dataset model 5).getD 0 := by
  fin_cases dataset <;> fin_cases model <;> fin_cases baseline <;>
    norm_num [reportedCifar]

example : reportedCifar 0 0 0 = some 9564 := by decide

/-- Table 4, §4.2, transfer-learning means, in the same optimizer order.
Model order: ResNet50, ResNet101, DenseNet121, MobileNetV3. -/
def reportedTransfer : Fin 2 → Fin 4 → Fin 6 → ℕ :=
  ![![![9650, 9645, 9635, 9645, 9603, 9713],
       ![9707, 9670, 9665, 9684, 9663, 9722],
       ![9480, 9477, 9308, 9441, 9476, 9503],
       ![9176, 9092, 8645, 9172, 9139, 9278]],
    ![![8212, 8201, 8064, 8055, 8170, 8223],
       ![8401, 8243, 8136, 8226, 8265, 8447],
       ![7598, 7565, 7106, 7610, 7608, 7692],
       ![7186, 6611, 5969, 6985, 6887, 7238]]]

/-- All reported transfer-learning means favor AdaFisher, Table 4, §4.2. -/
theorem reportedTransfer_best : ∀ dataset model (baseline : Fin 5),
    reportedTransfer dataset model baseline.castSucc < reportedTransfer dataset model 5 := by
  decide

/-- Baseline ImageNet Top-1 scores from Table 3, §4.1, including the
literature comparisons. Their training setups are not all identical. -/
def reportedImageNetBaselines : Fin 8 → ℕ :=
  ![6778, 7096, 7282, 7640, 7634, 7666, 7520, 7510]

/-- The reported single-GPU AdaFisher Top-1 score exceeds every listed
baseline in Table 3, §4.1. This checks the reported numbers only. -/
theorem reportedImageNet_top1_best : ∀ baseline, reportedImageNetBaselines baseline < 7695 := by
  decide

/-- Table 5, §4.3, test perplexity in hundredths. Dataset order: WikiText-2,
PTB. Optimizer order: AdamW, AdaHessian, Shampoo, AdaFisherW. -/
def reportedPerplexity : Fin 2 → Fin 4 → Option ℕ :=
  ![![some 17506, some 40769, some 172775, some 15272],
    ![some 4470, some 5943, none, some 4115]]

/-- AdaFisherW has the lowest reported test PPL, Table 5, §4.3.
The absent Shampoo/PTB measurement is not assigned a score. -/
theorem reportedPerplexity_best (dataset : Fin 2) (baseline : Fin 3) :
    ∀ value, reportedPerplexity dataset baseline.castSucc = some value →
      (reportedPerplexity dataset 3).getD 0 < value := by
  fin_cases dataset <;> fin_cases baseline <;> norm_num [reportedPerplexity]

example : reportedPerplexity 0 0 = some 17506 := by decide

/-- Reported seconds per epoch for 1,2,3,4 GPUs, Appendix A.4, Table 6. -/
def reportedGpuSeconds : Fin 4 → ℕ := ![2882, 1438, 963, 720]

/-- The reported timings are within seven GPU-seconds per epoch of ideal
linear scaling, Appendix A.4, Table 6. This is a finite arithmetic statement,
not a performance guarantee for other hardware or batch sizes. -/
theorem reportedGpu_linear_error : ∀ gpu : Fin 4,
    Int.natAbs ((reportedGpuSeconds gpu : ℤ) * (gpu.val + 1) - 2882) ≤ 7 := by
  decide

end Transformer.AdaFisher

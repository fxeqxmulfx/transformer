/-
# Mask granularity need not be equivalent under diagonal preconditioning

Audit of the conjecture in arXiv:2602.15322v1, Section 2, Impacts of
structured masking. The table's near-equivalence is proved separately.
The claimed explanation is qualitative, not a quantified identity.
The theorem below refutes a universal equality interpretation even with
identity preconditioning and the objective's actual gradient direction.
-/

import Transformer.Magma.Section2_Expectations
import Transformer.Magma.Section2_Moments
import Transformer.Magma.Section5_SmoothnessRefutation

noncomputable section

namespace Transformer.Magma

/-- On (x+y)^2/2 at (1,1), identity preconditioning and rate=1/4 give
the update (1/2,1/2). Sharing a survival bit for the two coordinates gives
expected loss 1; independent element masks give 3/4, at p=1/2.
Thus diagonal preconditioning does not make mask granularity irrelevant
as a universal mathematical fact. This does not refute the observed
near-equivalence on the paper's particular C4 experiment. Source:
arXiv:2602.15322v1, Section 2, structured-masking conjecture and eq:implicit_reg. -/
theorem identity_preconditioning_granularity_counterexample :
    let gx := deriv (fun z => coupledQuadratic z 1) 1
    let gy := deriv (fun z => coupledQuadratic 1 z) 1
    let shared := maskExpectation (1 / 2) (fun mask : Unit → Bool =>
      coupledQuadratic (1 - maskCoefficient (1 / 2) (mask ()) * gx / 4)
        (1 - maskCoefficient (1 / 2) (mask ()) * gy / 4))
    let element := maskExpectation (1 / 2) (fun mask : Fin 2 → Bool =>
      coupledQuadratic (1 - maskCoefficient (1 / 2) (mask 0) * gx / 4)
        (1 - maskCoefficient (1 / 2) (mask 1) * gy / 4))
    shared = 1 ∧ element = 3 / 4 ∧ shared ≠ element := by
  dsimp only
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y]
  have hshared : maskExpectation (1 / 2) (fun mask : Unit → Bool =>
      coupledQuadratic (1 - maskCoefficient (1 / 2) (mask ()) * (1 + 1) / 4)
        (1 - maskCoefficient (1 / 2) (mask ()) * (1 + 1) / 4)) = 1 := by
    rw [maskExpectation_coordinate (1 / 2) ()
      (fun bit => coupledQuadratic (1 - maskCoefficient (1 / 2) bit * (1 + 1) / 4)
        (1 - maskCoefficient (1 / 2) bit * (1 + 1) / 4))]
    norm_num [maskCoefficient, coupledQuadratic]
  have hpoly (mask : Fin 2 → Bool) :
      coupledQuadratic (1 - maskCoefficient (1 / 2) (mask 0) * (1 + 1) / 4)
        (1 - maskCoefficient (1 / 2) (mask 1) * (1 + 1) / 4) =
      2 - maskCoefficient (1 / 2) (mask 0) - maskCoefficient (1 / 2) (mask 1) +
        (1 / 8) * (maskCoefficient (1 / 2) (mask 0) ^ 2 +
          2 * (maskCoefficient (1 / 2) (mask 0) * maskCoefficient (1 / 2) (mask 1)) +
          maskCoefficient (1 / 2) (mask 1) ^ 2) := by
    unfold coupledQuadratic
    ring
  have helement : maskExpectation (1 / 2) (fun mask : Fin 2 → Bool =>
      coupledQuadratic (1 - maskCoefficient (1 / 2) (mask 0) * (1 + 1) / 4)
        (1 - maskCoefficient (1 / 2) (mask 1) * (1 + 1) / 4)) = 3 / 4 := by
    simp only [hpoly, maskExpectation_add, maskExpectation_sub, maskExpectation_const,
      maskExpectation_mul, maskCoefficient_mean (1 / 2) (by norm_num),
      maskCoefficient_second (1 / 2) (by norm_num),
      maskCoefficient_cross (1 / 2) (by norm_num) (0 : Fin 2) 1 (by decide)]
    norm_num
  rw [hshared, helement]
  norm_num

end Transformer.Magma

import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic

/-!
# The actual gradient of a normalized parallelogram loss

Source: Liu et al., arXiv:2205.10343v2, section 3.2, equations
`eq:l_eff` and `eq:Ei_dynamics`, and appendix "Conservation laws of the
effective theory", equation `eq:l_eff_app`.

The three scalar embeddings have one nontrivial parallelogram constraint,
`E₀ + E₂ = E₁ + E₁`. Constant duplication/averaging factors in the numerator
are suppressed, giving a positive rescaling of time, as in the appendix.
The denominator is the sum of squared embeddings. It is differentiated
as a function of the embeddings, even if it is conserved along a flow.

Every component below is an actual partial derivative of the quotient.
The next module checks conservation and the source's linear-flow claim.
No transformer/AdamW dynamics is assumed to obey this effective model.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Parallelogram residual for `(0, 2)` versus `(1, 1)`. Source:
arXiv:2205.10343v2, section 3.2, numerator in equation `eq:l_eff`. -/
def residual (x y z : ℝ) : ℝ := x + z - 2 * y

/-- Unnormalized squared parallelogram loss, with constant averaging
suppressed. Source: arXiv:2205.10343v2, section 3.2, `ℓ₀` in `eq:l_eff`. -/
def numerator (x y z : ℝ) : ℝ := residual x y z ^ 2

/-- Squared representation norm, not a frozen scalar. Source:
arXiv:2205.10343v2, section 3.2, `Z₀` in equation `eq:l_eff`. -/
def squaredNorm (x y z : ℝ) : ℝ := x ^ 2 + y ^ 2 + z ^ 2

/-- Actual normalized loss for the one-constraint specialization. Source:
arXiv:2205.10343v2, section 3.2 and appendix conservation laws. -/
noncomputable def normalizedLoss (x y z : ℝ) : ℝ :=
  numerator x y z / squaredNorm x y z

/-- The gradient is defined through partial derivatives of the actual
loss, rather than by the desired conservation formula. Source:
arXiv:2205.10343v2, equation `eq:Ei_dynamics`. -/
noncomputable def quotientGradient (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (deriv (fun u => normalizedLoss u y z) x,
    deriv (fun u => normalizedLoss x u z) y,
    deriv (fun u => normalizedLoss x y u) z)

/-- First numerator derivative before quotient normalization. Source:
arXiv:2205.10343v2, section 3.2, `ℓ₀` in equation `eq:l_eff`. -/
theorem numerator_deriv_x (x y z : ℝ) :
    HasDerivAt (fun u => numerator u y z) (2 * residual x y z) x := by
  convert (((hasDerivAt_id x).add_const z).sub_const (2 * y)).pow 2 using 1
  · rfl
  · unfold residual
    norm_num

/-- The middle numerator derivative includes the repeated index. Source:
arXiv:2205.10343v2, section 3.2, parallelogram loss `eq:l_eff`. -/
theorem numerator_deriv_y (x y z : ℝ) :
    HasDerivAt (fun u => numerator x u z) (-4 * residual x y z) y := by
  convert (((hasDerivAt_id y).const_mul 2).const_sub (x + z)).pow 2 using 1
  · rfl
  · unfold residual
    norm_num
    ring

/-- Last numerator derivative before normalization. Source:
arXiv:2205.10343v2, section 3.2, equation `eq:l_eff`. -/
theorem numerator_deriv_z (x y z : ℝ) :
    HasDerivAt (fun u => numerator x y u) (2 * residual x y z) z := by
  convert (((hasDerivAt_id z).const_add x).sub_const (2 * y)).pow 2 using 1
  · rfl
  · unfold residual
    norm_num

/-- Weight of the first embedding in the quotient gradient. Source:
arXiv:2205.10343v2, section 3.2, quotient rule for `eq:l_eff`;
the derivative of `Z₀` is retained. -/
theorem normalizedLoss_deriv_x (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun u => normalizedLoss u y z)
      ((2 * residual x y z * squaredNorm x y z - numerator x y z * (2 * x)) /
        squaredNorm x y z ^ 2) x := by
  have hd : HasDerivAt (fun u => squaredNorm u y z) (2 * x) x := by
    convert (((hasDerivAt_id x).pow 2).add_const (y ^ 2)).add_const (z ^ 2) using 1
    · rfl
    · norm_num
  exact (numerator_deriv_x x y z).div hd hZ

example : squaredNorm 1 0 0 ≠ 0 := by
  unfold squaredNorm
  norm_num

/-- The repeated middle embedding has residual coefficient -2. Source:
arXiv:2205.10343v2, section 3.2, `eq:l_eff`; both numerator and denominator
are differentiated, including this repeated index. -/
theorem normalizedLoss_deriv_y (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun u => normalizedLoss x u z)
      ((-4 * residual x y z * squaredNorm x y z - numerator x y z * (2 * y)) /
        squaredNorm x y z ^ 2) y := by
  have hd : HasDerivAt (fun u => squaredNorm x u z) (2 * y) y := by
    convert (((hasDerivAt_id y).pow 2).const_add (x ^ 2)).add_const (z ^ 2) using 1
    · rfl
    · norm_num
  exact (numerator_deriv_y x y z).div hd hZ

example : squaredNorm 0 1 0 ≠ 0 := by
  unfold squaredNorm
  norm_num

/-- Quotient derivative for the last embedding. Source:
arXiv:2205.10343v2, section 3.2, `eq:l_eff` and `eq:Ei_dynamics`. -/
theorem normalizedLoss_deriv_z (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun u => normalizedLoss x y u)
      ((2 * residual x y z * squaredNorm x y z - numerator x y z * (2 * z)) /
        squaredNorm x y z ^ 2) z := by
  have hd : HasDerivAt (fun u => squaredNorm x y u) (2 * z) z := by
    convert ((hasDerivAt_id z).pow 2).const_add (x ^ 2 + y ^ 2) using 1
    · rfl
    · norm_num
  exact (numerator_deriv_z x y z).div hd hZ

example : squaredNorm 0 0 1 ≠ 0 := by
  unfold squaredNorm
  norm_num

/-- Closed gradient formula verified against all actual partial
derivatives. Source: arXiv:2205.10343v2, section 3.2 and the appendix's
first quotient-rule equation; no constancy assumption for `Z₀`. -/
theorem quotientGradient_eq (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    quotientGradient x y z =
      ((2 * residual x y z * squaredNorm x y z - numerator x y z * (2 * x)) /
          squaredNorm x y z ^ 2,
        (-4 * residual x y z * squaredNorm x y z - numerator x y z * (2 * y)) /
          squaredNorm x y z ^ 2,
        (2 * residual x y z * squaredNorm x y z - numerator x y z * (2 * z)) /
          squaredNorm x y z ^ 2) := by
  unfold quotientGradient
  rw [(normalizedLoss_deriv_x x y z hZ).deriv,
    (normalizedLoss_deriv_y x y z hZ).deriv,
    (normalizedLoss_deriv_z x y z hZ).deriv]

example : squaredNorm 1 (-2) 1 ≠ 0 := by
  unfold squaredNorm
  norm_num

end Transformer.Grokking.EffectiveTheory

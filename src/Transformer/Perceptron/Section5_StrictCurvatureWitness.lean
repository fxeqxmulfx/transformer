module

public import Transformer.Perceptron.Section5_CircleGeometry
public import Transformer.Perceptron.CircleDeriv
public import Transformer.Perceptron.Dirac
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# A complete nonempty witness of the strict angular curvature condition

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron` and
`eq:strictSOPD-sufficient`. Use the quadratic primitive `φ(s)=s²`
of the identity activation, one active neuron with weight `-4`
and vector minus the first basis vector, and a Dirac probability
at that basis vector. The second neuron has weight zero.

Its potential on the circle is `-4 cos²θ`, whose actual second
angular derivative is `8 cos(2θ)`. The measure is stationary
because its perceptron drift is radial. Every representative of
its support point has cosine one, so the potential curvature
there is eight. At `β=κ=1`, `2 exp 1+2<8` verifies the full
source condition. None of these statements assumes strict SOPD.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **The actual one-neuron potential is `-4cos²θ`.**
Source: arXiv:2601.21366v2, §2.3, the perceptron potential
with `φ'=2σ`, and §5.5, `eq:strictSOPD-sufficient`.
This is the two-neuron presentation with an inactive second neuron. -/
theorem quadratic_pin_potential_circle (θ : ℝ) :
    potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (circlePoint θ : EucSpace 2) = -4 * Real.cos θ ^ 2 := by
  rw [potential, Fin.sum_univ_two]
  simp only [Pi.single_eq_same, Pi.single_eq_of_ne (show (1 : Idx 2) ≠ 0 by decide),
    zero_mul, add_zero]
  rw [inner_neg_left, inner_basePoint_circlePoint, neg_sq]

/-- The witness's potential is minimal at the pinned circle point,
among individual circle points. Source: arXiv:2601.21366v2, §5.5,
the potential curvature in `rem:strictSOPD-perceptron`.
This is a pointwise potential comparison, without a claim of
minimizing the coupled energy over probability measures. -/
theorem quadratic_pin_potential_min (θ : ℝ) :
    potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) (circlePoint 0 : EucSpace 2) ≤
    potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) (circlePoint θ : EucSpace 2) := by
  rw [quadratic_pin_potential_circle, quadratic_pin_potential_circle, Real.cos_zero]
  nlinarith [Real.cos_sq_le_one θ]

/-- **Its actual angular second derivative is `8cos(2θ)`.**
Source: arXiv:2601.21366v2, §5.5, the potential second-derivative
display in `rem:strictSOPD-perceptron`.
The derivative is computed from the actual primitive and activation,
not supplied as a value of a Hessian. -/
theorem quadratic_pin_second_angular_derivative (θ : ℝ) :
    deriv^[2] (fun t : ℝ => potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (circlePoint t : EucSpace 2)) θ = 8 * Real.cos (2 * θ) := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * (fun s : ℝ => s) s) s :=
    fun s => by simpa using hasDerivAt_pow 2 s
  have hσ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s :=
    fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s)
  have hvel : inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2) (circleVel θ) =
      -Real.sin θ := by
    rw [inner_circleVel]
    simp
  rw [secondDeriv_potential_circlePoint (fun s : ℝ => s ^ 2) (fun s : ℝ => s)
    (fun _ : ℝ => (1 : ℝ)) hφ hσ, Fin.sum_univ_two]
  simp only [Pi.single_eq_same, Pi.single_eq_of_ne (show (1 : Idx 2) ≠ 0 by decide),
    inner_zero_left, inner_neg_left, inner_basePoint_circlePoint, hvel]
  rw [Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq θ]

/-- **The actual Dirac probability is stationary at every temperature.**
Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`.
Its interaction is radial at its own atom, and the perceptron
drift there is the multiple `-4x`. -/
theorem isStationary_quadratic_pin_dirac (β : ℝ) :
    IsStationary β (fun s : ℝ => s) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) := by
  apply isStationary_diracProb_of_radial β _ _ _ (basePoint 1) (-4)
  rw [Fin.sum_univ_two]
  simp [inner_neg_left, ← neg_smul]

/-- **The quadratic pin meets the source's curvature condition at
every representative of its actual support point.**
Source: arXiv:2601.21366v2, §5.5, `eq:strictSOPD-sufficient`.
The assumptions `exp β≤3`, `κ≤1` give `2 exp β+2κ≤8`.
Positivity of `κ` is needed by the strictness conclusion, but
not by this pointwise verification of its numerical inequality. -/
theorem quadratic_pin_curvature_condition (β κ : ℝ) (hβ : Real.exp β ≤ 3) (hκ : κ ≤ 1)
    (θ : ℝ) (hθ : circlePoint θ ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support) :
    2 * Real.exp β + 2 * κ ≤ deriv^[2]
      (fun t : ℝ => potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (circlePoint t : EucSpace 2)) θ := by
  have heq : circlePoint θ = (basePoint 1 : SSphere 2) :=
    Interpolation.eq_of_mem_support_dirac hθ
  have hcos : Real.cos θ = 1 := by
    have h := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) heq
    rw [circlePoint_coord_zero] at h
    simpa [basePoint] using h
  rw [quadratic_pin_second_angular_derivative, Real.cos_two_mul, hcos]
  norm_num
  linarith

/-- **Every hypothesis of the source's strict-curvature statement is
jointly realized by actual data.** Source: arXiv:2601.21366v2,
§5.5, `rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
The activation and primitive derivatives, stationarity, positive
temperature and positive `κ`, and the all-representatives support
condition are proved together. This theorem will witness the
same inputs of the finite-support and Hessian consequences. -/
theorem strict_curvature_quadratic_dirac_witness :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) ∧
    IsStationary 1 (fun s : ℝ => s) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧ (0 : ℝ) < 1 ∧
    ∀ θ : ℝ, circlePoint θ ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support →
      2 * Real.exp 1 + 2 * 1 ≤ deriv^[2]
        (fun t : ℝ => potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
          (circlePoint t : EucSpace 2)) θ := by
  refine ⟨one_pos, fun s => ?_, fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s),
    isStationary_quadratic_pin_dirac 1, one_pos, ?_⟩
  · simpa using hasDerivAt_pow 2 s
  · exact quadratic_pin_curvature_condition 1 1 Real.exp_one_lt_three.le le_rfl

/-- The pointwise numerical hypotheses of the curvature verification
are satisfiable at the actual support point, `β=κ=1`, `θ=0`. -/
example : Real.exp 1 ≤ 3 ∧ (1 : ℝ) ≤ 1 ∧ circlePoint 0 ∈
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support := by
  refine ⟨Real.exp_one_lt_three.le, le_rfl, ?_⟩
  have heq : circlePoint 0 = (basePoint 1 : SSphere 2) := by
    apply Subtype.ext
    simp [circlePoint, greatCircle]
  rw [heq]
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : (basePoint 1 : SSphere 2) ∈ U := mem_of_mem_nhds hU
  change 0 < Measure.dirac (basePoint 1) U
  simp [hx]

end Transformer.Perceptron

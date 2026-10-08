import Transformer.Grokking.Composition.Basic

/-!
# Actual curvature of a regularized compositional CE model

Source: Nanda et al., arXiv:2301.05217v1, appendix section Further
speculations on grokking, subsections An intuitive explanation of
grokking and Hypothesis: Phase Transitions are inherent to composition.
The source describes competition with weight size and multi-part circuits;
it does not derive the two-component model or its constants below.

Explicit specialization: add `lambda * (x^2 + y^2) / 2` to the actual
bilinear binary CE. Lambda is a coupled L2 penalty, not native AdamW's
decoupled weight decay. Derive its true gradient and all four second
coordinate derivatives at the all-zero state. The resulting Hessian
has an aligned and an opposed mode with different curvature signs.

The threshold `lambda = 1/2` concerns local finite-dimensional curvature.
No thermodynamic limit, learned transformer circuit, full AdamW dynamics
or delayed generalization is inferred from this calculation.
-/

namespace Transformer.Grokking.Composition

/-- Actual CE plus a stated quadratic size penalty. Source comparison:
arXiv:2301.05217v1, appendix competition explanation; explicit deviation:
coupled L2 regularization of two scalar components, not native AdamW. -/
noncomputable def penalizedLoss (lam x y : ℝ) : ℝ :=
  coupledLoss x y + lam * (x ^ 2 + y ^ 2) / 2

/-- Actual partial derivatives of that stated objective. Source:
the regularized specialization of arXiv:2301.05217v1's appendix above;
stationarity or a preferred circuit is not supplied by the definition. -/
noncomputable def penalizedGradient (lam x y : ℝ) : ℝ × ℝ :=
  (deriv (fun u => penalizedLoss lam u y) x, deriv (fun v => penalizedLoss lam x v) y)

/-- Actual second coordinate derivatives of the verified gradient.
Source: the appendix composition/size hypothesis of arXiv:2301.05217v1,
specialized here; both coordinates and all four entries are retained. -/
noncomputable def penalizedHessian (lam x y : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((deriv (fun u => (penalizedGradient lam u y).1) x,
    deriv (fun v => (penalizedGradient lam x v).1) y),
   (deriv (fun u => (penalizedGradient lam u y).2) x,
    deriv (fun v => (penalizedGradient lam x v).2) y))

/-- Quadratic form of the actual origin Hessian. Source: the explicit
regularized CE specialization of arXiv:2301.05217v1's appendix; it uses
the computed derivatives, not a matrix chosen to have a desired spectrum. -/
noncomputable def originCurvature (lam u v : ℝ) : ℝ :=
  u * ((penalizedHessian lam 0 0).1.1 * u + (penalizedHessian lam 0 0).1.2 * v) +
    v * ((penalizedHessian lam 0 0).2.1 * u + (penalizedHessian lam 0 0).2.2 * v)

/-- Differentiate the actual first coordinate, including the size
penalty. Source: arXiv:2301.05217v1, appendix competition hypothesis;
the stated L2 specialization differs from decoupled AdamW. -/
theorem penalizedLoss_deriv_first (lam x y : ℝ) :
    HasDerivAt (fun u => penalizedLoss lam u y)
      (-y / (Real.exp (x * y) + 1) + lam * x) x := by
  have hp := ((((hasDerivAt_id x).pow 2).add_const (y ^ 2)).const_mul lam).div_const 2
  convert (coupledLoss_deriv_first x y).add hp using 1
  · funext u
    rfl
  · dsimp
    ring

/-- Differentiate the actual second coordinate as well. Source:
arXiv:2301.05217v1, appendix competition hypothesis; no desired
gradient flow is inserted in the objective. -/
theorem penalizedLoss_deriv_second (lam x y : ℝ) :
    HasDerivAt (fun v => penalizedLoss lam x v)
      (-x / (Real.exp (x * y) + 1) + lam * y) y := by
  have hp := ((((hasDerivAt_id y).pow 2).const_add (x ^ 2)).const_mul lam).div_const 2
  convert (coupledLoss_deriv_second x y).add hp using 1
  · funext v
    rfl
  · dsimp
    ring

/-- The true gradient has a component-coupling term and a size term.
Source: arXiv:2301.05217v1, appendix composition and competition
hypotheses, for this explicitly stated two-component objective. -/
theorem penalizedGradient_eq (lam x y : ℝ) :
    penalizedGradient lam x y =
      (-y / (Real.exp (x * y) + 1) + lam * x, -x / (Real.exp (x * y) + 1) + lam * y) := by
  unfold penalizedGradient
  rw [(penalizedLoss_deriv_first lam x y).deriv, (penalizedLoss_deriv_second lam x y).deriv]

/-- Exact gradient restrictions retain cross-component information
even though the unregularized axis loss is flat. Source:
arXiv:2301.05217v1, appendix compositional hypothesis, scalar CE specialization. -/
theorem penalized_axis_gradients (lam t : ℝ) :
    penalizedGradient lam t 0 = (lam * t, -t / 2) ∧
      penalizedGradient lam 0 t = (-t / 2, lam * t) := by
  rw [penalizedGradient_eq, penalizedGradient_eq]
  norm_num

/-- All four actual Hessian entries at the all-absent state. Source:
arXiv:2301.05217v1, appendix circuit hypothesis; this derives the
cross derivatives from actual CE rather than stipulating a saddle matrix. -/
theorem penalized_origin_hessian (lam : ℝ) :
    penalizedHessian lam 0 0 = ((lam, -1 / 2), (-1 / 2, lam)) := by
  have hd1 : HasDerivAt (fun t => (penalizedGradient lam t 0).1) lam 0 := by
    convert (hasDerivAt_id (0 : ℝ)).const_mul lam using 1
    · funext t
      rw [(penalized_axis_gradients lam t).1]
      rfl
    · norm_num
  have hd2 : HasDerivAt (fun t => (penalizedGradient lam 0 t).2) lam 0 := by
    convert (hasDerivAt_id (0 : ℝ)).const_mul lam using 1
    · funext t
      rw [(penalized_axis_gradients lam t).2]
      rfl
    · norm_num
  have hc1 : HasDerivAt (fun t => (penalizedGradient lam 0 t).1) (-1 / 2) 0 := by
    convert (hasDerivAt_id (0 : ℝ)).mul_const (-1 / 2) using 1
    · funext t
      rw [(penalized_axis_gradients lam t).2]
      dsimp
      ring
    · norm_num
  have hc2 : HasDerivAt (fun t => (penalizedGradient lam t 0).2) (-1 / 2) 0 := by
    convert (hasDerivAt_id (0 : ℝ)).mul_const (-1 / 2) using 1
    · funext t
      rw [(penalized_axis_gradients lam t).1]
      dsimp
      ring
    · norm_num
  unfold penalizedHessian
  rw [hd1.deriv, hc1.deriv, hc2.deriv, hd2.deriv]

/-- The actual Hessian form exposes the learned-component interaction.
Source: arXiv:2301.05217v1, appendix composition/size hypothesis;
the coefficient one comes from the computed binary CE cross derivatives. -/
theorem originCurvature_eq (lam u v : ℝ) :
    originCurvature lam u v = lam * (u ^ 2 + v ^ 2) - u * v := by
  unfold originCurvature
  rw [penalized_origin_hessian]
  dsimp
  ring

/-- Aligned and opposed directions have different actual curvature.
Source: arXiv:2301.05217v1, appendix compositional hypothesis, in the
explicit regularized bilinear model; no thermodynamic phase is asserted. -/
theorem aligned_and_opposed_curvature (lam : ℝ) :
    originCurvature lam 1 1 = 2 * lam - 1 ∧ originCurvature lam 1 (-1) = 2 * lam + 1 := by
  rw [originCurvature_eq, originCurvature_eq]
  constructor <;> ring

/-- A genuine negative-curvature direction coexists with a positive
one below the stated penalty threshold. Source: arXiv:2301.05217v1,
appendix circuit hypothesis; the finite-dimensional constant is derived here. -/
theorem opposed_curvature_signs_below_half (lam : ℝ) (hl : 0 ≤ lam) (hh : lam < 1 / 2) :
    originCurvature lam 1 1 < 0 ∧ 0 < originCurvature lam 1 (-1) := by
  rw [(aligned_and_opposed_curvature lam).1, (aligned_and_opposed_curvature lam).2]
  constructor <;> linarith

example : 0 ≤ (1 / 10 : ℝ) ∧ (1 / 10 : ℝ) < 1 / 2 := by norm_num

/-- Nonnegative origin curvature above the threshold is a local
calculation, not global convexity of transformer parameters. Source:
arXiv:2301.05217v1, appendix competition hypothesis, with stated L2 penalty. -/
theorem origin_curvature_nonneg_above_half (lam u v : ℝ) (hl : 1 / 2 ≤ lam) :
    0 ≤ originCurvature lam u v := by
  rw [originCurvature_eq]
  have hp : 0 ≤ (lam - 1 / 2) * (u ^ 2 + v ^ 2) := by positivity
  nlinarith [sq_nonneg (u - v)]

example : (1 / 2 : ℝ) ≤ 1 := by norm_num

end Transformer.Grokking.Composition

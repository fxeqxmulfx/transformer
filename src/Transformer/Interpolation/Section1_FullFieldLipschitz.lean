module

public import Transformer.Interpolation.Section1_FullFieldBounds
public import Transformer.Perspective.DrivenFlow

/-!
# Uniform spatial constants for the complete original network field

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`,
and §1.1, the norm estimate in `thm: targets.atoms`.
The normalized attention estimate is independent of its probability law.
The perceptron difference is bounded by the product of its actual input
and output norms. Tangent projection adds twice the unprojected speed.

All original matrices and the bias remain in these estimates. A common
component bound K gives the explicit spatial constant
2K+(2 exp(2K)+5)K². This constant applies along any evolving probability,
without assuming a characteristic representation or weak uniqueness.
-/

@[expose] public section

open Real MeasureTheory
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual ambient perceptron argument is globally Lipschitz.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem lipschitzWith_perceptron_argument (W U : ParamMatrix d) (b : EucSpace d) :
    LipschitzWith (‖W‖₊ * ‖U‖₊) (fun x : EucSpace d =>
      W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max ((U x + b) i) 0))) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simpa only [NNReal.coe_mul, coe_nnnorm, dist_eq_norm] using norm_perceptron_argument_sub_le W U b x y

/-- At a fixed original time and law, the complete original field is
continuous at every ambient query. Source auxiliary:
arXiv:2411.04551v3, §1, the characteristic field in `eq: vf`. -/
theorem continuous_fullVF (θ : TimeParams d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Continuous (fullVF d θ ν t) := by
  let N (x : EucSpace d) := (θ t).W
    ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (((θ t).U x + (θ t).b) i) 0))
  have hN : Continuous N := (lipschitzWith_perceptron_argument (θ t).W (θ t).U (θ t).b).continuous
  have hD : Continuous (fun x => (θ t).V (attentionAverage (θ t).B ν x) + N x) :=
    (((θ t).V).continuous.comp (continuous_attentionAverage (θ t).B ν)).add hN
  have he (x : EucSpace d) : fullVF d θ ν t x =
      proj d x ((θ t).V (attentionAverage (θ t).B ν x) + N x) := by
    simp only [fullVF, attentionAverage, Perspective.partitionMu, one_mul, N]
    rfl
  simp_rw [show fullVF d θ ν t = fun x => proj d x
    ((θ t).V (attentionAverage (θ t).B ν x) + N x) from funext he, proj]
  exact hD.sub ((continuous_id.inner (𝕜 := ℝ) hD).smul continuous_id)

/-- The complete original field has a spatial constant uniform over all
probability laws. Source auxiliary: arXiv:2411.04551v3, §1, the
Lipschitz well-posedness of `eq: vf` and `eq: cauchy.pb`. -/
theorem norm_fullVF_sub_le (θ : TimeParams d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ‖fullVF d θ ν t x - fullVF d θ ν t y‖ ≤
      (‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) + ‖(θ t).W‖ * ‖(θ t).U‖ +
        2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖))) * ‖x - y‖ := by
  let D (z : EucSpace d) := (θ t).V (attentionAverage (θ t).B ν z) +
    (θ t).W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (((θ t).U z + (θ t).b) i) 0))
  have hfield (z : EucSpace d) : fullVF d θ ν t z = proj d z (D z) := by
    simp only [fullVF, D, attentionAverage, Perspective.partitionMu, one_mul]
    rfl
  have hv : ‖(θ t).V (attentionAverage (θ t).B ν y)‖ ≤ ‖(θ t).V‖ := by
    apply (((θ t).V).le_opNorm _).trans
    simpa only [mul_one] using mul_le_mul_of_nonneg_left
      (norm_attentionAverage_le_one (θ t).B ν y) (norm_nonneg ((θ t).V))
  have hUy : ‖(θ t).U y + (θ t).b‖ ≤ ‖(θ t).U‖ + ‖(θ t).b‖ :=
    (norm_add_le _ _).trans (add_le_add (by simpa only [hy, mul_one] using (θ t).U.le_opNorm y) le_rfl)
  have hw : ‖(θ t).W ((EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun i => max (((θ t).U y + (θ t).b) i) 0))‖ ≤
      ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖) :=
    (((θ t).W).le_opNorm _).trans (mul_le_mul_of_nonneg_left
      ((norm_coordinate_relu_le _).trans hUy) (norm_nonneg _))
  have hDy : ‖D y‖ ≤ ‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖) :=
    (norm_add_le _ _).trans (add_le_add hv hw)
  have ha : ‖(θ t).V (attentionAverage (θ t).B ν x) -
      (θ t).V (attentionAverage (θ t).B ν y)‖ ≤
      (‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖)) * ‖x - y‖ := by
    rw [← map_sub]
    apply (((θ t).V).le_opNorm _).trans
    have hh := norm_attentionAverage_sub_le (θ t).B ν 1 x y hx.le hy.le
    simpa only [mul_one, mul_assoc] using mul_le_mul_of_nonneg_left hh (norm_nonneg ((θ t).V))
  have hdiff : ‖D x - D y‖ ≤
      (‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) + ‖(θ t).W‖ * ‖(θ t).U‖) * ‖x - y‖ := by
    have he : D x - D y =
        ((θ t).V (attentionAverage (θ t).B ν x) - (θ t).V (attentionAverage (θ t).B ν y)) +
        ((θ t).W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (((θ t).U x + (θ t).b) i) 0)) -
          (θ t).W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (((θ t).U y + (θ t).b) i) 0))) := by
      dsimp only [D]
      abel
    rw [he, add_mul]
    exact (norm_add_le _ _).trans (add_le_add ha (norm_perceptron_argument_sub_le _ _ _ x y))
  have hsplit : proj d x (D x) - proj d y (D y) =
      proj d x (D x - D y) + (proj d x (D y) - proj d y (D y)) := by
    simp only [proj, inner_sub_right, sub_smul]
    module
  rw [hfield x, hfield y, hsplit]
  calc
    _ ≤ ‖proj d x (D x - D y)‖ + ‖proj d x (D y) - proj d y (D y)‖ := norm_add_le _ _
    _ ≤ ‖D x - D y‖ + 2 * ‖x - y‖ * ‖D y‖ :=
      add_le_add (norm_proj_le hx _) (Perspective.norm_proj_sub_proj_le_mul hx hy _)
    _ ≤ _ := by nlinarith [mul_le_mul_of_nonneg_left hDy (by positivity : 0 ≤ 2 * ‖x - y‖)]

/-- The explicit component bound gives one spatial constant for every
original law. Source auxiliary: arXiv:2411.04551v3, §1.1,
`thm: targets.atoms`, and §1, `eq: vf`. -/
theorem norm_fullVF_sub_le_of_component_bounds (θ : TimeParams d)
    (ν : Perspective.ProbSphere d) (t K : ℝ) (x y : EucSpace d)
    (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (hV : ‖(θ t).V‖ ≤ K) (hB : ‖(θ t).B‖ ≤ K) (hW : ‖(θ t).W‖ ≤ K)
    (hU : ‖(θ t).U‖ ≤ K) (hb : ‖(θ t).b‖ ≤ K) :
    ‖fullVF d θ ν t x - fullVF d θ ν t y‖ ≤
      (2 * K + (2 * exp (2 * K) + 5) * K ^ 2) * ‖x - y‖ := by
  have hK : 0 ≤ K := (norm_nonneg _).trans hV
  have ha : ‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) ≤
      K * (2 * exp (2 * K) * K) := by gcongr
  have hwu := mul_le_mul hW hU (norm_nonneg _) hK
  have hg := add_le_add hV (mul_le_mul hW (add_le_add hU hb) (by positivity) hK)
  have hconst : ‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) +
      ‖(θ t).W‖ * ‖(θ t).U‖ + 2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖)) ≤
      2 * K + (2 * exp (2 * K) + 5) * K ^ 2 := by nlinarith
  exact (norm_fullVF_sub_le θ ν t x y hx hy).trans
    (mul_le_mul_of_nonneg_right hconst (norm_nonneg _))

/-- The actual parameter norms themselves give a sphere Lipschitz
constant, with no bound hypothesis on the blocks. Source auxiliary:
arXiv:2411.04551v3, §1, the regularity of `eq: vf`. -/
theorem lipschitzWith_fullVF_sphere (θ : TimeParams d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    LipschitzWith ⟨‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) + ‖(θ t).W‖ * ‖(θ t).U‖ +
      2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖)), by positivity⟩
      (fun x : SSphere d => fullVF d θ ν t x) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  change ‖fullVF d θ ν t x - fullVF d θ ν t y‖ ≤ _ * ‖(x : EucSpace d) - (y : EucSpace d)‖
  exact norm_fullVF_sub_le θ ν t x y
    (mem_sphere_zero_iff_norm.mp x.2) (mem_sphere_zero_iff_norm.mp y.2)

/-- The original field is continuous on the whole sphere for every
law and time. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_fullVF_sphere (θ : TimeParams d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Continuous (fun x : SSphere d => fullVF d θ ν t x) :=
  (lipschitzWith_fullVF_sphere θ ν t).continuous

/-- The complete sphere-restricted field has the displayed uniform
Lipschitz constant. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem lipschitzWith_fullVF_sphere_of_component_bounds (θ : TimeParams d)
    (ν : Perspective.ProbSphere d) (t K : ℝ) (hK : 0 ≤ K)
    (hV : ‖(θ t).V‖ ≤ K) (hB : ‖(θ t).B‖ ≤ K) (hW : ‖(θ t).W‖ ≤ K)
    (hU : ‖(θ t).U‖ ≤ K) (hb : ‖(θ t).b‖ ≤ K) :
    LipschitzWith ⟨2 * K + (2 * exp (2 * K) + 5) * K ^ 2, by positivity⟩
      (fun x : SSphere d => fullVF d θ ν t x) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  change ‖fullVF d θ ν t x - fullVF d θ ν t y‖ ≤ _ * ‖(x : EucSpace d) - (y : EucSpace d)‖
  exact norm_fullVF_sub_le_of_component_bounds θ ν t K x y
    (mem_sphere_zero_iff_norm.mp x.2) (mem_sphere_zero_iff_norm.mp y.2) hV hB hW hU hb

/-- Every spatial hypothesis is realized by nonzero original matrices
and bias at two distinct unit queries, with an actual Dirac law. -/
example : ∃ (θ : TimeParams 2) (ν : Perspective.ProbSphere 2)
    (x y : EucSpace 2) (K : ℝ),
    ‖x‖ = 1 ∧ ‖y‖ = 1 ∧ x ≠ y ∧ 0 ≤ K ∧ (ν : Measure (SSphere 2)) Set.univ = 1 ∧
    ‖(θ 0).V‖ ≤ K ∧ ‖(θ 0).B‖ ≤ K ∧ ‖(θ 0).W‖ ≤ K ∧
    ‖(θ 0).U‖ ≤ K ∧ ‖(θ 0).b‖ ≤ K := by
  let p : SSphere 2 := basePoint 1
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let θ : TimeParams 2 := fun _ => ⟨I, I, I, I, (p : EucSpace 2)⟩
  let q : SSphere 2 := antipode 2 p
  have hp : ‖(p : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hq : ‖(q : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp q.2
  have hne : (p : EucSpace 2) ≠ (q : EucSpace 2) := by
    intro he
    exact (antipode_ne 2 p) (Subtype.ext he.symm)
  refine ⟨θ, Perspective.diracProb 2 p, p, q, 1, hp, hq, hne, zero_le_one, by simp,
    ?_, ?_, ?_, ?_, hp.le⟩ <;> simp [θ, I]

end Transformer.Interpolation

/-
# Homogenized Transformers — second moments of a random linear map

The computation behind consequence (i) of assumption (G) of arXiv:2604.01978v1.
Under (G) the value matrix `V` is independent of the query-key matrix `A`, and
its entries are uncorrelated with variance `σ_V²`.  For a bounded vector `m(A)`
that sees only `A`,

  `E⟨u, V m(A)⟩² = σ_V² ‖u‖² E‖m(A)‖²`  and
  `E‖Proj_e V m(A)‖² = σ_V² (d-1) E‖m(A)‖²`  for a unit vector `e`

(`integral_inner_toEuclideanLin_sq`, `integral_norm_proj_toEuclideanLin_sq`).
The first is the linear form `Σ_{ik} V_{ik} u_i m_k` of
`Homogenized.LinearFormMoments`, whose coefficients have square norm
`‖u‖²‖m‖²`; the second follows from `‖Proj_e w‖² = ‖w‖² - ⟨e, w⟩²`.

Source: arXiv:2604.01978v1, §2.3.3, item (i).
-/

import Transformer.Basic
import Transformer.Homogenized.LinearFormMoments
import Mathlib.Analysis.Matrix.MeasurableSpace
import Mathlib.Analysis.Normed.Lp.MeasurableSpace

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {d : ℕ}

/-- `⟨u, M a⟩ = Σ_{ik} M_{ik} u_i a_k`: a linear form in the entries of `M`. -/
theorem inner_toEuclideanLin_eq_sum (M : Matrix (Fin d) (Fin d) ℝ) (u a : EucSpace d) :
    inner (𝕜 := ℝ) u (Matrix.toEuclideanLin M a) =
      ∑ p : Fin d × Fin d, M p.1 p.2 * (u p.1 * a p.2) := by
  simp [EuclideanSpace.inner_eq_star_dotProduct, Matrix.toEuclideanLin, Matrix.mulVec,
    dotProduct, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- The coefficients of that form have square norm `‖u‖² ‖a‖²`. -/
theorem sum_mul_apply_sq (u a : EucSpace d) :
    ∑ p : Fin d × Fin d, (u p.1 * a p.2) ^ 2 = ‖u‖ ^ 2 * ‖a‖ ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type,
    Finset.sum_mul_sum]
  simp only [mul_pow]

/-- `‖Proj_e w‖² = ‖w‖² - ⟨e, w⟩²` for a unit vector `e`. -/
theorem norm_proj_sq {e : EucSpace d} (he : ‖e‖ = 1) (w : EucSpace d) :
    ‖proj d e w‖ ^ 2 = ‖w‖ ^ 2 - inner (𝕜 := ℝ) e w ^ 2 := by
  rw [proj, norm_sub_sq_real, real_inner_smul_right, norm_smul, real_inner_comm w e, he]
  simp
  ring

/-- The square norm is the sum of the squares of the coordinates. -/
theorem norm_sq_eq_sum_inner_single_sq (w : EucSpace d) :
    ‖w‖ ^ 2 = ∑ i, inner (𝕜 := ℝ) (EuclideanSpace.single i (1 : ℝ)) w ^ 2 := by
  simp [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.inner_single_left]

/-- The coefficients `u_i a_k` of that form are measurable in `a`. -/
theorem measurable_coef (u : EucSpace d) :
    Measurable fun (z : EucSpace d) (p : Fin d × Fin d) => u p.1 * z p.2 :=
  Measurable.of_eval fun p => measurable_const.mul (by fun_prop)

/-- …and bounded by `‖u‖ C` when `‖a‖ ≤ C`. -/
theorem abs_coef_le {C : ℝ} {z : EucSpace d} (hz : ‖z‖ ≤ C) (u : EucSpace d)
    (p : Fin d × Fin d) : |u p.1 * z p.2| ≤ ‖u‖ * C := by
  rw [abs_mul, ← Real.norm_eq_abs, ← Real.norm_eq_abs]
  exact mul_le_mul (PiLp.norm_apply_le u p.1) ((PiLp.norm_apply_le z p.2).trans hz)
    (norm_nonneg _) (norm_nonneg _)

variable {V : Ω → Matrix (Fin d) (Fin d) ℝ} {Y : Ω → EucSpace d}

/-- `⟨u, V Y⟩²` is integrable when the entries of `V` are square-integrable and
`Y` is bounded. -/
theorem integrable_inner_toEuclideanLin_sq [IsFiniteMeasure P] (hY : Measurable Y)
    (hint : ∀ p q : Fin d × Fin d, Integrable (fun ω => V ω p.1 p.2 * V ω q.1 q.2) P)
    {C : ℝ} (hC : ∀ ω, ‖Y ω‖ ≤ C) (u : EucSpace d) :
    Integrable (fun ω => inner (𝕜 := ℝ) u (Matrix.toEuclideanLin (V ω) (Y ω)) ^ 2) P := by
  simp_rw [inner_toEuclideanLin_eq_sum]
  exact integrable_sum_mul_sq (X := fun ω p => V ω p.1 p.2)
    (c := fun (z : EucSpace d) (p : Fin d × Fin d) => u p.1 * z p.2) hY hint (measurable_coef u)
    fun ω => abs_coef_le (hC ω) u

/-- **`E⟨u, V m⟩² = σ_V² ‖u‖² E‖m‖²`.**  If the entries of `V` are uncorrelated
with variance `σ²` and `V` is independent of the bounded vector `Y`, the linear
form `⟨u, V Y⟩` has second moment `σ² ‖u‖² E‖Y‖²`.

Source: arXiv:2604.01978v1, §2.3.3, item (i) (the computation behind it). -/
theorem integral_inner_toEuclideanLin_sq [IsFiniteMeasure P] (hV : Measurable V)
    (hY : Measurable Y) (hVY : IndepFun V Y P) {σ2 : ℝ}
    (hmom : ∀ p q : Fin d × Fin d,
      ∫ ω, V ω p.1 p.2 * V ω q.1 q.2 ∂P = if p = q then σ2 else 0)
    (hint : ∀ p q : Fin d × Fin d, Integrable (fun ω => V ω p.1 p.2 * V ω q.1 q.2) P)
    {C : ℝ} (hC : ∀ ω, ‖Y ω‖ ≤ C) (u : EucSpace d) :
    ∫ ω, inner (𝕜 := ℝ) u (Matrix.toEuclideanLin (V ω) (Y ω)) ^ 2 ∂P =
      σ2 * ‖u‖ ^ 2 * ∫ ω, ‖Y ω‖ ^ 2 ∂P := by
  have hflat : Measurable fun (M : Matrix (Fin d) (Fin d) ℝ) (p : Fin d × Fin d) => M p.1 p.2 :=
    Measurable.of_eval fun _ => Matrix.measurable_apply
  have key := integral_sum_mul_sq (X := fun ω p => V ω p.1 p.2)
    (c := fun (z : EucSpace d) (p : Fin d × Fin d) => u p.1 * z p.2) (hflat.comp hV) hY
    (hVY.comp hflat measurable_id) hmom hint (measurable_coef u) fun ω => abs_coef_le (hC ω) u
  simp_rw [inner_toEuclideanLin_eq_sum]
  refine key.trans ?_
  simp_rw [sum_mul_apply_sq]
  rw [integral_const_mul, mul_assoc]

/-- **`E‖Proj_e V m‖² = σ_V² (d-1) E‖m‖²`.**  Under the hypotheses of
`integral_inner_toEuclideanLin_sq`, projecting `V Y` onto the tangent space at
a unit vector `e` removes one of `d` directions of equal variance.

Source: arXiv:2604.01978v1, §2.3.3, item (i) (the computation behind it). -/
theorem integral_norm_proj_toEuclideanLin_sq [IsFiniteMeasure P] (hV : Measurable V)
    (hY : Measurable Y) (hVY : IndepFun V Y P) {σ2 : ℝ}
    (hmom : ∀ p q : Fin d × Fin d,
      ∫ ω, V ω p.1 p.2 * V ω q.1 q.2 ∂P = if p = q then σ2 else 0)
    (hint : ∀ p q : Fin d × Fin d, Integrable (fun ω => V ω p.1 p.2 * V ω q.1 q.2) P)
    {C : ℝ} (hC : ∀ ω, ‖Y ω‖ ≤ C) {e : EucSpace d} (he : ‖e‖ = 1) :
    ∫ ω, ‖proj d e (Matrix.toEuclideanLin (V ω) (Y ω))‖ ^ 2 ∂P =
      σ2 * ((d : ℝ) - 1) * ∫ ω, ‖Y ω‖ ^ 2 ∂P := by
  have h : ∀ ω, ‖proj d e (Matrix.toEuclideanLin (V ω) (Y ω))‖ ^ 2 =
      ∑ i, inner (𝕜 := ℝ) (EuclideanSpace.single i (1 : ℝ))
          (Matrix.toEuclideanLin (V ω) (Y ω)) ^ 2 -
        inner (𝕜 := ℝ) e (Matrix.toEuclideanLin (V ω) (Y ω)) ^ 2 := fun ω => by
    rw [norm_proj_sq he, norm_sq_eq_sum_inner_single_sq]
  have hI := integrable_inner_toEuclideanLin_sq hY hint hC
  simp_rw [h]
  rw [integral_sub (integrable_finsetSum _ fun i _ => hI _) (hI e),
    integral_finsetSum _ fun i _ => hI _]
  simp only [integral_inner_toEuclideanLin_sq hV hY hVY hmom hint hC, PiLp.norm_single, he,
    norm_one, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- The hypotheses of `norm_proj_sq`, `abs_coef_le`,
`integrable_inner_toEuclideanLin_sq`, `integral_inner_toEuclideanLin_sq` and
`integral_norm_proj_toEuclideanLin_sq` are satisfiable: `V = 0`, of variance
`0`, `Y = 0`, and a coordinate vector for `e`. -/
example (d : ℕ) :
    IndepFun (fun _ : Unit => (0 : Matrix (Fin d) (Fin d) ℝ)) (fun _ : Unit => (0 : EucSpace d))
        (Measure.dirac ()) ∧
      (∀ p q : Fin d × Fin d, ∫ _ω, (0 : Matrix (Fin d) (Fin d) ℝ) p.1 p.2 *
        (0 : Matrix (Fin d) (Fin d) ℝ) q.1 q.2 ∂(Measure.dirac ()) = if p = q then 0 else 0) ∧
      ‖(0 : EucSpace (d + 1))‖ ≤ 0 ∧ ‖(EuclideanSpace.single 0 1 : EucSpace (d + 1))‖ = 1 :=
  ⟨indepFun_const_left _ _, fun _ _ => by simp, by simp, by simp [PiLp.norm_single]⟩

end Homogenized
end Transformer

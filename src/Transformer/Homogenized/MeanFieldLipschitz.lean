/-
# Homogenized Transformers — the diffusion kernel is Lipschitz

Formalization of `prop:satisfying_MF` of arXiv:2604.01978v1, *Homogenized
Transformers*: the coefficient `G_μ(x,·)` of `eq:G_def` is Lipschitz jointly
in the measure, for `W₂`, and in the point.  This is the input the proof of
`thm:PoC_wellposedness` runs on.

**What the source says and what is carried here.**  The source displays the
constant

  `C = O(β² e^{4β‖Ā‖} e^{β²dσ_A²})`,

and that rate is false: `not_forall_satisfying_MF_rate` refutes it even at one
fixed model, with the constant allowed to depend on all model data but `β`.

* It vanishes as `β ↓ 0`, and the left-hand side does not.  At small `β` the
  attention is nearly uniform and `m_A[μ](x)` nearly independent of `x`, but
  `Proj_x` still depends on `x`, and `Proj_x u - Proj_y u` is of order
  `‖x-y‖`, not `β²‖x-y‖`.  This is what the refutation proves.
* Read with a constant uniform over models, it fails a second time: it carries
  no scale of the value matrix.  The identity the proof opens with reads

  `‖G_μ(x,·)-G_ν(y,·)‖²_{L²} = dσ_V² ‖Proj_x m_A[μ](x) - Proj_y m_A[ν](y)‖²_{L²}
    + ‖Proj_x V̄ Cov^A_μ(x) - Proj_y V̄ Cov^A_ν(y)‖²_{L²}`,

  whose two sides scale together under `(σ_V, V̄) ↦ (λσ_V, λV̄)`, while the
  displayed `C` mentions neither.  `lem:Kernel_regularity` of the same paper
  does carry the factor `dσ_V²`.  This is a remark; the refutation does not
  rest on it.

The witness is `d = 2`, `Ā = 0` and `A ≡ 0`, so both exponentials are `1` at
every `β`; one Rademacher entry in `V`, by `isHighOrderLaw_rademacher`; and
`μ = ν = δ_{e₁}`, `x = e₁`, `y = e₂`, for which the left-hand side is `1` and
the right-hand side at most `2Kβ²`.

What survives is the qualitative statement, and it is exactly what the proof of
`thm:PoC_wellposedness` uses of this proposition — "the coefficient `G_μ(x,·)`
is globally Lipschitz in `x` and `μ`": a constant produced after the model data
`(β, σ_V, σ_A, ρ*)` is fixed.  That is `satisfying_MF`.

Source: arXiv:2604.01978v1, `prop:satisfying_MF`.
-/

import Transformer.Homogenized.RademacherLaw
import Transformer.Homogenized.MeanField
import Transformer.Wasserstein

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The fields at a point mass -/

/-- The attention field at a point mass is the value map read at the atom: the
normalizer cancels the single weight, at every temperature and every head. -/
theorem attnFieldOf_dirac {d : ℕ} (β : ℝ) (θ : HeadParam d) (z x : EucSpace d) :
    attnFieldOf β θ (Measure.dirac z) x = valueMap θ z := by
  rw [attnFieldOf, integral_dirac, integral_dirac, smul_smul,
    inv_mul_cancel₀ (attnWeight_pos β θ x z).ne', one_smul]

/-- **`eq:G_def` at a point mass**: `G_{δ_z}(x,θ) = Proj_x(V z - 𝔼V z)`.  The
head enters only through its value matrix, and the point only through the
projection. -/
theorem GfieldOf_dirac {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) (z x : EucSpace d)
    (θ : HeadParam d) :
    GfieldOf β ρ (Measure.dirac z) x θ
      = proj d x (valueMap θ z - ∫ θ', valueMap θ' z ∂ρ) := by
  rw [GfieldOf, fluctOf, meanFieldOf, attnFieldOf_dirac]
  simp only [attnFieldOf_dirac]

/-! ### The proposition -/

/-- **Proposition (prop:satisfying_MF).**  Under `ass:high_order_short` the
diffusion kernel of `eq:G_def` is Lipschitz in its two arguments:

  `‖G_μ(x,·) - G_ν(y,·)‖_{L²(ρ*)} ≤ C (W₂(μ,ν) + ‖x-y‖)`

for all probability measures `μ, ν` on the sphere and all unit `x, y`.

The constant is produced after the model data is fixed.  The rate the source
displays for it is refuted by `not_forall_satisfying_MF_rate`; the module
docstring records what it says and why it is not carried.

Not proved here.

Source: arXiv:2604.01978v1, `prop:satisfying_MF`. -/
theorem satisfying_MF {d : ℕ} (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : HasHighOrderLaw d σV σA ρ) :
    ∃ L : ℝ, 0 < L ∧
      ∀ (μ ν : Measure (EucSpace d)), IsProbabilityMeasure μ → IsProbabilityMeasure ν →
        μ {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 → ν {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        Real.sqrt (∫ θ, ‖GfieldOf β ρ μ x θ - GfieldOf β ρ ν y θ‖ ^ 2 ∂ρ) ≤
          L * (Wasserstein.W2 μ ν + ‖x - y‖) := by
  sorry

/-- The hypothesis of `satisfying_MF` is satisfiable, and not only at a law
whose fluctuation vanishes: the Rademacher law of `isHighOrderLaw_rademacher`
satisfies `ass:high_order_short`. -/
example : HasHighOrderLaw 2 1 0 radLaw := hasHighOrderLaw_rademacher

/-! ### The displayed rate is false -/

/-- The value matrix of the Rademacher law read at `e₁`: the first column,
which is `±e₁`. -/
theorem valueMap_radMatrix (b : Bool) :
    valueMap ((radMatrix b, 0) : HeadParam 2) (EuclideanSpace.single 0 1)
      = (if b then (1 : ℝ) else -1) •
        (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2) := by
  ext i
  cases b <;> simp [valueMap, Matrix.toLpLin_apply, radMatrix_apply]
  split_ifs <;> norm_num

/-- The mean value map of the Rademacher law vanishes: the two atoms are
opposite. -/
theorem integral_valueMap_radLaw :
    ∫ θ, valueMap θ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2) ∂radLaw = 0 := by
  rw [integral_radLaw, valueMap_radMatrix true, valueMap_radMatrix false]
  norm_num

/-- `Proj_{e₁}(c e₁) - Proj_{e₂}(c e₁) = -c e₁`: the first projection kills a
multiple of its own direction and the second leaves it alone.  This is the
`β`-free part of the left-hand side of `prop:satisfying_MF`. -/
theorem norm_proj_sub_proj_sq (c : ℝ) :
    ‖proj 2 (EuclideanSpace.single 0 1)
          (c • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))
        - proj 2 (EuclideanSpace.single 1 1)
          (c • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))‖ ^ 2 = c ^ 2 := by
  have h0 : proj 2 (EuclideanSpace.single 0 1)
      (c • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)) = 0 := by
    rw [proj, real_inner_smul_right, real_inner_self_eq_norm_sq]
    simp [PiLp.norm_single]
  have h1 : proj 2 (EuclideanSpace.single 1 1)
      (c • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))
      = c • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2) := by
    rw [proj, real_inner_smul_right]
    simp [EuclideanSpace.inner_single_left]
  rw [h0, h1, zero_sub, norm_neg, norm_smul, PiLp.norm_single, Real.norm_eq_abs,
    Real.norm_eq_abs]
  simp [sq_abs]

/-- At the Rademacher law, with base measure the point mass at `e₁` and the two
points `e₁` and `e₂`, the left-hand side of `prop:satisfying_MF` is `1` — at
every temperature, the query-key matrix of that law being `0`. -/
theorem sqrt_integral_norm_GfieldOf_radLaw (β : ℝ) :
    Real.sqrt (∫ θ,
        ‖GfieldOf β radLaw (Measure.dirac (EuclideanSpace.single 0 1))
            (EuclideanSpace.single 0 1) θ
          - GfieldOf β radLaw (Measure.dirac (EuclideanSpace.single 0 1))
            (EuclideanSpace.single 1 1) θ‖ ^ 2 ∂radLaw) = 1 := by
  have hG : ∀ (x : EucSpace 2) (θ : HeadParam 2),
      GfieldOf β radLaw (Measure.dirac (EuclideanSpace.single 0 1)) x θ
        = proj 2 x (valueMap θ (EuclideanSpace.single 0 1)) := fun x θ => by
    rw [GfieldOf_dirac, integral_valueMap_radLaw, sub_zero]
  simp only [hG]
  rw [integral_radLaw, valueMap_radMatrix true, valueMap_radMatrix false,
    show (if true then (1 : ℝ) else -1) = 1 by norm_num,
    show (if false then (1 : ℝ) else -1) = -1 by norm_num,
    norm_proj_sub_proj_sq, norm_proj_sub_proj_sq]
  norm_num

/-- **The rate of `prop:satisfying_MF` is false.**  Not even for one fixed
model: there is a model `ass:high_order_short` allows for which no `K`, however
it depends on the model data `(d, σ_V, σ_A, ρ*, Ā)`, satisfies

  `‖G_μ(x,·) - G_ν(y,·)‖_{L²(ρ*)} ≤ K β² e^{4β‖Ā‖} e^{β²dσ_A²} (W₂(μ,ν) + ‖x-y‖)`

for every `β > 0`.  This is the weakest reading of `O(·)`: the constant is
only required not to depend on `β`, which the display does make explicit, nor
on the points and measures the inequality is quantified over.  The bound on
`Ā` is
carried as an operator bound `opA`, so the refutation does not depend on which
matrix norm `‖Ā‖` is read in.

The witness is `d = 2` and the Rademacher law of `isHighOrderLaw_rademacher`:
`Ā = 0` and `A ≡ 0`, so `σ_A = 0` and both exponentials are `1`; `μ = ν` is
the point mass at `e₁`, so `W₂(μ,ν) = 0`; and `x = e₁`, `y = e₂`.  The
left-hand side is `1` at every `β`, by `sqrt_integral_norm_GfieldOf_radLaw`,
while the right-hand side is at most `2Kβ²`, which is `< 1` at `β = 1/(K+2)`.

Source: arXiv:2604.01978v1, `prop:satisfying_MF`. -/
theorem not_forall_satisfying_MF_rate :
    ¬ ∀ (d : ℕ) (C σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (Ω : Type) [MeasurableSpace Ω]
        (P : Measure Ω) (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
        (mV mA : Matrix (Fin d) (Fin d) ℝ),
        IsHighOrderLaw d C σV σA ρ P Vr Wr Wr' mV mA →
      ∀ opA : ℝ, (∀ v : EucSpace d, ‖Matrix.toEuclideanLin mA v‖ ≤ opA * ‖v‖) →
      ∃ K : ℝ, 0 < K ∧ ∀ β : ℝ, 0 < β →
      ∀ (μ ν : Measure (EucSpace d)), IsProbabilityMeasure μ → IsProbabilityMeasure ν →
        μ {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 → ν {z : EucSpace d | ‖z‖ = 1}ᶜ = 0 →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        Real.sqrt (∫ θ, ‖GfieldOf β ρ μ x θ - GfieldOf β ρ ν y θ‖ ^ 2 ∂ρ) ≤
          K * β ^ 2 * Real.exp (4 * β * opA) *
              Real.exp (β ^ 2 * (d : ℝ) * (σA : ℝ) ^ 2) *
            (Wasserstein.W2 μ ν + ‖x - y‖) := by
  intro hall
  obtain ⟨K, hK, h⟩ := hall 2 1 1 0 radLaw Bool fairCoin radMatrix 0 0 0 0
    isHighOrderLaw_rademacher 0 (fun v => by simp)
  have hK2 : (0 : ℝ) < K + 2 := by linarith
  have hβ : (0 : ℝ) < 1 / (K + 2) := by positivity
  have hsph : MeasurableSet {z : EucSpace 2 | ‖z‖ = 1} :=
    (isClosed_eq continuous_norm continuous_const).measurableSet
  have hnorm : ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 := by
    simp [PiLp.norm_single]
  have hnorm' : ‖(EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 := by
    simp [PiLp.norm_single]
  have hsupp : (Measure.dirac (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))
      {z : EucSpace 2 | ‖z‖ = 1}ᶜ = 0 := by
    rw [Measure.dirac_apply' _ hsph.compl,
      Set.indicator_of_notMem (by simp)]
  have key := h (1 / (K + 2)) hβ
    (Measure.dirac (EuclideanSpace.single 0 1))
    (Measure.dirac (EuclideanSpace.single 0 1)) inferInstance inferInstance hsupp hsupp
    (EuclideanSpace.single 0 1) (EuclideanSpace.single 1 1) hnorm hnorm'
  rw [sqrt_integral_norm_GfieldOf_radLaw,
    Wasserstein.W2_self (Measure.dirac (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))]
    at key
  have hxy : ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
      - (EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2)‖ ≤ 2 := by
    refine le_trans (norm_sub_le _ _) ?_
    rw [hnorm, hnorm']
    norm_num
  rw [show (4 : ℝ) * (1 / (K + 2)) * 0 = 0 by ring, Real.exp_zero,
    show ((1 : ℝ) / (K + 2)) ^ 2 * ((2 : ℕ) : ℝ) * (((0 : ℝ≥0) : ℝ)) ^ 2 = 0 by norm_num,
    Real.exp_zero, mul_one, mul_one, zero_add] at key
  have hsq : (0 : ℝ) < (1 / (K + 2)) ^ 2 := by positivity
  have hbound : K * (1 / (K + 2)) ^ 2 *
      ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
        - (EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2)‖
      ≤ 2 * K * (1 / (K + 2)) ^ 2 := by
    have hmul := mul_le_mul_of_nonneg_left hxy (mul_nonneg hK.le hsq.le)
    linarith
  have hlt : 2 * K * (1 / (K + 2)) ^ 2 < 1 := by
    rw [div_pow, one_pow, mul_one_div, div_lt_one (by positivity)]
    nlinarith
  linarith

end Homogenized
end Transformer

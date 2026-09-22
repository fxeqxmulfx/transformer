/-
# §3.2 — The interaction energy through the moments of `μ`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2.

The groundwork of `eq: dissipation.softmax`.  The energy is a quadratic form in
`μ`, and the continuity equation only differentiates linear functionals
`∫ φ dμ(t)`.  The monomials `m_I(x) = ∏_j x_{I_j}` split the kernel:

  `⟨x, y⟩^k = Σ_{I : [k] → [d]} m_I(x) m_I(y)`   (`inner_pow_eq_sum_mono`),

so `∫∫ ⟨x, y⟩^k dμ dμ = Σ_I (∫ m_I dμ)²` (`integral_prod_inner_pow`), a sum of
squares of linear functionals, and the exponential kernel is the sum of these
over `k` with weights `β^k/k!`.
-/

import Transformer.Perspective.Section2_EnergyKernel
import Mathlib.MeasureTheory.Integral.Prod

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- The monomial `m_I(x) = ∏_j x_{I_j}`. -/
noncomputable def mono {k : ℕ} (I : Fin k → Fin d) (x : EucSpace d) : ℝ := ∏ j, x (I j)

/-- `⟨x, y⟩^k = Σ_I m_I(x) m_I(y)`. -/
theorem inner_pow_eq_sum_mono (k : ℕ) (x y : EucSpace d) :
    inner (𝕜 := ℝ) x y ^ k = ∑ I : Fin k → Fin d, mono I x * mono I y := by
  rw [PiLp.inner_apply, ← Fin.prod_const, Finset.prod_univ_sum, Fintype.piFinset_univ]
  refine Finset.sum_congr rfl fun I _ => ?_
  simp only [mono, ← Finset.prod_mul_distrib, RCLike.inner_apply, conj_trivial, mul_comm]

/-- `m_I` is continuous. -/
theorem continuous_mono {k : ℕ} (I : Fin k → Fin d) : Continuous (mono I) := by
  unfold mono; fun_prop

/-- `m_I` is smooth, so it is a test function for `eq: CE`. -/
theorem contDiff_mono {k : ℕ} (I : Fin k → Fin d) (n : ℕ∞) : ContDiff ℝ n (mono I) := by
  unfold mono
  exact contDiff_prod fun j _ => (EuclideanSpace.proj (I j) : EucSpace d →L[ℝ] ℝ).contDiff

/-- `|m_I(x)| ≤ 1` on the sphere. -/
theorem abs_mono_le_one {k : ℕ} (I : Fin k → Fin d) (x : SSphere d) :
    |mono I (x : EucSpace d)| ≤ 1 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [mono, Finset.abs_prod]
  refine Finset.prod_le_one₀ (fun _ _ => abs_nonneg _) fun j _ => ?_
  have := PiLp.norm_apply_le (x : EucSpace d) (I j)
  rwa [hx, Real.norm_eq_abs] at this

/-- `m_I` is integrable against any finite measure on the sphere. -/
theorem integrable_mono {k : ℕ} (I : Fin k → Fin d) (ν : Measure (SSphere d))
    [IsFiniteMeasure ν] : Integrable (fun x : SSphere d => mono I (x : EucSpace d)) ν :=
  (integrable_const (1 : ℝ)).mono'
    ((continuous_mono I).comp continuous_subtype_val).aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => by rw [Real.norm_eq_abs]; exact abs_mono_le_one I x)

/-- `∫∫ ⟨x, y⟩^k dν dν = Σ_I (∫ m_I dν)²`. -/
theorem integral_prod_inner_pow (k : ℕ) (ν : Measure (SSphere d)) [IsFiniteMeasure ν] :
    ∫ p : SSphere d × SSphere d, inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ k
        ∂ν.prod ν =
      ∑ I : Fin k → Fin d, (∫ x : SSphere d, mono I (x : EucSpace d) ∂ν) ^ 2 := by
  simp_rw [inner_pow_eq_sum_mono]
  rw [integral_finsetSum _ fun I _ => ?_]
  · refine Finset.sum_congr rfl fun I _ => ?_
    rw [integral_prod_mul (fun x : SSphere d => mono I (x : EucSpace d))
      (fun x : SSphere d => mono I (x : EucSpace d)), sq]
  · exact (integrable_mono I ν).mul_prod (integrable_mono I ν)

/-- The derivative of `x ↦ ⟨x, y⟩^k` in the direction `w`:
`Σ_I m_I(y) Dm_I(x)[w] = k ⟨x, y⟩^{k-1} ⟨y, w⟩`. -/
theorem sum_mono_mul_fderiv_mono (k : ℕ) (x y w : EucSpace d) :
    ∑ I : Fin k → Fin d, mono I y * fderiv ℝ (mono I) x w =
      k * inner (𝕜 := ℝ) x y ^ (k - 1) * inner (𝕜 := ℝ) y w := by
  have hd : ∀ I : Fin k → Fin d, DifferentiableAt ℝ (mono I) x := fun I =>
    ((contDiff_mono I 1).differentiable one_ne_zero).differentiableAt
  have hsum : HasFDerivAt (fun z => ∑ I : Fin k → Fin d, mono I y * mono I z)
      (∑ I : Fin k → Fin d, mono I y • fderiv ℝ (mono I) x) x :=
    HasFDerivAt.fun_sum fun I _ => ((hd I).hasFDerivAt.const_mul (mono I y))
  have hpow : HasFDerivAt (fun z => inner (𝕜 := ℝ) y z ^ k)
      ((k • inner (𝕜 := ℝ) y x ^ (k - 1)) • innerSL ℝ y) x :=
    (innerSL ℝ y).hasFDerivAt.pow k
  have heq : (fun z => ∑ I : Fin k → Fin d, mono I y * mono I z) =
      fun z => inner (𝕜 := ℝ) y z ^ k := by
    funext z; rw [inner_pow_eq_sum_mono]
  rw [heq] at hsum
  have := congrArg (fun L : EucSpace d →L[ℝ] ℝ => L w) (hsum.unique hpow)
  simp only [FunLike.coe_sum, Finset.sum_apply, smul_apply,
    smul_eq_mul, innerSL_apply_apply, nsmul_eq_mul] at this
  rw [this, real_inner_comm x y]

/-- `⟨∇φ(x), w⟩ = Dφ(x)[w]`. -/
theorem inner_gradient_eq_fderiv (φ : EucSpace d → ℝ) (x w : EucSpace d) :
    inner (𝕜 := ℝ) (gradient φ x) w = fderiv ℝ φ x w := by
  rw [gradient, InnerProductSpace.toDual_symm_apply]

/-- A continuous function on a compact space is integrable for a finite measure. -/
theorem integrable_of_continuous_compact {X E : Type*} [TopologicalSpace X] [CompactSpace X]
    [MeasurableSpace X] [OpensMeasurableSpace X] [NormedAddCommGroup E] {f : X → E}
    (hf : Continuous f) (ρ : Measure X) [IsFiniteMeasure ρ] : Integrable f ρ :=
  hf.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace f)

/-- What the continuity equation produces from the `k`-th term of the energy,
summed back over `I`:
`Σ_I (∫ m_I dν) (∫ ⟨∇m_I, w⟩ dν) = ∫∫ k ⟨x, y⟩^{k-1} ⟨y, w(x)⟩ dν(x) dν(y)`. -/
theorem sum_integral_mono_mul_integral_gradient (k : ℕ) (ν : Measure (SSphere d))
    [IsFiniteMeasure ν] (w : SSphere d → EucSpace d) (hw : Continuous w) :
    ∑ I : Fin k → Fin d, (∫ y : SSphere d, mono I (y : EucSpace d) ∂ν) *
        ∫ x : SSphere d, inner (𝕜 := ℝ) (gradient (mono I) (x : EucSpace d)) (w x) ∂ν =
      ∫ p : SSphere d × SSphere d, k * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)
          ^ (k - 1) * inner (𝕜 := ℝ) (p.2 : EucSpace d) (w p.1) ∂ν.prod ν := by
  simp_rw [inner_gradient_eq_fderiv, ← sum_mono_mul_fderiv_mono]
  have hc : ∀ I : Fin k → Fin d,
      Continuous fun x : SSphere d => fderiv ℝ (mono I) (x : EucSpace d) (w x) := fun I =>
    (((contDiff_mono I 1).continuous_fderiv one_ne_zero).comp continuous_subtype_val).clm_apply hw
  rw [integral_finsetSum _ fun I _ => ?_]
  swap
  · exact integrable_of_continuous_compact
      (((continuous_mono I).comp (continuous_subtype_val.comp continuous_snd)).mul
        ((hc I).comp continuous_fst)) _
  refine Finset.sum_congr rfl fun I _ => ?_
  simp_rw [mul_comm (mono I _)]
  rw [integral_prod_mul (fun x : SSphere d => fderiv ℝ (mono I) (x : EucSpace d) (w x))
    (fun y : SSphere d => mono I (y : EucSpace d)), mul_comm]

end Perspective
end Transformer

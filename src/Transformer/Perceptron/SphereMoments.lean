/-
# Moments on the sphere, and the exponential kernel that sees them

The injectivity of `μ ↦ f^μ` in `lem: quadpol`, at the level of measures
(`eq_of_integral_exp_inner_eq`), and the two facts it rests on.

* A finite measure on `𝕊^{d-1}` is determined by its moments
  `m_I(μ) = ∫ x_{I 0} ⋯ x_{I (k-1)} dμ(x)` (`measure_eq_of_integral_mono_eq`):
  the coordinates separate the points of the sphere, so by Stone–Weierstrass
  the algebra they generate is dense in `C(𝕊^{d-1})`, and a finite measure is
  determined by its integrals against a dense algebra.
* `∫∫ e^{β x·y} db(y) da(x) = Σ_k β^k/k! Σ_I m_I(a) m_I(b)`
  (`hasSum_integral_exp_inner`), the two-measure form of the expansion
  `Perspective.hasSum_interactionEnergy` runs on.

For `β > 0` every coefficient `β^k/k!` is positive, so the kernel is positive
definite: `T(a, b) = ∫∫ e^{β x·y} db da` satisfies
`T(μ₁,μ₁) - T(μ₁,μ₂) - T(μ₂,μ₁) + T(μ₂,μ₂) = Σ_k β^k/k! Σ_I (m_I(μ₁) - m_I(μ₂))²`.
`f^{μ₁} = f^{μ₂}` makes the left side vanish, hence every moment agrees.

The source proves injectivity through the Funk–Hecke formula instead: the
spherical-harmonic coefficients of `f^ν` are `λ_j(β) m_{jℓ}(ν)` with
`λ_j(β) > 0`.  Both routes end in the same Stone–Weierstrass step; this one
needs no spherical harmonics, which Mathlib does not have.

Source: arXiv:2601.21366v2, `lem: quadpol` (injectivity) and its proof.
-/

import Transformer.Perspective.Section2_EnergySeries
import Mathlib.MeasureTheory.Measure.FiniteMeasureExt

open scoped BigOperators Nat BoundedContinuousFunction
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Moments determine the measure -/

/-- The coordinate `x ↦ x_i` of the sphere, as a bounded continuous function. -/
noncomputable def coordBCF (i : Fin d) : SSphere d →ᵇ ℝ :=
  BoundedContinuousFunction.mkOfCompact ⟨fun x => (x : EucSpace d) i, by fun_prop⟩

@[simp] theorem coordBCF_apply (i : Fin d) (x : SSphere d) :
    coordBCF i x = (x : EucSpace d) i := rfl

/-- A product of coordinates is a monomial `x ↦ x_{I 0} ⋯ x_{I (k-1)}`. -/
theorem exists_mono_of_mem_closure {g : SSphere d →ᵇ ℝ}
    (hg : g ∈ Submonoid.closure (Set.range coordBCF ∪ star (Set.range coordBCF))) :
    ∃ k, ∃ I : Fin k → Fin d, ∀ x : SSphere d, g x = Perspective.mono I (x : EucSpace d) := by
  induction hg using Submonoid.closure_induction with
  | mem g hg =>
    obtain ⟨i, rfl⟩ : ∃ i, coordBCF (d := d) i = g := by
      rcases hg with hg | hg
      · exact hg
      · obtain ⟨i, hi⟩ := Set.mem_star.1 hg
        exact ⟨i, by rw [hi]; ext x; simp⟩
    exact ⟨1, fun _ => i, fun x => by simp [Perspective.mono]⟩
  | one => exact ⟨0, Fin.elim0, fun x => by simp [Perspective.mono]⟩
  | mul f g _ _ hf hg =>
    obtain ⟨k, I, hI⟩ := hf
    obtain ⟨l, J, hJ⟩ := hg
    refine ⟨k + l, Fin.append I J, fun x => ?_⟩
    rw [BoundedContinuousFunction.mul_apply, hI, hJ, Perspective.mono, Perspective.mono,
      Perspective.mono, Fin.prod_univ_add]
    simp [Fin.append_left, Fin.append_right]

/-- **A finite measure on `𝕊^{d-1}` is determined by its moments.**  If
`∫ x_{I 0} ⋯ x_{I (k-1)} dμ₁ = ∫ x_{I 0} ⋯ x_{I (k-1)} dμ₂` for every `k` and
every `I : Fin k → Fin d`, then `μ₁ = μ₂`.

The step the source takes as "spherical polynomials are uniformly dense in
`C⁰(𝕊^{d-1})`" and "uniqueness in the Riesz representation theorem": the
coordinates separate points, so the star algebra they generate satisfies the
hypothesis of Mathlib's Stone–Weierstrass uniqueness theorem for finite measures.

Source: arXiv:2601.21366v2, proof of `lem: quadpol` (injectivity). -/
theorem measure_eq_of_integral_mono_eq (μ₁ μ₂ : Measure (SSphere d))
    [IsFiniteMeasure μ₁] [IsFiniteMeasure μ₂]
    (h : ∀ k (I : Fin k → Fin d), ∫ x, Perspective.mono I (x : EucSpace d) ∂μ₁ =
      ∫ x, Perspective.mono I (x : EucSpace d) ∂μ₂) : μ₁ = μ₂ := by
  refine ext_of_forall_mem_subalgebra_integral_eq_of_pseudoEMetric_complete_countable
    (A := StarAlgebra.adjoin ℝ (Set.range (coordBCF (d := d)))) ?_ ?_
  · intro x y hxy
    obtain ⟨i, hi⟩ : ∃ i, (x : EucSpace d) i ≠ (y : EucSpace d) i := by
      by_contra hne
      push Not at hne
      exact hxy (Subtype.ext (PiLp.ext hne))
    refine ⟨_, ⟨BoundedContinuousFunction.toContinuousMapStarₐ ℝ (coordBCF i), ?_, rfl⟩, ?_⟩
    · exact StarSubalgebra.mem_map.2
        ⟨coordBCF i, StarAlgebra.subset_adjoin ℝ _ ⟨i, rfl⟩, rfl⟩
    · simpa using hi
  · intro g hg
    have hg' : g ∈ Subalgebra.toSubmodule
        (Algebra.adjoin ℝ (Set.range (coordBCF (d := d)) ∪ star (Set.range coordBCF))) := hg
    rw [Algebra.adjoin_eq_span] at hg'
    clear hg
    induction hg' using Submodule.span_induction with
    | mem g hg =>
      obtain ⟨k, I, hI⟩ := exists_mono_of_mem_closure hg
      simp_rw [hI]
      exact h k I
    | zero => simp
    | add f g _ _ hf hg =>
      simp only [BoundedContinuousFunction.coe_add, Pi.add_apply]
      rw [integral_add (f.integrable _) (g.integrable _),
        integral_add (f.integrable _) (g.integrable _), hf, hg]
    | smul a f _ hf =>
      simp only [BoundedContinuousFunction.coe_smul, smul_eq_mul]
      rw [integral_const_mul, integral_const_mul, hf]

/-- The hypothesis of `measure_eq_of_integral_mono_eq` is satisfiable: a
measure and itself. -/
example : ∀ k (I : Fin k → Fin 1),
    ∫ x, Perspective.mono I (x : EucSpace 1) ∂(0 : Measure (SSphere 1)) =
      ∫ x, Perspective.mono I (x : EucSpace 1) ∂(0 : Measure (SSphere 1)) :=
  fun _ _ => rfl

/-! ### The kernel `e^{β x·y}` in moments -/

/-- `∫∫ (x·y)^k da(x) db(y) = Σ_I m_I(a) m_I(b)`, the two-measure form of
`Perspective.integral_prod_inner_pow`. -/
theorem integral_prod_inner_pow_eq_sum (k : ℕ) (a b : Measure (SSphere d))
    [IsFiniteMeasure a] [IsFiniteMeasure b] :
    ∫ p : SSphere d × SSphere d, inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ k
        ∂a.prod b =
      ∑ I : Fin k → Fin d, (∫ x : SSphere d, Perspective.mono I (x : EucSpace d) ∂a) *
        ∫ y : SSphere d, Perspective.mono I (y : EucSpace d) ∂b := by
  simp_rw [Perspective.inner_pow_eq_sum_mono]
  rw [integral_finsetSum _ fun I _ =>
    (Perspective.integrable_mono I a).mul_prod (Perspective.integrable_mono I b)]
  exact Finset.sum_congr rfl fun I _ =>
    integral_prod_mul (fun x : SSphere d => Perspective.mono I (x : EucSpace d))
      (fun y : SSphere d => Perspective.mono I (y : EucSpace d))

/-- **`∫∫ e^{β x·y} db(y) da(x) = Σ_k β^k/k! Σ_I m_I(a) m_I(b)`**, for
probability measures `a`, `b` on the sphere. -/
theorem hasSum_integral_exp_inner (β : ℝ) (a b : Measure (SSphere d))
    [IsProbabilityMeasure a] [IsProbabilityMeasure b] :
    HasSum (fun k : ℕ => β ^ k / k ! * ∑ I : Fin k → Fin d,
        (∫ x : SSphere d, Perspective.mono I (x : EucSpace d) ∂a) *
          ∫ y : SSphere d, Perspective.mono I (y : EucSpace d) ∂b)
      (∫ x : SSphere d, ∫ y : SSphere d,
        exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂b ∂a) := by
  have hK := Perspective.continuous_inner_sphere (d := d)
  have h := Perspective.hasSum_integral_of_abs_le (a.prod b)
    (F := fun k p => β ^ k / k ! * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ k)
    (G := fun p => exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)))
    (u := fun k => |β| ^ k / k !) (fun k => continuous_const.mul (hK.pow k))
    (Real.summable_pow_div_factorial |β|) (fun k p => ?_)
    (fun p => Perspective.hasSum_exp_mul β _)
  · have hint : Integrable (fun p : SSphere d × SSphere d =>
        exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))) (a.prod b) :=
      Perspective.integrable_of_continuous_compact (continuous_const.mul hK).rexp _
    rw [← integral_prod _ hint]
    convert h using 2 with k
    rw [integral_const_mul, integral_prod_inner_pow_eq_sum]
  · rw [abs_mul, abs_pow, abs_div, abs_pow, Nat.abs_cast]
    refine mul_le_of_le_one_right (by positivity) (pow_le_one₀ (abs_nonneg _) ?_)
    exact Perspective.abs_inner_sphere_le_one p.1 p.2

/-! ### Injectivity -/

/-- **Lemma (lem: quadpol), injectivity, for measures.**  For `β > 0`, if
`∫ e^{β x·y} dμ₁(y) = ∫ e^{β x·y} dμ₂(y)` for every `x ∈ 𝕊^{d-1}`, then the
probability measures `μ₁` and `μ₂` are equal.

The source's proof expands `f^{μ₁ - μ₂}` in spherical harmonics; this one
expands the kernel in monomials, see the module docstring.  The conclusion is
the same.

Source: arXiv:2601.21366v2, `lem: quadpol`. -/
theorem eq_of_integral_exp_inner_eq (β : ℝ) (hβ : 0 < β) (μ₁ μ₂ : Measure (SSphere d))
    [IsProbabilityMeasure μ₁] [IsProbabilityMeasure μ₂]
    (h : ∀ x : SSphere d,
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₁ =
        ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₂) :
    μ₁ = μ₂ := by
  have hT : ∀ a : Measure (SSphere d),
      ∫ x, ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₁ ∂a =
        ∫ x, ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂μ₂ ∂a :=
    fun a => congrArg (fun f : SSphere d → ℝ => ∫ x, f x ∂a) (funext h)
  have h1 := (hasSum_integral_exp_inner β μ₁ μ₁).sub (hasSum_integral_exp_inner β μ₁ μ₂)
  have h2 := (hasSum_integral_exp_inner β μ₂ μ₁).sub (hasSum_integral_exp_inner β μ₂ μ₂)
  rw [hT, sub_self] at h1 h2
  have h3 := h1.sub h2
  rw [sub_self] at h3
  -- the terms of `T(μ₁,μ₁) - T(μ₁,μ₂) - T(μ₂,μ₁) + T(μ₂,μ₂)`
  have h4 : HasSum (fun k : ℕ => β ^ k / k ! * ∑ I : Fin k → Fin d,
      ((∫ x, Perspective.mono I (x : EucSpace d) ∂μ₁) -
        ∫ x, Perspective.mono I (x : EucSpace d) ∂μ₂) ^ 2) 0 := by
    convert h3 using 1
    funext k
    simp only [← mul_sub, ← Finset.sum_sub_distrib]
    congr 1
    exact Finset.sum_congr rfl fun I _ => by ring
  rw [hasSum_zero_iff_of_nonneg fun k => by positivity] at h4
  refine measure_eq_of_integral_mono_eq _ _ fun k I => ?_
  have hk := congrFun h4 k
  rw [Pi.zero_apply, mul_eq_zero, or_iff_right (by positivity)] at hk
  have := (Finset.sum_eq_zero_iff_of_nonneg fun I _ => sq_nonneg _).1 hk I (Finset.mem_univ _)
  exact sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 this)

/-- The hypotheses of `eq_of_integral_exp_inner_eq` are satisfiable: `β = 1`
and `μ₁ = μ₂ = δ_{e₀}`, `e₀ ∈ 𝕊^0`. -/
example : (0 : ℝ) < 1 ∧
    IsProbabilityMeasure (Measure.dirac (⟨EuclideanSpace.single 0 1, by simp⟩ : SSphere 1)) ∧
    ∀ x : SSphere 1,
      ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 1) (y : EucSpace 1))
          ∂Measure.dirac (⟨EuclideanSpace.single 0 1, by simp⟩ : SSphere 1) =
        ∫ y, exp (1 * inner (𝕜 := ℝ) (x : EucSpace 1) (y : EucSpace 1))
          ∂Measure.dirac (⟨EuclideanSpace.single 0 1, by simp⟩ : SSphere 1) :=
  ⟨one_pos, inferInstance, fun _ => rfl⟩

end Perceptron
end Transformer

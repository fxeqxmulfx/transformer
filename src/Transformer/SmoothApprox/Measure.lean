/-
# Testing measures against smooth functions

Probability measures on a finite-dimensional space that live in a common
ball are determined by their integrals of smooth compactly supported
functions (`ext_of_forall_contDiff`).  For a curve `t ↦ μ_t` of such measures,
continuity of `t ↦ ∫ g dμ_t` for every smooth `g` with compact support gives
continuity of `p ↦ ∫ f(p, x) dμ_{τ(p)}(x)` for every jointly continuous `f`
with values in a finite-dimensional space (`continuous_integral_param`).

Both rest on the uniform approximation of `SmoothApprox.Basic`, and on a
smooth cutoff equal to `1` on the ball where the measures live.  This is how
arXiv:2305.05465v6, §6, `d:solconti`, passes from the test functions `C_c^∞`
to the continuous integrands of `e:vectorfield`.
-/

import Transformer.SmoothApprox.Basic
import Mathlib.MeasureTheory.Measure.HasOuterApproxClosed
import Mathlib.Topology.UniformSpace.UniformApproximation

open MeasureTheory Metric Filter Topology

namespace Transformer
namespace SmoothApprox

/-- **In finite dimension, weakly continuous is continuous**: a map is
continuous as soon as its images under all continuous linear functionals are. -/
theorem continuous_of_forall_clm {P : Type*} [TopologicalSpace P] {F : Type*}
    [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F] {f : P → F}
    (h : ∀ φ : F →L[ℝ] ℝ, Continuous fun p => φ (f p)) : Continuous f := by
  let e := (Module.finBasis ℝ F).equivFunL
  have he : Continuous fun p => e (f p) := continuous_pi fun i => by
    simpa using h ((ContinuousLinearMap.proj i).comp (e : F →L[ℝ] Fin _ → ℝ))
  exact (e.symm.continuous.comp he).congr fun p => by simp

variable {G : Type*} [NormedAddCommGroup G] [NormedSpace ℝ G] [FiniteDimensional ℝ G]
  [MeasurableSpace G] [BorelSpace G]
  {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]

omit [NormedAddCommGroup G] [NormedSpace ℝ G] [FiniteDimensional ℝ G] [BorelSpace G]
  [CompleteSpace F] in
/-- Functions uniformly within `ε` of each other have integrals within `ε`
against a probability measure. -/
theorem norm_integral_sub_le_of_forall_le {μ : Measure G} [IsProbabilityMeasure μ] {f g : G → F}
    (hf : Integrable f μ) (hg : Integrable g μ) {ε : ℝ} (h : ∀ x, ‖g x - f x‖ ≤ ε) :
    ‖∫ x, g x ∂μ - ∫ x, f x ∂μ‖ ≤ ε := by
  rw [← integral_sub hg hf]
  simpa using norm_integral_le_of_norm_le_const (μ := μ) (Eventually.of_forall h)

omit [MeasurableSpace G] [BorelSpace G] in
/-- **A smooth cutoff**: a smooth function with compact support, with values
in `[0, 1]`, equal to `1` on `B̄(0,R)`. -/
theorem exists_cutoff (R : ℝ) :
    ∃ χ : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) χ ∧ HasCompactSupport χ ∧
      (∀ x, 0 ≤ χ x ∧ χ x ≤ 1) ∧ ∀ x ∈ closedBall (0 : G) R, χ x = 1 := by
  let φ : ContDiffBump (0 : G) := ⟨|R| + 1, |R| + 2, by positivity, by linarith⟩
  refine ⟨φ, φ.contDiff, φ.hasCompactSupport, fun x => ⟨φ.nonneg, φ.le_one⟩, fun x hx => ?_⟩
  refine φ.one_of_mem_closedBall (closedBall_subset_closedBall ?_ hx)
  change R ≤ |R| + 1
  linarith [le_abs_self R]

/-- A sequence of smooth functions with compact support converging uniformly
to a continuous function with compact support. -/
theorem exists_seq_contDiff {f : G → F} (hf : Continuous f) (hc : HasCompactSupport f) :
    ∃ g : ℕ → G → F, (∀ n, ContDiff ℝ (⊤ : ℕ∞) (g n) ∧ HasCompactSupport (g n)) ∧
      ∀ n x, ‖g n x - f x‖ ≤ 1 / (n + 1) := by
  choose g hg using fun n : ℕ => exists_contDiff_near hf hc (ε := 1 / (n + 1)) (by positivity)
  exact ⟨g, fun n => ⟨(hg n).1, (hg n).2.1⟩, fun n => (hg n).2.2⟩

/-- **Smooth functions with compact support test continuous ones**: two
probability measures with the same integrals of the former have the same
integrals of the latter. -/
theorem integral_eq_of_forall_contDiff {μ ν : Measure G} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν]
    (h : ∀ g : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∫ x, g x ∂μ = ∫ x, g x ∂ν)
    {f : G → ℝ} (hf : Continuous f) (hc : HasCompactSupport f) : ∫ x, f x ∂μ = ∫ x, f x ∂ν := by
  obtain ⟨g, hg, hgf⟩ := exists_seq_contDiff hf hc
  have hlim : ∀ m : Measure G, IsProbabilityMeasure m →
      Tendsto (fun n => ∫ x, g n x ∂m) atTop (𝓝 (∫ x, f x ∂m)) := by
    intro m _
    refine tendsto_iff_norm_sub_tendsto_zero.2 (squeeze_zero (fun _ => norm_nonneg _)
      (fun n => norm_integral_sub_le_of_forall_le (hf.integrable_of_hasCompactSupport hc)
        ((hg n).1.continuous.integrable_of_hasCompactSupport (hg n).2) (hgf n))
      tendsto_one_div_add_atTop_nhds_zero_nat)
  refine tendsto_nhds_unique (hlim μ inferInstance) ?_
  simpa only [fun n => h (g n) (hg n).1 (hg n).2] using hlim ν inferInstance

/-- **Smooth functions with compact support determine a measure** among
probability measures living in a common ball. -/
theorem ext_of_forall_contDiff {μ ν : Measure G} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {R : ℝ} (hμ : μ (closedBall 0 R)ᶜ = 0) (hν : ν (closedBall 0 R)ᶜ = 0)
    (h : ∀ g : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∫ x, g x ∂μ = ∫ x, g x ∂ν) :
    μ = ν := by
  obtain ⟨χ, hχ, hχc, -, hχ1⟩ := exists_cutoff (G := G) R
  refine ext_of_forall_integral_eq_of_IsFiniteMeasure fun f => ?_
  have hcut : ∀ m : Measure G, m (closedBall 0 R)ᶜ = 0 →
      ∫ x, f x ∂m = ∫ x, χ x * f x ∂m := fun m hm => by
    refine integral_congr_ae ?_
    filter_upwards [(ae_iff.2 hm : ∀ᵐ x ∂m, x ∈ closedBall (0 : G) R)] with x hx
    rw [hχ1 x hx, one_mul]
  rw [hcut μ hμ, hcut ν hν]
  exact integral_eq_of_forall_contDiff h (hχ.continuous.mul f.continuous) hχc.mul_right

/-- **A curve tested against smooth functions is tested against smooth
vector-valued ones**, in finite dimension. -/
theorem continuous_integral_contDiff {μ : ℝ → Measure G} [∀ t, IsProbabilityMeasure (μ t)]
    (hμ : ∀ g : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    [FiniteDimensional ℝ F] {g : G → F} (hg : ContDiff ℝ (⊤ : ℕ∞) g) (hc : HasCompactSupport g) :
    Continuous fun t => ∫ x, g x ∂μ t := by
  refine continuous_of_forall_clm fun φ => ?_
  have hint : ∀ t, Integrable g (μ t) := fun t => hg.continuous.integrable_of_hasCompactSupport hc
  simp_rw [← φ.integral_comp_comm (hint _)]
  exact hμ _ (φ.contDiff.comp hg) (hc.comp_left φ.map_zero)

/-- **A curve tested against smooth functions is tested against continuous
ones** with compact support, by uniform approximation. -/
theorem continuous_integral_of_hasCompactSupport {μ : ℝ → Measure G}
    [∀ t, IsProbabilityMeasure (μ t)]
    (hμ : ∀ g : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    [FiniteDimensional ℝ F] {g : G → F} (hg : Continuous g) (hc : HasCompactSupport g) :
    Continuous fun t => ∫ x, g x ∂μ t := by
  obtain ⟨g', hg', hgg'⟩ := exists_seq_contDiff hg hc
  refine TendstoUniformly.continuous (F := fun n t => ∫ x, g' n x ∂μ t) (p := atTop) ?_
    (Frequently.of_forall fun n => continuous_integral_contDiff hμ (hg' n).1 (hg' n).2)
  rw [Metric.tendstoUniformly_iff]
  intro ε hε
  obtain ⟨N, hN⟩ := exists_nat_one_div_lt hε
  filter_upwards [eventually_ge_atTop N] with n hn t
  rw [dist_eq_norm, norm_sub_rev]
  refine (norm_integral_sub_le_of_forall_le (hg.integrable_of_hasCompactSupport hc)
    ((hg' n).1.continuous.integrable_of_hasCompactSupport (hg' n).2) (hgg' n)).trans_lt ?_
  calc 1 / ((n : ℝ) + 1) ≤ 1 / ((N : ℝ) + 1) := by gcongr
    _ < ε := hN

/-- **Weak continuity with a parameter.**  If `t ↦ μ_t` lives near each time
in a fixed ball and is continuous against smooth functions with compact
support, then `p ↦ ∫ f(p, x) dμ_{τ(p)}(x)` is continuous for every jointly
continuous `f` and continuous `τ`. -/
theorem continuous_integral_param {μ : ℝ → Measure G} [∀ t, IsProbabilityMeasure (μ t)]
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : G → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    [FiniteDimensional ℝ F] {P : Type*} [TopologicalSpace P] {τ : P → ℝ} (hτ : Continuous τ)
    {f : P → G → F} (hf : Continuous (Function.uncurry f)) :
    Continuous fun p => ∫ x, f p x ∂μ (τ p) := by
  refine continuous_iff_continuousAt.2 fun p₀ => ?_
  obtain ⟨R, hR⟩ := hR (τ p₀)
  obtain ⟨χ, hχ, hχc, hχ01, hχ1⟩ := exists_cutoff (G := G) R
  have hcut : (fun p => ∫ x, χ x • f p x ∂μ (τ p)) =ᶠ[𝓝 p₀] fun p => ∫ x, f p x ∂μ (τ p) := by
    filter_upwards [hτ.continuousAt.eventually hR] with p hp
    refine integral_congr_ae ?_
    filter_upwards [(ae_iff.2 hp : ∀ᵐ x ∂μ (τ p), x ∈ closedBall (0 : G) R)] with x hx
    rw [hχ1 x hx, one_smul]
  refine ContinuousAt.congr ?_ hcut
  have hfp : ∀ p, Continuous (f p) := fun p => hf.comp (continuous_const.prodMk continuous_id)
  have hint : ∀ p t, Integrable (fun x => χ x • f p x) (μ t) := fun p t =>
    (hχ.continuous.smul (hfp p)).integrable_of_hasCompactSupport hχc.smul_right
  rw [ContinuousAt, Metric.tendsto_nhds]
  intro ε hε
  have h1 := Metric.tendsto_nhds.1 ((continuous_integral_of_hasCompactSupport hμ
    (g := fun x => χ x • f p₀ x) (hχ.continuous.smul (hfp p₀)) hχc.smul_right).continuousAt.comp
      (hτ.continuousAt (x := p₀))) (ε / 2) (by positivity)
  have h2 : ∀ᶠ p in 𝓝 p₀, ∀ x ∈ tsupport χ, dist (f p x) (f p₀ x) < ε / 4 := by
    refine hχc.isCompact.eventually_forall_of_forall_eventually fun y _ => ?_
    have hc1 : Tendsto (fun z : P × G => f z.1 z.2) (𝓝 (p₀, y)) (𝓝 (f p₀ y)) := hf.continuousAt
    have hc2 : Tendsto (fun z : P × G => f p₀ z.2) (𝓝 (p₀, y)) (𝓝 (f p₀ y)) :=
      ((hfp p₀).comp continuous_snd).continuousAt
    have := hc1.dist hc2
    rw [dist_self] at this
    exact this.eventually (gt_mem_nhds (by positivity))
  filter_upwards [h1, h2] with p hp1 hp2
  have h3 : ‖∫ x, χ x • f p x ∂μ (τ p) - ∫ x, χ x • f p₀ x ∂μ (τ p)‖ ≤ ε / 4 := by
    refine norm_integral_sub_le_of_forall_le (hint _ _) (hint _ _) fun x => ?_
    by_cases hx : x ∈ tsupport χ
    · rw [← smul_sub, norm_smul, Real.norm_of_nonneg (hχ01 x).1, ← dist_eq_norm]
      exact (mul_le_of_le_one_left dist_nonneg (hχ01 x).2).trans (hp2 x hx).le
    · rw [image_eq_zero_of_notMem_tsupport hx]
      simpa using (by positivity : (0 : ℝ) ≤ ε / 4)
  calc dist (∫ x, χ x • f p x ∂μ (τ p)) (∫ x, χ x • f p₀ x ∂μ (τ p₀))
      ≤ dist (∫ x, χ x • f p x ∂μ (τ p)) (∫ x, χ x • f p₀ x ∂μ (τ p))
        + dist (∫ x, χ x • f p₀ x ∂μ (τ p)) (∫ x, χ x • f p₀ x ∂μ (τ p₀)) := dist_triangle _ _ _
    _ < ε / 4 + ε / 2 := by
        rw [dist_eq_norm]
        exact add_lt_add_of_le_of_lt h3 hp1
    _ < ε := by linarith

/-- The hypotheses of the results above are satisfiable: the constant curve at the Dirac
mass at `0 ∈ ℝ`, carried by `B̄(0, 0)`, the zero function and the identity of `ℝ`. -/
example : IsProbabilityMeasure (Measure.dirac (0 : ℝ)) ∧
    (∀ t₀ : ℝ, ∃ R : ℝ, ∀ᶠ _ in 𝓝 t₀, Measure.dirac (0 : ℝ) (closedBall 0 R)ᶜ = 0) ∧
    Measure.dirac (0 : ℝ) (closedBall 0 0)ᶜ = 0 ∧
    (∀ g : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous (fun _ : ℝ => ∫ x, g x ∂Measure.dirac (0 : ℝ)) ∧
        ∫ x, g x ∂Measure.dirac (0 : ℝ) = ∫ x, g x ∂Measure.dirac (0 : ℝ)) ∧
    (∀ φ : ℝ →L[ℝ] ℝ, Continuous fun p : ℝ => φ p) ∧ Integrable (0 : ℝ → ℝ) (Measure.dirac 0) ∧
    (∀ x : ℝ, ‖(0 : ℝ → ℝ) x - (0 : ℝ → ℝ) x‖ ≤ 0) ∧ ContDiff ℝ (⊤ : ℕ∞) (0 : ℝ → ℝ) ∧
    HasCompactSupport (0 : ℝ → ℝ) ∧ Continuous (Function.uncurry fun (_ x : ℝ) => x) :=
  ⟨inferInstance, fun _ => ⟨0, Eventually.of_forall fun _ => by simp⟩, by simp,
    fun _ _ _ => ⟨continuous_const, rfl⟩, fun φ => φ.continuous, integrable_zero _ _ _,
    fun _ => by simp, contDiff_const, HasCompactSupport.zero, continuous_snd⟩

end SmoothApprox
end Transformer

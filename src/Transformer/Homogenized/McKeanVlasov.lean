/-
# Homogenized Transformers — the mean-field limit with common noise

Formalization of `def:weak_spde` and `def:nonlinear_SDE_common` of
arXiv:2604.01978v1, *Homogenized Transformers*, §4: the two solution concepts
for the limiting dynamics of one token — the nonlinear Fokker–Planck SPDE
`eq: the.spde`, driven by the noise that all tokens share, and its trajectorial
characterization `eq:non_linear_SDE_common`.

**What the source writes, and what is written here.**  Both definitions are
stated in the source through the cylindrical Wiener process `W` on `L²(ρ*)`:
the SPDE carries `∫_0^t∫_Θ ⟨∇φ·G_{μ(s)}(·,θ), μ(s)⟩ W(ds,dθ)` and the SDE
carries `∫_Θ G_{μ(t)}(x(t),θ) W(dθ,dt)`.  Mathlib has no stochastic integral,
so neither display can be transcribed.  Each is written instead as the
martingale problem it is equivalent to: the compensated process is a
martingale, and its square minus `∫_0^t∫_Θ (…)² ρ*(dθ) ds` — the quadratic
variation that a cylindrical noise on `L²(ρ*)` produces — is a martingale too.
Those two conditions, over all test functions, pin the law of a solution; what
they do not record, and what the source's displays do, is *which* Wiener
process drives the equation.  That is why the well-posedness of
`thm:PoC_wellposedness` can only be uniqueness in law here, and not the
pathwise uniqueness the source's proof establishes: two solutions of a
martingale problem on one probability space may be driven by two different
Wiener processes, and then they do not agree pathwise.
-/

import Transformer.Homogenized.MvGenerator
import Mathlib.MeasureTheory.Measure.GiryMonad
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The nonlinear Fokker–Planck SPDE -/

/-- The compensated process of `def:weak_spde`:

  `N^φ(t) = ⟨μ(t),φ⟩ - ⟨μ(0),φ⟩ - ∫_0^t ⟨𝖫_{μ(s)}φ, μ(s)⟩ ds`,

which the source's identity equates with the stochastic integral against the
common noise.

Source: arXiv:2604.01978v1, `def:weak_spde`. -/
noncomputable def spdeMart {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} (μ : ℝ → Ω → Measure (EucSpace d)) (φ : EucSpace d → ℝ)
    (t : ℝ) (ω : Ω) : ℝ :=
  (∫ y, φ y ∂(μ t ω)) - (∫ y, φ y ∂(μ 0 ω))
    - ∫ s in (0 : ℝ)..t, ∫ y, mvGenerator β ρ (μ s ω) φ y ∂(μ s ω)

/-- The quadratic variation that the cylindrical noise on `L²(ρ*)` gives to
`N^φ`:  `⟨N^φ⟩(t) = ∫_0^t ∫_Θ ⟨∇φ·G_{μ(s)}(·,θ), μ(s)⟩² ρ*(dθ) ds`.

Source: arXiv:2604.01978v1, `def:weak_spde`, `eq:cross_variation_kernel`. -/
noncomputable def spdeBracket {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} (μ : ℝ → Ω → Measure (EucSpace d)) (φ : EucSpace d → ℝ)
    (t : ℝ) (ω : Ω) : ℝ :=
  ∫ s in (0 : ℝ)..t, ∫ θ, spdeNoise β ρ (μ s ω) φ θ ^ 2 ∂ρ

/-- **Definition (def:weak_spde).**  An `(ℱ_t^W)`-adapted process `(μ(t))` with
values in `𝒫(𝕊^{d-1})` is a weak solution of the nonlinear Fokker–Planck SPDE
`eq: the.spde` when, for every `C^∞` test function `φ`, the compensated process
`N^φ` of `spdeMart` is a martingale with quadratic variation `spdeBracket` —
the martingale problem that the source's stochastic integral is equivalent to.

Source: arXiv:2604.01978v1, `def:weak_spde`, `eq: the.spde`. -/
structure IsWeakSpdeSolution {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (𝒢 : Filtration ℝ m)
    (μ : ℝ → Ω → Measure (EucSpace d)) : Prop where
  /-- `μ(t)` is a probability measure. -/
  prob : ∀ t ω, IsProbabilityMeasure (μ t ω)
  /-- `μ(t)` is carried by the sphere. -/
  sphere : ∀ t ω, μ t ω {y : EucSpace d | ‖y‖ = 1}ᶜ = 0
  /-- `(μ(t))` is adapted to the noise filtration `(ℱ_t^W)`. -/
  adapted : ∀ t ∈ Set.Icc (0 : ℝ) T, Measurable[𝒢 t] (μ t)
  /-- The compensated process is a martingale. -/
  mart : ∀ φ : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
    IsMartingaleOn T 𝒢 P (spdeMart β ρ μ φ)
  /-- Its quadratic variation is the one the common noise produces. -/
  bracket : ∀ φ : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
    IsMartingaleOn T 𝒢 P
      (fun t ω => spdeMart β ρ μ φ t ω ^ 2 - spdeBracket β ρ μ φ t ω)

/-! ### The McKean–Vlasov SDE -/

/-- The compensated process of `eq:non_linear_SDE_common`:

  `M^φ(t) = φ(x(t)) - φ(x(0)) - ∫_0^t 𝖫_{μ(s)}φ(x(s)) ds`.

Source: arXiv:2604.01978v1, `def:nonlinear_SDE_common`. -/
noncomputable def mvMart {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d))
    (φ : EucSpace d → ℝ) (t : ℝ) (ω : Ω) : ℝ :=
  φ (x t ω) - φ (x 0 ω) - ∫ s in (0 : ℝ)..t, mvGenerator β ρ (μ s ω) φ (x s ω)

/-- **Definition (def:nonlinear_SDE_common).**  The pair `(x, μ)` solves the
McKean–Vlasov SDE `eq:non_linear_SDE_common` associated with `eq: the.spde`:
`x` is `(ℱ_t)`-adapted with values in `𝕊^{d-1}` and starts at `x_0`, `μ` is
adapted to the noise filtration `𝒢 = (ℱ_t^W)` and starts at `μ_0`, `μ(t)` is
the conditional law of `x(t)` given `ℱ_t^W`, and `dx = ∫_Θ G_{μ(t)}(x(t),θ)
W(dθ,dt)`.

**What the source says and what is written here.**  The last display is the
martingale problem for the generator `𝖫_{μ(t)}`, for the reason given in the
module docstring.  The source's `x_0` is `ℱ_0`-measurable, independent of `W`,
with law `μ_0`; independence of `W` is what makes `Law(x_0 | ℱ_0^W) = μ_0`
deterministic, and it is that conclusion — `initLaw` — that is carried, since
`W` itself is not available to be independent of.

Source: arXiv:2604.01978v1, `def:nonlinear_SDE_common`, `eq:non_linear_SDE_common`. -/
structure IsMcKeanVlasovSolution {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ m)
    (x₀ : Ω → EucSpace d) (μ₀ : Measure (EucSpace d))
    (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)) : Prop where
  /-- The noise filtration is coarser than the one carrying the token. -/
  noise_le : ∀ t, 𝒢 t ≤ ℱ t
  /-- `x(t) ∈ 𝕊^{d-1}`. -/
  sphere : ∀ t ω, ‖x t ω‖ = 1
  /-- `(x(t))` is `(ℱ_t)`-adapted. -/
  adapted : ∀ t ∈ Set.Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (x t)
  /-- `(μ(t))` is `(ℱ_t^W)`-adapted. -/
  adaptedLaw : ∀ t ∈ Set.Icc (0 : ℝ) T, Measurable[𝒢 t] (μ t)
  /-- `x(0) = x_0` almost surely. -/
  init : ∀ᵐ ω ∂P, x 0 ω = x₀ ω
  /-- `Law(x_0) = μ_0`. -/
  lawInit : Measure.map x₀ P = μ₀
  /-- `μ(0) = Law(x_0 | ℱ_0^W) = μ_0`. -/
  initLaw : ∀ ω, μ 0 ω = μ₀
  /-- `μ(t)` is a probability measure. -/
  prob : ∀ t ω, IsProbabilityMeasure (μ t ω)
  /-- `μ(t) = Law(x(t) | ℱ_t^W)` almost surely, tested on Borel sets. -/
  condLaw : ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ A : Set (EucSpace d), MeasurableSet A →
    (fun ω => (μ t ω A).toReal)
      =ᵐ[P] condExp (𝒢 t) P (fun ω => Set.indicator A (1 : EucSpace d → ℝ) (x t ω))
  /-- `eq:non_linear_SDE_common`, as the martingale problem for `𝖫_{μ(t)}`. -/
  mart : ∀ φ : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
    IsMartingaleOn T ℱ P (mvMart β ρ x μ φ)
  /-- Its quadratic variation is the one the common noise produces. -/
  bracket : ∀ φ : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
    IsMartingaleOn T ℱ P (fun t ω => mvMart β ρ x μ φ t ω ^ 2
      - ∫ s in (0 : ℝ)..t, ∫ θ, fderiv ℝ φ (x s ω) (GfieldOf β ρ (μ s ω) (x s ω) θ) ^ 2 ∂ρ)

/-! ### The trivial head

Under `ρ* = δ_0` all the fields vanish, and a token sitting still at a unit
vector, with `μ(t) = δ_x`, solves both `def:nonlinear_SDE_common` and
`def:weak_spde`: every compensated process is the constant `0`.  This is the
witness that neither definition is contradictory. -/

theorem isWeakSpdeSolution_dirac_zero {d : ℕ} (β T : ℝ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (𝒢 : Filtration ℝ m) (e : EucSpace d)
    (he : ‖e‖ = 1) :
    IsWeakSpdeSolution β T (Measure.dirac (0 : HeadParam d)) P 𝒢
      (fun _ _ => Measure.dirac e) := by
  have hmart : ∀ φ : EucSpace d → ℝ,
      spdeMart β (Measure.dirac (0 : HeadParam d)) (fun (_ : ℝ) (_ : Ω) => Measure.dirac e)
        φ = fun _ _ => (0 : ℝ) := by
    intro φ; funext t ω; simp [spdeMart]
  have hbr : ∀ φ : EucSpace d → ℝ,
      spdeBracket β (Measure.dirac (0 : HeadParam d)) (fun (_ : ℝ) (_ : Ω) => Measure.dirac e)
        φ = fun _ _ => (0 : ℝ) := by
    intro φ; funext t ω; simp [spdeBracket]
  refine ⟨fun _ _ => inferInstance, fun _ _ => ?_, fun _ _ => measurable_const,
    fun φ _ => ?_, fun φ _ => ?_⟩
  · rw [Measure.dirac_apply]
    simp [he]
  · rw [hmart φ]; exact isMartingaleOn_const T 𝒢 P 0
  · simp only [hmart φ, hbr φ]
    simpa using isMartingaleOn_const T 𝒢 P 0

theorem isMcKeanVlasovSolution_dirac_zero {d : ℕ} (β T : ℝ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) (e : EucSpace d)
    (he : ‖e‖ = 1) :
    IsMcKeanVlasovSolution β T (Measure.dirac (0 : HeadParam d)) P ℱ ℱ
      (fun _ => e) (Measure.dirac e) (fun _ _ => e) (fun _ _ => Measure.dirac e) := by
  have hmart : ∀ φ : EucSpace d → ℝ,
      mvMart β (Measure.dirac (0 : HeadParam d)) (fun (_ : ℝ) (_ : Ω) => e)
        (fun (_ : ℝ) (_ : Ω) => Measure.dirac e) φ = fun _ _ => (0 : ℝ) := by
    intro φ; funext t ω; simp [mvMart]
  refine ⟨fun _ => le_rfl, fun _ _ => he, fun _ _ => stronglyMeasurable_const,
    fun _ _ => measurable_const, Filter.Eventually.of_forall fun _ => rfl, ?_,
    fun _ => rfl, fun _ _ => inferInstance, fun t _ A hA => ?_, fun φ _ => ?_, fun φ _ => ?_⟩
  · rw [Measure.map_const]
    simp
  · have hc : ((Measure.dirac e) A).toReal = Set.indicator A (1 : EucSpace d → ℝ) e := by
      rw [Measure.dirac_apply]
      by_cases h : e ∈ A <;> simp [h]
    show (fun _ : Ω => ((Measure.dirac e) A).toReal)
      =ᵐ[P] condExp (ℱ t) P (fun _ : Ω => Set.indicator A (1 : EucSpace d → ℝ) e)
    rw [hc, condExp_const (ℱ.le t)]
  · rw [hmart φ]; exact isMartingaleOn_const T ℱ P 0
  · simp only [hmart φ]
    simpa using isMartingaleOn_const T ℱ P 0

/-- Both definitions are satisfiable: the trivial head `ρ* = δ_0`, a token
sitting at `basePoint d`, and the deterministic law `μ(t) = δ_{basePoint d}`,
on the one-point probability space. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer

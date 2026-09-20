/-
# Homogenized Transformers — the logistic limit of the overlap

Formalization of the second part of `thm:large_beta_meta` of
arXiv:2604.01978v1, *Homogenized Transformers*, §2.5: on the time scale `td`,
and at a temperature growing fast enough that `d β^{-1/2} = o(1)`, the overlap
of two tokens under one common noise converges in law to the solution of the
logistic SDE `eq: sde.logistic` started at `0`.

The first part — the slow-motion bound on the original time scale — is
`Metastability.lean`.
-/

import Transformer.Homogenized.Metastability
import Transformer.Homogenized.Logistic
import Mathlib.Analysis.SpecificLimits.Basic

open scoped BigOperators ENNReal NNReal Topology
open Real MeasureTheory Filter

namespace Transformer
namespace Homogenized

/-! ### Paths -/

/-- A real process restricted to `[0,T]`, read as a point of `C([0,T], ℝ)`.

Convergence in law of processes is convergence of the laws of these points, so
a process only has one once its paths are known to be continuous — which is
why `hZ` is an argument and not an afterthought. -/
noncomputable def pathOn {Ω : Type*} (T : ℝ) (Z : ℝ → Ω → ℝ)
    (hZ : ∀ ω, ContinuousOn (fun t => Z t ω) (Set.Icc (0 : ℝ) T)) (ω : Ω) :
    C(Set.Icc (0 : ℝ) T, ℝ) :=
  ⟨Set.domRestrict _ (fun t => Z t ω), (hZ ω).domRestrict⟩

/-- The overlap on the slow time scale, `R̂_d(t) = ⟨x⁽¹⁾(td), x⁽²⁾(td)⟩`.

Source: arXiv:2604.01978v1, proof of `thm:large_beta_meta`, Step 5. -/
noncomputable def rescaledOverlap {d : ℕ} {Ω : Type*} (x₁ x₂ : ℝ → Ω → EucSpace d)
    (t : ℝ) (ω : Ω) : ℝ :=
  overlap x₁ x₂ ((d : ℝ) * t) ω

@[simp] theorem rescaledOverlap_zero {d : ℕ} {Ω : Type*} (x₁ x₂ : ℝ → Ω → EucSpace d)
    (ω : Ω) : rescaledOverlap x₁ x₂ 0 ω = overlap x₁ x₂ 0 ω := by
  simp [rescaledOverlap]

/-! ### The limit -/

/-- **Theorem (thm:large_beta_meta), second part.**  If `β = β(d)` grows fast
enough that `d β^{-1/2} = o(1)`, then

  `(⟨x⁽¹⁾(td), x⁽²⁾(td)⟩)_{t∈[0,T]} ⟹ (R_∞(t))_{t∈[0,T]}`  as `d → ∞`,

where `R_∞` solves `eq: sde.logistic` with `R_∞(0) = 0`.

**What the source says and what is written here.**

* The family is indexed by dimension `d + 1`.  At dimension `0` there is no
  unit vector, so `IsLowTemperature 0 σ …` is contradictory — `σ` would have to
  be a probability measure carried by the empty sphere — and a family indexed
  by `d` would have unsatisfiable hypotheses and say nothing.

* The horizon of the pair is `(d+1)T`, since `R̂_d(t) = R((d+1)t)` for
  `t ∈ [0,T]` reads `R` on `[0,(d+1)T]`.

* `d β^{-1/2} = o(1)` is `(d+1)·effBeta^{-1/2} → 0`.  The `β` is again the
  temperature of the normalized query-key matrix of `lem:delta_method`, as in
  `large_beta_metastability`; `effBeta` records the translation.

* Convergence in law is written by testing against bounded continuous
  functionals of the path, which is what it means.  No measurable structure is
  put on `C([0,T],ℝ)` here, so `F ∘ pathOn` is not known to be measurable and
  its integrability is hypothesized; boundedness alone does not give it.

* Continuity of `t ↦ ⟨x⁽¹⁾(td), x⁽²⁾(td)⟩` is an added hypothesis.
  `IsMcKeanVlasovSolution` is a martingale problem and carries no path
  regularity, whereas the source's `x(t)` is a continuous semimartingale by
  construction; without continuity the path is not a point of `C([0,T],ℝ)` and
  the statement cannot be made.

* The source says `R_∞` is "the unique solution" of `eq: sde.logistic`.  What
  is asserted here is that the limit *is* a solution started at `0`;
  uniqueness in law for that equation is a separate claim about
  `IsLogisticSolution`, not about this limit, and asserting it here would be
  asserting two theorems under one name.

Not proved here.

Source: arXiv:2604.01978v1, `thm:large_beta_meta`, `eq: sde.logistic`. -/
theorem large_beta_logistic_limit :
    ∀ (ρmin ρmax L T : ℝ), 0 < T →
    ∀ (β : ℕ → ℝ) (σV σA : ℕ → ℝ≥0), (∀ d, 0 < β d) → (∀ d, 0 < σA d) →
      (∀ d, ((σV d : ℝ)) ^ 2 = 1 / ((d : ℝ) + 1)) →
    ∀ ρ : ∀ d : ℕ, Measure (HeadParam (d + 1)),
      (∀ d, IsGaussianHeadLaw (d + 1) (σV d) (σA d) (ρ d)) →
      Tendsto (fun d : ℕ => ((d : ℝ) + 1) * effBeta (d + 1) (β d) (σA d) ^ (-(1 : ℝ) / 2))
        atTop (𝓝 0) →
    ∀ (Ω : ℕ → Type) (m : ∀ d, MeasurableSpace (Ω d)) (P : ∀ d, Measure (Ω d)),
      (∀ d, IsProbabilityMeasure (P d)) →
    ∀ (ℱ 𝒢 : ∀ d, Filtration ℝ (m d)) (σ μ₀ : ∀ d, Measure (EucSpace (d + 1)))
      (x₁ x₂ : ∀ d : ℕ, ℝ → Ω d → EucSpace (d + 1))
      (μ : ∀ d : ℕ, ℝ → Ω d → Measure (EucSpace (d + 1)))
      (dens : ∀ d : ℕ, Ω d → ℝ → EucSpace (d + 1) → ℝ),
      (∀ d ω, IsLowTemperature (d + 1) (σ d) (fun t => μ d t ω) (dens d ω) ρmin ρmax L) →
      (∀ d, IsCoupledPair (β d) (((d : ℝ) + 1) * T) (ρ d) (P d) (ℱ d) (𝒢 d)
        (μ₀ d) (x₁ d) (x₂ d) (μ d)) →
    ∀ hcont : ∀ d ω, ContinuousOn (fun t => rescaledOverlap (x₁ d) (x₂ d) t ω)
        (Set.Icc (0 : ℝ) T),
      ∃ (Ω' : Type) (m' : MeasurableSpace Ω') (Q : Measure Ω') (𝒦 : Filtration ℝ m')
        (R : ℝ → Ω' → ℝ) (hR : ∀ ω, ContinuousOn (fun t => R t ω) (Set.Icc (0 : ℝ) T)),
        IsProbabilityMeasure Q ∧ IsLogisticSolution Q 𝒦 R ∧ (∀ ω, R 0 ω = 0) ∧
          ∀ F : C(Set.Icc (0 : ℝ) T, ℝ) → ℝ, Continuous F → (∃ M : ℝ, ∀ g, |F g| ≤ M) →
            (∀ d, Integrable
              (fun ω => F (pathOn T (rescaledOverlap (x₁ d) (x₂ d)) (hcont d) ω)) (P d)) →
            Integrable (fun ω => F (pathOn T R hR ω)) Q →
            Tendsto
              (fun d => ∫ ω, F (pathOn T (rescaledOverlap (x₁ d) (x₂ d)) (hcont d) ω) ∂(P d))
              atTop (𝓝 (∫ ω, F (pathOn T R hR ω) ∂Q)) := by
  sorry

/-- Every hypothesis of `large_beta_logistic_limit` that does not involve the
dynamics is satisfiable at once, in **every** dimension: the temperature
`β(d) = (d+1)³` with `σ_A = 1` makes `(d+1)·effBeta^{-1/2} = 1/(d+1) → 0`, the
Gaussian head law of assumption (G) exists at the standard scaling
`σ_V² = 1/(d+1)`, and the uniform measure satisfies `ass:low-temperature` with
density `1`, `ρ_min = ρ_max = 1`, `L = 0`.

What is *not* witnessed is `IsCoupledPair` in every dimension.  In dimension
`1` the frozen uniform pair of `OneDim.lean` is one, as
`large_beta_metastability`'s witness records; from dimension `2` on, the field
`eq:G_def` does not vanish at a Gaussian head law, no frozen pair can solve the
martingale problem, and constructing a solution is `thm:PoC_wellposedness`,
which is a `sorry` in `WellPosed.lean`.  Exhibiting a family here would be
claiming that construction. -/
example :
    Tendsto
      (fun d : ℕ => ((d : ℝ) + 1) * effBeta (d + 1) (((d : ℝ) + 1) ^ 3) 1 ^ (-(1 : ℝ) / 2))
      atTop (𝓝 0) ∧
    ∀ d : ℕ, (0 : ℝ) < ((d : ℝ) + 1) ^ 3 ∧ (0 : ℝ≥0) < 1 ∧
      ((stdSigmaV (d + 1) : ℝ)) ^ 2 = 1 / ((d : ℝ) + 1) ∧
      IsGaussianHeadLaw (d + 1) (stdSigmaV (d + 1)) 1
        (gaussHeadLaw (d + 1) (stdSigmaV (d + 1)) 1) ∧
      IsLowTemperature (d + 1) (uniformAmbient (d + 1)) (fun _ => uniformAmbient (d + 1))
        (fun _ _ => 1) 1 1 0 := by
  constructor
  · have heq : (fun d : ℕ => ((d : ℝ) + 1) *
        effBeta (d + 1) (((d : ℝ) + 1) ^ 3) 1 ^ (-(1 : ℝ) / 2))
        = fun d : ℕ => 1 / ((d : ℝ) + 1) := by
      funext d
      have hpos : (0 : ℝ) < (d : ℝ) + 1 := by positivity
      have hb : effBeta (d + 1) (((d : ℝ) + 1) ^ 3) 1 = (((d : ℝ) + 1) ^ 2) ^ 2 := by
        simp only [effBeta, NNReal.coe_one, one_pow, mul_one, Nat.cast_add, Nat.cast_one]
        ring
      rw [hb, show (-(1 : ℝ) / 2) = -(1 / 2 : ℝ) by ring, Real.rpow_neg (by positivity),
        ← Real.sqrt_eq_rpow, Real.sqrt_sq (by positivity)]
      field_simp
    rw [heq]
    exact tendsto_one_div_add_atTop_nhds_zero_nat
  · intro d
    refine ⟨by positivity, one_pos, ?_, isGaussianHeadLaw_gaussHeadLaw (d + 1) _ _,
      isLowTemperature_uniformAmbient (Nat.succ_pos d)⟩
    rw [stdSigmaV_sq]
    push_cast
    ring

end Homogenized
end Transformer

/-
# Homogenized Transformers — slow motion of the two-point overlap

Formalization of the first part of `thm:large_beta_meta` of arXiv:2604.01978v1,
*Homogenized Transformers*, §2.5: under `ass:low-temperature`, two tokens
driven by the **same** common noise keep their overlap for a time of order `d`.

The second part — the logistic limit of the overlap on the time scale `td` —
is `LogisticLimit.lean`.
-/

import Transformer.Homogenized.OneDim
import Transformer.Homogenized.SlowMotion
import Transformer.Homogenized.GaussianEnsemble

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The effective temperature -/

/-- The **effective inverse temperature** `β d σ_A²`.

Assumption (G) takes the query-key matrix to be `A = W W'ᵀ` with
`W_ij, W'_ij ∼ N(0, σ_A²)`, so `‖A x‖` is of order `d σ_A²`; the attention
weight `attnWeight β θ x y = e^{β⟨Ax,y⟩}` therefore concentrates at the scale
`β d σ_A²` and not at the scale `β`.  The proof of `thm:large_beta_meta`
normalizes it away — `lem:delta_method` is stated for `Â = (dσ_A²)^{-1}WᵀW'` —
and then writes its error as `O(β^{-1/2})` in the temperature of `Â`.  That
temperature is this one.

`thm:clustering_small_beta` carries the same scale in the open: its error term
`d²σ_A⁴β²` is `(β d σ_A²)²`.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`, `lem:delta_method`. -/
noncomputable def effBeta (d : ℕ) (β : ℝ) (σA : ℝ≥0) : ℝ := β * d * (σA : ℝ) ^ 2

theorem effBeta_pos {d : ℕ} (hd : 0 < d) {β : ℝ} (hβ : 0 < β) {σA : ℝ≥0} (hA : 0 < σA) :
    0 < effBeta d β σA := by
  have hd' : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  have hA' : (0 : ℝ) < (σA : ℝ) := hA
  unfold effBeta
  positivity

/-! ### Slow motion -/

/-- **Theorem (thm:large_beta_meta), first part.**  Under
`ass:low-temperature`, let `x⁽¹⁾` and `x⁽²⁾` solve `eq:non_linear_SDE_common`
under the same common noise, with initial data drawn independently from `μ(0)`.
Then there is `C > 0` such that for every `T > 0` and every `δ ∈ (0,1)`, with
probability at least `1 - δ`,

  `sup_{t∈[0,T]} |⟨x⁽¹⁾(t),x⁽²⁾(t)⟩ - ⟨x⁽¹⁾(0),x⁽²⁾(0)⟩|
      ≤ T/d + C√(T/d · log(1/δ)) + C·T·(d^{-3/2} + β_eff^{-1/2})`.

On the original time scale the overlap therefore moves by `O(1)` only after a
time of order `d`: the slow transient of §2.5.

**What the source says and what is written here.**

* The source's quantifier list reads "for every `T>0` and every `δ∈(0,1)` and
  `t≥0`".  There is no free `t` — it is bound by the `sup_{t∈[0,T]}` — so the
  `t ≥ 0` is dropped.

* The source displays the last term as `r(β,d)` with
  `|r(β,d)| = O(d^{-3/2}+β^{-1/2})`, but its own Step 4 produces
  `T·sup_{t∈[0,T]}|ε_{d,β}(t)|` with `|ε_{d,β}| = O(d^{-3/2}+β^{-1/2})`.  The
  factor `T` belongs there and is restored; without it the statement is false
  for large `T`, since the error accumulates over the whole interval.  The `4`
  under Step 4's square root is absorbed into `C`, as the display does.

* The `β` of `β^{-1/2}` is the temperature of the **normalized** query-key
  matrix: `lem:delta_method` is stated for `Â = (dσ_A²)^{-1}WᵀW'`, whereas
  assumption (G) — and `attnWeight` — use `A = WW'ᵀ`.  It is therefore
  `effBeta`, and `σ_A > 0` is required, without which that quantity is `0` and
  the bound says nothing.

* `σ_V² = 1/d` is the "standard scaling, which we adopt henceforth" of §2.5,
  and it is carried as a hypothesis because the proof uses it: Step 3 computes
  the bracket from `E⟨m_{β,A}[μ](x⁽¹⁾),m_{β,A}[μ](x⁽²⁾)⟩` with no `V` in sight,
  which is `E⟨Vu,Vv⟩ = σ_V² d ⟨u,v⟩ = ⟨u,v⟩` only at that scaling.  `σ_A` is
  left general, as the source leaves it.

* `C` is produced after `ρ_min, ρ_max, L` and before `d, β, T, δ`.  The source
  calls `C` universal, but `lemma:Laplace_method` produces constants depending
  on the density bounds of `ass:low-temperature`, and the assumption itself
  claims only that `ρ_min, ρ_max` are independent of `d` and `β`.  Placing `C`
  after the three constants and before `d` and `β` is what that says.

Not proved here.

Source: arXiv:2604.01978v1, `thm:large_beta_meta`. -/
theorem large_beta_metastability (ρmin ρmax L : ℝ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ), 0 < d → ∀ β : ℝ, 0 < β → ∀ σV σA : ℝ≥0,
        ((σV : ℝ)) ^ 2 = 1 / (d : ℝ) → 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ T : ℝ, 0 < T → ∀ δ ∈ Set.Ioo (0 : ℝ) 1,
      ∀ (Ω : Type) (m : MeasurableSpace Ω) (P : Measure Ω), IsProbabilityMeasure P →
      ∀ (ℱ 𝒢 : Filtration ℝ m) (σ μ₀ : Measure (EucSpace d))
        (x₁ x₂ : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d))
        (dens : Ω → ℝ → EucSpace d → ℝ),
        (∀ ω, IsLowTemperature d σ (fun t => μ t ω) (dens ω) ρmin ρmax L) →
        IsCoupledPair β T ρ P ℱ 𝒢 μ₀ x₁ x₂ μ →
        ENNReal.ofReal (1 - δ) ≤
          P {ω | ∀ t ∈ Set.Icc (0 : ℝ) T,
            |overlap x₁ x₂ t ω - overlap x₁ x₂ 0 ω| ≤
              T / (d : ℝ) + C * Real.sqrt (T / (d : ℝ) * Real.log (1 / δ))
                + C * T * ((d : ℝ) ^ (-(3 : ℝ) / 2)
                  + effBeta d β σA ^ (-(1 : ℝ) / 2))} := by
  sorry

/-- The hypotheses of `large_beta_metastability` are satisfiable, at
`ρ_min = ρ_max = 1` and `L = 0`.

Dimension `1` is what makes a joint witness available: there the tangent space
of the sphere is trivial, so `eq:G_def` vanishes at *every* head law
(`OneDim.lean`) — in particular at the honest Gaussian law of assumption (G)
with `σ_V² = 1/d` and `σ_A = 1 > 0`, which `δ_0` is not.  The two tokens are
then frozen at independent uniform draws, their common conditional law is the
uniform measure `σ_1`, and that same measure satisfies `ass:low-temperature`
with density `1`.

In dimension `d ≥ 2` no such pair is available here: producing one is
`thm:PoC_wellposedness`, which is a `sorry`. -/
example :
    ((stdSigmaV 1 : ℝ)) ^ 2 = 1 / ((1 : ℕ) : ℝ) ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
      IsProbabilityMeasure ((uniformAmbient 1).prod (uniformAmbient 1)) ∧
      IsLowTemperature 1 (uniformAmbient 1) (fun _ => uniformAmbient 1)
        (fun _ _ => 1) 1 1 0 ∧
      IsCoupledPair (1 : ℝ) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1)
        ((uniformAmbient 1).prod (uniformAmbient 1))
        (Filtration.const ℝ inferInstance le_rfl) (Filtration.const ℝ ⊥ bot_le)
        (uniformAmbient 1)
        (fun _ ω => clampSphere (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ω.1)
        (fun _ ω => clampSphere (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ω.2)
        (fun _ _ => uniformAmbient 1) := by
  obtain ⟨hprob, -, -⟩ := isUniformAmbient_uniformAmbient (d := 1) one_pos
  refine ⟨stdSigmaV_sq 1, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _,
    ⟨by norm_num, by norm_num⟩, inferInstance,
    isLowTemperature_uniformAmbient one_pos,
    isCoupledPair_uniformAmbient_one 1 1 _ (by simp [PiLp.norm_single])⟩

end Homogenized
end Transformer

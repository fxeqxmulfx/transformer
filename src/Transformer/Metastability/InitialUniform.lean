/-
# Metastability — On the initial configuration, uniform initialization
  (§4 of 2410.06833v1)

* `Proposition prop: concentration unif` — `eq: upto-t`, the orthogonal
                                           approximation of a uniform sample,
* `Corollary coro: cm`, `eq: technical.cond` — uniform points are
                                           `(β, ε)`-separated,
* the low-dimensional bound on the probability of being `(β, ε)`-separated.

All three bound the probability of an event under the *uniform* measure on
`𝕊^{d-1}`, which this development does not construct.  It is pinned down
instead of built: `IsUniformFamily` asks for a probability measure on each
sphere invariant under every linear isometry of `ℝ^d`, and such a measure is
unique, so quantifying over all families satisfying it is not a strengthening.
This is `Perspective.UniformTuple`'s device, one sphere at a time.

`coro: cm` is proved, from `prop: concentration unif` carried as an explicit
hypothesis and from `Metastability.AlphaDist`, whose `α(ε)` estimate corrects
the one the source's proof uses.  The other two are not proved.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Transformer.Metastability.AlphaDist
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Constructions.Pi

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- The uniform law on `𝕊^{d-1}`: a probability measure invariant under every
linear isometry of the ambient `ℝ^d`.

Such a measure is unique — a rotation-invariant Borel probability measure on
`𝕊^{d-1}` *is* `σ_d` — so the statements below, read against every measure
satisfying this, say exactly what the survey says about the uniform one.  The
invariance is stated through `Perspective.sphereMap`, which keeps the Haar
machinery out. -/
def IsUniformOn (d : ℕ) (ν : Measure (SSphere d)) : Prop :=
  IsProbabilityMeasure ν ∧
    ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      ν.map (Perspective.sphereMap d U) = ν

/-- The same, dimension by dimension: a uniform law on every sphere at once. -/
def IsUniformFamily (σ : ∀ d : ℕ, Measure (SSphere d)) : Prop :=
  ∀ d : ℕ, IsUniformOn d (σ d)

/-- The law of `n` i.i.d. draws from a measure `ν` on `𝕊^{d-1}`. -/
noncomputable def iidSphere (d n : ℕ) (ν : Measure (SSphere d)) :
    Measure (Idx n → SSphere d) :=
  Measure.pi fun _ : Idx n => ν

/-- **Equation (eq: upto-t).**  The event that the sample is uniformly close
to a pairwise-orthogonal configuration:

  `∃ (w_1,…,w_n)` pairwise orthogonal with `‖x_i - w_i‖ ≤ √(4 log d / d)`. -/
def nearOrthogonal (d n : ℕ) : Set (Idx n → SSphere d) :=
  { X | ∃ w : Idx n → SSphere d,
      (∀ i j : Idx n, i ≠ j →
        inner (𝕜 := ℝ) ((w i : EucSpace d)) ((w j : EucSpace d)) = 0) ∧
      ∀ i : Idx n, ‖(X i : EucSpace d) - (w i : EucSpace d)‖
        ≤ Real.sqrt (4 * Real.log d / (d : ℝ)) }

/-- **Proposition (prop: concentration unif).**

For `n ≥ 2` there is `d⋆(n) > n` such that for every `d ≥ d⋆(n)`, an i.i.d.
uniform sample on `𝕊^{d-1}` lies in `nearOrthogonal` with probability at
least `1 - 2 n² d^{-1/64}`.

Not proved here.

Source: arXiv:2410.06833v1, §4. -/
theorem concentration_unif (n : ℕ) (hn : 2 ≤ n) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n < d_star ∧ ∀ d : ℕ, d_star ≤ d →
        1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
          ≤ (iidSphere d n (σ d)).real (nearOrthogonal d n) := by
  sorry

/-- The hypothesis of `concentration_unif` is satisfiable: `n = 2`. -/
example : 2 ≤ 2 := le_rfl

/-- **Corollary (coro: cm) with equation (eq: technical.cond).**

For `d ≥ max(d⋆(n), 381)` and `β > 0` satisfying

  `4√(4 log d / d) + (48 log d)/d + β⁻¹ log(n² d / (2 log d)) < 1`,

an i.i.d. uniform sample on `𝕊^{d-1}` is `(β, ε)`-separated with
`ε = 4 log d / d` with probability at least `1 - 2 n² d^{-1/64}`.

The concentration statement the corollary rests on, `concentration_unif`, is
unproved, so it is carried here as the explicit hypothesis `h_conc` rather
than used: what is proved is that the corollary follows from it, and the
dependence is legible in the signature.  Given `h_conc` the rest is
`isSeparated_of_near_orthogonal` — the near-orthogonal configurations are
`(β, ε)`-separated — and monotonicity of the measure.

**What the source says and what is changed here.**  The technical condition is
different.  `eq: technical.cond` reads

  `(16 log² d)/d² + (40 log d)/d + β⁻¹ log(n² d / (2 log d)) < 1`,

whose first two terms are `ε² + 10ε`; that is `eq: gamma` rewritten with the
proof's bound `α(ε) ≤ ε² + 2ε`, which is false — see
`αDist_le_of_orthogonal`, where the correct order is `4√ε`.  With the correct
bound `α(ε) ≤ 4ε + 4√ε` the condition `γ(β) > 0` becomes
`4√ε + 12ε + β⁻¹ log(2n²/ε) < 1`, which is what is assumed above, written out
at `ε = 4 log d / d` and with `2n²/ε = n² d / (2 log d)` as in the source.
The corrected condition is strictly stronger than the printed one for small
`ε`, so this is a correction and not a weakening.

The source's `n ≥ 2` is not needed once `h_conc` is assumed, and is dropped.

Source: arXiv:2410.06833v1, §4, `coro: cm`, `eq: technical.cond`. -/
theorem uniform_separated (n : ℕ)
    (h_conc : ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n < d_star ∧ ∀ d : ℕ, d_star ≤ d →
        1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
          ≤ (iidSphere d n (σ d)).real (nearOrthogonal d n)) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n ≤ d_star ∧ 381 ≤ d_star ∧
        ∀ d : ℕ, d_star ≤ d → ∀ β : ℝ, 0 < β →
          4 * Real.sqrt (4 * Real.log d / (d : ℝ))
              + (48 * Real.log d / (d : ℝ))
              + β⁻¹ * Real.log ((n : ℝ)^2 * d / (2 * Real.log d)) < 1 →
            1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
              ≤ (iidSphere d n (σ d)).real
                  { X | isSeparated d n β (4 * Real.log d / (d : ℝ)) X } := by
  intro σ hσ
  obtain ⟨d₀, hd₀n, hd₀⟩ := h_conc σ hσ
  refine ⟨max d₀ 381, hd₀n.le.trans (le_max_left _ _), le_max_right _ _, ?_⟩
  intro d hd β hβ hcond
  have h381 : (381 : ℝ) ≤ (d : ℝ) := by exact_mod_cast (le_max_right d₀ 381).trans hd
  have hdpos : (0 : ℝ) < (d : ℝ) := by linarith
  have hlogpos : 0 < Real.log (d : ℝ) := Real.log_pos (by linarith)
  set ε : ℝ := 4 * Real.log d / (d : ℝ) with hεdef
  have hε : 0 ≤ ε := by
    rw [hεdef]; positivity
  have hargs : (n : ℝ)^2 * d / (2 * Real.log d) = 2 * (n : ℝ)^2 / ε := by
    rw [hεdef]
    field_simp
    ring
  have h12 : 12 * ε = 48 * Real.log d / (d : ℝ) := by
    rw [hεdef]; ring
  have hsub : nearOrthogonal d n ⊆ { X | isSeparated d n β ε X } := by
    rintro X ⟨w, hw, hclose⟩
    refine isSeparated_of_near_orthogonal d n β ε hε X w hw hclose ?_
    rw [← hargs, h12]
    exact hcond
  have : IsProbabilityMeasure (σ d) := (hσ d).1
  have : IsProbabilityMeasure (iidSphere d n (σ d)) := by
    unfold iidSphere; infer_instance
  exact (hd₀ d ((le_max_left d₀ 381).trans hd)).trans (measureReal_mono hsub)

/-- The hypothesis of `uniform_separated` is satisfiable: at `n = 0` there is
nothing to place, every configuration is near-orthogonal, and the bound reads
`1 ≤ 1`. -/
example : ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
    ∃ d_star : ℕ, 0 < d_star ∧ ∀ d : ℕ, d_star ≤ d →
      1 - 2 * ((0 : ℕ) : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
        ≤ (iidSphere d 0 (σ d)).real (nearOrthogonal d 0) := by
  intro σ hσ
  refine ⟨1, one_pos, fun d _ => ?_⟩
  have : IsProbabilityMeasure (σ d) := (hσ d).1
  have : IsProbabilityMeasure (iidSphere d 0 (σ d)) := by
    unfold iidSphere; infer_instance
  have huniv : nearOrthogonal d 0 = Set.univ :=
    Set.eq_univ_of_forall fun _ => ⟨Fin.elim0, fun i => i.elim0, fun i => i.elim0⟩
  rw [huniv]
  simp

/-- **Low-dimensional bound.**

For `d = 2`, `n ≥ 2` and `0 < ε < 1/16`, the probability that an i.i.d.
uniform sample on `𝕊^1` is `(β, ε)`-separated decays exponentially in `n`:
there is `c ∈ (0, 1)`, depending on `β` and `ε` alone, with

  `ℙ((x_1,…,x_n) is (β, ε)-separated) ≤ c^n`.

Not proved here.

Source: arXiv:2410.06833v1, §4. -/
theorem low_dim_decay (β ε : ℝ) (hε : 0 < ε) (hε16 : ε < 1 / 16) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ c : ℝ, 0 < c ∧ c < 1 ∧
        ∀ n : ℕ, 2 ≤ n →
          (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X } ≤ c ^ n := by
  sorry

/-- The hypotheses of `low_dim_decay` are satisfiable: `ε = 1/32`. -/
example : (0 : ℝ) < 1 / 32 ∧ (1 : ℝ) / 32 < 1 / 16 := by norm_num

end Metastability
end Transformer

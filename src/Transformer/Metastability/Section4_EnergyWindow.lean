/-
# A nonempty energy window for metastability

The energy-level question of arXiv:2410.06833v1, §4,
`sec: energy.levels`, is resolved in the existing large-`β` formulation
by a window near consensus. The strict uniform energy gap makes this
window nonempty, and the pair-deficit estimate supplies a common cap.

The chosen interval may extend beyond the attainable energy range.
Consensus still witnesses nonemptiness, and the lower endpoint forces
the required cap condition for every actual configuration in the interval.
-/

import Transformer.Metastability.Section4_CapRegime
import Transformer.Metastability.EnergyLevel
import Transformer.Metastability.Section4_UniformEnergy

open Real MeasureTheory

namespace Transformer.Metastability
variable (d n : ℕ)

/-- An explicit temperature threshold gives the nonempty energy window.
For every invariant probability law and every `β ≥ 1000 n² + 2`, the
constants may be chosen from its strict gap to consensus. The actual
consensus configuration witnesses nonemptiness, and each configuration in
the window satisfies both dynamical conclusions of `IsMetastable`.

Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. The source fixes any
`β > 0`; this sufficient large-`β` regime is the correction documented in
`energy_level_metastability` below. No minimum width is imposed by the source. -/
theorem energy_level_window_at_threshold (hd : 2 ≤ d) (hn : 2 ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) (β : ℝ)
    (hβ : 1000 * (n : ℝ) ^ 2 + 2 ≤ β) :
    ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ c₁ < c₂ ∧ c₂ < 1 ∧
      (∃ X₀ : SphereTuple d n,
        c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂iidSphere d n ν ∧
        Eβ d n β X₀ - ∫ U, Eβ d n β U ∂iidSphere d n ν ≤ c₂) ∧
      ∀ X₀ : SphereTuple d n,
        c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂iidSphere d n ν →
        Eβ d n β X₀ - ∫ U, Eβ d n β U ∂iidSphere d n ν ≤ c₂ →
          IsMetastable d n β X₀ := by
  have hβ1 : 1 < β := by nlinarith [sq_nonneg (n : ℝ)]
  have hβ0 : 0 < β := by linarith
  let m := ∫ U, Eβ d n β U ∂iidSphere d n ν
  let η := 1 / (2 * β) - m
  have hη : 0 < η := sub_pos.mpr (mean_energy_lt_max hn β hβ0 ν hν)
  have hη1 : η < 1 := (uniform_energy_gap_mem_Ioo hn β hβ1 ν hν).2
  let δ := (Real.exp β - Real.exp (β * (1 - (1 / 100)))) /
    (2 * β * Real.exp β * (n : ℝ) ^ 2)
  have hδ : 0 < δ := cap_energy_tolerance_pos β (1 / 100) hβ0 (by norm_num) (by omega)
  let a := min δ η / 2
  have ha : 0 < a := half_pos (lt_min hδ hη)
  have haδ : a < δ :=
    lt_of_lt_of_le (half_lt_self (lt_min hδ hη)) (min_le_left _ _)
  have haη : a < η :=
    lt_of_lt_of_le (half_lt_self (lt_min hδ hη)) (min_le_right _ _)
  let c₁ := η - a
  let c₂ := (η + 1) / 2
  refine ⟨c₁, c₂, by dsimp [c₁]; linarith, by dsimp [c₁, c₂]; linarith,
    by dsimp [c₂]; linarith, ?_, ?_⟩
  · let w : SSphere d := ⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ),
      by simp⟩
    refine ⟨fun _ => w, ?_, ?_⟩
    all_goals rw [energy_consensus (by omega) β hβ0 w]
    · change η - a ≤ η
      linarith
    · change η ≤ (η + 1) / 2
      linarith
  · intro X hlow
    have hE : 1 / (2 * β) - δ < Eβ d n β X := by
      change η - a ≤ Eβ d n β X - m at hlow
      dsimp [η] at hlow
      linarith
    exact fun _ => isMetastable_of_high_energy (by omega) β hβ X hE

/-- **Problem (sec: energy.levels).** *Metastability from an energy level.*

Fix `d, n ≥ 2`, and let `U_1,…,U_n` be i.i.d. uniform on `𝕊^{d-1}`.  Can one
find `1 > c₂ > c₁ > 0`, depending on `β`, such that every
`(x_1,…,x_n) ∈ (𝕊^{d-1})^n` with

  `c₂ ≥ 𝖤_β(x_1,…,x_n) - 𝔼[𝖤_β(U_1,…,U_n)] ≥ c₁`

is metastable in the sense of `thm: metastability` (`IsMetastable`)?

The expectation is the integral against the product `iidSphere d n ν` of
any rotation-invariant probability law. `isUniformOn_uniformLaw` exhibits
such a law, and `mean_energy_lt_max` proves its energy gap from consensus.

The large-`β` nonempty-window formulation below is proved here with
`β₀ = 1000 n² + 2`. Write `m = 𝔼[𝖤_β(U)]`, `η = 1/(2β) - m > 0`,
`δ = (e^β - e^{β(1-1/100)})/(2β e^β n²)` and `a = min(δ,η)/2`.
Take `c₁ = η - a` and `c₂ = (η+1)/2`. Consensus lies in this window;
every configuration in it has deficit less than `δ`, so all its tokens
lie in one height-`1/100` cap and the proved metastability theorem applies.
This is an affirmative answer for a window near consensus; it does not
establish the source's broader interpretation about every symmetry-breaking
configuration or a prescribed window width.

**What the source says and what is changed here.**

* *`IsMetastable`.*  The source asks for "metastability, as stated in
  `thm: metastability`".  The earlier predicate kept only the conclusion of the
  theorem and admitted `k = 0` caps, for which it held for every configuration;
  it is now the theorem with its hypotheses (a nonempty cover of the initial
  configuration by caps, `γ(β) > 0`, the bounds of `eq: lambda.3`) and its
  conclusion, see `IsMetastable`.
* *The window is nonempty.*  The source does not say how large the window
  `[c₁, c₂]` is, and read literally the question is answered by an empty
  window: `printed_energy_level_trivial` gives, for `β > 1/2`, a `c₁ ≥ 1/(2β)`
  with no configuration in it.  A nonempty window is required here.  That does
  not make the window large: the remark of the source that "any configuration
  which breaks the symmetry of uniformly distributed random points will lead to
  metastability" would ask for more, and the source gives no size to encode.
  A window of configurations with strongly clustered tokens is not excluded by
  the statement.
* *Large `β`.*  The source fixes `β > 0`.  `thm: metastability` needs `β > 1`,
  and `γ(β) > 0` forces `β > (1/2) log(32 n²)` (as `α ≥ -1` and `ε < 1/16`),
  so below that no configuration is metastable in the sense of `IsMetastable`
  and no nonempty window can work.  The statement is for `β ≥ β₀`, some `β₀`;
  the regime of the source is the low temperature limit `β → +∞`.

Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem energy_level_metastability (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ ν : Measure (SSphere d), IsUniformOn d ν →
      ∃ β₀ : ℝ, ∀ β : ℝ, β₀ ≤ β →
        ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ c₁ < c₂ ∧ c₂ < 1 ∧
          (∃ X₀ : SphereTuple d n,
            c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ∧
            Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ≤ c₂) ∧
          ∀ X₀ : SphereTuple d n,
            c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) →
            Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ≤ c₂ →
              IsMetastable d n β X₀ := by
  intro ν hν
  exact ⟨1000 * (n : ℝ) ^ 2 + 2,
    fun β hβ => energy_level_window_at_threshold d n hd hn ν hν β hβ⟩

/-- The explicit-threshold theorem's complete hypotheses are witnessed by
the uniform circle law, two tokens and `β = 10000`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    1000 * ((2 : ℕ) : ℝ) ^ 2 + 2 ≤ (10000 : ℝ) :=
  ⟨le_rfl, le_rfl, isUniformOn_uniformLaw (by norm_num), by norm_num⟩

/-- Every hypothesis is exhibited on the circle with two tokens and the
constructed invariant probability law. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, le_rfl, isUniformOn_uniformLaw (by norm_num)⟩

end Transformer.Metastability

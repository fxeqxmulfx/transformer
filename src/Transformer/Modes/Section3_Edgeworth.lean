import Transformer.Modes.Section3_BR

/-
# The number of modes of a Gaussian KDE — the standardized summand `Y(t)`

§2.3–§3.1 of arXiv:2412.09080v3: the standardized single-sample vector
`Y(t) = Σ_t^{-1/2}(G(t) - 𝔼G(t), G'(t) - 𝔼G'(t))` of `eq:Yi`, the change of
variables `eq:qt` between the density `p_t` of `n^{-1/2} Σ (G, G')(t, Xᵢ)` and
the density `q_t` of `n^{-1/2} Σ Yᵢ(t)`, and `lem:eta`.

**What the source says and what is carried here.**

* `Σ_t^{-1/2}` is `whiten` (see `Section3_Hermite`); `Law(Y(t))` is `lawY`.

* `eq:qt` presupposes that `q_t` exists.  For a single summand it does not
  (`Y(t)` lives on a curve), and the source gets existence for five summands
  only in `lem: pt.bdd`.  The identity is therefore stated as an equivalence:
  `q` is a density of `n^{-1/2} Σ Yᵢ` iff `(det Σ_t)^{-1/2} q(Σ_t^{-1/2}[· - μ_t])`
  is a density of `n^{-1/2} Σ (Gᵢ, Gᵢ')`.  That is the change of variables
  itself, and it carries `eq:qt` in both directions.

* `lem:eta`, first half, "cumulants of order `s` are clearly `O(η_s)`" with a
  constant depending on `s` only, holds for every law with exponential moments
  (a cumulant is a polynomial in moments, each of which Lyapunov's
  inequality bounds by a power of `η_s`); it is stated so.  The second half,
  `η_s ≲ (β e^{t²})^{(s-2)/4}`, is uniform on `T`, as its proof in §5 uses
  `t ∈ T`.

Source: arXiv:2412.09080v3, `eq:Yi`, `eq:qt`, `lem:eta` and its proof in §5.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- One sample of `Y(t) = Σ_t^{-1/2}(G(t) - 𝔼G(t), G'(t) - 𝔼G'(t))`, as a
function of the sample point `X = x`.  arXiv:2412.09080v3, `eq:Yi`. -/
noncomputable def singleY (β t x : ℝ) : ℝ × ℝ :=
  whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
    (bigG β t x - meanG β t, bigG' β t x - meanG' β t)

/-- The law of `Y(t)` for `X ~ N(0,1)`.  arXiv:2412.09080v3, `eq:Yi`. -/
noncomputable def lawY (β t : ℝ) : Measure (ℝ × ℝ) := (gaussianReal 0 1).map (singleY β t)

/-- `Y(t)` is a measurable function of the sample point.  arXiv:2412.09080v3,
`eq:Yi`. -/
theorem measurable_singleY (β t : ℝ) : Measurable (singleY β t) := by
  unfold singleY whiten bigG bigG'
  fun_prop

/-- `Law(Y(t))` is a probability measure.  arXiv:2412.09080v3, `eq:Yi`. -/
instance (β t : ℝ) : IsProbabilityMeasure (lawY β t) :=
  (Measure.isProbabilityMeasure_map_iff (measurable_singleY β t).aemeasurable).2 inferInstance

/-- `η_s = 𝔼‖Y(t)‖^s`.  arXiv:2412.09080v3, `lem:eta`. -/
noncomputable def etaMoment (β t : ℝ) (s : ℕ) : ℝ := ∫ z, eucl z ^ s ∂lawY β t

/-- The normalized sum `n^{-1/2} Σᵢ (G(t, Xᵢ), G'(t, Xᵢ))`, whose density is
`p_t`.  arXiv:2412.09080v3, §2.2. -/
noncomputable def sumGG' (n : ℕ) (β t : ℝ) (X : Fin n → ℝ) : ℝ × ℝ :=
  ((Real.sqrt n)⁻¹ * ∑ i, bigG β t (X i), (Real.sqrt n)⁻¹ * ∑ i, bigG' β t (X i))

/-- **Lemma (lem:eta), cumulants.**  A cumulant of order `s = a + b ≥ 3` is
`O(η_s)`, with a constant depending on `(a, b)` only.
arXiv:2412.09080v3, `lem:eta` and its proof in §5 ("cumulants of order `s` are
clearly `O(η_s)`"), stated for every law with exponential moments. -/
theorem abs_cumulantOf_le (a b : ℕ) (hs : 3 ≤ a + b) :
    ∃ C : ℝ, ∀ μ : Measure (ℝ × ℝ), IsProbabilityMeasure μ → HasExpMoments μ →
      |cumulantOf μ a b| ≤ C * ∫ x, eucl x ^ (a + b) ∂μ := by
  sorry

/-- **Lemma (lem:eta), moments.**  `η_s = 𝔼‖Y(t)‖^s ≲ (β e^{t²})^{(s-2)/4}`,
uniformly on `T`.  arXiv:2412.09080v3, `lem:eta` and its proof in §5. -/
theorem etaMoment_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) (s : ℕ) (hs : 3 ≤ s) :
    ∃ C : ℝ, ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      etaMoment (B k) t s ≤ C * (B k * Real.exp (t ^ 2)) ^ (((s : ℝ) - 2) / 4) := by
  sorry

/-- The hypotheses of `abs_cumulantOf_le` are satisfiable, and so are those
of the implication it asserts. -/
example : 3 ≤ 3 + 0 ∧ IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨le_rfl, inferInstance, hasExpMoments_stdGauss2⟩

/-- The hypotheses of `etaMoment_le` are satisfiable. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧ 3 ≤ 3 :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, le_rfl⟩

end Modes
end Transformer

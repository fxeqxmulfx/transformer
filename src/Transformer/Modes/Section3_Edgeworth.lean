import Transformer.Modes.Section3_BR
import Transformer.Modes.Section3_CumulantCalculus
import Transformer.Modes.Section3_MomentRoots

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
  `t ∈ T`. The moment bound is proved in `Section3_EtaMoment`, after
  standardization and the pointwise bounds for the summand.

Source: arXiv:2412.09080v3, `eq:Yi`, `eq:qt`, `lem:eta` and its proof in §5.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology

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

/-- The algebraic moment expression obtained by actual mixed logarithmic
derivatives. The zero-order expression is zero because a probability MGF
is one at the origin. The equality with `cumulantOf` is proved below.
Source: arXiv:2412.09080v3, §3.1 and the proof of `lem:eta` in §5.3. -/
def cumulantExpression : ℕ → ℕ → MomentExpression
  | 0, 0 => MomentExpression.scale 0 (MomentExpression.moment 0 0)
  | a + 1, 0 => (MomentExpression.diff true)^[a] (MomentExpression.logDerivative true)
  | a, b + 1 => (MomentExpression.diff true)^[a]
      ((MomentExpression.diff false)^[b] (MomentExpression.logDerivative false))

/-- The moment expansion has exactly the order of its mixed derivative.
Source: arXiv:2412.09080v3, proof of `lem:eta` in §5.3. -/
theorem degree_cumulantExpression (a b : ℕ) :
    MomentExpression.Degree (cumulantExpression a b) (a + b) := by
  cases b with
  | zero =>
      cases a with
      | zero => exact MomentExpression.Degree.scale 0 (MomentExpression.Degree.moment 0 0)
      | succ a =>
          simpa [cumulantExpression, Nat.add_comm] using
            (MomentExpression.degree_logDerivative true).iterate_diff true a
  | succ b =>
      simpa [cumulantExpression, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        ((MomentExpression.degree_logDerivative false).iterate_diff false b).iterate_diff true a

/-- The actual mixed derivatives of the logarithm equal their moment
expansion. Exponential moments justify each derivative on a neighbourhood.
Source: arXiv:2412.09080v3, §3.1, definition of `κ^α`, and `lem:eta`. -/
theorem cumulantOf_eq_expression {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) (a b : ℕ) :
    cumulantOf μ a b = (cumulantExpression a b).eval (fun i j => expMoment μ i j 0 0) := by
  obtain ⟨ε, hε, hE⟩ := hexp
  have h0 : |(0 : ℝ)| < ε := by simpa using hε
  cases b with
  | zero =>
      cases a with
      | zero => simp [cumulantOf, cumulantExpression, MomentExpression.eval,
          ← expMoment_zero_zero]
      | succ a =>
          have h := MomentExpression.iteratedDeriv_logAlong hE h0 h0 true a
          simpa [cumulantOf, cumulantExpression, MomentExpression.evalAlong,
            ← expMoment_zero_zero] using h
  | succ b =>
      have heq : (fun u => iteratedDeriv (b + 1)
          (fun v => log (expMoment μ 0 0 u v)) 0) =ᶠ[𝓝 (0 : ℝ)]
          fun u => ((MomentExpression.diff false)^[b] (MomentExpression.logDerivative false)
            ).evalAlong μ true 0 u := by
        filter_upwards [continuous_abs.continuousAt.eventually_lt continuousAt_const h0]
          with u hu
        have h := MomentExpression.iteratedDeriv_logAlong hE hu h0 false b
        simpa [MomentExpression.evalAlong] using h
      simp only [cumulantOf, ← expMoment_zero_zero]
      rw [heq.iteratedDeriv_eq a]
      have h := MomentExpression.iteratedDeriv_evalAlong hE h0 h0
        ((MomentExpression.diff false)^[b] (MomentExpression.logDerivative false)) true a
      rw [cumulantExpression]
      exact h

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, hasExpMoments_stdGauss2⟩

/-- **Lemma (lem:eta), cumulants.**  A cumulant of order `s = a + b ≥ 3` is
`O(η_s)`, with a constant depending on `(a, b)` only.
arXiv:2412.09080v3, `lem:eta` and its proof in §5 ("cumulants of order `s` are
clearly `O(η_s)`"), stated for every law with exponential moments. -/
theorem abs_cumulantOf_le (a b : ℕ) (hs : 3 ≤ a + b) :
    ∃ C : ℝ, ∀ μ : Measure (ℝ × ℝ), IsProbabilityMeasure μ → HasExpMoments μ →
      |cumulantOf μ a b| ≤ C * ∫ x, eucl x ^ (a + b) ∂μ := by
  refine ⟨(cumulantExpression a b).size, fun μ hμ hexp => ?_⟩
  have := hμ
  let M : ℝ := ∫ x, eucl x ^ (a + b) ∂μ
  have hM : 0 ≤ M := integral_nonneg fun x => by unfold eucl; positivity
  have hn : 0 < a + b := by omega
  rw [cumulantOf_eq_expression hexp a b]
  have h := MomentExpression.eval_bound (degree_cumulantExpression a b) le_rfl
    (fun i j => expMoment μ i j 0 0) (M ^ ((a + b : ℕ) : ℝ)⁻¹)
    (moment_root_nonneg M hM (a + b)) (expMoment_origin_zero_zero μ)
    (fun i j hij => expMoment_origin_le_root hexp (a + b) i j hn hij)
  rw [moment_root_pow_self M hM (a + b) hn] at h
  exact h

/-- The hypotheses of `abs_cumulantOf_le` are satisfiable, and so are those
of the implication it asserts. -/
example : 3 ≤ 3 + 0 ∧ IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨le_rfl, inferInstance, hasExpMoments_stdGauss2⟩

end Modes
end Transformer

module

public import Transformer.Perspective.AppendixD_PhaseTransition
public import Transformer.Perspective.AppendixD_PairContraction
public import Transformer.Perspective.Section2_SphereFlow
public import Transformer.Perspective.Section6_EquiangularFlow

/-!
# The complete quantitative phase-transition comparison

ArXiv:2312.10794v5, §6.2, `thm: phase.transition.curve`, proved by combining
Appendix D's exact orthogonal approximation and flow stability with a
uniform version of the proved hemisphere chart contraction.

Both the original trajectory and the orthogonal comparison trajectory have
initial height at least `1/(2n)` in a suitable hemisphere. Their pairwise
inner products therefore differ by at most
`8n³ exp(-exp(-2β)t/n²)`, uniformly in dimension and in the initial data.
The comparison trajectory has the prescribed scalar Gram matrix by uniqueness.
Together with the first stability branch this gives the source's minimum.

This argument supplies actual constants and avoids assuming differentiability
of a minimum over particles or using any of the remaining clustering claims.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory
open Transformer.Metastability

namespace Transformer.Perspective

variable {d n : ℕ}

/-- The source's comparison radius is eventually below the height tolerance
needed for a uniform hemisphere. Source: arXiv:2312.10794v5, Appendix D,
`eq: d.large`; the growth `d/log(d) → ∞` was proved in that appendix's API. -/
theorem exists_small_comparison_radius (n : ℕ) (hn : 0 < n) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d →
      uniformProjectionRadius d ≤ 1 / (2 * (n : ℝ)) := by
  obtain ⟨D, hD⟩ := exists_le_div_log (4 * (n : ℝ) ^ 2)
  refine ⟨max D 2, fun d hd => ?_⟩
  have hd2 : 2 ≤ d := (le_max_right _ _).trans hd
  have hdpos : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hlog : 0 < Real.log (d : ℝ) := Real.log_pos (by exact_mod_cast (show 1 < d by omega))
  have hprod := (le_div_iff₀ hlog).mp (hD d ((le_max_left _ _).trans hd))
  apply Real.sqrt_le_iff.mpr
  refine ⟨by positivity, ?_⟩
  rw [div_pow, one_pow]
  apply (div_le_div_iff₀ hdpos (by positivity : 0 < (2 * (n : ℝ)) ^ 2)).mpr
  nlinarith

example : (0 : ℕ) < 2 := by decide

/-- The two branches of `eq: upto-t` hold for every initial tuple having
the actual orthogonal approximation. Source: arXiv:2312.10794v5,
Appendix D, Step 3. The exponential constants are uniform over all such tuples. -/
theorem curve_comparison_of_orthogonal_approximation (β : ℝ) (hβ : 0 ≤ β) (hn : 2 ≤ n)
    (X₀ Y₀ : SphereTuple d n)
    (hY₀ : ∀ i j : Idx n, i ≠ j → inner (𝕜 := ℝ) (Y₀ i : EucSpace d) (Y₀ j : EucSpace d) = 0)
    (hsmall : uniformProjectionRadius d ≤ 1 / (2 * (n : ℝ)))
    (hclose : ∀ i : Idx n, ‖(X₀ i : EucSpace d) - (Y₀ i : EucSpace d)‖ ≤ uniformProjectionRadius d)
    (γ : ℝ → ℝ) (hγ : ybetaODE_SA n β γ) (X : ℝ → SphereTuple d n)
    (hX₀ : X 0 = X₀) (hX : SA d n β X) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d) - γ t| ≤
        min (2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d))
          (8 * (n : ℝ) ^ 3 * Real.exp (-(Real.exp (-(2 * β)) / (n : ℝ) ^ 2 * t))) := by
  have hnpos : 0 < n := by omega
  obtain ⟨Y, hYinit, hY⟩ := exists_SA β Y₀
  have h0 : tokenGram (Y 0) = equiGram n (γ 0) := by
    rw [hYinit, hγ.1]
    exact tokenGram_orthogonal d n Y₀ hY₀
  have hcommon : ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      inner (𝕜 := ℝ) (Y t i : EucSpace d) (Y t j : EucSpace d) = γ t := by
    intro t ht i j hij
    have h := tokenGram_eq_equiGram d n β Y hY γ hγ.2 h0 t
    simpa only [tokenGram, equiGram, ite_eq_right hij] using congrFun (congrFun h i) j
  obtain ⟨wx, hwx⟩ := uniform_hemisphere_of_near_orthogonal hnpos X₀ Y₀ hY₀ hsmall hclose
  obtain ⟨wy, hwy⟩ := uniform_hemisphere_of_near_orthogonal hnpos Y₀ Y₀ hY₀
    (δ := 0) (by positivity) (fun i => by simp)
  have hlate := sa_pair_inner_difference_uniform hβ hnpos X Y hX hY wx wy
    (by rw [hX₀]; exact hwx) (by rw [hYinit]; exact hwy)
  have hearly := ineq_first_part d n β hβ X Y γ hX hY
    (by rw [hX₀, hYinit]; exact hclose) hcommon
  intro t ht i j hij
  apply le_min (hearly t ht i j hij)
  simpa only [hcommon t ht i j hij] using hlate t ht i j

/-- Two actual orthogonal initial tokens and their genuine global attention
flow realize the deterministic comparison hypotheses in a large dimension. -/
example : ∃ (d : ℕ) (X₀ : SphereTuple d 2) (X : ℝ → SphereTuple d 2) (γ : ℝ → ℝ),
    (0 : ℝ) ≤ 0 ∧ (2 : ℕ) ≤ 2 ∧
    (∀ i j : Idx 2, i ≠ j → inner (𝕜 := ℝ) (X₀ i : EucSpace d) (X₀ j : EucSpace d) = 0) ∧
    uniformProjectionRadius d ≤ 1 / (2 * (2 : ℝ)) ∧
    (∀ i : Idx 2, ‖(X₀ i : EucSpace d) - (X₀ i : EucSpace d)‖ ≤ uniformProjectionRadius d) ∧
    ybetaODE_SA 2 0 γ ∧ X 0 = X₀ ∧ SA d 2 0 X := by
  obtain ⟨D, hD⟩ := exists_small_comparison_radius 2 (by decide)
  let d := max D 2
  let X₀ : SphereTuple d 2 := fun i =>
    ⟨EuclideanSpace.single (Fin.castLE (by omega : 2 ≤ d) i) 1, by simp⟩
  obtain ⟨X, hX₀, hX⟩ := exists_SA 0 X₀
  obtain ⟨γ, hγ0, hγ⟩ := exists_saAngle 2 0 (by decide)
  refine ⟨d, X₀, X, γ, le_rfl, le_rfl, ?_, hD d (le_max_left _ _), ?_,
    ⟨hγ0, fun t => (hγ t).1⟩, hX₀, hX⟩
  · intro i j hij
    simp [X₀, EuclideanSpace.inner_single_left, hij]
  · intro i; simp only [sub_self, norm_zero]; exact Real.sqrt_nonneg _

/-- The full source comparison holds with constants chosen before dimension
and initialization. Source: arXiv:2312.10794v5, §6.2,
`thm: phase.transition.curve`, `eq: upto-t`; this is its Appendix D assembly.

We choose `C = 8n³` and `λ = exp(-2β)/n²`. The dimension threshold depends
only on `n`: it ensures the orthogonal approximation event and makes its
distance tolerance small enough to give the uniform initial hemisphere.
The conclusion holds for every invariant uniform law and every solution
of the prescribed scalar equation. -/
theorem uniform_phase_transition_curve (n : ℕ) (β : ℝ) (hβ : 0 ≤ β) (hn : 2 ≤ n) :
    ∃ C lam : ℝ, 0 < C ∧ 0 < lam ∧
      ∃ d_star : ℕ, n ≤ d_star ∧ ∀ d : ℕ, d_star ≤ d →
        ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
          ∀ γ : ℝ → ℝ, ybetaODE_SA n β γ →
            1 - ENNReal.ofReal (2 * (n : ℝ) ^ 2 * (d : ℝ) ^ (-(1 / 64 : ℝ))) ≤
              P {X₀ : SphereTuple d n | ∀ X : ℝ → SphereTuple d n,
                X 0 = X₀ → SA d n β X → ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
                  |inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d) - γ t| ≤
                    min (2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d))
                      (C * Real.exp (-(lam * t)))} := by
  have hnpos : 0 < n := by omega
  have hnreal : (0 : ℝ) < n := by exact_mod_cast hnpos
  refine ⟨8 * (n : ℝ) ^ 3, Real.exp (-(2 * β)) / (n : ℝ) ^ 2, by positivity, by positivity, ?_⟩
  obtain ⟨D₀, hD₀n, hD₀⟩ := almost_orthogonal n hn
  obtain ⟨D₁, hD₁⟩ := exists_small_comparison_radius n hnpos
  refine ⟨max D₀ D₁, hD₀n.trans (le_max_left _ _), ?_⟩
  intro d hd P hP γ hγ
  have : IsProbabilityMeasure P := by
    rcases hP with ⟨ν, hν, hνinv, rfl⟩
    have : IsProbabilityMeasure ν := hν
    infer_instance
  have hb := hD₀ d ((le_max_left _ _).trans hd) P hP
  have hsmall := hD₁ d ((le_max_right _ _).trans hd)
  rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_sub 1 (by positivity)]
  apply (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top P _)).mpr
  apply hb.trans
  apply measureReal_mono _ (measure_ne_top P _)
  rintro X₀ ⟨Y₀, hY₀, hclose⟩
  exact fun X hXinit hX => curve_comparison_of_orthogonal_approximation β hβ hn
    X₀ Y₀ hY₀ hsmall hclose γ hγ X hXinit hX

example : (0 : ℝ) ≤ 0 ∧ (2 : ℕ) ≤ 2 := ⟨le_rfl, le_rfl⟩

end Transformer.Perspective

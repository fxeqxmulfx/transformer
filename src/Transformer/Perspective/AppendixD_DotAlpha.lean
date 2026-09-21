/-
# `e:dotalpha` and `e:mineqalpha` — the derivative of `α` at the limit

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The lower bound on the derivative of `α(t) = min_i ⟨x_i(t), x⋆⟩` that both
step 2 of `lem: hemisphere.clustering` (§6.1) and Appendix D use, for `x⋆` the
common limit of particles starting in an open hemisphere.  Each then bounds
`α(t)` from below its own way: step 2 by `1/2`
(`Perspective.step2_alpha_diff_ineq`), Appendix D by `α(1/n)`
(`Perspective.diff_ineq_alpha`).
-/

import Transformer.Perspective.Section5_HemisphereCone

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equations (e:dotalpha) and (e:mineqalpha), combined.**

  `α̇(t) ≥ (1/(n e^{2β})) α(t) (1 - α(t))`   wherever `α(t) ≥ 0`,

for `α(t) = min_i ⟨x_i(t), x⋆⟩` and `x⋆` the limit of
`lem: hemisphere.clustering`.  This is the part the survey's two derivations
share: step 2 of the lemma (`e:dotalpha.step2`, `e:mineqalpha.step2`) and
Appendix D (`e:dotalpha`, `e:mineqalpha`) differ only in how they bound
`α(t)` from below afterwards.  `e:dotalpha` is `α̇ ≥ Σ_j a_{ij}(1 - ⟨x_i,x_j⟩)
α(t)`, `e:mineqalpha` is `min_j ⟨x_i,x_j⟩ ≤ α(t)` through the cone of
`limit_mem_cone`, and `a_{ij} ≥ n^{-1} e^{-2β}` joins them.

The hypotheses are those of `diff_ineq_alpha`, whose docstring explains each;
`hαt : 0 ≤ α t` is what makes the bracket `⟨x_j,x⋆⟩ - ⟨x_i,x_j⟩ α(t)` dominate
`α(t)(1 - ⟨x_i,x_j⟩)`.

Source: arXiv:2312.10794v5, §6.1 `e:dotalpha.step2`, `e:mineqalpha.step2`;
Appendix D `e:dotalpha`, `e:mineqalpha`. -/
theorem dot_alpha_ge (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ)
    (hX : SA d n β X) (hα : IsMinInner d n X x_star α)
    (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s)
    (t : ℝ) (ht0 : 0 ≤ t) (hαt0 : 0 ≤ α t) :
    ∃ c : ℝ, HasDerivAt α c t ∧
      ((n : ℝ) * Real.exp (2 * β))⁻¹ * (α t * (1 - α t)) ≤ c := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  obtain ⟨i, hi⟩ := (hα t).2
  -- `α ≤ 1` on the sphere
  have hαle : α t ≤ 1 := by
    rw [hi]
    have hcs := real_inner_le_norm ((X t i : EucSpace d)) ((x_star : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X t i).2,
      mem_sphere_zero_iff_norm.mp x_star.2, one_mul] at hcs
  -- the derivative of the attaining curve, `eq: therighthandside`
  have hfd := step1_rhs d n β X hX ((x_star : EucSpace d)) t i
  set D : ℝ := (partitionSA d n β X t i)⁻¹ *
      ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))) *
          (inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((x_star : EucSpace d))
            - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
                * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)))
    with hDdef
  -- Fermat: where `α` is differentiable its derivative is that of the attaining curve
  have hc : HasDerivAt α D t := by
    have hgd : HasDerivAt
        (fun s => inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((x_star : EucSpace d)) - α s)
        (D - deriv α t) t := hfd.sub (hdα t).hasDerivAt
    have hlm : IsLocalMin
        (fun s => inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((x_star : EucSpace d)) - α s) t := by
      refine Filter.Eventually.of_forall fun s => ?_
      have h1 : inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)) - α t = 0 := by
        rw [← hi]; ring
      simp only
      rw [h1]
      exact sub_nonneg.mpr ((hα s).1 i)
    have hzero := hlm.hasDerivAt_eq_zero hgd
    have hDα : deriv α t = D := by linarith
    rw [← hDα]
    exact (hdα t).hasDerivAt
  refine ⟨D, hc, ?_⟩
  -- the weights of `eq: SA` at `i`
  have hq1 : ∀ j : Idx n,
      inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) ≤ 1 := by
    intro j
    have hcs := real_inner_le_norm ((X t i : EucSpace d)) ((X t j : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X t i).2,
      mem_sphere_zero_iff_norm.mp (X t j).2, one_mul] at hcs
  have hqm1 : ∀ j : Idx n,
      (-1 : ℝ) ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) := by
    intro j
    have hcs := abs_real_inner_le_norm ((X t i : EucSpace d)) ((X t j : EucSpace d))
    rw [mem_sphere_zero_iff_norm.mp (X t i).2,
      mem_sphere_zero_iff_norm.mp (X t j).2, one_mul] at hcs
    exact (abs_le.mp hcs).1
  have hZpos : 0 < partitionSA d n β X t i :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty
  have hZle : partitionSA d n β X t i ≤ (n : ℝ) * Real.exp β := by
    have hterm : ∀ k : Idx n, k ∈ (Finset.univ : Finset (Idx n)) →
        Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t k : EucSpace d)))
          ≤ Real.exp β := by
      intro k _
      exact Real.exp_le_exp.mpr (by nlinarith [hq1 k])
    calc partitionSA d n β X t i ≤ ∑ _k : Idx n, Real.exp β :=
          Finset.sum_le_sum hterm
      _ = (n : ℝ) * Real.exp β := by
          simp [Finset.sum_const, Finset.card_univ]
  -- every bracket is non-negative, and every weight is at least `(n e^{2β})⁻¹`
  have hTnn : ∀ j : Idx n,
      0 ≤ Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))) *
        (inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((x_star : EucSpace d))
          - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
              * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) := by
    intro j
    refine mul_nonneg (Real.exp_nonneg _) ?_
    have hpj : α t ≤ inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((x_star : EucSpace d)) :=
      (hα t).1 j
    rw [← hi]
    nlinarith [hq1 j, hαt0]
  -- `e:mineqalpha`, through the cone of step 2
  obtain ⟨η, ⟨hη0, hη1⟩, hcone⟩ := limit_mem_cone d n hn β w X hX hw x_star hlim t ht0
  obtain ⟨j₀, hj₀'⟩ := exists_inner_le_of_mem_convexHull d n hn (X t) _ hcone i
  have hj₀ : inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))
      ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)) := by
    rw [real_inner_smul_right, ← hi] at hj₀'
    rw [← hi]
    nlinarith
  have hsingle :
      Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))) *
        (inner (𝕜 := ℝ) ((X t j₀ : EucSpace d)) ((x_star : EucSpace d))
          - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))
              * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)))
        ≤ ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))) *
              (inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((x_star : EucSpace d))
                - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
                    * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) :=
    Finset.single_le_sum (fun j _ => hTnn j) (Finset.mem_univ j₀)
  -- the weight at `j₀`
  have hwexp : Real.exp (-β)
      ≤ Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))) :=
    Real.exp_le_exp.mpr (by nlinarith [hqm1 j₀])
  have hZinv : ((n : ℝ) * Real.exp β)⁻¹ ≤ (partitionSA d n β X t i)⁻¹ := by
    gcongr
  have hAeq : ((n : ℝ) * Real.exp (2 * β))⁻¹
      = ((n : ℝ) * Real.exp β)⁻¹ * Real.exp (-β) := by
    rw [Real.exp_neg, two_mul, Real.exp_add]
    field_simp
  have hweight : ((n : ℝ) * Real.exp (2 * β))⁻¹
      ≤ (partitionSA d n β X t i)⁻¹ *
          Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))) := by
    rw [hAeq]
    exact mul_le_mul hZinv hwexp (Real.exp_pos _).le (inv_nonneg.mpr hZpos.le)
  -- the bracket at `j₀`, by `e:mineqalpha`
  have hbracket : α t * (1 - α t)
      ≤ inner (𝕜 := ℝ) ((X t j₀ : EucSpace d)) ((x_star : EucSpace d))
          - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))
              * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)) := by
    have hpj : α t ≤ inner (𝕜 := ℝ) ((X t j₀ : EucSpace d)) ((x_star : EucSpace d)) :=
      (hα t).1 j₀
    rw [← hi] at hj₀ ⊢
    nlinarith [hαt0]
  calc ((n : ℝ) * Real.exp (2 * β))⁻¹ * (α t * (1 - α t))
    _ ≤ ((partitionSA d n β X t i)⁻¹ *
          Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d)))) *
          (inner (𝕜 := ℝ) ((X t j₀ : EucSpace d)) ((x_star : EucSpace d))
            - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))
                * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) :=
        mul_le_mul hweight hbracket (by nlinarith [hαt0, hαle]) (by positivity)
    _ ≤ D := by
        rw [hDdef, mul_assoc]
        exact mul_le_mul_of_nonneg_left hsingle (inv_nonneg.mpr hZpos.le)

end Perspective
end Transformer

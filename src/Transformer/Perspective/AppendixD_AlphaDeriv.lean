/-
# Appendix D — the differential inequality for `α`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:dotalpha`, `e:mineqalpha` and `e:diffineqalpha`: the lower bound on the
derivative of `α(t) = min_i ⟨x_i(t), x⋆⟩` that Appendix D integrates into
`e:productcloseto1` (`Perspective.product_close_to_one`).

The survey writes the three equations for the `x⋆` that `lem: hemisphere.clustering`
produces, under the hemisphere assumption `𝒜`, and it imports from step 2 of
that lemma the fact that `x⋆` is a convex combination of the particles.  With
`x⋆` left free the inequality is false, which `not_forall_diff_ineq_alpha`
records; the hypotheses below are exactly what the survey's own derivation
uses.
-/

import Transformer.Perspective.AppendixD_Alpha
import Transformer.Perspective.Section5_Hemisphere

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:diffineqalpha).** *The differential inequality for `α`.*

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`   for `t ≥ 1/n`.

**What the source says and what is changed here.**  The conclusion is the
survey's, verbatim.  Four hypotheses are added, and each of them is used in
the survey's own derivation without appearing in the displayed equation.

*`hβ : 0 ≤ β`.*  The step "`a_{ij}(t) ≥ n^{-1} e^{-2β}`" is `e^{β⟨x_i,x_j⟩} ≥
e^{-β}` over `Z_{β,i} ≤ n e^{β}`, and both need `β ≥ 0`.

*`hinit`, the hemisphere assumption.*  Appendix D runs under `𝒜`, and `x⋆` is
the limit point of `lem: hemisphere.clustering`, around which the particles
already sit in an open hemisphere.  It enters twice: through step 1, which
makes `α` non-decreasing (`hemisphere_step1_monotone`) and hence `α(1/n) ≤
α(t)`, and through `0 ≤ α(t)`, without which `⟨x_j,x⋆⟩ - ⟨x_i,x_j⟩ α(t) ≥
α(t)(1 - ⟨x_i,x_j⟩)` points the wrong way.

*`hhull`.*  `e:mineqalpha` is the statement that `x⋆` is a convex combination
of the particles — the survey takes it from step 2 of the same lemma.

*`hdα`, differentiability.*  `α` is a minimum of finitely many smooth curves,
so it need not be differentiable where the minimising index changes; the
survey's `α̇` is a Dini derivative.  The statement asserts an honest
`HasDerivAt`, so differentiability is carried as a hypothesis — at a time
where `α` *is* differentiable, Fermat's theorem forces its derivative to agree
with that of the attaining curve, which is what the proof uses.

With `x⋆` free the inequality is false: see `not_forall_diff_ineq_alpha`.

Source: arXiv:2312.10794v5, Appendix D, `e:dotalpha`, `e:mineqalpha`,
`e:diffineqalpha`. -/
theorem diff_ineq_alpha (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ)
    (hX : SA d n β X) (hα : IsMinInner d n X x_star α)
    (hinit : ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((x_star : EucSpace d)))
    (hhull : ∀ s : ℝ, ((x_star : EucSpace d)) ∈
      convexHull ℝ (Set.range fun k : Idx n => ((X s k : EucSpace d))))
    (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hninv : (0 : ℝ) < (n : ℝ)⁻¹ := inv_pos.mpr hnR
  have hmono : MonotoneOn α (Set.Ici (0 : ℝ)) :=
    hemisphere_step1_monotone d n β x_star X α hX hα hinit
  have hα0 : 0 < α 0 := by
    obtain ⟨i, hi⟩ := (hα 0).2
    rw [hi]; exact hinit i
  intro t ht
  have ht0 : (0 : ℝ) ≤ t := le_trans hninv.le ht
  have hαt0 : 0 ≤ α t :=
    le_trans hα0.le (hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht0) ht0)
  have hαstep : α ((n : ℝ)⁻¹) ≤ α t :=
    hmono (Set.mem_Ici.mpr hninv.le) (Set.mem_Ici.mpr ht0) ht
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
  obtain ⟨j₀, hj₀⟩ :=
    exists_inner_le_of_mem_convexHull d n hn (X t) x_star (hhull t) i
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
  have hApos : (0 : ℝ) < ((n : ℝ) * Real.exp (2 * β))⁻¹ := by positivity
  calc ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t)
      ≤ ((n : ℝ) * Real.exp (2 * β))⁻¹ * (α t * (1 - α t)) := by
        rw [mul_assoc]
        exact mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_right hαstep (by linarith)) hApos.le
    _ ≤ ((partitionSA d n β X t i)⁻¹ *
          Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d)))) *
          (inner (𝕜 := ℝ) ((X t j₀ : EucSpace d)) ((x_star : EucSpace d))
            - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j₀ : EucSpace d))
                * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d))) :=
        mul_le_mul hweight hbracket (by nlinarith [hαt0, hαle]) (by positivity)
    _ ≤ D := by
        rw [hDdef, mul_assoc]
        exact mul_le_mul_of_nonneg_left hsingle (inv_nonneg.mpr hZpos.le)

/-- The hypotheses of `diff_ineq_alpha` are satisfiable: two particles sitting
together at `basePoint 0`, measured against their own position, where `α ≡ 1`,
`x⋆` is in the hull because it *is* one of the particles, and both sides of
`e:diffineqalpha` vanish. -/
example :
    ∀ t : ℝ, (((2 : ℕ) : ℝ))⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) c t ∧
        (((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ)))⁻¹ * 1 * (1 - 1) ≤ c :=
  diff_ineq_alpha 1 2 two_pos 0 le_rfl (fun _ _ => basePoint 0) (basePoint 0)
    (fun _ => 1) (SA_const_consensus 1 2 two_pos 0 (basePoint 0))
    (by
      have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
        mem_sphere_zero_iff_norm.mp (basePoint 0).2
      have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
          (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
        rw [real_inner_self_eq_norm_mul_norm, hx]; ring
      exact fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩)
    (fun _ => by
      have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
        mem_sphere_zero_iff_norm.mp (basePoint 0).2
      rw [real_inner_self_eq_norm_mul_norm, hx]; norm_num)
    (fun _ => subset_convexHull ℝ _ ⟨0, rfl⟩)
    (fun _ => differentiableAt_const 1)

/-- **With `x⋆` free the differential inequality is false.**

The survey's `x⋆` is the limit point of `lem: hemisphere.clustering`, and
`e:mineqalpha` reads it as a convex combination of the particles.  Left free
it is any point of the sphere, and then the inequality fails at rest.

One particle standing at `p = basePoint 1 ∈ 𝕊^1` is a solution of `eq: SA`
(`SA_const_consensus`).  Measure it against `x⋆ = (p + q)/√2`, the unit vector
half-way to the second coordinate axis: `α ≡ ⟨p, x⋆⟩ = √2/2` is constant, so
`α̇ = 0`, while the right-hand side of `e:diffineqalpha` at `n = 1`, `β = 0` is

  `α(1) (1 - α(1)) = (√2/2)(1 - √2/2) > 0`.

The configuration satisfies every hypothesis of `diff_ineq_alpha` except
`hhull`: `x⋆` is not the particle, so it is not in the hull of the
configuration.  It also has `α > 0` throughout, so it is not positivity that
fails.

Source: arXiv:2312.10794v5, Appendix D, `e:diffineqalpha`. -/
theorem not_forall_diff_ineq_alpha :
    ¬ ∀ (d n : ℕ) (β : ℝ) (X : ℝ → SphereTuple d n) (x_star : SSphere d)
        (α : ℝ → ℝ), SA d n β X → IsMinInner d n X x_star α →
        ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
          ∃ c : ℝ, HasDerivAt α c t ∧
            ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c := by
  intro h
  have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs0 : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hnorm : ‖((Real.sqrt 2 / 2) • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)
      + EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) : EucSpace 2)‖ = 1 := by
    simp [EuclideanSpace.norm_eq, Fin.sum_univ_two]
    nlinarith
  set v : SSphere 2 := ⟨_, mem_sphere_zero_iff_norm.mpr hnorm⟩ with hvdef
  have hval : inner (𝕜 := ℝ) (((basePoint 1 : SSphere 2)) : EucSpace 2)
      ((v : EucSpace 2)) = Real.sqrt 2 / 2 := by
    rw [hvdef, basePoint]
    simp [PiLp.inner_apply]
  have hmin : IsMinInner 2 1 (fun _ _ => basePoint 1) v (fun _ => Real.sqrt 2 / 2) :=
    fun _ => ⟨fun _ => le_of_eq hval.symm, ⟨0, hval.symm⟩⟩
  obtain ⟨c, hc, hle⟩ := h 2 1 0 (fun _ _ => basePoint 1) v (fun _ => Real.sqrt 2 / 2)
    (SA_const_consensus 2 1 one_pos 0 (basePoint 1)) hmin 1 (by norm_num)
  rw [← (hasDerivAt_const (1 : ℝ) (Real.sqrt 2 / 2)).unique hc] at hle
  norm_num at hle
  nlinarith

end Perspective
end Transformer

/-
# Appendix D — the differential inequality for `α`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:dotalpha`, `e:mineqalpha` and `e:diffineqalpha`: the lower bound on the
derivative of `α(t) = min_i ⟨x_i(t), x⋆⟩` that Appendix D integrates into
`e:productcloseto1` (`Perspective.product_close_to_one`).

The survey writes the three equations for the `x⋆` that `lem: hemisphere.clustering`
produces — the common limit of the particles, which start in an open
hemisphere — and it imports from step 2 of that lemma the fact that `x⋆` lies
in the cone of the particles.  Both are hypotheses of `diff_ineq_alpha` in
that form: the hemisphere and the limit, from which the cone is proved
(`Perspective.limit_mem_cone`).
-/

import Transformer.Perspective.AppendixD_Alpha
import Transformer.Perspective.Section5_HemisphereCone

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:diffineqalpha).** *The differential inequality for `α`.*

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`   for `t ≥ 1/n`.

**What the source says and what is changed here.**  The conclusion is the
survey's, verbatim.  The hypotheses are the survey's setting, each used in its
own derivation without appearing in the displayed equation.

*`hβ : 0 ≤ β`.*  The step "`a_{ij}(t) ≥ n^{-1} e^{-2β}`" is `e^{β⟨x_i,x_j⟩} ≥
e^{-β}` over `Z_{β,i} ≤ n e^{β}`, and both need `β ≥ 0`.

*`hw` and `hlim`: `x⋆` is the limit of `lem: hemisphere.clustering`.*  The
particles start in the open hemisphere around some `w`, and `x⋆` is their
common limit.  Step 2 of the lemma then puts `x⋆` in the cone of the particles
at every time (`limit_mem_cone`), which is what `e:mineqalpha` uses: with
`η x⋆` in the hull, `min_j ⟨x_i,x_j⟩ ≤ η α(t) ≤ α(t)`.

*`hpos : 0 < α(1/n)`.*  This is what the survey draws from `e:1/n`,
`α(1/n) ≥ γ_β(1/n)/2 > 0`, which rests on the high-dimensional estimates of
Appendix D (`Perspective.alpha_at_one_over_n`).  From it step 1, run from time
`1/n` along `x⋆`, makes `α` non-decreasing — "we gather that `α(t) ≥ α(1/n)`
for `t ≥ 1/n`" — and hence `0 ≤ α(t)`, without which `⟨x_j,x⋆⟩ - ⟨x_i,x_j⟩ α(t)
≥ α(t)(1 - ⟨x_i,x_j⟩)` points the wrong way.

*`hdα`, differentiability.*  `α` is a minimum of finitely many smooth curves,
so it need not be differentiable where the minimising index changes; the
survey's `α̇` is a Dini derivative.  The statement asserts an honest
`HasDerivAt`, so differentiability is carried as a hypothesis — at a time
where `α` *is* differentiable, Fermat's theorem forces its derivative to agree
with that of the attaining curve, which is what the proof uses.

Source: arXiv:2312.10794v5, Appendix D, `e:dotalpha`, `e:mineqalpha`,
`e:diffineqalpha`. -/
theorem diff_ineq_alpha (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ)
    (hX : SA d n β X) (hα : IsMinInner d n X x_star α)
    (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (hpos : 0 < α ((n : ℝ)⁻¹))
    (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hninv : (0 : ℝ) < (n : ℝ)⁻¹ := inv_pos.mpr hnR
  -- step 1 from time `1/n` along `x⋆`: `α` does not decrease
  have hmono : MonotoneOn (fun s => α (s + (n : ℝ)⁻¹)) (Set.Ici (0 : ℝ)) :=
    hemisphere_step1_monotone d n β x_star (fun s => X (s + (n : ℝ)⁻¹))
      (fun s => α (s + (n : ℝ)⁻¹)) (SA_shift d n β X hX _) (fun s => hα _)
      (fun i => by rw [zero_add]; exact hpos.trans_le ((hα _).1 i))
  intro t ht
  have ht0 : (0 : ℝ) ≤ t := le_trans hninv.le ht
  have hαstep : α ((n : ℝ)⁻¹) ≤ α t := by
    have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr (sub_nonneg.mpr ht))
      (sub_nonneg.mpr ht)
    simpa using this
  have hαt0 : 0 ≤ α t := hpos.le.trans hαstep
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
together at `basePoint 0`, with `w = x⋆` their position, where `α ≡ 1` and both
sides of `e:diffineqalpha` vanish. -/
example :
    ∀ t : ℝ, (((2 : ℕ) : ℝ))⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) c t ∧
        (((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ)))⁻¹ * 1 * (1 - 1) ≤ c := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact diff_ineq_alpha 1 2 two_pos 0 le_rfl (fun _ _ => basePoint 0) (basePoint 0)
    (fun _ => 1) (SA_const_consensus 1 2 two_pos 0 (basePoint 0))
    (fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩) (basePoint 0)
    (fun _ => by rw [hxx]; norm_num) (fun _ => tendsto_const_nhds) one_pos
    (fun _ => differentiableAt_const 1)

end Perspective
end Transformer

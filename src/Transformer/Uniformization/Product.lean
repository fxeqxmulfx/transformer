/-
# Local uniformization: the product step

Near `0` let `F = m(x') · V(x) · P(x', y)` with `P = y^d + Σ_i b_i(x') y^i` centered. Let
`S` be the set of the `i` with `b_i ≢ 0` and, for `i, i' ∈ S`, let
`D_{ii'} = b_i^{d - i'} - c_{ii'} b_{i'}^{d - i}` with `c_{ii'} ∈ {1, 2}` such that
`D_{ii'} ≢ 0`. Monomial charts on `ℝⁿ` of `G = m · Π_S b_i · Π_{S × S} D_{ii'}`, which
exist by induction on the dimension, times an interval in `y`, are charts in which `m`,
the `b_i` and the `D_{ii'}` have normal crossings: by `Coefficients`, `F` has monomial
charts at every point of them. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Uniformization.Coefficients

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial)

variable {n : ℕ}

/-- **The product step.** A function of `x'` that does not vanish identically times a
unit times a centered monic polynomial in the last variable with analytic coefficients
has monomial charts at `0`, given monomial charts in dimension `n` and for such products
whose polynomial at the point is not `y^d`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem hasMonomialCharts_product {d : ℕ}
    (hunif : ∀ (f : Base n → ℝ) (z : Base n), AnalyticAt ℝ f z → ¬ f =ᶠ[𝓝 z] 0 →
      HasMonomialCharts f z)
    (hcent : ∀ (m : Base n → ℝ) (V : Base (n + 1) → ℝ) (c : Fin d → Base n → ℝ)
      (x₀ : Base (n + 1)), AnalyticAt ℝ m (Fin.init x₀) → ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0 →
      AnalyticAt ℝ V x₀ → V x₀ ≠ 0 → (∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) →
      (∀ i : Fin d, (i : ℕ) + 1 = d → c i (Fin.init x₀) = 0) →
      distinguishedPolynomial d c (Fin.init x₀) ≠ X ^ d →
      HasMonomialCharts (fun x => m (Fin.init x) *
        (V x * preparedPolynomial d c (Fin.init x, x (Fin.last n)))) x₀)
    {m : Base n → ℝ} {V : Base (n + 1) → ℝ} {b : Fin d → Base n → ℝ}
    (hm : AnalyticAt ℝ m 0) (hm0 : ¬ m =ᶠ[𝓝 0] 0) (hV : AnalyticAt ℝ V 0) (hV0 : V 0 ≠ 0)
    (hb : ∀ i, AnalyticAt ℝ (b i) 0)
    (hcenter : ∀ i : Fin d, (i : ℕ) + 1 = d → ∀ x', b i x' = 0) :
    HasMonomialCharts (fun x => m (Fin.init x) *
      (V x * preparedPolynomial d b (Fin.init x, x (Fin.last n)))) 0 := by
  classical
  set S : Finset (Fin d) := Finset.univ.filter fun i => ¬ b i =ᶠ[𝓝 0] 0
  have hS : ∀ i ∈ S, ¬ b i =ᶠ[𝓝 0] 0 := fun i hi => (Finset.mem_filter.1 hi).2
  have hS0 : ∀ i ∉ S, b i =ᶠ[𝓝 0] 0 := fun i hi =>
    not_not.1 fun h => hi (Finset.mem_filter.2 ⟨Finset.mem_univ i, h⟩)
  have hc : ∀ i ∈ S, ∀ i' ∈ S, ∃ c : ℝ, c ≠ 0 ∧
      ¬ (fun x => b i x ^ (d - i') - c * b i' x ^ (d - i)) =ᶠ[𝓝 0] 0 := fun i _ i' hi' =>
    exists_sub_mul_not_eventuallyEq_zero (not_eventuallyEq_zero_pow (hb i') (hS i' hi') _)
  choose! c hc0 hcD using hc
  let D : Fin d × Fin d → Base n → ℝ := fun ii x =>
    b ii.1 x ^ (d - ii.2) - c ii.1 ii.2 * b ii.2 x ^ (d - ii.1)
  have hDA : ∀ x', (∀ i, AnalyticAt ℝ (b i) x') → ∀ ii, AnalyticAt ℝ (D ii) x' :=
    fun x' hb' ii =>
      ((hb' ii.1).fun_pow _).fun_sub (analyticAt_const.fun_mul ((hb' ii.2).fun_pow _))
  have hPb : AnalyticAt ℝ (fun x => ∏ i ∈ S, b i x) 0 :=
    Finset.analyticAt_fun_prod _ fun i _ => hb i
  have hPD : AnalyticAt ℝ (fun x => ∏ ii ∈ S ×ˢ S, D ii x) 0 :=
    Finset.analyticAt_fun_prod _ fun ii _ => hDA 0 hb ii
  have hG0 : ¬ (fun x => m x * ((∏ i ∈ S, b i x) * ∏ ii ∈ S ×ˢ S, D ii x)) =ᶠ[𝓝 0] 0 :=
    not_eventuallyEq_zero_mul hm (hPb.fun_mul hPD) hm0 (not_eventuallyEq_zero_mul hPb hPD
      (not_eventuallyEq_zero_prod S (fun i _ => hb i) hS)
      (not_eventuallyEq_zero_prod (S ×ˢ S) (fun ii _ => hDA 0 hb ii) fun ii hii =>
        hcD ii.1 (Finset.mem_product.1 hii).1 ii.2 (Finset.mem_product.1 hii).2))
  -- the balls where the data are defined
  obtain ⟨r, hr, hgood, hgoodV⟩ : ∃ r > 0, (∀ x' ∈ Metric.ball (0 : Base n) r,
      AnalyticAt ℝ m x' ∧ (∀ i, AnalyticAt ℝ (b i) x') ∧ ∀ i ∉ S, b i x' = 0) ∧
      ∀ x ∈ Metric.ball (0 : Base (n + 1)) r, AnalyticAt ℝ V x ∧ V x ≠ 0 := by
    have h1 : ∀ᶠ x' in 𝓝 (0 : Base n), AnalyticAt ℝ m x' ∧ (∀ i, AnalyticAt ℝ (b i) x') ∧
        ∀ i ∉ S, b i x' = 0 :=
      hm.eventually_analyticAt.and ((eventually_all.2 fun i => (hb i).eventually_analyticAt).and
        (eventually_all.2 fun i => by
          by_cases hi : i ∈ S
          · exact Eventually.of_forall fun x h => absurd hi h
          · exact (hS0 i hi).mono fun x hx _ => hx))
    obtain ⟨r₁, hr₁, h₁⟩ := Metric.eventually_nhds_iff_ball.1 h1
    obtain ⟨r₂, hr₂, h₂⟩ := Metric.eventually_nhds_iff_ball.1
      (hV.eventually_analyticAt.and (hV.continuousAt.eventually_ne hV0))
    exact ⟨min r₁ r₂, lt_min hr₁ hr₂,
      fun x' hx' => h₁ x' (Metric.ball_subset_ball (min_le_left _ _) hx'),
      fun x hx => h₂ x (Metric.ball_subset_ball (min_le_right _ _) hx)⟩
  obtain ⟨T, hTfin, hT, hTcover⟩ := (hunif _ 0 (hm.fun_mul (hPb.fun_mul hPD)) hG0).shrink
    (Metric.ball_mem_nhds (0 : Base n) hr)
  refine hasMonomialCharts_of_cover ((fun e : (Base n → Base n) × Set (Base n) =>
      ((fun x : Base (n + 1) => (Fin.snoc (e.1 (Fin.init x)) (x (Fin.last n)) : Base (n + 1))),
        prodIcc e.2 (-(r / 2)) (r / 2))) '' T) (hTfin.image _) ?_ ?_
  · rintro _ ⟨e, heT, rfl⟩
    obtain ⟨⟨hK, hψ, hNC⟩, hsub⟩ := hT e heT
    refine ⟨IsCompact.prodIcc hK _ _, fun x hx => AnalyticAt.snoc ((hψ _ (mem_prodIcc.1 hx).1).comp
      (analyticAt_init x)) (analyticAt_last x), fun x₀ hx₀ => ?_⟩
    obtain ⟨w', y₀, rfl⟩ : ∃ w' y₀, x₀ = (Fin.snoc w' y₀ : Base (n + 1)) :=
      ⟨_, _, (Fin.snoc_init_self x₀).symm⟩
    rw [mem_prodIcc, Fin.init_snoc, Fin.snoc_last] at hx₀
    obtain ⟨hw', hy₀⟩ := hx₀
    have hψw : e.1 w' ∈ Metric.ball (0 : Base n) r := hsub ⟨w', hw', rfl⟩
    obtain ⟨hmw, hbw, -⟩ := hgood _ hψw
    have hψA : AnalyticAt ℝ e.1 w' := hψ w' hw'
    have hbA : ∀ i, AnalyticAt ℝ (fun x' => b i (e.1 x')) w' := fun i => (hbw i).comp hψA
    have hPb' : AnalyticAt ℝ (fun x' => ∏ i ∈ S, b i (e.1 x')) w' :=
      Finset.analyticAt_fun_prod _ fun i _ => hbA i
    have hPD' : AnalyticAt ℝ (fun x' => ∏ ii ∈ S ×ˢ S, D ii (e.1 x')) w' :=
      Finset.analyticAt_fun_prod _ fun ii _ => (hDA _ hbw ii).comp hψA
    have hN : NormalCrossingAt (fun x' => m (e.1 x') *
        ((∏ i ∈ S, b i (e.1 x')) * ∏ ii ∈ S ×ˢ S, D ii (e.1 x'))) w' := hNC w' hw'
    have hrest := hN.of_mul_right (hmw.comp hψA) (hPb'.fun_mul hPD')
    let Ψ : Base (n + 1) → Base (n + 1) := fun x => Fin.snoc (e.1 (Fin.init x)) (x (Fin.last n))
    have hΨw : Ψ (Fin.snoc w' y₀) = Fin.snoc (e.1 w') y₀ := by
      simp only [Ψ, Fin.init_snoc, Fin.snoc_last]
    have hΨr : Ψ (Fin.snoc w' y₀) ∈ Metric.ball (0 : Base (n + 1)) r := by
      rw [hΨw, mem_ball_zero_iff]
      rw [mem_ball_zero_iff] at hψw
      exact lt_of_le_of_lt (norm_snoc_le (le_max_left _ (r / 2))
        ((abs_le.2 hy₀).trans (le_max_right _ _))) (max_lt hψw (half_lt_self hr))
    have hψA' : AnalyticAt ℝ e.1 (Fin.init (Fin.snoc w' y₀ : Base (n + 1))) := by
      rwa [Fin.init_snoc]
    have hF : (fun x => m (Fin.init x) *
        (V x * preparedPolynomial d b (Fin.init x, x (Fin.last n)))) ∘ Ψ =
        fun x => m (e.1 (Fin.init x)) * (V (Ψ x) *
          preparedPolynomial d (fun i x' => b i (e.1 x')) (Fin.init x, x (Fin.last n))) := by
      funext x
      simp only [Function.comp_apply, Ψ, Fin.init_snoc, Fin.snoc_last]
      rfl
    change HasMonomialCharts ((fun x => m (Fin.init x) *
      (V x * preparedPolynomial d b (Fin.init x, x (Fin.last n)))) ∘ Ψ) (Fin.snoc w' y₀)
    rw [hF]
    refine hasMonomialCharts_product_point (M := fun x' => m (e.1 x'))
      (V := fun x => V (Ψ x)) (B := fun i x' => b i (e.1 x')) hcent
      (hN.of_mul (hmw.comp hψA) (hPb'.fun_mul hPD')) hbA
      (fun i hi => (hrest.of_mul hPb' hPD').of_prod S
        (F := fun i x' => b i (e.1 x')) (fun i _ => hbA i) hi)
      (fun i hi => ?_) (fun i hi x' => hcenter i hi _)
      (fun i hi i' hi' => ⟨c i i', hc0 i hi i' hi', (hrest.of_mul_right hPb' hPD').of_prod
        (S ×ˢ S) (F := fun ii x' => D ii (e.1 x')) (fun ii _ => (hDA _ hbw ii).comp hψA)
        (Finset.mem_product.2 ⟨hi, hi'⟩ : (i, i') ∈ S ×ˢ S)⟩)
      (((hgoodV _ hΨr).1).comp (AnalyticAt.snoc (hψA'.comp (analyticAt_init _))
        (analyticAt_last _))) (hgoodV _ hΨr).2
    filter_upwards [hψA.continuousAt.preimage_mem_nhds (Metric.isOpen_ball.mem_nhds hψw)]
      with x' hx'
    exact (hgood _ hx').2.2 i hi
  · refine mem_of_superset (prodIcc_mem_nhds (z := (0 : Base (n + 1))) hTcover (half_pos hr))
      fun x hx => ?_
    obtain ⟨hx1, hx2⟩ := mem_prodIcc.1 hx
    simp only [Pi.zero_apply, zero_sub, zero_add] at hx2
    rw [Set.mem_iUnion₂] at hx1 ⊢
    obtain ⟨e, heT, w', hw', hew'⟩ := hx1
    refine ⟨_, ⟨e, heT, rfl⟩, Fin.snoc w' (x (Fin.last n)), mem_prodIcc.2 ⟨?_, ?_⟩, ?_⟩
    · rwa [Fin.init_snoc]
    · rwa [Fin.snoc_last]
    · dsimp only
      rw [Fin.init_snoc, Fin.snoc_last, hew', Fin.snoc_init_self]

end Transformer.Uniformization

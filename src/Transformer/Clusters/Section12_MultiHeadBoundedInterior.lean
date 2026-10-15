/-
# The bounded interior token in the original discrete multi-head equation

ArXiv:2305.05465v6, §7, `l:boundedother`, with §3's discrete
rescaling and the separate original positive heads in §12.
The actual normalized rate configuration has both signs in this case.
Its weighted head mean has a unique zero. At each fixed query the
actual normalized means converge to that limiting mean, while their
values at the actual bounded token's scaled query tend to zero.
Monotonicity traps that query on both sides of the unique limiting zero.

This argument uses the original discrete coordinate equation and its
bounded increments. It supplies the convergence required in the paper's
informal drift argument without assuming a query limit, uniform
convergence in the query, or a differential equation for the iterates.
The exact original head scores then converge separately.
-/

import Transformer.Clusters.Section12_MultiHeadRates
import Transformer.Clusters.Section12_MultiHeadScaledDrift
import Transformer.Clusters.Section12_MultiHeadMean

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

/-- Monotone approximating means with asymptotically zero values trap their queries at a strict limiting zero.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, made precise for §12's discrete equation. -/
theorem tendsto_query_of_monotone_zero {α : Type*} {F : Filter α}
    (G : ℝ → ℝ) (g : α → ℝ → ℝ) (r : α → ℝ) (θ : ℝ)
    (hG : StrictMono G) (hθ : G θ = 0) (hg : ∀ k, Monotone (g k))
    (hlim : ∀ a, Tendsto (fun k => g k a) F (nhds (G a)))
    (hzero : Tendsto (fun k => g k (r k)) F (nhds 0)) :
    Tendsto r F (nhds θ) := by
  apply tendsto_order.mpr
  constructor
  · intro a ha
    have hneg : G a < 0 := by simpa only [hθ] using hG ha
    have hleft := (tendsto_order.mp (hlim a)).2 (G a / 2) (by linarith)
    have hright := (tendsto_order.mp hzero).1 (G a / 2) (by linarith)
    filter_upwards [hleft, hright] with k hk hz
    by_contra hn
    have he := hg k (show r k ≤ a by linarith)
    linarith
  · intro b hb
    have hpos : 0 < G b := by simpa only [hθ] using hG hb
    have hleft := (tendsto_order.mp (hlim b)).1 (G b / 2) (by linarith)
    have hright := (tendsto_order.mp hzero).2 (G b / 2) (by linarith)
    filter_upwards [hleft, hright] with k hk hz
    by_contra hn
    have he := hg k (show b ≤ r k by linarith)
    linarith

variable {m H : ℕ}

/-- A bounded original token's actual scaled query converges to the weighted mean's zero when rates have both signs.
Source: arXiv:2305.05465v6, §7, `l:boundedother`, for §12's original positive discrete head sum. -/
theorem multiHeadTransformer_bounded_query_limit {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (h : Idx H) (i : Idx (m + 1)) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (γ : Idx (m + 1) → ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    {p q : Idx (m + 1)} (hp : 0 < γ p) (hq : γ q < 0) :
    ∃ θ : ℝ,
      multiHeadScalarDrift (fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1))
        (fun a => (V a unit1) 0) γ θ = 0 ∧
      Tendsto (fun k => multiHeadLayerFactor Δt V ^ k * X k i 0) atTop (nhds θ) := by
  let c := fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1)
  let v := fun a => (V a unit1) 0
  have hc : ∀ a, 0 < c a := fun a => (scalar_head_parameters_pos (Q a) (K a) (V a) (hV a) (hQK a)).1
  have hv : ∀ a, 0 < v a := fun a => (scalar_head_parameters_pos (Q a) (K a) (V a) (hV a) (hQK a)).2
  obtain ⟨θ, hθ, _⟩ := existsUnique_multiHeadScalarDrift_root c v hc hv γ hp hq h
  refine ⟨θ, hθ, ?_⟩
  apply tendsto_query_of_monotone_zero (multiHeadScalarDrift c v γ)
    (fun k => multiHeadScalarDrift c v (fun j => X k j 0 / multiHeadLayerFactor Δt V ^ k))
    (fun k => multiHeadLayerFactor Δt V ^ k * X k i 0) θ
  · exact multiHeadScalarDrift_strictMono c v hc hv γ (show γ p ≠ γ q by linarith) h
  · exact hθ
  · exact fun k => multiHeadScalarDrift_monotone c v (fun a => (hc a).le)
      (fun a => (hv a).le) (fun j => X k j 0 / multiHeadLayerFactor Δt V ^ k)
  · intro a
    exact tendsto_multiHeadScalarDrift c v hγ (tendsto_const_nhds (x := a))
  · exact multiHeadTransformer_bounded_scaled_drift_zero hΔt hV hX h i hb

/-- Converging actual normalized anchors and an actual scaled query give the limit of each original head weight.
Source: arXiv:2305.05465v6, §7, final score limit in `l:boundedother`, for each original head of §12. -/
theorem multiHead_attention_tendsto_of_query {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (X : ℕ → Idx (m + 1) → EucSpace 1) (i : Idx (m + 1)) (γ : Idx (m + 1) → ℝ) (θ : ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    (hθ : Tendsto (fun k => multiHeadLayerFactor Δt V ^ k * X k i 0) atTop (nhds θ))
    (h : Idx H) (j : Idx (m + 1)) :
    Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop
      (nhds (Perspective.softmaxWeight (fun l => inner (𝕜 := ℝ) (Q h unit1) (K h unit1) * θ * γ l) j)) := by
  have hs : ∀ l, Tendsto
      (fun k => inner (𝕜 := ℝ) (Q h unit1) (K h unit1) *
        (multiHeadLayerFactor Δt V ^ k * X k i 0) * (X k l 0 / multiHeadLayerFactor Δt V ^ k))
      atTop (nhds (inner (𝕜 := ℝ) (Q h unit1) (K h unit1) * θ * γ l)) :=
    fun l => (hθ.const_mul _).mul (hγ l)
  simpa only [multiHead_attention_rescaled hΔt Q K V hV X h] using tendsto_softmaxWeight_finite hs j

/-- Every original head of a bounded token with rates of both signs converges to its own root-score row.
Source: arXiv:2305.05465v6, §7, `l:boundedother`, for the separate original heads of §12. -/
theorem multiHeadTransformer_bounded_interior_rows {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (h : Idx H) (i : Idx (m + 1)) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (γ : Idx (m + 1) → ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    {p q : Idx (m + 1)} (hp : 0 < γ p) (hq : γ q < 0) :
    ∃ θ : ℝ, ∀ a j, Tendsto (fun k => attentionMatrix (Q a) (K a) (X k) i j) atTop
      (nhds (Perspective.softmaxWeight
        (fun l => inner (𝕜 := ℝ) (Q a unit1) (K a unit1) * θ * γ l) j)) := by
  obtain ⟨θ, _, hθ⟩ := multiHeadTransformer_bounded_query_limit hΔt hV hQK hX h i hb γ hγ hp hq
  exact ⟨θ, fun a j => multiHead_attention_tendsto_of_query hΔt.le hV X i γ θ hγ hθ a j⟩

/-- A varying query and varying monotone means jointly realize the trapping lemma's premises. -/
example : ∃ (g : ℕ → ℝ → ℝ) (r : ℕ → ℝ), StrictMono (fun x : ℝ => x) ∧
    (∀ k, Monotone (g k)) ∧ (∀ a, Tendsto (fun k => g k a) atTop (nhds a)) ∧
    Tendsto (fun k => g k (r k)) atTop (nhds 0) := by
  let e := fun k : ℕ => (1 / 2 : ℝ) ^ k
  have he : Tendsto e atTop (nhds 0) := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  refine ⟨fun k a => a + e k, e, ?_, ?_, ?_, ?_⟩
  · intro a b hab; exact hab
  · intro k a b hab; linarith
  · intro a
    simpa only [add_zero] using (tendsto_const_nhds (x := a)).add he
  · simpa only [add_zero] using he.add he

/-- A genuine original three-token trajectory with two positive heads jointly realizes
boundedness, distinctness, rates of both signs, and the actual scaled-query limit. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1) (γ : Idx 3 → ℝ),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ k, |X k 1 0| ≤ 0) ∧
    (∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds (γ j))) ∧
    0 < γ 2 ∧ γ 0 < 0 ∧
    Tendsto (fun k => multiHeadLayerFactor 1 V ^ k * X k 1 0) atTop (nhds 0) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, hn, hord⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  have hd : ∀ i j, i ≠ j → X 0 i ≠ X 0 j := by
    intro i j hij he
    have hc := congrArg (fun z : EucSpace 1 => z 0) he
    rcases lt_or_gt_of_ne hij with hlt | hgt
    · have hs := hord i j hlt; linarith
    · have hs := hord j i hgt; linarith
  obtain ⟨_, _, _, _, γ, _, _, hγ, _, hpos, hneg, _⟩ :=
    multiHeadTransformer_exists_token_rates one_pos hV hQK hX hd 0
  refine ⟨V, X, γ, one_pos, hV, hQK, hX, ?_, hγ, (hpos 2 hp).2, (hneg 0 hn).2, ?_⟩
  · intro k; rw [hz k]; norm_num
  · simpa only [hz, PiLp.zero_apply, mul_zero] using (tendsto_const_nhds (x := (0 : ℝ)) :
      Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0))

end Transformer.Clusters

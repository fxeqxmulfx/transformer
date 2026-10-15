/-
# A bounded original token cannot stay uniformly positive under vanishing negative head terms

ArXiv:2305.05465v6, §7, proof of `l:onlyone`, from
`e:minorationpourxn`, for §12's original finite head sum.
The separate original head estimates give a uniformly positive summed
drift if a query stays above a positive threshold and all negative
anchor contributions vanish. In discrete time this gives a fixed
positive coordinate increment at every sufficiently late layer.
An elementary finite-step sum contradicts any global upper bound.

The result is the discrete version of `frequently_lt_of_bounded`;
reflection will exclude two bounded tokens beside an escaping one.
The joint witness has three actual original tokens, the middle one
identically zero and the others escaping in opposite directions.
-/

import Transformer.Clusters.Section12_MultiHeadNegativePart
import Transformer.Clusters.Section12_MultiHeadSymmetry

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- Fixed eventual increments give an explicit linear bound along the actual discrete indices.
Source: arXiv:2305.05465v6, §7, the final integration in the proof of `l:onlyone`, in discrete time. -/
theorem discrete_linear_lower_bound_after (f : ℕ → ℝ) (η : ℝ) (T : ℕ)
    (hstep : ∀ k, T ≤ k → f k + η ≤ f (k + 1)) :
    ∀ l : ℕ, f T + (l : ℝ) * η ≤ f (T + l) := by
  intro l
  induction l with
  | zero => simp only [Nat.cast_zero, zero_mul, add_zero]; exact le_rfl
  | succ l ih =>
      have hs := hstep (T + l) (by omega)
      push_cast
      convert (show f T + ((l : ℝ) + 1) * η ≤ f (T + l + 1) by linarith) using 1

/-- A globally upper-bounded sequence cannot have fixed positive increments at all sufficiently late layers.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, the boundedness contradiction in discrete time. -/
theorem discrete_not_uniform_increment_of_bounded (f : ℕ → ℝ) {R η : ℝ}
    (hR : ∀ k, f k ≤ R) (hη : 0 < η) (T : ℕ) :
    ¬ ∀ k, T ≤ k → f k + η ≤ f (k + 1) := by
  intro hstep
  obtain ⟨l, hl⟩ := exists_nat_gt ((R - f T) / η)
  have hbig : R - f T < (l : ℝ) * η := (div_lt_iff₀ hη).mp hl
  have hg := discrete_linear_lower_bound_after f η T hstep l
  linarith [hR (T + l)]

/-- Every separate original positive head contributes its positive query bound to the actual summed drift.
Source: arXiv:2305.05465v6, §12's head sum and §7, `e:minorationpourxn`. -/
theorem multiHeadScalarDrift_eventually_ge (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 ≤ v h) (y : ℕ → Idx (m + 1) → ℝ) (i : Idx (m + 1))
    {δ : ℝ} (hδ : 0 < δ) (hr : ∀ᶠ k in atTop, δ ≤ y k i)
    (htok : ∀ j, (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ y k j) ∨
      Tendsto (fun k => y k j) atTop atBot) :
    ∀ᶠ k in atTop, (∑ h, v h) * (δ / (2 * ((m : ℝ) + 1))) ≤
      multiHeadScalarDrift c v (y k) (y k i) := by
  have hh : ∀ h, ∀ᶠ k in atTop, δ / (2 * ((m : ℝ) + 1)) ≤ scalarHeadAverage (c h) (y k) (y k i) :=
    fun h => scalarHeadAverage_eventually_ge_of_positive_query (c h) (hc h) y i hδ hr htok
  filter_upwards [eventually_all.mpr hh] with k hk
  have hs := Finset.sum_le_sum fun h (_ : h ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hk h) (hv h)
  rw [← Finset.sum_mul] at hs
  exact hs

/-- A bounded original token falls below every positive threshold at arbitrarily late layers
if each anchor approaches the nonnegative half-line or tends to negative infinity.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, `e:minorationpourxn`, for §12's original equation. -/
theorem multiHeadTransformer_bounded_frequently_lt {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H) (i : Idx (m + 1))
    {δ R : ℝ} (hδ : 0 < δ) (hR : ∀ k, X k i 0 ≤ R)
    (htok : ∀ j, (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ X k j 0) ∨
      Tendsto (fun k => X k j 0) atTop atBot) : ∃ᶠ k in atTop, X k i 0 < δ := by
  by_contra hn
  simp only [not_frequently, not_lt] at hn
  let Λ := multiHeadValueCoefficient V
  have hΛ : 0 < Λ := (multiHead_coefficients_pos (m + 1) Q K V hV hQK h).1
  let κ := Λ * (δ / (2 * ((m : ℝ) + 1)))
  have hκ : 0 < κ := by dsimp [κ]; positivity
  have hd : ∀ᶠ k in atTop, κ ≤
      multiHeadScalarDrift (fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1))
        (fun a => (V a unit1) 0) (fun j => X k j 0) (X k i 0) := by
    exact multiHeadScalarDrift_eventually_ge _ _
      (fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).1)
      (fun a => (scalar_head_parameters_pos _ _ _ (hV a) (hQK a)).2.le)
      (fun k j => X k j 0) i hδ hn htok
  have hinc : ∀ᶠ k in atTop, X k i 0 + Δt * κ ≤ X (k + 1) i 0 := by
    filter_upwards [hd] with k hk
    rw [multiHeadTransformer_coord hX k i]
    have hm := mul_le_mul_of_nonneg_left hk hΔt.le
    linarith
  obtain ⟨T, hT⟩ := eventually_atTop.mp hinc
  exact discrete_not_uniform_increment_of_bounded (fun k => X k i 0) hR (mul_pos hΔt hκ) T hT

/-- A nonconstant scalar sequence realizes the actual discrete summation premise. -/
example : ∃ f : ℕ → ℝ, ∀ k, 0 ≤ k → f k + (1 : ℝ) ≤ f (k + 1) := by
  refine ⟨fun k => (k : ℝ), ?_⟩
  intro k hk
  push_cast
  exact le_rfl

/-- A genuinely bounded scalar sequence realizes the bounded-increment contradiction's premises. -/
example : ∃ f : ℕ → ℝ, (∀ k, f k ≤ (0 : ℝ)) ∧ (0 : ℝ) < 1 :=
  ⟨fun _ => 0, fun _ => le_rfl, one_pos⟩

/-- Positive bounded original anchors realize all premises of the summed static head estimate. -/
example : ∃ (c v : Idx 2 → ℝ) (y : ℕ → Idx 2 → ℝ),
    (∀ h, 0 < c h) ∧ (∀ h, 0 ≤ v h) ∧ (∀ᶠ k in atTop, (1 : ℝ) ≤ y k 0) ∧
    (∀ j, ∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ y k j) := by
  refine ⟨![1, 2], ![1, 3], fun _ => ![1, 2], ?_, ?_, Eventually.of_forall fun _ => by norm_num, ?_⟩
  · intro h; fin_cases h <;> norm_num
  · intro h; fin_cases h <;> norm_num
  · intro j ε hε
    apply Eventually.of_forall
    intro k
    fin_cases j <;> norm_num <;> linarith

/-- An actual stationary interior token and two escaping original tokens jointly realize
all matrix, boundedness, and anchor-alternative premises of the frequently-small conclusion. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ (∀ k, X k 1 0 ≤ 0) ∧
    (∀ j, (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ X k j 0) ∨
      Tendsto (fun k => X k j 0) atTop atBot) := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (Q h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (Q h) (Q h) :=
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, hn, _⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  refine ⟨Q, Q, Q, X, one_pos, hV, hQK, hX, ?_, ?_⟩
  · intro k; rw [hz k]; norm_num
  · intro j
    fin_cases j
    · exact Or.inr hn
    · left
      intro ε hε
      apply Eventually.of_forall
      intro k
      change -ε ≤ X k 1 0
      rw [hz k]
      norm_num
      linarith
    · exact Or.inl fun ε _ => tendsto_atTop.mp hp (-ε)

end Transformer.Clusters

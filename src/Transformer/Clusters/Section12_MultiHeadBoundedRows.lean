module

public import Transformer.Clusters.Section12_MultiHeadBoundedInterior
public import Transformer.Clusters.Section12_MultiHeadMeanDefect

/-!
# The exceptional bounded row of the original discrete multi-head Transformer

ArXiv:2305.05465v6, §7, `l:boundedxn` and `l:boundedother`,
for §3's discrete setting and §12's original positive heads.
The normalized original anchors converge at the full original layer
factor. A bounded token supplies a zero self rate and an actual
scaled weighted drift tending to zero.

If every off-diagonal rate is negative, the weighted mean-defect bound
makes every original head's off-diagonal weights vanish; finite row
normalization leaves the self weight one. Simultaneous sign reversal
of the original trajectory supplies the positive-rate case, preserving
each original attention matrix. Rates of both signs instead use the
unique weighted-mean zero and the actual bounded scaled-query limit.
These exhaust nonzero off-diagonal rates, including the single-token
case where the off-diagonal family is empty.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- Negative off-diagonal original rates force the bounded token's row to its self basis row in every head.
Source: arXiv:2305.05465v6, §7, `l:boundedxn`, for the actual discrete heads in §12. -/
theorem multiHeadTransformer_bounded_negative_rates {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ a, IsPosDefOp (V a))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (h : Idx H) (i : Idx (m + 1)) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (γ : Idx (m + 1) → ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    (hneg : ∀ j, j ≠ i → γ j < 0) (j : Idx (m + 1)) :
    Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = i then 1 else 0)) := by
  let c := fun a => inner (𝕜 := ℝ) (Q a unit1) (K a unit1)
  let v := fun a => (V a unit1) 0
  let z := fun k j => X k j 0 / multiHeadLayerFactor Δt V ^ k
  let r := fun k => multiHeadLayerFactor Δt V ^ k * X k i 0
  have hv : ∀ a, 0 < v a := fun a => value_coordinate_pos (V a) (hV a)
  have hz : Tendsto (fun k => z k i) atTop (nhds 0) :=
    bounded_div_power_zero (fun k => X k i 0) _ (multiHeadLayerFactor_gt_one hΔt V hV h) hb
  have hn : ∀ l, l ≠ i → ∃ a : ℝ, a < 0 ∧ Tendsto (fun k => z k l) atTop (nhds a) :=
    fun l hl => ⟨γ l, hneg l hl, hγ l⟩
  have hzero : Tendsto (fun k => multiHeadScalarDrift c v (z k) (r k)) atTop (nhds 0) :=
    multiHeadTransformer_bounded_scaled_drift_zero hΔt hV hX h i hb
  have hoff : ∀ l, l ≠ i → Tendsto
      (fun k => Perspective.softmaxWeight (fun b => c h * r k * z k b) l) atTop (nhds 0) :=
    fun l hl => tendsto_multiHead_oneSided_off c v hv z r i hz hn hzero h l hl
  have ht : Tendsto (fun k => Perspective.softmaxWeight (fun b => c h * r k * z k b) j)
      atTop (nhds (if j = i then 1 else 0)) := by
    by_cases hji : j = i
    · subst j
      simpa only [ite_true] using tendsto_softmaxWeight_self_of_off_zero
        (fun k b => c h * r k * z k b) i hoff
    · simpa only [ite_eq_right hji] using hoff j hji
  simpa only [multiHead_attention_rescaled hΔt.le Q K V hV X h] using ht

/-- Positive off-diagonal original rates give the same self-row limit by symmetry of the actual original dynamics.
Source: arXiv:2305.05465v6, §7, the left-endpoint case of `l:boundedxn`, for §12's separate heads. -/
theorem multiHeadTransformer_bounded_positive_rates {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ a, IsPosDefOp (V a))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (h : Idx H) (i : Idx (m + 1)) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (γ : Idx (m + 1) → ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    (hpos : ∀ j, j ≠ i → 0 < γ j) (j : Idx (m + 1)) :
    Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = i then 1 else 0)) := by
  have hbneg : ∃ R : ℝ, ∀ k, |(-X k i) 0| ≤ R := by
    obtain ⟨R, hR⟩ := hb
    exact ⟨R, fun k => by simpa only [PiLp.neg_apply, abs_neg] using hR k⟩
  have hγneg : ∀ l, Tendsto (fun k => (-X k l) 0 / multiHeadLayerFactor Δt V ^ k)
      atTop (nhds (-γ l)) := by
    intro l
    simpa only [PiLp.neg_apply, neg_div] using (hγ l).neg
  have ht := multiHeadTransformer_bounded_negative_rates hΔt hV (multiHeadTransformer_neg hX)
    h i hbneg (fun l => -γ l) hγneg (fun l hl => by linarith [hpos l hl]) j
  simpa only [attentionMatrix_neg_one] using ht

/-- Every original bounded row converges when its actual normalized off-diagonal rates are nonzero.
Source: arXiv:2305.05465v6, §7, `l:boundedxn` and `l:boundedother`, for §12's original heads. -/
theorem multiHeadTransformer_bounded_row_exists {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ a, IsPosDefOp (V a))
    (hQK : ∀ a, IsPosDefQK (Q a) (K a))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (h : Idx H) (i : Idx (m + 1)) (hb : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (γ : Idx (m + 1) → ℝ)
    (hγ : ∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor Δt V ^ k) atTop (nhds (γ j)))
    (hne : ∀ j, j ≠ i → γ j ≠ 0) :
    ∃ P : Idx (m + 1) → ℝ, ∀ j,
      Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (P j)) := by
  by_cases hp : ∃ p, 0 < γ p
  · obtain ⟨p, hp⟩ := hp
    by_cases hn : ∃ q, γ q < 0
    · obtain ⟨q, hq⟩ := hn
      obtain ⟨θ, hθ⟩ := multiHeadTransformer_bounded_interior_rows hΔt hV hQK hX h i hb γ hγ hp hq
      exact ⟨Perspective.softmaxWeight (fun l => inner (𝕜 := ℝ) (Q h unit1) (K h unit1) * θ * γ l), hθ h⟩
    · have hpos : ∀ j, j ≠ i → 0 < γ j := by
        intro j hj
        have hnj : ¬ γ j < 0 := fun hh => hn ⟨j, hh⟩
        have hneq := hne j hj
        rcases lt_or_gt_of_ne hneq with hn' | hp'
        · exact (hnj hn').elim
        · exact hp'
      exact ⟨fun j => if j = i then 1 else 0,
        multiHeadTransformer_bounded_positive_rates hΔt hV hX h i hb γ hγ hpos⟩
  · have hneg : ∀ j, j ≠ i → γ j < 0 := by
      intro j hj
      have hpj : ¬ 0 < γ j := fun hh => hp ⟨j, hh⟩
      have hneq := hne j hj
      rcases lt_or_gt_of_ne hneq with hn' | hp'
      · exact hn'
      · exact (hpj hp').elim
    exact ⟨fun j => if j = i then 1 else 0,
      multiHeadTransformer_bounded_negative_rates hΔt hV hX h i hb γ hγ hneg⟩

/-- A converging original discrete attention row retains nonnegativity and unit total mass.
Source: arXiv:2305.05465v6, §7, the final probability argument for `t:boolean`, for §12's heads. -/
theorem attentionMatrix_discrete_row_limit_probability {d n : ℕ} (hn : 0 < n)
    (Q K : ParamMatrix d) (X : ℕ → Idx n → EucSpace d) (i : Idx n) (P : Idx n → ℝ)
    (hP : ∀ j, Tendsto (fun k => attentionMatrix Q K (X k) i j) atTop (nhds (P j))) :
    (∀ j, 0 ≤ P j) ∧ ∑ j, P j = 1 := by
  refine ⟨fun j => ge_of_tendsto (hP j) (Eventually.of_forall fun k => attentionMatrix_nonneg Q K (X k) i j), ?_⟩
  have ht : Tendsto (fun k => ∑ j, attentionMatrix Q K (X k) i j) atTop (nhds (∑ j, P j)) :=
    tendsto_finsetSum _ fun j _ => hP j
  have hs : Tendsto (fun k => ∑ j, attentionMatrix Q K (X k) i j) atTop (nhds 1) := by
    simpa only [sum_attentionMatrix hn] using (tendsto_const_nhds (x := (1 : ℝ)) :
      Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1))
  exact tendsto_nhds_unique ht hs

/-- The genuine original stationary single-token trajectory with two positive heads
jointly realizes every one-sided and bounded-row premise, including the empty off-diagonal family. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 1 → EucSpace 1) (γ : Idx 1 → ℝ),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (V h) (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ k, |X k 0 0| ≤ 0) ∧
    (∀ j, Tendsto (fun k => X k j 0 / multiHeadLayerFactor 1 V ^ k) atTop (nhds (γ j))) ∧
    (∀ j, j ≠ 0 → γ j < 0) ∧ (∀ j, j ≠ 0 → 0 < γ j) ∧ (∀ j, j ≠ 0 → γ j ≠ 0) ∧
    (∀ j, Tendsto (fun k => attentionMatrix (V 0) (V 0) (X k) 0 j) atTop (nhds 1)) := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  let X : ℕ → Idx 1 → EucSpace 1 := fun _ _ => 0
  have hV : ∀ h, IsPosDefOp (V h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (V h) (V h) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  have hX : MultiHeadTransformer 1 V V V X := by intro k i; simp [X]
  have hatt : ∀ k (j : Idx 1), attentionMatrix (V 0) (V 0) (X k) 0 j = 1 := by
    intro k j
    have hj : j = 0 := Subsingleton.elim _ _
    rw [hj]
    simpa only [Fin.sum_univ_one] using sum_attentionMatrix one_pos (V 0) (V 0) (X k) 0
  refine ⟨V, X, fun _ => 0, one_pos, hV, hQK, hX, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k; norm_num [X]
  · intro j
    simpa only [X, PiLp.zero_apply, zero_div] using (tendsto_const_nhds (x := (0 : ℝ)) :
      Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0))
  · intro j hj; fin_cases j; exact (hj rfl).elim
  · intro j hj; fin_cases j; exact (hj rfl).elim
  · intro j hj; fin_cases j; exact (hj rfl).elim
  · intro j
    simpa only [hatt] using (tendsto_const_nhds (x := (1 : ℝ)) :
      Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1))

end Transformer.Clusters

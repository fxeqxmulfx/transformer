module

public import Transformer.Clusters.Section12_MultiHeadBoundedUpper
public import Transformer.Clusters.Section12_MultiHeadBoundedLower
public import Mathlib.Data.Set.Card

/-!
# At most one bounded token in the original discrete multi-head dynamics

ArXiv:2305.05465v6, §7, `l:onlyone`, extended to the original
positive heads of §12, `sec:conclusion`.
Beside a positively escaping token, every bounded original coordinate
has nonpositive limsup. Reflecting the original equation, the proved
frequently-small estimate excludes two bounded coordinates retaining
a positive gap. The original noncollision estimate supplies that gap.

If two bounded original tokens had distinct initial states, the actual
spreading and escape classification would provide an escaping token.
The preceding argument or its sign reversal gives a contradiction.
Consequently, bounded original tokens have identical initial states;
under the source's pairwise-distinct initial data there is at most one.
Every other original token tends to one of the two infinities. These
conclusions are proved from the actual original head matrices and
finite-step equation; no escape premise is added to the final results.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ} {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1}
variable {X : ℕ → Idx (m + 1) → EucSpace 1}
variable (hΔt : 0 < Δt) (hV : ∀ h, IsPosDefOp (V h))
variable (hQK : ∀ h, IsPosDefQK (Q h) (K h)) (hX : MultiHeadTransformer Δt Q K V X) (h : Idx H)

include hΔt hV hQK hX h

/-- Two bounded original tokens have identical initial states beside any positive escaping token.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, for §12's original positive-head equation. -/
theorem multiHeadTransformer_bounded_initial_eq_of_positive_escape (M : Idx (m + 1))
    (hM : Tendsto (fun k => X k M 0) atTop atTop) (i j : Idx (m + 1))
    (hi : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) (hj : ∃ R : ℝ, ∀ k, |X k j 0| ≤ R) : X 0 i = X 0 j := by
  obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  have htok : ∀ l, (∀ ε : ℝ, 0 < ε → ∀ᶠ k in atTop, -ε ≤ (-X k l) 0) ∨
      Tendsto (fun k => (-X k l) 0) atTop atBot := by
    intro l
    rcases halt l with hb | hp | hn
    · left
      intro ε hε
      have he := multiHeadTransformer_bounded_eventually_le hΔt hV hQK hX h M hM l
        (fun k => (abs_le.mp (hb k)).2) hε
      filter_upwards [he] with k hk
      change -ε ≤ -(X k l 0)
      linarith
    · exact Or.inr (tendsto_neg_atTop_atBot.comp hp)
    · left
      intro ε hε
      filter_upwards [tendsto_atBot.mp hn ε] with k hk
      change -ε ≤ -(X k l 0)
      linarith
  have hkey (a b : Idx (m + 1)) (ha : ∃ A : ℝ, ∀ k, |X k a 0| ≤ A)
      (hb : ∃ B : ℝ, ∀ k, |X k b 0| ≤ B) (hab : X 0 a 0 < X 0 b 0) : False := by
    obtain ⟨A, hA⟩ := ha
    obtain ⟨B, hB⟩ := hb
    let g := X 0 b 0 - X 0 a 0
    have hg : 0 < g := sub_pos.mpr hab
    have hup := multiHeadTransformer_bounded_eventually_le hΔt hV hQK hX h M hM b
      (fun k => (abs_le.mp (hB k)).2) (half_pos hg)
    have hbn : ∀ k, (-X k a) 0 ≤ A := by
      intro k
      change -(X k a 0) ≤ A
      linarith [(abs_le.mp (hA k)).1]
    have hf := multiHeadTransformer_bounded_frequently_lt hΔt hV hQK (multiHeadTransformer_neg hX)
      h a (half_pos hg) hbn htok
    obtain ⟨k, hk, hbk⟩ := (hf.and_eventually hup).exists
    change -(X k a 0) < g / 2 at hk
    have hgap := multiHeadTransformer_initial_gap hΔt.le hV hQK hX a b hab.le k
    change g ≤ X k b 0 - X k a 0 at hgap
    linarith
  by_contra hne
  have hc : X 0 i 0 ≠ X 0 j 0 := by
    intro he
    apply hne
    rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j), he]
  rcases lt_or_gt_of_ne hc with hlt | hgt
  · exact hkey i j hi hj hlt
  · exact hkey j i hj hi hgt

/-- Bounded original tokens necessarily have identical original initial states, without any escape premise.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, for the original discrete heads of §12. -/
theorem multiHeadTransformer_bounded_initial_eq (i j : Idx (m + 1))
    (hi : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) (hj : ∃ R : ℝ, ∀ k, |X k j 0| ≤ R) : X 0 i = X 0 j := by
  by_contra hne
  have hc : X 0 i 0 ≠ X 0 j 0 := by
    intro he
    apply hne
    rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j), he]
  have hesc : ∃ a, Tendsto (fun k => X k a 0) atTop atTop ∨ Tendsto (fun k => X k a 0) atTop atBot := by
    rcases lt_or_gt_of_ne hc with hlt | hgt
    · exact multiHeadTransformer_exists_escaping hΔt hV hQK hX i j hlt h
    · exact multiHeadTransformer_exists_escaping hΔt hV hQK hX j i hgt h
  rcases hesc with ⟨M, hp | hn⟩
  · exact hne (multiHeadTransformer_bounded_initial_eq_of_positive_escape hΔt hV hQK hX h M hp i j hi hj)
  · have hin : ∃ R : ℝ, ∀ k, |(-X k i) 0| ≤ R := by
      obtain ⟨R, hR⟩ := hi
      refine ⟨R, ?_⟩
      intro k
      change |-(X k i 0)| ≤ R
      rw [abs_neg]
      exact hR k
    have hjn : ∃ R : ℝ, ∀ k, |(-X k j) 0| ≤ R := by
      obtain ⟨R, hR⟩ := hj
      refine ⟨R, ?_⟩
      intro k
      change |-(X k j 0)| ≤ R
      rw [abs_neg]
      exact hR k
    have he := multiHeadTransformer_bounded_initial_eq_of_positive_escape hΔt hV hQK
      (multiHeadTransformer_neg hX) h M (tendsto_neg_atTop_iff.mpr hn) i j hin hjn
    apply hne
    simpa only [neg_neg] using congrArg (fun z : EucSpace 1 => -z) he

/-- At most one original token remains bounded under the original pairwise-distinct data hypothesis.
Source: arXiv:2305.05465v6, §7, `l:onlyone`, for the original positive-head iteration of §12. -/
theorem multiHeadTransformer_ncard_bounded_le_one
    (hdist : ∀ i j : Idx (m + 1), i ≠ j → X 0 i ≠ X 0 j) :
    {i : Idx (m + 1) | ∃ R : ℝ, ∀ k, |X k i 0| ≤ R}.ncard ≤ 1 := by
  rw [Set.ncard_le_one]
  intro i hi j hj
  by_contra hij
  exact hdist i j hij (multiHeadTransformer_bounded_initial_eq hΔt hV hQK hX h i j hi hj)

/-- All but one original token tend to an infinity, under exactly the original distinctness premise.
Source: arXiv:2305.05465v6, §7, `l:onlyone` and `l:unboundedparticles`, extended to §12. -/
theorem multiHeadTransformer_except_one_escape
    (hdist : ∀ i j : Idx (m + 1), i ≠ j → X 0 i ≠ X 0 j) :
    ∃ i₀ : Idx (m + 1), ∀ i, i ≠ i₀ →
      Tendsto (fun k => X k i 0) atTop atTop ∨ Tendsto (fun k => X k i 0) atTop atBot := by
  obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  by_cases hb : ∃ i : Idx (m + 1), ∃ A : ℝ, ∀ k, |X k i 0| ≤ A
  · obtain ⟨i₀, hi₀⟩ := hb
    refine ⟨i₀, ?_⟩
    intro i hne
    rcases halt i with hi | hp | hn
    · exact (hdist i i₀ hne (multiHeadTransformer_bounded_initial_eq hΔt hV hQK hX h i i₀ ⟨R, hi⟩ hi₀)).elim
    · exact Or.inl hp
    · exact Or.inr hn
  · refine ⟨0, ?_⟩
    intro i hne
    rcases halt i with hi | hp | hn
    · exact (hb ⟨i, R, hi⟩).elim
    · exact Or.inl hp
    · exact Or.inr hn

/-- Two genuine positive original heads, distinct original data, and an actually
bounded interior token jointly realize all the original bounded-token premises. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ (∀ k, |X k 1 0| ≤ 0) ∧
    Tendsto (fun k => X k 2 0) atTop atTop ∧
    (∀ i j : Idx 3, i ≠ j → X 0 i ≠ X 0 j) := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (Q h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (Q h) (Q h) :=
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, _, hord⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  refine ⟨Q, Q, Q, X, one_pos, hV, hQK, hX, ?_, hp, ?_⟩
  · intro k; rw [hz k]; norm_num
  · intro i j hij he
    have hc := congrArg (fun z : EucSpace 1 => z 0) he
    rcases lt_or_gt_of_ne hij with hlt | hgt
    · have hs := hord i j hlt; linarith
    · have hs := hord j i hgt; linarith

end Transformer.Clusters

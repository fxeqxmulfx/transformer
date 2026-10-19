module

public import Transformer.Clusters.Section12_MultiHeadSymmetry

/-!
# Bounded original tokens beside escaping tokens

ArXiv:2305.05465v6, §7, proof of `l:onlyone`, for the original
multi-head discrete equation of §12, `sec:conclusion`.
A token bounded above must eventually lie below every positive epsilon
if any original token tends to positive infinity. In discrete time the
proved reference-token pull gives this directly: a positive query at
least epsilon would make the very next coordinate exceed its bound.
The proof retains the actual sum of the original head coefficients.

Sign reversal gives the corresponding lower statement beside a token
tending to negative infinity. A bounded original token therefore tends
to zero when both directions of escape occur. Since original ordered
gaps cannot shrink, two bounded tokens must then have identical original
initial states. No source distinctness assumption is added to these
auxiliary statements. The joint witness is an actual constructed
three-token trajectory with its stationary interior token and two
escaping outer tokens, for two genuine positive original heads.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}
variable {Δt : ℝ}
variable {Q K V : Idx H → ParamMatrix 1}
variable {X : ℕ → Idx (m + 1) → EucSpace 1}
variable (hΔt : 0 < Δt)
variable (hV : ∀ h, IsPosDefOp (V h))
variable (hQK : ∀ h, IsPosDefQK (Q h) (K h))
variable (hX : MultiHeadTransformer Δt Q K V X)
variable (h : Idx H)

include hΔt hV hQK hX h

/-- An original token bounded above eventually lies below every positive epsilon beside positive escape.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, from `e:lowerboundxi`, for §12's discrete sum. -/
theorem multiHeadTransformer_bounded_eventually_le (M : Idx (m + 1))
    (hM : Tendsto (fun k => X k M 0) atTop atTop)
    (i : Idx (m + 1))
    {R : ℝ}
    (hR : ∀ k, X k i 0 ≤ R)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ k in atTop, X k i 0 ≤ ε := by
  let Λ := multiHeadValueCoefficient V
  let B := multiHeadPullCoefficient (m + 1) Q K V
  obtain ⟨hΛ, hB⟩ := multiHead_coefficients_pos (m + 1) Q K V hV hQK h
  let A := (R + Δt * (B / ε) + 1) / (Δt * Λ)
  filter_upwards [tendsto_atTop.mp hM A] with k hk
  by_contra hle
  have hi : ε < X k i 0 := lt_of_not_ge hle
  have hip : 0 < X k i 0 := hε.trans hi
  have hrec := multiHeadTransformer_reference_pull hΔt.le hV hQK hX k i M hip
  change X k i 0 + Δt * (Λ * X k M 0 - B / X k i 0) ≤ X (k + 1) i 0 at hrec
  have hdiv : B / X k i 0 ≤ B / ε := div_le_div_of_nonneg_left hB hε hi.le
  have hmul := mul_le_mul_of_nonneg_left hdiv hΔt.le
  have hlarge : R + Δt * (B / ε) + 1 ≤ X k M 0 * (Δt * Λ) :=
    (div_le_iff₀ (mul_pos hΔt hΛ)).mp hk
  have hbound := hR (k + 1)
  nlinarith

/-- An original token bounded below eventually lies above every negative epsilon beside negative escape.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, the symmetric estimate, for §12. -/
theorem multiHeadTransformer_bounded_eventually_ge (L : Idx (m + 1))
    (hL : Tendsto (fun k => X k L 0) atTop atBot)
    (i : Idx (m + 1))
    {R : ℝ}
    (hR : ∀ k, -R ≤ X k i 0)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ k in atTop, -ε ≤ X k i 0 := by
  have hn : Tendsto (fun k => (-X k L) 0) atTop atTop := tendsto_neg_atTop_iff.mpr hL
  have hRn : ∀ k, (-X k i) 0 ≤ R := by
    intro k
    change -(X k i 0) ≤ R
    linarith [hR k]
  have he := multiHeadTransformer_bounded_eventually_le hΔt hV hQK
    (multiHeadTransformer_neg hX) h L hn i hRn hε
  filter_upwards [he] with k hk
  change -(X k i 0) ≤ ε at hk
  linarith

/-- A bounded original token tends to zero when actual tokens escape in both directions.
Source: arXiv:2305.05465v6, §7, proof of `l:onlyone`, for §12's original positive-head iteration. -/
theorem multiHeadTransformer_bounded_tendsto_zero (L M : Idx (m + 1))
    (hL : Tendsto (fun k => X k L 0) atTop atBot)
    (hM : Tendsto (fun k => X k M 0) atTop atTop)
    (i : Idx (m + 1))
    {R : ℝ} (hR : ∀ k, |X k i 0| ≤ R) :
    Tendsto (fun k => X k i 0) atTop (nhds 0) := by
  apply tendsto_order.mpr
  constructor
  · intro a ha
    have he := multiHeadTransformer_bounded_eventually_ge hΔt hV hQK hX h L hL i
      (fun k => (abs_le.mp (hR k)).1) (show 0 < -a / 2 by linarith)
    filter_upwards [he] with k hk
    linarith
  · intro b hb
    have he := multiHeadTransformer_bounded_eventually_le hΔt hV hQK hX h M hM i
      (fun k => (abs_le.mp (hR k)).2) (show 0 < b / 2 by linarith)
    filter_upwards [he] with k hk
    linarith

/-- Two bounded tokens have identical original initial states when escape occurs in both directions.
Source: arXiv:2305.05465v6, §7, `e:infdist` and proof of `l:onlyone`, for the original heads of §12. -/
theorem multiHeadTransformer_bounded_initial_eq_of_two_escape (L M : Idx (m + 1))
    (hL : Tendsto (fun k => X k L 0) atTop atBot)
    (hM : Tendsto (fun k => X k M 0) atTop atTop)
    (i j : Idx (m + 1))
    (hi : ∃ R : ℝ, ∀ k, |X k i 0| ≤ R)
    (hj : ∃ R : ℝ, ∀ k, |X k j 0| ≤ R) : X 0 i = X 0 j := by
  obtain ⟨Ri, hRi⟩ := hi
  obtain ⟨Rj, hRj⟩ := hj
  have hti := multiHeadTransformer_bounded_tendsto_zero hΔt hV hQK hX h L M hL hM i hRi
  have htj := multiHeadTransformer_bounded_tendsto_zero hΔt hV hQK hX h L M hL hM j hRj
  have hordered (a b : Idx (m + 1)) (ha : Tendsto (fun k => X k a 0) atTop (nhds 0))
      (hb : Tendsto (fun k => X k b 0) atTop (nhds 0)) (hab : X 0 a 0 ≤ X 0 b 0) :
      X 0 b 0 - X 0 a 0 ≤ 0 := by
    have ht : Tendsto (fun k => X k b 0 - X k a 0) atTop (nhds 0) := by
      simpa only [sub_self] using hb.sub ha
    exact ge_of_tendsto ht (Eventually.of_forall
      (multiHeadTransformer_initial_gap hΔt.le hV hQK hX a b hab))
  have he : X 0 i 0 = X 0 j 0 := by
    rcases le_total (X 0 i 0) (X 0 j 0) with hij | hji
    · have hs := hordered i j hti htj hij
      linarith
    · have hs := hordered j i htj hti hji
      linarith
  rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j), he]

/-- Genuine positive original heads jointly realize boundedness, both directions
of actual escape, and the original vector update with an interior token fixed at zero. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ Tendsto (fun k => X k 2 0) atTop atTop ∧
    Tendsto (fun k => X k 0 0) atTop atBot ∧ (∀ k, |X k 1 0| ≤ 0) := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ h, IsPosDefOp (Q h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (Q h) (Q h) :=
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, hn, _⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  refine ⟨Q, Q, Q, X, one_pos, hV, hQK, hX, hp, hn, ?_⟩
  intro k
  rw [hz k]
  norm_num

end Transformer.Clusters

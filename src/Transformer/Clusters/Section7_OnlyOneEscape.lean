/-
# The emergence of clusters in self-attention dynamics — bounded tokens beside an escaping one

Two steps of the proof of `l:onlyone` (§7 of arXiv:2305.05465v6), on the
scalar curves of `e:Idnonresca`, once the largest token `x_M` runs off to
`+∞`:

* `eventually_le_of_bounded` — a token bounded from above ends up below any
  `ε > 0`: at a time when it sits above `ε`, `e:lowerboundxi` pushes it up at a
  speed that grows with `x_M`, and a barrier carries it past its bound.  This is
  the source's "if `x_i(t) > ε` for an unbounded sequence of times then
  `x_i(t) → +∞`".
* `frequently_lt_of_bounded` — a bounded token that stays below `-δ` is
  impossible when the other tokens either approach `(-∞, 0]` or run off to
  `+∞`: the smallest token pulls it down at speed at least `δ/(2n)`
  (`e:minorationpourxn`, mirrored).

Source: arXiv:2305.05465v6, proof of `l:onlyone`.
-/

import Transformer.Clusters.Section7_OnlyOneDrift

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **A bounded token ends up below every `ε > 0`** while the largest token
runs off to `+∞`.

Source: arXiv:2305.05465v6, proof of `l:onlyone`, from `e:lowerboundxi`. -/
theorem eventually_le_of_bounded (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (M : Idx (m + 1)) (hmax : ∀ t, 0 ≤ t → ∀ j, x t j ≤ x t M)
    (hM : Tendsto (fun t => x t M) atTop atTop) (k : Idx (m + 1)) {R : ℝ}
    (hR : ∀ t, 0 ≤ t → x t k ≤ R) {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ t in atTop, x t k ≤ ε := by
  by_contra h
  obtain ⟨T, hT⟩ := eventually_atTop.1
    (hM.eventually (eventually_ge_atTop (((m : ℝ) + 1) * (((m : ℝ) + 1) / ε + 1))))
  obtain ⟨t₁, ht₁, hx₁⟩ := frequently_atTop.1 (not_eventually.1 h) (max T 0)
  push Not at hx₁
  set L : ℝ → ℝ := fun t => ε + (1 / 2) * (t - t₁)
  have hL : ∀ y, HasDerivAt L (1 / 2) y := fun y => by
    show HasDerivAt (fun t => ε + (1 / 2) * (t - t₁)) (1 / 2) y
    simpa using (((hasDerivAt_id y).sub_const t₁).const_mul (1 / 2 : ℝ)).const_add ε
  have hcont : Continuous fun s => x s k :=
    continuous_iff_continuousAt.2 fun t => (hder t k).continuousAt
  have hlin : ∀ t, t₁ ≤ t → L t ≤ x t k := by
    intro t ht
    refine image_le_of_deriv_right_lt_deriv_boundary' (f' := fun _ => 1 / 2)
      (B' := fun y => ∑ j, Perspective.softmaxWeight (fun l => x y k * x y l) j * x y j)
      (by fun_prop) (fun y _ => (hL y).hasDerivWithinAt) (by simp [L, hx₁.le])
      hcont.continuousOn (fun y _ => (hder y k).hasDerivWithinAt) ?_ ⟨ht, le_rfl⟩
    rintro y ⟨hy, -⟩ hLa
    have hy₁ : max T 0 ≤ y := ht₁.trans hy
    have hεy : ε ≤ x y k := by rw [← hLa]; simp only [L]; nlinarith
    have hpos : 0 < x y k := hε.trans_le hεy
    have hd := div_sub_le_drift_of_max (x y) k M hpos (hmax y (le_of_max_le_right hy₁))
    have h1 : ((m : ℝ) + 1) / ε + 1 ≤ x y M / ((m : ℝ) + 1) := by
      rw [le_div_iff₀ (by positivity)]
      linarith [hT y (le_of_max_le_left hy₁)]
    have h2 : ((m : ℝ) + 1) / x y k ≤ ((m : ℝ) + 1) / ε :=
      div_le_div_of_nonneg_left (by positivity) hε hεy
    linarith
  have h := hlin (t₁ + 2 * (|R| + 1)) (by linarith [abs_nonneg R])
  have h' := hR (t₁ + 2 * (|R| + 1)) (by linarith [abs_nonneg R, le_of_max_le_right ht₁])
  simp only [L] at h
  linarith [le_abs_self R]

/-- `e^{-δz} z ≤ 2/(δ²z)` for `δ, z > 0`, from `e^u ≥ u²/2`. -/
theorem exp_neg_mul_mul_le {δ z : ℝ} (hδ : 0 < δ) (hz : 0 < z) :
    Real.exp (-(δ * z)) * z ≤ 2 / (δ ^ 2 * z) := by
  have h := Real.quadratic_le_exp_of_nonneg (mul_pos hδ hz).le
  rw [Real.exp_neg, inv_mul_eq_div, div_le_div_iff₀ (Real.exp_pos _) (by positivity)]
  nlinarith [mul_pos hδ hz]

/-- **A bounded token does not stay below `-δ`** when every token either ends
up above `-η` for every `η > 0`, or runs off to `-∞` — written for `y = -x`,
where `y_L` is the largest coordinate: `y_k < δ` at arbitrarily large times.

Source: arXiv:2305.05465v6, proof of `l:onlyone`, from `e:minorationpourxn`. -/
theorem frequently_lt_of_bounded (y : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => y s k)
      (∑ j, Perspective.softmaxWeight (fun l => y t k * y t l) j * y t j) t)
    (L : Idx (m + 1)) (hmax : ∀ t, 0 ≤ t → ∀ j, y t j ≤ y t L) (k : Idx (m + 1))
    {δ R : ℝ} (hδ : 0 < δ) (hR : ∀ t, 0 ≤ t → y t k ≤ R)
    (htok : ∀ j, (∀ η, 0 < η → ∀ᶠ t in atTop, -η ≤ y t j) ∨
      Tendsto (fun t => y t j) atTop atBot) : ∃ᶠ t in atTop, y t k < δ := by
  by_contra hk
  simp only [not_frequently, not_lt] at hk
  set η := δ / (2 * ((m : ℝ) + 1) ^ 2)
  have hη : 0 < η := by positivity
  have hterm : ∀ j, ∀ᶠ t in atTop, -η ≤ min 0 (Real.exp (y t k * y t j) * y t j) := by
    intro j
    rcases htok j with hj | hj
    · filter_upwards [hj η hη, hk] with t ht htk
      refine le_min (neg_nonpos.2 hη.le) ?_
      rcases le_or_gt 0 (y t j) with h | h
      · exact (neg_nonpos.2 hη.le).trans (mul_nonneg (Real.exp_pos _).le h)
      · have he : Real.exp (y t k * y t j) ≤ 1 :=
          Real.exp_le_one_iff.2 (mul_nonpos_of_nonneg_of_nonpos (hδ.le.trans htk) h.le)
        nlinarith
    · filter_upwards [hj.eventually (eventually_le_atBot (-(2 / (δ ^ 2 * η)))), hk]
        with t ht htk
      refine le_min (neg_nonpos.2 hη.le) ?_
      have hz : 0 < -y t j := by
        have : 0 < 2 / (δ ^ 2 * η) := by positivity
        linarith
      have h1 := exp_neg_mul_mul_le hδ hz
      have h2 : 2 / (δ ^ 2 * -y t j) ≤ η := by
        rw [div_le_iff₀ (by positivity)]
        have : 2 / (δ ^ 2 * η) * (δ ^ 2 * η) = 2 := div_mul_cancel₀ _ (by positivity)
        nlinarith [mul_le_mul_of_nonneg_left (show 2 / (δ ^ 2 * η) ≤ -y t j by linarith)
          (by positivity : (0 : ℝ) ≤ δ ^ 2 * η)]
      have h3 : Real.exp (y t k * y t j) ≤ Real.exp (-(δ * -y t j)) :=
        Real.exp_le_exp.2 (by nlinarith)
      nlinarith [mul_le_mul_of_nonneg_right h3 hz.le]
  set κ := δ / (2 * ((m : ℝ) + 1))
  have hκ : 0 < κ := by positivity
  obtain ⟨T, hT⟩ := eventually_atTop.1 ((eventually_all.2 hterm).and hk)
  set T' := max T 0
  have hdrift : ∀ t, T' ≤ t →
      κ ≤ ∑ j, Perspective.softmaxWeight (fun l => y t k * y t l) j * y t j := by
    intro t ht
    obtain ⟨h1, h2⟩ := hT t (le_of_max_le_left ht)
    have hmx := hmax t (le_of_max_le_right ht)
    have hd := add_sum_min_le_drift (y t) k L (hδ.trans_le h2) hmx
    have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => h1 j
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    push_cast at hsum
    have hL : δ / ((m : ℝ) + 1) ≤ y t L / ((m : ℝ) + 1) :=
      div_le_div_of_nonneg_right (h2.trans (hmx k)) (by positivity)
    have he : ((m : ℝ) + 1) * -η = -(δ / ((m : ℝ) + 1)) + κ := by
      simp only [η, κ]; field_simp; ring
    linarith
  have hanti : AntitoneOn (fun t => κ * t - y t k) (Set.Ici T') :=
    antitoneOn_Ici_of_hasDerivAt
      (fun t _ => ((hasDerivAt_id t).const_mul κ).sub (hder t k))
      fun t ht => by simp only [mul_one]; linarith [hdrift t ht]
  set t₂ := T' + (|R| + 1) / κ
  have ht₂ : T' ≤ t₂ := le_add_of_nonneg_right (by positivity)
  have h1 := hanti (Set.mem_Ici.2 le_rfl) (Set.mem_Ici.2 ht₂) ht₂
  have h2 : δ ≤ y T' k := (hT T' (le_max_left _ _)).2
  have h3 := hR t₂ ((le_max_right T 0).trans ht₂)
  have h4 : κ * t₂ = κ * T' + (|R| + 1) := by simp only [t₂]; field_simp
  simp only at h1
  linarith [le_abs_self R]

end Clusters
end Transformer

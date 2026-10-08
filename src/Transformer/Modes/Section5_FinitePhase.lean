import Transformer.Modes.Section5_PhaseBounds

/-!
# Combining local phase estimates across finitely many derivative zeros

In arXiv:2412.09080v3, §5.5, the oscillatory integral is split into
stationary and nonstationary intervals. The local estimate in
`Section5_UnitPhase` instead requires one sign of the second derivative.
This module partitions at a finite cover of its zeros in the interior.

A continuous function without an interior zero has one sign on the
entire closed interval; zeros at the endpoints are allowed. For a finite
cover `Z`, induction cuts at a member inside the interval and applies the
local estimate to both sides. Each cut doubles the bound, giving the
explicit factor `2 ^ Z.card`. Members outside the interval are harmless.

The resulting estimate still uses the actual first two derivatives,
linked by `HasDerivAt`, and their quantitative lower bound. The finite
cover of second-derivative zeros is an explicit hypothesis. For Gaussian
phases it will come from the roots of the quartic polynomial multiplying
the nonzero Gaussian factor; no root locations or separations are needed.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- A continuous second derivative has one sign if it has no interior
zero, allowing zeros at either endpoint.
Source: arXiv:2412.09080v3, §5.5, the intervals between derivative zeros. -/
theorem phase_sign_of_no_interior_zero {a b : ℝ} {q : ℝ → ℝ} (hab : a ≤ b)
    (hcq : ContinuousOn q (Set.uIcc a b)) (hne : ∀ t ∈ Set.Ioo a b, q t ≠ 0) :
    (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0) := by
  classical
  by_cases hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t
  · exact Or.inl hq
  push Not at hq
  obtain ⟨x, hx, hqx⟩ := hq
  right
  intro t ht
  by_contra hqt
  have hqt : 0 < q t := not_le.mp hqt
  rw [Set.uIcc_of_le hab] at hx ht
  rcases le_total x t with hxt | htx
  · have hs : Set.Icc x t ⊆ Set.uIcc a b := by
      intro r hr
      rw [Set.uIcc_of_le hab]
      exact ⟨hx.1.trans hr.1, hr.2.trans ht.2⟩
    obtain ⟨c, hc, he⟩ := intermediate_value_Icc hxt (hcq.mono hs) ⟨hqx.le, hqt.le⟩
    have hcx : x < c := lt_of_le_of_ne hc.1 (by
      intro h
      rw [← h] at he
      exact hqx.ne he)
    have hct : c < t := lt_of_le_of_ne hc.2 (by
      intro h
      rw [h] at he
      exact hqt.ne' he)
    exact hne c ⟨hx.1.trans_lt hcx, hct.trans_le ht.2⟩ he
  · have hs : Set.Icc t x ⊆ Set.uIcc a b := by
      intro r hr
      rw [Set.uIcc_of_le hab]
      exact ⟨ht.1.trans hr.1, hr.2.trans hx.2⟩
    obtain ⟨c, hc, he⟩ := intermediate_value_Icc' htx (hcq.mono hs) ⟨hqx.le, hqt.le⟩
    have hct : t < c := lt_of_le_of_ne hc.1 (by
      intro h
      rw [← h] at he
      exact hqt.ne' he)
    have hcx : c < x := lt_of_le_of_ne hc.2 (by
      intro h
      rw [h] at he
      exact hqx.ne he)
    exact hne c ⟨ht.1.trans_lt hct, hcx.trans_le hx.2⟩ he


example : (0 : ℝ) ≤ 1 ∧ ContinuousOn (fun t : ℝ => t) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) 1, t ≠ 0) := by
  exact ⟨zero_le_one, continuous_id.continuousOn, fun _ ht => ht.1.ne'⟩

/-- A finite cover of the second-derivative zeros gives a uniform local
estimate without a sign assumption on the whole interval.
Source: arXiv:2412.09080v3, §5.5, combining stationary and nonstationary pieces. -/
theorem unit_interval_phase_test_finite_zeros {a b ρ κ : ℝ} {φ p q : ℝ → ℝ} (Z : Finset ℝ)
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hmin : ∀ t ∈ Set.uIcc a b, κ ≤ max |p t| |q t|)
    (hcover : ∀ t ∈ Set.Ioo a b, q t = 0 → t ∈ Z) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) := by
  classical
  have hC : 0 ≤ 10 / Real.sqrt (|ρ| * κ) := by positivity
  induction Z using Finset.induction_on generalizing a b with
  | empty =>
    have hsign := phase_sign_of_no_interior_zero hab hcq (fun t ht he => by
      have h := hcover t ht he
      exact Finset.notMem_empty t h)
    simpa only [Finset.card_empty, pow_zero, one_mul] using
      unit_interval_phase_test hab hlen hρ hκ hφ hp hcq hsign hmin
  | @insert z Z hz ih =>
    have hsub (x y : ℝ) (hax : a ≤ x) (hxy : x ≤ y) (hyb : y ≤ b) :
        Set.uIcc x y ⊆ Set.uIcc a b := by
      intro t ht
      rw [Set.uIcc_of_le hxy] at ht
      rw [Set.uIcc_of_le hab]
      exact ⟨hax.trans ht.1, ht.2.trans hyb⟩
    by_cases hzab : z ∈ Set.Ioo a b
    · have hsl := hsub a z le_rfl hzab.1.le hzab.2.le
      have hsr := hsub z b hzab.1.le hzab.2.le le_rfl
      have hl := ih hzab.1.le (by linarith [hzab.2]) (fun t ht => hφ t (hsl ht))
        (fun t ht => hp t (hsl ht)) (hcq.mono hsl) (fun t ht => hmin t (hsl ht))
        (fun t ht he => by
          have h := Finset.mem_insert.mp (hcover t ⟨ht.1, ht.2.trans hzab.2⟩ he)
          exact h.resolve_left ht.2.ne)
      have hr := ih hzab.2.le (by linarith [hzab.1]) (fun t ht => hφ t (hsr ht))
        (fun t ht => hp t (hsr ht)) (hcq.mono hsr) (fun t ht => hmin t (hsr ht))
        (fun t ht he => by
          have h := Finset.mem_insert.mp (hcover t ⟨hzab.1.trans ht.1, ht.2⟩ he)
          exact h.resolve_left ht.1.ne')
      have hcφ : ContinuousOn φ (Set.uIcc a b) :=
        continuousOn_of_forall_continuousAt fun t ht => (hφ t ht).continuousAt
      have hK : ContinuousOn (oscillatoryKernel ρ φ) (Set.uIcc a b) := by
        unfold oscillatoryKernel
        fun_prop
      rw [← intervalIntegral.integral_add_adjacent_intervals
        ((hK.mono hsl).intervalIntegrable) ((hK.mono hsr).intervalIntegrable)]
      calc _ ≤ (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) +
            (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) := norm_add_le_of_le hl hr
        _ = _ := by rw [Finset.card_insert_of_notMem hz, pow_succ]; ring
    · have h := ih hab hlen hφ hp hcq hmin (fun t ht he => by
        have h := Finset.mem_insert.mp (hcover t ht he)
        exact h.resolve_left (fun heq => hzab (heq ▸ ht)))
      calc _ ≤ (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) := h
        _ ≤ (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) +
              (2 : ℝ) ^ Z.card * (10 / Real.sqrt (|ρ| * κ)) :=
          le_add_of_nonneg_right (mul_nonneg (by positivity) hC)
        _ = _ := by rw [Finset.card_insert_of_notMem hz, pow_succ]; ring

example : (-1 / 2 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) - (-1 / 2) ≤ 1 ∧
    (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2),
      HasDerivAt (fun s : ℝ => s + s ^ 3) (1 + 3 * t ^ 2) t) ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2),
      HasDerivAt (fun s : ℝ => 1 + 3 * s ^ 2) (6 * t) t) ∧
    ContinuousOn (fun t : ℝ => 6 * t) (Set.uIcc (-1 / 2 : ℝ) (1 / 2)) ∧
    (∀ t ∈ Set.uIcc (-1 / 2 : ℝ) (1 / 2), (1 : ℝ) ≤ max |1 + 3 * t ^ 2| |6 * t|) ∧
    (∀ t ∈ Set.Ioo (-1 / 2 : ℝ) (1 / 2), 6 * t = 0 → t ∈ ({0} : Finset ℝ)) := by
  refine ⟨by norm_num, by norm_num, one_ne_zero, one_pos, ?_, ?_, by fun_prop, ?_, ?_⟩
  · intro t _
    simpa using (hasDerivAt_id t).fun_add ((hasDerivAt_id t).fun_pow 3)
  · intro t _
    have h := (hasDerivAt_const t (1 : ℝ)).fun_add (((hasDerivAt_id t).fun_pow 2).const_mul 3)
    exact h.congr_deriv (by simp only [id_eq]; ring)
  · intro t _
    exact (by nlinarith [sq_nonneg t] : (1 : ℝ) ≤ 1 + 3 * t ^ 2).trans
      ((le_abs_self _).trans (le_max_left _ _))
  · intro t _ he
    rw [Finset.mem_singleton]
    linarith

end Transformer.Modes

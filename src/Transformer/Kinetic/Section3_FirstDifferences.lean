/-
# First differences of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 1, `eq:est-Gl`: moving
the initial angle of token `l` by `h` moves token `a` at time `t` by at most

  `|h| e^{L₁ t} Ĝ^l_a(t)`,

where `L₁` is a Lipschitz constant of `w_β'`, `Ĝ` is the profile of
`Section3_Supersolution` built with any `α ≥ 2 e^{2|λ|} L₁` and `γ ∈ (0, 1]`.
Tokens before `l` do not move, token `l` moves by at most `|h| e^{L₁ t}`, and a
later token `a` by at most `|h| e^{L₁t} αt e^{αt/γ} ((a+1)/(l+1))^γ / (a+1)`.

The source bounds the Glauber derivative `D_l θ_a`, which is an average of such
differences; the bound is the same.
-/

import Transformer.Kinetic.Section3_Flow
import Transformer.Kinetic.Section3_Fence
import Transformer.Kinetic.Section3_Supersolution

open Real

namespace Transformer
namespace Kinetic

/-- A sum over the predecessors of a token is a sum over `range`. -/
theorem sum_filter_lt_eq_sum_range {N : ℕ} (a : Idx N) (f : ℕ → ℝ) :
    ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), f b = ∑ c ∈ Finset.range a, f c := by
  rw [Finset.filter_gt_eq_Iio, ← Nat.Iio_eq_range, ← Fin.map_valEmbedding_Iio,
    Finset.sum_map]
  rfl

/-- The weighted predecessor sum of a nonnegative sequence is at most
`2e^{2|λ|}/(a+1)` times its plain sum over `range a`. -/
theorem sum_alibiWeight_mul_le (lam : ℝ) {N : ℕ} (a : Idx N) {f : ℕ → ℝ}
    (hf : ∀ c, 0 ≤ f c) :
    ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), alibiWeight lam N a b * f b ≤
      2 * Real.exp (2 * |lam|) / ((a : ℝ) + 1) * ∑ c ∈ Finset.range a, f c := by
  rw [← sum_filter_lt_eq_sum_range, Finset.mul_sum]
  exact Finset.sum_le_sum fun b hb => mul_le_mul_of_nonneg_right
    (alibiWeight_le_succ lam N (Finset.mem_filter.1 hb).2) (hf b)

/-- The difference of two flows obeys the triangular inequality of Step 1. -/
theorem abs_deriv_flow_sub_le (lam β : ℝ) (N : ℕ) {L₁ : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (θ θ' : Idx N → ℝ) (a : Idx N) :
    |gptField lam β N θ' a - gptField lam β N θ a| ≤ L₁ * |θ' a - θ a| +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * |θ' b - θ b| := by
  refine (abs_gptField_sub_le lam β N hL θ θ' a).trans ?_
  have hw := sum_alibiWeight_le_one lam N a
  have h1 : ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
      alibiWeight lam N a b * (L₁ * |θ' a - θ a|) ≤ L₁ * |θ' a - θ a| := by
    rw [← Finset.sum_mul]
    exact (mul_le_mul_of_nonneg_right hw (by positivity)).trans_eq (one_mul _)
  rw [show (∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
      alibiWeight lam N a b * (L₁ * (|θ' a - θ a| + |θ' b - θ b|))) =
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        alibiWeight lam N a b * (L₁ * |θ' a - θ a|) +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * |θ' b - θ b| by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun b _ => by ring]
  linarith

/-- **First differences (`eq:est-Gl`).** -/
theorem abs_flowMap_add_single_sub_le (lam β : ℝ) (N : ℕ) {L₁ α γ T : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (x : Idx N → ℝ) (l : Idx N) (h : ℝ) (a : Idx N) {t : ℝ} (ht : t ∈ Set.Icc 0 T) :
    |flowMap lam β N t (x + Pi.single l h) a - flowMap lam β N t x a| ≤
      |h| * Real.exp (L₁ * t) * profile α γ l a t := by
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  set y : Idx N → ℝ → ℝ := fun a t =>
    flowMap lam β N t (x + Pi.single l h) a - flowMap lam β N t x a with hy
  set W : Idx N → ℝ → ℝ := fun a t => |h| * (Real.exp (L₁ * t) * profile α γ l a t) with hWdef
  have hdy : ∀ a t, HasDerivAt (y a) (gptField lam β N (flowMap lam β N t (x + Pi.single l h)) a -
      gptField lam β N (flowMap lam β N t x) a) t := fun a t =>
    (hasDerivAt_flowMap lam β N _ t a).sub (hasDerivAt_flowMap lam β N x t a)
  have hdW : ∀ a t, HasDerivAt (W a) (|h| * (Real.exp (L₁ * t) * L₁ * profile α γ l a t +
      Real.exp (L₁ * t) * profileDeriv α γ l a t)) t := fun a t => by
    have he : HasDerivAt (fun t => Real.exp (L₁ * t)) (Real.exp (L₁ * t) * L₁) t := by
      simpa using ((hasDerivAt_id t).const_mul L₁).exp
    exact (he.mul (hasDerivAt_profile α hγ0.ne' l a t)).const_mul |h|
  have h0 : ∀ a, |y a 0| ≤ W a 0 := fun a => by
    simp only [hy, hWdef, flowMap_zero, Pi.add_apply, profile_zero, mul_zero, Real.exp_zero,
      one_mul, add_sub_cancel_left]
    by_cases hal : a = l
    · subst hal
      simp
    · simp [hal, Fin.val_eq_val]
  have hbound : ∀ a, ∀ t ∈ Set.Ico 0 T,
      |gptField lam β N (flowMap lam β N t (x + Pi.single l h)) a -
        gptField lam β N (flowMap lam β N t x) a| ≤ L₁ * |y a t| +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * |y b t| + 0 := fun a t _ => by
    rw [add_zero]
    exact abs_deriv_flow_sub_le lam β N hL hL₁ _ _ a
  have hsuper : ∀ a, ∀ t ∈ Set.Ico 0 T, L₁ * W a t +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), L₁ * alibiWeight lam N a b * W b t + 0 ≤
      |h| * (Real.exp (L₁ * t) * L₁ * profile α γ l a t +
        Real.exp (L₁ * t) * profileDeriv α γ l a t) := fun a t ht => by
    have hp : ∀ c : ℕ, 0 ≤ profile α γ l c t := fun c => profile_nonneg hα0 hγ0 l c ht.1
    have h1 := sum_alibiWeight_mul_le lam a hp
    have h2 := sum_profile_le hα0 hγ0 hγ1 l a ht.1
    have hc : 0 ≤ ∑ c ∈ Finset.range a, profile α γ l c t := Finset.sum_nonneg fun c _ => hp c
    have h3 : L₁ * (2 * Real.exp (2 * |lam|) / ((a : ℝ) + 1)) ≤ α / ((a : ℝ) + 1) := by
      rw [mul_div_assoc', div_le_div_iff_of_pos_right (by positivity)]
      linarith
    have hsum : ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * W b t ≤
        |h| * (Real.exp (L₁ * t) * profileDeriv α γ l a t) := by
      calc ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), L₁ * alibiWeight lam N a b * W b t
          = |h| * Real.exp (L₁ * t) * (L₁ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
              alibiWeight lam N a b * profile α γ l b t) := by
            rw [Finset.mul_sum, Finset.mul_sum]
            exact Finset.sum_congr rfl fun b _ => by simp only [hWdef]; ring
        _ ≤ |h| * Real.exp (L₁ * t) * (L₁ * (2 * Real.exp (2 * |lam|) / ((a : ℝ) + 1) *
              ∑ c ∈ Finset.range a, profile α γ l c t)) := by gcongr
        _ ≤ |h| * Real.exp (L₁ * t) * (α / ((a : ℝ) + 1) *
              ∑ c ∈ Finset.range a, profile α γ l c t) := by
            rw [← mul_assoc L₁]
            gcongr
        _ ≤ |h| * Real.exp (L₁ * t) * profileDeriv α γ l a t := by gcongr
        _ = |h| * (Real.exp (L₁ * t) * profileDeriv α γ l a t) := by ring
    have hWa : L₁ * W a t = |h| * (Real.exp (L₁ * t) * L₁ * profile α γ l a t) := by
      simp only [hWdef]
      ring
    linarith
  have := abs_le_of_triangular (fun a b => mul_nonneg hL₁ (alibiWeight_nonneg lam N a b))
    hdy hdW h0 hbound hsuper a t ht
  simp only [hy, hWdef] at this
  linarith

/-- Every shift is a shift in `[-π, π]` plus whole turns. -/
theorem exists_shift_le_pi (d : ℝ) : ∃ h : ℝ, |h| ≤ π ∧ ∃ n : ℤ, d = h + n * (2 * π) := by
  have hm := toIcoMod_mem_Ico Real.two_pi_pos (-π) d
  refine ⟨toIcoMod Real.two_pi_pos (-π) d, abs_le.2 ⟨hm.1, by linarith [hm.2]⟩,
    toIcoDiv Real.two_pi_pos (-π) d, ?_⟩
  rw [← self_sub_toIcoDiv_zsmul, zsmul_eq_mul]
  ring

/-- `update y l z` is `y` moved at `l` by `z - y l`. -/
theorem update_eq_add_single {N : ℕ} (y : Idx N → ℝ) (l : Idx N) (z : ℝ) :
    Function.update y l z = y + Pi.single l (z - y l) := by
  funext m
  by_cases hm : m = l
  · subst hm
    simp
  · simp [hm]

/-- **Periodic reduction.** A `2π`-periodic test of the flow does not see whole
turns of an initial angle. -/
theorem periodic_flowMap_add_single (lam β : ℝ) (N : ℕ) {φ : ℝ → ℝ}
    (hφ : Function.Periodic φ (2 * π)) (y : Idx N → ℝ) (l : Idx N) (h : ℝ) (n : ℤ) (t : ℝ)
    (a : Idx N) :
    φ (flowMap lam β N t (y + Pi.single l (h + n * (2 * π))) a) =
      φ (flowMap lam β N t (y + Pi.single l h) a) := by
  rw [Pi.single_add, ← add_assoc, flowMap_add_single, Pi.add_apply]
  by_cases hal : a = l
  · subst hal
    rw [Pi.single_eq_same]
    exact hφ.int_mul n _
  · rw [Pi.single_eq_of_ne hal, add_zero]

/-- A periodic test sees a move of one initial angle as a move by at most `π`. -/
theorem exists_periodic_update (lam β : ℝ) (N : ℕ) {φ : ℝ → ℝ}
    (hφ : Function.Periodic φ (2 * π)) (y : Idx N → ℝ) (l : Idx N) (z : ℝ) :
    ∃ h : ℝ, |h| ≤ π ∧ ∀ t a, φ (flowMap lam β N t (Function.update y l z) a) =
      φ (flowMap lam β N t (y + Pi.single l h) a) := by
  obtain ⟨h, hh, n, hn⟩ := exists_shift_le_pi (z - y l)
  refine ⟨h, hh, fun t a => ?_⟩
  rw [update_eq_add_single, hn, periodic_flowMap_add_single lam β N hφ]

/-- The hypothesis of `sum_alibiWeight_mul_le` holds for `f = 1`. -/
example := sum_alibiWeight_mul_le 0 (N := 2) 1 (f := fun _ => 1) (fun _ => zero_le_one)

/-- The hypotheses of `abs_deriv_flow_sub_le` and `abs_flowMap_add_single_sub_le` hold for
the kernel's own constant `L₁`, `α = 2e^{2|λ|}L₁`, `γ = 1` and `t = 0 ≤ T = 1`. -/
example (lam β : ℝ) (x : Idx 2 → ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  And.intro (abs_deriv_flow_sub_le lam β 2 hw.2.2.2.1 hw.1 x x 1)
    (abs_flowMap_add_single_sub_le lam β 2 hw.2.2.2.1 hw.1 le_rfl (γ := 1) one_pos le_rfl x 0
      1 1 (T := 1) (t := 0) ⟨le_rfl, zero_le_one⟩)

/-- The hypotheses of `periodic_flowMap_add_single` and `exists_periodic_update` hold for
the periodic test `sin`. -/
example (lam β : ℝ) (y : Idx 2 → ℝ) :=
  And.intro (periodic_flowMap_add_single lam β 2 Real.sin_periodic y 0 1 1 0 0)
    (exists_periodic_update lam β 2 Real.sin_periodic y 0 1)

end Kinetic
end Transformer

/-
# The attention weights and the kernel's Lipschitz constants

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 1. The field of
`eq:gpt-dyn-alibi` averages `w_β'(θ_j - θ_k)` over the predecessors `k < j`
with the normalized ALiBi weights `ω_{j,k} = e^{-λ(j-k)/N} / 𝒵_{N,j}`.

* `alibiWeight` is `ω_{j,k}`; the weights of a token with a predecessor sum to
  one (`sum_alibiWeight`), and each is at most `e^{2|λ|} / j`
  (`alibiWeight_le`), the source's `ω_{j,k} ≲ j⁻¹`.
* `w_β'` and its derivative are Lipschitz (`exists_lipschitz_wBetaDeriv`):
  both are continuous and `2π`-periodic, so their derivatives are bounded.
-/

import Transformer.Kinetic.Defs
import Mathlib.Topology.Instances.Real.Lemmas

open Real

namespace Transformer
namespace Kinetic

/-- `ω_{j,k} = e^{-λ(j-k)/N} / 𝒵_{N,j}`, the weight token `j` gives to token `k`
in `eq:gpt-dyn-alibi`. -/
noncomputable def alibiWeight (lam : ℝ) (N : ℕ) (j k : Idx N) : ℝ :=
  Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ))) / alibiZ lam N j

/-- The field of `eq:gpt-dyn-alibi` is the `ω`-average of `w_β'(θ_j - θ_k)`. -/
theorem gptField_eq_sum (lam β : ℝ) (N : ℕ) (θ : Idx N → ℝ) (j : Idx N) :
    gptField lam β N θ j = ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
      alibiWeight lam N j k * wBetaDeriv β (θ j - θ k) := by
  rw [gptField, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [alibiWeight]
  ring

theorem alibiWeight_nonneg (lam : ℝ) (N : ℕ) (j k : Idx N) : 0 ≤ alibiWeight lam N j k := by
  unfold alibiWeight alibiZ
  exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)

/-- Each ALiBi exponent lies in `[-|λ|, |λ|]`, since `0 < j - k < N`. -/
theorem abs_alibi_exponent_le (lam : ℝ) (N : ℕ) {j k : Idx N} (hkj : k < j) :
    |-(lam / N) * ((j : ℝ) - (k : ℝ))| ≤ |lam| := by
  have hN : (0 : ℝ) < N := by exact_mod_cast (Fin.pos j)
  have hjk : (0 : ℝ) ≤ (j : ℝ) - (k : ℝ) := by
    have : (k : ℝ) < j := by exact_mod_cast hkj
    linarith
  have hjN : (j : ℝ) - (k : ℝ) ≤ N := by
    have : (j : ℝ) < N := by exact_mod_cast j.2
    have : (0 : ℝ) ≤ k := Nat.cast_nonneg _
    linarith
  rw [abs_mul, abs_neg, abs_div, abs_of_pos hN, abs_of_nonneg hjk]
  calc |lam| / N * ((j : ℝ) - (k : ℝ)) ≤ |lam| / N * N :=
        mul_le_mul_of_nonneg_left hjN (div_nonneg (abs_nonneg _) hN.le)
    _ = |lam| := by field_simp

theorem card_filter_lt (N : ℕ) (j : Idx N) :
    (Finset.univ.filter (fun k : Idx N => k < j)).card = (j : ℕ) := by
  rw [Finset.filter_gt_eq_Iio, Fin.card_Iio]

/-- `𝒵_{N,j} ≥ j e^{-|λ|}`. -/
theorem alibiZ_ge (lam : ℝ) (N : ℕ) (j : Idx N) :
    (j : ℝ) * Real.exp (-|lam|) ≤ alibiZ lam N j := by
  have h : ∀ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
      Real.exp (-|lam|) ≤ Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ))) := by
    intro k hk
    rw [Finset.mem_filter] at hk
    exact Real.exp_le_exp.2 (neg_le_of_abs_le (abs_alibi_exponent_le lam N hk.2))
  have := Finset.card_nsmul_le_sum _ _ _ h
  rwa [card_filter_lt, nsmul_eq_mul] at this

/-- A token with a predecessor has a positive partition function. -/
theorem alibiZ_pos (lam : ℝ) (N : ℕ) {j : Idx N} (hj : (j : ℕ) ≠ 0) : 0 < alibiZ lam N j := by
  have hj' : (0 : ℝ) < j := by exact_mod_cast Nat.pos_of_ne_zero hj
  exact (mul_pos hj' (Real.exp_pos _)).trans_le (alibiZ_ge lam N j)

/-- The weights of a token with a predecessor sum to one. -/
theorem sum_alibiWeight (lam : ℝ) (N : ℕ) {j : Idx N} (hj : (j : ℕ) ≠ 0) :
    ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j), alibiWeight lam N j k = 1 := by
  unfold alibiWeight
  rw [← Finset.sum_div, div_eq_one_iff_eq (alibiZ_pos lam N hj).ne']
  rfl

/-- The weights of any token sum to at most one. -/
theorem sum_alibiWeight_le_one (lam : ℝ) (N : ℕ) (j : Idx N) :
    ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j), alibiWeight lam N j k ≤ 1 := by
  by_cases hj : (j : ℕ) = 0
  · have : Finset.univ.filter (fun k : Idx N => k < j) = ∅ := by
      rw [Finset.filter_gt_eq_Iio]
      ext k
      simp [Fin.lt_def, hj]
    simp [this]
  · exact (sum_alibiWeight lam N hj).le

/-- `ω_{j,k} ≤ e^{2|λ|} / j`, the source's `ω_{j,k} ≲ j⁻¹`. -/
theorem alibiWeight_le (lam : ℝ) (N : ℕ) {j k : Idx N} (hkj : k < j) :
    alibiWeight lam N j k ≤ Real.exp (2 * |lam|) / j := by
  have hj : (j : ℕ) ≠ 0 := by
    have : (k : ℕ) < j := hkj
    omega
  have hj' : (0 : ℝ) < j := by exact_mod_cast Nat.pos_of_ne_zero hj
  have hZ := alibiZ_ge lam N j
  have hnum : Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ))) ≤ Real.exp |lam| :=
    Real.exp_le_exp.2 (le_of_abs_le (abs_alibi_exponent_le lam N hkj))
  unfold alibiWeight
  rw [div_le_div_iff₀ (alibiZ_pos lam N hj) hj']
  calc Real.exp (-(lam / N) * ((j : ℝ) - (k : ℝ))) * j ≤ Real.exp |lam| * j :=
        mul_le_mul_of_nonneg_right hnum hj'.le
    _ = Real.exp (2 * |lam|) * ((j : ℝ) * Real.exp (-|lam|)) := by
        rw [show 2 * |lam| = |lam| + |lam| by ring, Real.exp_add]
        field_simp
        rw [← Real.exp_add]
        simp
    _ ≤ Real.exp (2 * |lam|) * alibiZ lam N j :=
        mul_le_mul_of_nonneg_left hZ (Real.exp_pos _).le

/-- `ω_{j,k} ≤ 2e^{2|λ|} / (j + 1)`, the form used with `1`-based positions. -/
theorem alibiWeight_le_succ (lam : ℝ) (N : ℕ) {j k : Idx N} (hkj : k < j) :
    alibiWeight lam N j k ≤ 2 * Real.exp (2 * |lam|) / ((j : ℝ) + 1) := by
  have hj : (1 : ℝ) ≤ j := by
    have : (k : ℕ) < j := hkj
    exact_mod_cast (show 1 ≤ (j : ℕ) by omega)
  refine (alibiWeight_le lam N hkj).trans ?_
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  have := Real.exp_pos (2 * |lam|)
  nlinarith

/-- `w_β'` is `C²`. -/
theorem contDiff_two_wBetaDeriv (β : ℝ) : ContDiff ℝ 2 (wBetaDeriv β) := by
  unfold wBetaDeriv
  fun_prop

/-- The derivative of a `C¹` periodic function is periodic. -/
theorem periodic_deriv {f : ℝ → ℝ} {c : ℝ} (hf : Function.Periodic f c) :
    Function.Periodic (deriv f) c := by
  intro x
  rw [← deriv_comp_add_const]
  congr 1
  funext y
  exact hf y

/-- A `C¹` function with a continuous periodic derivative is Lipschitz. -/
theorem exists_lipschitz_of_periodic {f : ℝ → ℝ} {c : ℝ} (hc : c ≠ 0)
    (hf : ContDiff ℝ 1 f) (hp : Function.Periodic f c) :
    ∃ L : ℝ, 0 ≤ L ∧ ∀ x y, |f x - f y| ≤ L * |x - y| := by
  obtain ⟨C, hC⟩ := isBounded_iff_forall_norm_le.1
    ((periodic_deriv hp).isBounded_of_continuous hc (hf.continuous_deriv le_rfl))
  have hC' : ∀ x, ‖deriv f x‖ ≤ C := fun x => hC _ ⟨x, rfl⟩
  refine ⟨max C 0, le_max_right _ _, fun x y => ?_⟩
  have := convex_univ.norm_image_sub_le_of_norm_deriv_le
    (fun z _ => (hf.differentiable one_ne_zero).differentiableAt) (fun z _ => (hC' z).trans
      (le_max_left C 0)) (Set.mem_univ y) (Set.mem_univ x)
  simpa [Real.norm_eq_abs] using this

/-- **The kernel's constants.** `w_β'` is Lipschitz, and so is its derivative.
They are the `‖w_β''‖_∞` and `‖w_β'''‖_∞` of the proof of `lem:correl`. -/
theorem exists_lipschitz_wBetaDeriv (β : ℝ) :
    ∃ L₁ L₂ : ℝ, 0 ≤ L₁ ∧ 0 ≤ L₂ ∧
      (∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x) ∧
      (∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) ∧
      (∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|) := by
  have hs := contDiff_two_wBetaDeriv β
  have hc : (2 * π) ≠ 0 := by positivity
  obtain ⟨L₁, hL₁, h₁⟩ := exists_lipschitz_of_periodic hc (hs.of_le (by norm_num))
    (wBetaDeriv_periodic β)
  have hd : ContDiff ℝ 1 (deriv (wBetaDeriv β)) :=
    (contDiff_succ_iff_deriv (n := 1)).1 (by simpa [one_add_one_eq_two] using hs) |>.2.2
  obtain ⟨L₂, hL₂, h₂⟩ := exists_lipschitz_of_periodic hc hd (periodic_deriv (wBetaDeriv_periodic β))
  exact ⟨L₁, L₂, hL₁, hL₂, fun x => ((hs.differentiable (by norm_num)) x).hasDerivAt, h₁, h₂⟩

/-- The hypotheses of the weight bounds are satisfiable: token `1` of two has the
predecessor `0`. -/
example : (0 : Idx 2) < 1 ∧ ((1 : Idx 2) : ℕ) ≠ 0 := ⟨by decide, by decide⟩

end Kinetic
end Transformer

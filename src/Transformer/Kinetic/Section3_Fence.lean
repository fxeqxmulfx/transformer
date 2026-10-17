/-
# Comparison for triangular differential inequalities

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Steps 1 and 3: the bounds on
the Glauber derivatives come from integrating differential inequalities of the
form

  `|y_a'| ≤ L |y_a| + Σ_{b<a} c_{a,b} |y_b| + s_a(t)`,

in which token `a` only sees its predecessors. The source iterates the
integrated inequality; here a supersolution `W` is compared with `|y|`, token
by token (`abs_le_of_triangular`). The one-token step `abs_le_of_deriv_le` is
Mathlib's fencing theorem, applied to `W + ε e^{(L+1)t}` and then `ε → 0`.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real

namespace Transformer
namespace Kinetic

/-- **Comparison.** If `|f'| ≤ L |f| + g` and `W' ≥ L W + g` on `[0, T)`, and
`|f 0| ≤ W 0`, then `|f| ≤ W` on `[0, T]`. -/
theorem abs_le_of_deriv_le {f f' W W' g : ℝ → ℝ} {L T : ℝ}
    (hf : ∀ t, HasDerivAt f (f' t) t) (hW : ∀ t, HasDerivAt W (W' t) t)
    (h0 : |f 0| ≤ W 0)
    (hbound : ∀ t ∈ Set.Ico 0 T, |f' t| ≤ L * |f t| + g t)
    (hsuper : ∀ t ∈ Set.Ico 0 T, L * W t + g t ≤ W' t) :
    ∀ t ∈ Set.Icc 0 T, |f t| ≤ W t := by
  have hε : ∀ ε : ℝ, 0 < ε → ∀ t ∈ Set.Icc 0 T,
      |f t| ≤ W t + ε * Real.exp ((L + 1) * t) := by
    intro ε hε
    have hB : ∀ t, HasDerivAt (fun t => W t + ε * Real.exp ((L + 1) * t))
        (W' t + ε * (Real.exp ((L + 1) * t) * (L + 1))) t := by
      intro t
      refine (hW t).add (HasDerivAt.const_mul ε ?_)
      exact ((hasDerivAt_id t).const_mul (L + 1)).exp.congr_deriv (by simp)
    have := image_norm_le_of_norm_deriv_right_lt_deriv_boundary (f := f) (f' := f') (a := 0)
      (b := T) (continuous_iff_continuousAt.2 fun t => (hf t).continuousAt).continuousOn
      (fun t _ => (hf t).hasDerivWithinAt) (B := fun t => W t + ε * Real.exp ((L + 1) * t))
      (by simp only [Real.norm_eq_abs, mul_zero, Real.exp_zero, mul_one]; linarith) hB
      (fun t ht hft => by
        rw [Real.norm_eq_abs] at hft ⊢
        have h1 := hbound t ht
        have h2 := hsuper t ht
        have he := Real.exp_pos ((L + 1) * t)
        rw [hft] at h1
        nlinarith)
    simpa [Real.norm_eq_abs] using this
  intro t ht
  refine le_of_forall_pos_le_add fun δ hδ => ?_
  have hpos := Real.exp_pos ((L + 1) * t)
  have := hε (δ / Real.exp ((L + 1) * t)) (div_pos hδ hpos) t ht
  rwa [div_mul_cancel₀ _ hpos.ne'] at this

/-- **Triangular comparison.** Token `a` sees only its predecessors: if
`|y_a'| ≤ L |y_a| + Σ_{b<a} c_{a,b} |y_b| + s_a` and
`W_a' ≥ L W_a + Σ_{b<a} c_{a,b} W_b + s_a` on `[0, T)`, with `c ≥ 0` and
`|y_a 0| ≤ W_a 0`, then `|y_a| ≤ W_a` on `[0, T]` for every token. -/
theorem abs_le_of_triangular {N : ℕ} {y y' W W' s : Idx N → ℝ → ℝ} {c : Idx N → Idx N → ℝ}
    {L T : ℝ} (hc : ∀ a b, 0 ≤ c a b)
    (hy : ∀ a t, HasDerivAt (y a) (y' a t) t) (hW : ∀ a t, HasDerivAt (W a) (W' a t) t)
    (h0 : ∀ a, |y a 0| ≤ W a 0)
    (hbound : ∀ a, ∀ t ∈ Set.Ico 0 T, |y' a t| ≤
      L * |y a t| + ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), c a b * |y b t| + s a t)
    (hsuper : ∀ a, ∀ t ∈ Set.Ico 0 T,
      L * W a t + ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), c a b * W b t + s a t ≤
        W' a t) :
    ∀ a, ∀ t ∈ Set.Icc 0 T, |y a t| ≤ W a t := by
  intro a
  induction a using WellFoundedLT.induction with
  | ind a ih =>
    refine abs_le_of_deriv_le (g := fun t =>
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), c a b * W b t + s a t) (hy a) (hW a)
      (h0 a) (fun t ht => ?_) (fun t ht => by linarith [hsuper a t ht])
    have hsum : ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), c a b * |y b t| ≤
        ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), c a b * W b t :=
      Finset.sum_le_sum fun b hb => mul_le_mul_of_nonneg_left
        (ih b (Finset.mem_filter.1 hb).2 t (Set.Ico_subset_Icc_self ht)) (hc a b)
    linarith [hbound a t ht]

/-- The hypotheses of `abs_le_of_deriv_le` are satisfiable: `f = W = 0`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧ |(0 : ℝ)| ≤ 0 ∧
    ∀ t ∈ Set.Ico (0 : ℝ) 1, |(0 : ℝ)| ≤ 0 * |(0 : ℝ)| + 0 :=
  ⟨fun t => hasDerivAt_const t 0, by simp, fun _ _ => by simp⟩

end Kinetic
end Transformer

/-
# Appendix A — two trajectories near the saddle start in the stable cone

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

Along two solutions of `e:gradfl` the cone form `q` of their difference has
derivative `B(P(x - y), P(F x - F y))` (`hasDerivAt_coneForm`), at least `2 q`
while both stay within `r`, `60 n r ≤ 1`, of the saddle `signedTuple e {i₀}`
(`Perspective.AppendixA_ConeGrowth`).  As `q` is bounded on the sphere, `q ≤ 0`
at time `0` (`coneForm_nonpos_of_stay`): the part of the source's center-stable
manifold theorem that the slice argument of `Perspective.AppendixA_StayNear`
needs.
-/

import Transformer.Perspective.AppendixA_ConeGrowth

open scoped BigOperators Topology
open Real Set Filter MeasureTheory

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `Proj_e` commutes with differentiation. -/
theorem hasDerivAt_proj (e : EucSpace d) {f : ℝ → EucSpace d} {f' : EucSpace d} {t : ℝ}
    (hf : HasDerivAt f f' t) : HasDerivAt (fun s => proj d e (f s)) (proj d e f') t := by
  have h := hf.fun_sub (((hasDerivAt_const t e).inner ℝ hf).smul_const e)
  simpa only [proj, inner_zero_left, add_zero] using h

/-- The hypothesis of `hasDerivAt_proj` holds for the line `s ↦ s v` in `ℝ²`. -/
example (v : EucSpace 2) : HasDerivAt (fun s : ℝ => s • v) v 0 := by
  simpa using (hasDerivAt_id (0 : ℝ)).smul_const v

/-- **The derivative of the cone form along two solutions of `e:gradfl`** is the
polarized form `B(P(x - y), P(F x - F y))`. -/
theorem hasDerivAt_coneForm (e : EucSpace d) (i₀ : Idx n) (X Y : ℝ → SphereTuple d n)
    (hX : E0GradientAscent d n X) (hY : E0GradientAscent d n Y) (t : ℝ) :
    HasDerivAt (fun s => coneForm e i₀ (fun k => (X s k : EucSpace d) - Y s k))
      (coneBil i₀ (fun k => proj d e ((X t k : EucSpace d) - Y t k))
        (fun k => proj d e (e0Field (fun l => (X t l : EucSpace d)) k
          - e0Field (fun l => (Y t l : EucSpace d)) k))) t := by
  have hu : ∀ k : Idx n, HasDerivAt (fun s => proj d e ((X s k : EucSpace d) - Y s k))
      (proj d e (e0Field (fun l => (X t l : EucSpace d)) k
        - e0Field (fun l => (Y t l : EucSpace d)) k)) t :=
    fun k => hasDerivAt_proj e ((hX t k).sub (hY t k))
  have hW := (HasDerivAt.fun_sum (u := Finset.univ) fun k _ => hu k).fun_add
    ((hu i₀).fun_const_smul ((n : ℝ) - 2))
  have hS := HasDerivAt.fun_sum (u := Finset.univ) fun k _ => (hu k).norm_sq
  have h := (hW.norm_sq.const_mul (2 / ((n : ℝ) * ((n : ℝ) - 1)))).fun_sub hS
  convert h using 1
  · rfl
  · simp only [coneBil, coneW, Finset.mul_sum]
    ring

/-- The cone form is bounded on differences of unit vectors. -/
theorem coneForm_le (e : SSphere d) (i₀ : Idx n) (x y : SphereTuple d n) :
    coneForm (e : EucSpace d) i₀ (fun k => (x k : EucSpace d) - y k)
      ≤ 2 / ((n : ℝ) * ((n : ℝ) - 1)) * (4 * n) ^ 2 := by
  have hN : (1 : ℝ) ≤ n := by exact_mod_cast i₀.pos
  have he : ‖(e : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp e.2
  have hc : 0 ≤ 2 / ((n : ℝ) * ((n : ℝ) - 1)) :=
    div_nonneg (by norm_num) (mul_nonneg (by linarith) (by linarith))
  have hu : ∀ k : Idx n, ‖proj d e ((x k : EucSpace d) - y k)‖ ≤ 2 := fun k =>
    (norm_proj_le he _).trans ((norm_sub_le _ _).trans (by
      rw [mem_sphere_zero_iff_norm.mp (x k).2, mem_sphere_zero_iff_norm.mp (y k).2]; norm_num))
  have hW : ‖coneW i₀ (fun k => proj d e ((x k : EucSpace d) - y k))‖ ≤ 4 * n := by
    rw [coneW]
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, Real.norm_eq_abs]
    have h1 : ‖∑ k : Idx n, proj d e ((x k : EucSpace d) - y k)‖ ≤ 2 * n :=
      (norm_sum_le _ _).trans ((Finset.sum_le_sum fun k _ => hu k).trans (by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; linarith))
    have h2 : |(n : ℝ) - 2| ≤ n := abs_le.mpr ⟨by linarith, by linarith⟩
    have h3 := mul_le_mul h2 (hu i₀) (norm_nonneg _) (by linarith)
    linarith
  have hS : 0 ≤ ∑ k : Idx n, ‖proj d e ((x k : EucSpace d) - y k)‖ ^ 2 :=
    Finset.sum_nonneg fun k _ => sq_nonneg _
  simp only [coneForm]
  have := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) hW 2) hc
  linarith

/-- A function with `g' ≥ 2 g` on `(0, ∞)` that stays below `K` on `[0, ∞)` is
nonpositive at `0`: `e^{-2t} g(t)` increases, so `g(t) ≥ g(0) e^{2t}`. -/
theorem nonpos_of_two_mul_le_deriv {g g' : ℝ → ℝ} {K : ℝ} (hg : ∀ t, HasDerivAt g (g' t) t)
    (hgr : ∀ t, 0 < t → 2 * g t ≤ g' t) (hK : ∀ t, 0 ≤ t → g t ≤ K) : g 0 ≤ 0 := by
  have hexp : ∀ t : ℝ, HasDerivAt (fun s => Real.exp (-2 * s)) (Real.exp (-2 * t) * (-2)) t :=
    fun t => by simpa using ((hasDerivAt_id t).const_mul (-2)).exp
  have hH := fun t => (hg t).fun_mul (hexp t)
  have hmono : MonotoneOn (fun s => g s * Real.exp (-2 * s)) (Ici 0) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
      (fun t _ => (hH t).continuousAt.continuousWithinAt) (fun t _ => (hH t).hasDerivWithinAt)
      fun t ht => ?_
    rw [interior_Ici] at ht
    have h2 := hgr t ht
    have hpos := Real.exp_pos (-2 * t)
    nlinarith
  by_contra hpos
  rw [not_le] at hpos
  have hK0 : 0 < K := lt_of_lt_of_le hpos (hK 0 le_rfl)
  have hT0 : 0 ≤ K / g 0 := div_nonneg hK0.le hpos.le
  have h1 := hmono (mem_Ici.mpr le_rfl) (mem_Ici.mpr hT0) hT0
  simp only [mul_zero, Real.exp_zero, mul_one] at h1
  have h3 : g 0 * Real.exp (2 * (K / g 0)) ≤ g (K / g 0) := by
    have := mul_le_mul_of_nonneg_right h1 (Real.exp_pos (2 * (K / g 0))).le
    rwa [mul_assoc, ← Real.exp_add, show -2 * (K / g 0) + 2 * (K / g 0) = 0 by ring,
      Real.exp_zero, mul_one] at this
  have h4 : g 0 * (K / g 0) = K := by field_simp
  nlinarith [mul_le_mul_of_nonneg_left (Real.add_one_le_exp (2 * (K / g 0))) hpos.le,
    hK (K / g 0) hT0]

/-- The hypotheses of `nonpos_of_two_mul_le_deriv` hold for `g = -1`, `K = 0`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (-1 : ℝ)) 0 t) ∧
    (∀ t : ℝ, 0 < t → 2 * (-1 : ℝ) ≤ 0) ∧ ∀ t : ℝ, 0 ≤ t → (-1 : ℝ) ≤ 0 :=
  ⟨fun t => hasDerivAt_const t _, fun _ _ => by norm_num, fun _ _ => by norm_num⟩

/-- **Two trajectories staying near the saddle start in the stable cone.** If two
solutions of `e:gradfl` stay within `r`, `60 n r ≤ 1`, of `signedTuple e {i₀}`
for all `t ≥ 0`, then `q(X 0 - Y 0) ≤ 0`: otherwise `q' ≥ 2q` would make `q`
grow like `e^{2t}`, while `q` is bounded. -/
theorem coneForm_nonpos_of_stay (hn : 3 ≤ n) (e : SSphere d) (i₀ : Idx n) {r : ℝ}
    (hr : 60 * (n : ℝ) * r ≤ 1) (X Y : ℝ → SphereTuple d n)
    (hX : E0GradientAscent d n X) (hY : E0GradientAscent d n Y)
    (hXr : ∀ t : ℝ, 0 ≤ t → ∀ k : Idx n, ‖(X t k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r)
    (hYr : ∀ t : ℝ, 0 ≤ t → ∀ k : Idx n, ‖(Y t k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r) :
    coneForm (e : EucSpace d) i₀ (fun k => (X 0 k : EucSpace d) - Y 0 k) ≤ 0 :=
  nonpos_of_two_mul_le_deriv (hasDerivAt_coneForm (e : EucSpace d) i₀ X Y hX hY)
    (fun t ht => two_mul_coneForm_le hn e i₀ hr (X t) (Y t) (hXr t ht.le) (hYr t ht.le))
    (fun t _ => coneForm_le e i₀ (X t) (Y t))

/-- The hypotheses of `coneForm_nonpos_of_stay` hold for the constant trajectories
at the saddle `signedTuple e {0}`, `n = 3`, `r = 0`. -/
example (e : SSphere 2) :
    E0GradientAscent 2 3 (fun _ => signedTuple e {0}) ∧ 60 * ((3 : ℕ) : ℝ) * 0 ≤ 1 ∧
      ∀ t : ℝ, 0 ≤ t → ∀ k : Idx 3,
        ‖(signedTuple e {0} k : EucSpace 2) - signedTuple e {0} k‖ ≤ 0 :=
  ⟨E0GradientAscent_const _ (isCriticalE0_signedTuple e {0}), by norm_num, fun _ _ _ => by simp⟩

end Perspective
end Transformer

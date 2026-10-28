module

public import Mathlib.Algebra.Polynomial.BigOperators
public import Mathlib.Analysis.Complex.Basic
public import Mathlib.Analysis.SpecificLimits.Basic

/-!
# From bounds on power sums to bounds on the numbers

If `‖∑_{ω ∈ s} ω^n‖ ≤ M R^n` for every large `n`, then every `ω ∈ s` has `‖ω‖ ≤ R`. For
`v ∈ s`, the polynomial `P = ∏_{w ∈ s, w ≠ v} (X − w)` vanishes at every other value, so
`∑_j P_j ∑_ω ω^{n + j} = ∑_ω ω^n P(ω) = (count v) v^n P(v)` with `P(v) ≠ 0`. Hence
`‖v‖^n ≤ D R^n` for every large `n`, with `D` independent of `n`, and `‖v‖ ≤ R`: otherwise
`D (R/‖v‖)^n → 0` would drop below `1`.

This turns Stepanov's bound over every extension `𝔽_{p^n}`, with a constant independent of `n`,
into the bound `√p` on the inverse roots of the `L`-function.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset Filter Topology

/-- If `a^n ≤ D b^n` for every `n ≥ n₀`, with `b ≥ 0`, then `a ≤ b`. -/
theorem le_of_pow_le_mul_pow {a b D : ℝ} (hb : 0 ≤ b) {n₀ : ℕ}
    (h : ∀ n, n₀ ≤ n → a ^ n ≤ D * b ^ n) : a ≤ b := by
  by_contra hab
  rw [not_le] at hab
  have ha : 0 < a := hb.trans_lt hab
  have hlim : Tendsto (fun n : ℕ => D * (b / a) ^ n) atTop (𝓝 0) := by
    simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one (div_nonneg hb ha.le)
      ((div_lt_one ha).mpr hab)).const_mul D
  obtain ⟨n, hn, hlt⟩ :=
    ((eventually_ge_atTop n₀).and (hlim.eventually (gt_mem_nhds one_pos))).exists
  have h1 : 1 ≤ D * (b / a) ^ n := by
    rw [div_pow, ← mul_div_assoc, le_div_iff₀ (pow_pos ha n), one_mul]
    exact h n hn
  exact absurd h1 (not_le.mpr hlt)

example := le_of_pow_le_mul_pow (a := 1) (b := 1) (D := 1) zero_le_one (n₀ := 0)
  fun n _ => by simp

/-- `∑_{ω ∈ s} ω^n P(ω) = ∑_j P_j ∑_{ω ∈ s} ω^{n + j}`. -/
theorem sum_map_pow_mul_eval {R : Type*} [CommRing R] (s : Multiset R) (P : R[X]) (n : ℕ) :
    (s.map fun w => w ^ n * P.eval w).sum =
      ∑ j ∈ range (P.natDegree + 1), P.coeff j * (s.map (· ^ (n + j))).sum := by
  induction s using Multiset.induction_on with
  | empty => simp
  | cons a t ih =>
    simp only [Multiset.map_cons, Multiset.sum_cons, mul_add, sum_add_distrib, ih]
    congr 1
    rw [eval_eq_sum_range, mul_sum]
    exact sum_congr rfl fun j _ => by ring

/-- **Transfer**: if `‖∑_{ω ∈ s} ω^n‖ ≤ M R^n` for every `n ≥ n₀`, with `R ≥ 0`, then every
`v ∈ s` has `‖v‖ ≤ R`. -/
theorem norm_le_of_norm_sum_pow_le {s : Multiset ℂ} {M R : ℝ} {n₀ : ℕ} (hR : 0 ≤ R)
    (h : ∀ n, n₀ ≤ n → ‖(s.map (· ^ n)).sum‖ ≤ M * R ^ n) {v : ℂ} (hv : v ∈ s) : ‖v‖ ≤ R := by
  classical
  set P : ℂ[X] := ∏ w ∈ s.toFinset.erase v, (X - C w) with hP
  have hPv : P.eval v ≠ 0 := by
    rw [hP, eval_prod, prod_ne_zero_iff]
    intro w hw
    rw [eval_sub, eval_X, eval_C, sub_ne_zero]
    exact (ne_of_mem_erase hw).symm
  have hPw (w : ℂ) (hwv : w ≠ v) (hw : w ∈ s) : P.eval w = 0 := by
    rw [hP, eval_prod]
    exact prod_eq_zero (mem_erase.mpr ⟨hwv, Multiset.mem_toFinset.mpr hw⟩) (by simp)
  have key (n : ℕ) : (s.count v : ℂ) * (v ^ n * P.eval v) =
      ∑ j ∈ range (P.natDegree + 1), P.coeff j * (s.map (· ^ (n + j))).sum := by
    rw [← sum_map_pow_mul_eval, Multiset.sum_map_eq_nsmul_single v
      fun w hwv hw => by rw [hPw w hwv hw, mul_zero], nsmul_eq_mul]
  have hc : (0 : ℝ) < s.count v := by exact_mod_cast Multiset.count_pos.mpr hv
  have hB : 0 < (s.count v : ℝ) * ‖P.eval v‖ := mul_pos hc (norm_pos_iff.mpr hPv)
  refine le_of_pow_le_mul_pow hR (n₀ := n₀)
    (D := (∑ j ∈ range (P.natDegree + 1), ‖P.coeff j‖ * M * R ^ j) /
      ((s.count v : ℝ) * ‖P.eval v‖)) fun n hn => ?_
  rw [div_mul_eq_mul_div, le_div_iff₀ hB, sum_mul]
  have hk := congrArg norm (key n)
  rw [norm_mul, norm_mul, norm_pow, Complex.norm_natCast] at hk
  calc ‖v‖ ^ n * ((s.count v : ℝ) * ‖P.eval v‖)
      = ‖∑ j ∈ range (P.natDegree + 1), P.coeff j * (s.map (· ^ (n + j))).sum‖ := by
        rw [← hk]
        ring
    _ ≤ ∑ j ∈ range (P.natDegree + 1), ‖P.coeff j * (s.map (· ^ (n + j))).sum‖ :=
        norm_sum_le _ _
    _ ≤ ∑ j ∈ range (P.natDegree + 1), ‖P.coeff j‖ * M * R ^ j * R ^ n := by
        refine sum_le_sum fun j _ => ?_
        rw [norm_mul, mul_assoc, mul_assoc, ← pow_add, add_comm j n]
        exact mul_le_mul_of_nonneg_left (h _ (by omega)) (norm_nonneg _)

example := norm_le_of_norm_sum_pow_le (s := {1}) (M := 1) (R := 1) zero_le_one (n₀ := 0)
  (fun n _ => by simp) (Multiset.mem_singleton_self 1)

end Transformer.Weil

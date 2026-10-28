module

public import Transformer.Weil.PowerSums

/-!
# Sums over the roots of a polynomial, from its coefficients

For `f, g ∈ K[X]`, `rootSum f g = ∑_j f_j p_j(g)` is `∑_i f(β_i)` when `g = ∏_i (X − β_i)`. It
is additive in `g` and commutes with field maps. The power sums `p_j`, `j < m`, of a monic `g`
depend only on the coefficients `A_b`, `b < m`, of its reverse `A`, and `p_m` moves by
`−m Δ` when `A_m` moves by `Δ` (Newton's recursion). So for `g` monic of degree `n ≥ m`,
`m = deg f ≥ 1`, adding `t X^{n − m}` to `g` moves `A_m` by `t` and `rootSum f g` by `−m f_m t`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K]

/-- `∑_i f(β_i)` over the roots `β_i` of `g`, as `∑_j f_j p_j(g)`. -/
noncomputable def rootSum (f g : K[X]) : K :=
  ∑ j ∈ range (f.natDegree + 1), f.coeff j * powerSum g j

theorem rootSum_mul (f : K[X]) {g h : K[X]} (hg : g ≠ 0) (hh : h ≠ 0) :
    rootSum f (g * h) = rootSum f g + rootSum f h := by
  simp only [rootSum, powerSum_mul hg hh, mul_add, sum_add_distrib]

example := rootSum_mul (K := ℚ) X X_ne_zero X_ne_zero

theorem rootSum_one (f : K[X]) : rootSum f 1 = 0 := by
  simp [rootSum, powerSum_one]

theorem rootSum_X_sub_C (f : K[X]) (β : K) : rootSum f (X - C β) = f.eval β := by
  simp only [rootSum, powerSum_X_sub_C, eval_eq_sum_range]

/-- `rootSum f (∏_{i ∈ s} (X − v_i)) = ∑_{i ∈ s} f(v_i)`. -/
theorem rootSum_prod_X_sub_C (f : K[X]) {ι : Type*} (s : Finset ι) (v : ι → K) :
    rootSum f (∏ i ∈ s, (X - C (v i))) = ∑ i ∈ s, f.eval (v i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [rootSum_one]
  | insert i s hi ih =>
    rw [prod_insert hi, sum_insert hi, rootSum_mul f (X_sub_C_ne_zero _)
      (monic_prod_of_monic _ _ fun i _ => monic_X_sub_C (v i)).ne_zero, rootSum_X_sub_C, ih]

/-- `rootSum` commutes with field maps. -/
theorem rootSum_map {L : Type*} [Field L] (φ : K →+* L) (f : K[X]) {g : K[X]} (hg : g.Monic) :
    rootSum (f.map φ) (g.map φ) = φ (rootSum f g) := by
  simp only [rootSum, natDegree_map_eq_of_injective φ.injective, coeff_map, powerSum_map φ hg,
    map_sum, map_mul]

example := rootSum_map (RingHom.id ℚ) X monic_X

/-- If `A′_b = A_b` for `b < m`, then `p′_j = p_j` for `1 ≤ j < m`. -/
theorem powerSum_eq_of_coeff_reverse_eq {g g' : K[X]} (hg : g.Monic) (hg' : g'.Monic) {m : ℕ}
    (h : ∀ b < m, g'.reverse.coeff b = g.reverse.coeff b) {j : ℕ} (hj : j ≠ 0) (hjm : j < m) :
    powerSum g' j = powerSum g j := by
  induction j using Nat.strong_induction_on with
  | _ j ih =>
  rw [powerSum_eq hg' hj, powerSum_eq hg hj, h j hjm]
  congr 1
  refine sum_congr rfl fun b hb => ?_
  have := mem_Ico.mp hb
  rw [h b (by omega), ih (j - b) (by omega) (by omega) (by omega)]

example := powerSum_eq_of_coeff_reverse_eq (K := ℚ) monic_X monic_X (m := 2) (fun _ _ => rfl)
  one_ne_zero one_lt_two

/-- If `A′_b = A_b` for `b < m`, then `p′_m = p_m − m (A′_m − A_m)`. -/
theorem powerSum_eq_sub_of_coeff_reverse_eq {g g' : K[X]} (hg : g.Monic) (hg' : g'.Monic)
    {m : ℕ} (h : ∀ b < m, g'.reverse.coeff b = g.reverse.coeff b) (hm : m ≠ 0) :
    powerSum g' m = powerSum g m - m * (g'.reverse.coeff m - g.reverse.coeff m) := by
  have hs : ∑ b ∈ Ico 1 m, g'.reverse.coeff b * powerSum g' (m - b) =
      ∑ b ∈ Ico 1 m, g.reverse.coeff b * powerSum g (m - b) :=
    sum_congr rfl fun b hb => by
      have := mem_Ico.mp hb
      rw [h b (by omega), powerSum_eq_of_coeff_reverse_eq hg hg' h (by omega) (by omega)]
  rw [powerSum_eq hg' hm, powerSum_eq hg hm, hs]
  ring

example := powerSum_eq_sub_of_coeff_reverse_eq (K := ℚ) monic_X monic_X (m := 1)
  (fun _ _ => rfl) one_ne_zero

/-- For `g` monic of degree `n ≥ m = deg f ≥ 1`, adding `t X^{n − m}` to `g` moves
`rootSum f g` by `−m f_m t`. -/
theorem rootSum_add_C_mul_X_pow {f g : K[X]} (hg : g.Monic) {m n : ℕ} (hm : f.natDegree = m)
    (hm0 : m ≠ 0) (hmn : m ≤ n) (hn : g.natDegree = n) (t : K) :
    rootSum f (g + C t * X ^ (n - m)) = rootSum f g - m * f.coeff m * t := by
  have hdeg : degree (C t * X ^ (n - m)) < degree g :=
    (degree_C_mul_X_pow_le _ _).trans_lt (by
      rw [degree_eq_natDegree hg.ne_zero, hn]
      exact_mod_cast (by omega : n - m < n))
  set g' := g + C t * X ^ (n - m) with hg'def
  have hg' : g'.Monic := hg.add_of_left hdeg
  have hn' : g'.natDegree = n := by rw [hg'def, natDegree_add_eq_left_of_degree_lt hdeg, hn]
  have hA (b : ℕ) : g'.reverse.coeff b = g.reverse.coeff b + if b = m then t else 0 := by
    rw [coeff_reverse, coeff_reverse, hn, hn', hg'def, coeff_add, coeff_C_mul_X_pow]
    have key : revAt n b = n - m ↔ b = m := by
      rcases le_or_gt b n with hb | hb
      · rw [revAt_le hb]
        omega
      · rw [revAt_eq_self_of_lt hb]
        omega
    simp only [key]
  have hlow : ∀ b < m, g'.reverse.coeff b = g.reverse.coeff b := fun b hb => by
    simp [hA, hb.ne]
  have hmid : g'.reverse.coeff m = g.reverse.coeff m + t := by simp [hA]
  have hs : ∑ j ∈ range m, f.coeff j * powerSum g' j = ∑ j ∈ range m, f.coeff j * powerSum g j :=
    sum_congr rfl fun j hj => by
      rcases eq_or_ne j 0 with rfl | hj0
      · rw [powerSum_zero, powerSum_zero, hn, hn']
      · rw [powerSum_eq_of_coeff_reverse_eq hg hg' hlow hj0 (mem_range.mp hj)]
  rw [rootSum, rootSum, hm, sum_range_succ, sum_range_succ, hs,
    powerSum_eq_sub_of_coeff_reverse_eq hg hg' hlow hm0, hmid]
  ring

example := rootSum_add_C_mul_X_pow (K := ℚ) (f := X) (g := X) monic_X natDegree_X one_ne_zero
  le_rfl natDegree_X 1

end Transformer.Weil

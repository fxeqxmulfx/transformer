module

public import Transformer.Weil.MulWeight

/-!
# `c_n = 0` through a unit modulo `f`

Let `f ∈ K[X]` have degree `m` and leading coefficient `λ`, and let `u` be prime to `f`. The map
`g ↦ g + ((u g) mod f − g mod f)` replaces the residue of `g` modulo `f` by that of `u g`. It
keeps the monics of each degree `n ≥ m`, and the same map for an inverse `a` of `u` modulo `f`
undoes it. For `deg u ≤ N`, the resultants with the formal degrees `m`, `N`, `n` satisfy
`λ^N Res(f, g') = Res(f, u) Res(f, g)`: `Res(f, u g) = Res(f, u) Res(f, g)`, and `Res(f, ·)` sees
only the residue modulo `f`, up to `λ^N` from the drop of the formal degree from `N + n` to `n`.
So if `χ(Res(f, u)) ≠ χ(λ)^N`, then `c_n = 0` for the weight `χ(Res(g, f))` and every `n ≥ m`.
`MulWeight` is the case of a constant `u = t`. (W. M. Schmidt, *Equations over Finite Fields*,
Ch. II.)
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K]

/-- `g ↦ g + ((u g) mod f − g mod f)`: the residue of `g` modulo `f` multiplied by `u`. -/
noncomputable def mulMod (f u g : K[X]) : K[X] :=
  g + ((u * g) % f - g % f)

/-- `deg (g / f) ≤ deg g − deg f`. -/
theorem natDegree_div_le_sub {f : K[X]} (hf : f ≠ 0) (g : K[X]) :
    (g / f).natDegree ≤ g.natDegree - f.natDegree := by
  rw [Polynomial.div_def]
  refine (natDegree_C_mul_le _ _).trans ?_
  rw [natDegree_divByMonic _ (monic_mul_leadingCoeff_inv hf),
    natDegree_mul_C (inv_ne_zero (leadingCoeff_ne_zero.mpr hf))]

example := natDegree_div_le_sub (K := ℚ) X_ne_zero X

section MulMod

variable {f : K[X]} (hf : f ≠ 0)

include hf

theorem mod_mod_self (g : K[X]) : g % f % f = g % f :=
  (mod_eq_self_iff hf).mpr (degree_mod_lt g hf)

theorem degree_mod_sub_mod_lt (u g : K[X]) : degree ((u * g) % f - g % f) < degree f :=
  (degree_sub_le _ _).trans_lt (max_lt (degree_mod_lt _ hf) (degree_mod_lt _ hf))

theorem mulMod_mod (u g : K[X]) : mulMod f u g % f = (u * g) % f := by
  rw [mulMod, add_mod, sub_mod, mod_mod_self hf, mod_mod_self hf]
  ring

/-- `a` inverse to `u` modulo `f` undoes `mulMod f u`. -/
theorem mulMod_mulMod {u a b : K[X]} (h : a * u + b * f = 1) (g : K[X]) :
    mulMod f a (mulMod f u g) = g := by
  have h1 : (a * mulMod f u g) % f = g % f := by
    rw [mul_mod, mulMod_mod hf, ← mul_mod,
      show a * (u * g) = g - f * (b * g) by linear_combination g * h, sub_mod,
      EuclideanDomain.mod_eq_zero.mpr (dvd_mul_right f (b * g)), sub_zero]
  calc mulMod f a (mulMod f u g)
      = mulMod f u g + ((a * mulMod f u g) % f - mulMod f u g % f) := rfl
    _ = g := by rw [h1, mulMod_mod hf, mulMod]; ring

theorem degree_mod_sub_mod_lt_degree {g : K[X]} (hg : g ≠ 0) (hfg : f.natDegree ≤ g.natDegree)
    (u : K[X]) : degree ((u * g) % f - g % f) < degree g :=
  (degree_mod_sub_mod_lt hf u g).trans_le
    ((degree_le_of_natDegree_le hfg).trans_eq (degree_eq_natDegree hg).symm)

theorem monic_mulMod {g : K[X]} (hg : g.Monic) (hfg : f.natDegree ≤ g.natDegree) (u : K[X]) :
    (mulMod f u g).Monic :=
  hg.add_of_left (degree_mod_sub_mod_lt_degree hf hg.ne_zero hfg u)

theorem natDegree_mulMod {g : K[X]} (hg : g ≠ 0) (hfg : f.natDegree ≤ g.natDegree) (u : K[X]) :
    (mulMod f u g).natDegree = g.natDegree :=
  natDegree_add_eq_left_of_degree_lt (degree_mod_sub_mod_lt_degree hf hg hfg u)

/-- `λ^N Res(f, g') = Res(f, u) Res(f, g)` for `g' = mulMod f u g`, `λ` the leading coefficient
of `f`, with the formal degrees `deg f`, `N ≥ deg u` and `deg g ≥ deg f`. -/
theorem resultant_mulMod {u : K[X]} {N : ℕ} (hN : u.natDegree ≤ N) {g : K[X]}
    (hfg : f.natDegree ≤ g.natDegree) :
    f.leadingCoeff ^ N * resultant f (mulMod f u g) f.natDegree g.natDegree =
      resultant f u f.natDegree N * resultant f g f.natDegree g.natDegree := by
  have hdiv := natDegree_div_le_sub hf
  have hmod (p : K[X]) : (p % f).natDegree ≤ g.natDegree :=
    natDegree_le_of_degree_le ((degree_mod_lt p hf).le.trans
      ((degree_le_natDegree).trans (by exact_mod_cast hfg)))
  have hug : (u * g).natDegree ≤ u.natDegree + g.natDegree := natDegree_mul_le
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hN
  have h1 : resultant f (u * g) f.natDegree (u.natDegree + k + g.natDegree) =
      resultant f u f.natDegree (u.natDegree + k) * resultant f g f.natDegree g.natDegree := by
    rw [show u.natDegree + k + g.natDegree = u.natDegree + g.natDegree + k by ring,
      resultant_add_right_deg _ _ _ _ _ hug, resultant_mul_right _ _ _ _ le_rfl,
      resultant_add_right_deg _ _ _ _ _ le_rfl]
    ring
  have h2 : resultant f (u * g) f.natDegree (u.natDegree + k + g.natDegree) =
      f.leadingCoeff ^ (u.natDegree + k) *
        resultant f ((u * g) % f) f.natDegree g.natDegree := by
    have hq := hdiv (u * g)
    conv_lhs => rw [← EuclideanDomain.mod_add_div (u * g) f]
    rw [resultant_add_mul_right _ _ _ _ _ (by omega) le_rfl, add_comm _ g.natDegree,
      resultant_add_right_deg _ _ _ _ _ (hmod _), coeff_natDegree]
  have h3 : mulMod f u g = (u * g) % f + f * (g / f) := by
    rw [mulMod]
    linear_combination -(EuclideanDomain.mod_add_div g f)
  rw [← h1, h2, h3, resultant_add_mul_right _ _ _ _ _ (by have := hdiv g; omega) le_rfl]

end MulMod

example := mod_mod_self (K := ℚ) X_ne_zero X

example := degree_mod_sub_mod_lt (K := ℚ) X_ne_zero 2 X

example := mulMod_mod (K := ℚ) X_ne_zero 2 X

example := mulMod_mulMod (K := ℚ) X_ne_zero (u := C 2) (a := C 2⁻¹) (b := 0)
  (by rw [zero_mul, add_zero, ← C_mul, inv_mul_cancel₀ two_ne_zero, C_1]) X

example := degree_mod_sub_mod_lt_degree (K := ℚ) X_ne_zero X_ne_zero le_rfl 2

example := monic_mulMod (K := ℚ) X_ne_zero monic_X le_rfl 2

example := natDegree_mulMod (K := ℚ) X_ne_zero X_ne_zero le_rfl 2

example := resultant_mulMod (K := ℚ) X_ne_zero (u := 2) (N := 0) (by simp) (g := X) le_rfl

/-- **`L` is a polynomial of degree `< m` through a unit modulo `f`**: if `u` is prime to
`f ≠ 0`, `deg u ≤ N` and `χ(Res(f, u)) ≠ χ(λ)^N`, `λ` the leading coefficient of `f` and
`m = deg f`, then `c_n = 0` for every `n ≥ m`. -/
theorem lCoeff_mulWeight_eq_zero_of_isCoprime [Fintype K] {χ : MulChar K ℂ} {f u : K[X]}
    (hf : f ≠ 0) (hu : IsCoprime u f) {N : ℕ} (hN : u.natDegree ≤ N)
    (hχ : χ (resultant f u f.natDegree N) ≠ χ f.leadingCoeff ^ N) {n : ℕ}
    (hn : f.natDegree ≤ n) : lCoeff (mulWeight χ f) n = 0 := by
  obtain ⟨a, b, hab⟩ := hu
  have hmem (v g : K[X]) (hg : g ∈ monics K n) : mulMod f v g ∈ monics K n := by
    obtain ⟨hgm, hgn⟩ := mem_monics.mp hg
    exact mem_monics.mpr ⟨monic_mulMod hf hgm (hgn ▸ hn) v,
      (natDegree_mulMod hf hgm.ne_zero (hgn ▸ hn) v).trans hgn⟩
  have hw (g : K[X]) (hg : g ∈ monics K n) :
      χ f.leadingCoeff ^ N * mulWeight χ f (mulMod f u g) =
        χ (resultant f u f.natDegree N) * mulWeight χ f g := by
    obtain ⟨hgm, hgn⟩ := mem_monics.mp hg
    have hR := resultant_mulMod hf hN (hgn ▸ hn : f.natDegree ≤ g.natDegree)
    rw [mulWeight, mulWeight, natDegree_mulMod hf hgm.ne_zero (hgn ▸ hn) u, resultant_comm,
      resultant_comm g, ← map_pow, ← map_mul, ← map_mul]
    congr 1
    linear_combination (-1) ^ (g.natDegree * f.natDegree) * hR
  have key : χ f.leadingCoeff ^ N * lCoeff (mulWeight χ f) n =
      χ (resultant f u f.natDegree N) * lCoeff (mulWeight χ f) n := by
    rw [lCoeff, mul_sum, mul_sum, ← sum_congr rfl hw]
    exact (sum_nbij' (mulMod f u) (mulMod f a) (hmem u) (hmem a)
      (fun g _ => mulMod_mulMod hf hab g)
      (fun g _ => mulMod_mulMod hf (b := b) (by linear_combination hab) g) fun _ _ => rfl).symm
  have h1 : (χ f.leadingCoeff ^ N - χ (resultant f u f.natDegree N)) *
      lCoeff (mulWeight χ f) n = 0 := by
    rw [sub_mul, key, sub_self]
  exact (mul_eq_zero.mp h1).resolve_left (sub_ne_zero.mpr (Ne.symm hχ))

/-- For the Legendre symbol of `𝔽₃`, `f = X` and the unit `u = t`, `t` a non-square:
`c_1 = 0`. -/
example : lCoeff (mulWeight ((quadraticChar (ZMod 3)).ringHomComp (Int.castRingHom ℂ))
    (X : (ZMod 3)[X])) 1 = 0 := by
  obtain ⟨t, ht⟩ := quadraticChar_exists_neg_one (F := ZMod 3)
    (by rw [ZMod.ringChar_zmod_n]; norm_num)
  have ht0 : t ≠ 0 := fun h => by rw [h, quadraticChar_zero] at ht; exact absurd ht (by norm_num)
  refine lCoeff_mulWeight_eq_zero_of_isCoprime X_ne_zero (u := C t)
    ⟨C t⁻¹, 0, by rw [zero_mul, add_zero, ← C_mul, inv_mul_cancel₀ ht0, C_1]⟩ (N := 0)
    (by rw [natDegree_C]) ?_ (by rw [natDegree_X])
  rw [natDegree_X, resultant_C_right, MulChar.ringHomComp_apply, pow_zero, pow_zero, pow_one,
    one_mul, ht]
  norm_num

end Transformer.Weil

module

public import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic
public import Mathlib.RingTheory.Polynomial.Resultant.Basic
public import Transformer.Weil.LFunction

/-!
# The `L`-function of `χ(f(x))`

For a multiplicative character `χ` of the finite field `K` and `f ∈ K[X]` of degree `m`, the weight
`w(g) = χ(Res(g, f))` is multiplicative on the monic polynomials, since
`Res(g h, f) = Res(g, f) Res(h, f)`. For monic `g` with roots `β_i`, `Res(g, f) = ∏_i f(β_i)`.

If `χ(t)^m ≠ 1` for some `t ≠ 0`, then `c_n = 0` for every `n ≥ m`. Write `g = f q + r` with
`deg r < m`. The map `Φ_t(g) = g + (t − 1) r = f q + t r` permutes the monics of degree `n ≥ m`,
with inverse `Φ_{1/t}`, and `Res(f q + t r, f) = t^m Res(r, f) = t^m Res(g, f)`. So
`c_n = χ(t)^m c_n`. (W. M. Schmidt, *Equations over Finite Fields*, Ch. II.)
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K]

/-- The weight `w(g) = χ(Res(g, f))` of `χ(f(x))`. -/
noncomputable def mulWeight (χ : MulChar K ℂ) (f g : K[X]) : ℂ :=
  χ (resultant g f)

theorem isMonicMul_mulWeight (χ : MulChar K ℂ) (f : K[X]) : IsMonicMul (mulWeight χ f) where
  map_one := by
    rw [mulWeight, natDegree_one, resultant_one_left, zero_mul, pow_zero, pow_zero, one_mul,
      map_one]
  map_mul hg hh := by
    rw [mulWeight, mulWeight, mulWeight, hg.natDegree_mul hh,
      resultant_mul_left _ _ _ _ le_rfl, map_mul]

/-- `Φ_t(g) = g + (t − 1)(g mod f)`. -/
noncomputable def scaleMod (f : K[X]) (t : K) (g : K[X]) : K[X] :=
  g + (t - 1) • (g % f)

section Scale

variable {f : K[X]} (hf : f ≠ 0)

include hf

theorem degree_smul_mod_lt (s : K) (g : K[X]) : degree (s • (g % f)) < degree f :=
  (degree_smul_le _ _).trans_lt (degree_mod_lt g hf)

theorem scaleMod_mod (t : K) (g : K[X]) : scaleMod f t g % f = t • (g % f) := by
  rw [scaleMod, add_mod, (mod_eq_self_iff hf).mpr (degree_smul_mod_lt hf _ g), sub_smul, one_smul]
  abel

theorem scaleMod_scaleMod {t : K} (ht : t ≠ 0) (g : K[X]) :
    scaleMod f t⁻¹ (scaleMod f t g) = g := by
  conv_lhs => rw [scaleMod]
  rw [scaleMod_mod hf, smul_smul, scaleMod, add_assoc, ← add_smul,
    show t - 1 + (t⁻¹ - 1) * t = 0 by rw [sub_mul, inv_mul_cancel₀ ht, one_mul]; ring,
    zero_smul, add_zero]

theorem degree_scaleMod_sub_lt {g : K[X]} (hg : g ≠ 0) (hfg : f.natDegree ≤ g.natDegree)
    (t : K) : degree ((t - 1) • (g % f)) < degree g :=
  (degree_smul_mod_lt hf _ g).trans_le
    ((degree_le_of_natDegree_le hfg).trans_eq (degree_eq_natDegree hg).symm)

theorem monic_scaleMod {g : K[X]} (hg : g.Monic) (hfg : f.natDegree ≤ g.natDegree) (t : K) :
    (scaleMod f t g).Monic :=
  hg.add_of_left (degree_scaleMod_sub_lt hf hg.ne_zero hfg t)

theorem natDegree_scaleMod {g : K[X]} (hg : g ≠ 0) (hfg : f.natDegree ≤ g.natDegree) (t : K) :
    (scaleMod f t g).natDegree = g.natDegree :=
  natDegree_add_eq_left_of_degree_lt (degree_scaleMod_sub_lt hf hg hfg t)

/-- `Res(Φ_t g, f) = t^m Res(g, f)`, with the formal degrees `n ≥ deg g` and `m = deg f ≤ n`. -/
theorem resultant_scaleMod (t : K) {g : K[X]} {n : ℕ} (hg : g.natDegree ≤ n)
    (hfn : f.natDegree ≤ n) :
    resultant (scaleMod f t g) f n f.natDegree = t ^ f.natDegree * resultant g f n f.natDegree := by
  have hq : (g / f).natDegree + f.natDegree ≤ n := by
    have h1 : (g / f).natDegree ≤ g.natDegree - f.natDegree := by
      rw [Polynomial.div_def]
      refine (natDegree_C_mul_le _ _).trans ?_
      rw [natDegree_divByMonic _ (monic_mul_leadingCoeff_inv hf),
        natDegree_mul_C (inv_ne_zero (leadingCoeff_ne_zero.mpr hf))]
    omega
  have hgqr := EuclideanDomain.div_add_mod g f
  have hΦ : scaleMod f t g = C t * (g % f) + f * (g / f) := by
    rw [scaleMod, smul_eq_C_mul, C_sub, C_1]
    linear_combination -hgqr
  have h0 : resultant g f n f.natDegree = resultant (g % f) f n f.natDegree := by
    conv_lhs => rw [← hgqr, add_comm]
    exact resultant_add_mul_left _ _ _ _ _ hq le_rfl
  rw [hΦ, resultant_add_mul_left _ _ _ _ _ hq le_rfl, resultant_C_mul_left, h0]

end Scale

example := degree_smul_mod_lt (K := ℚ) X_ne_zero 2 X

example := scaleMod_mod (K := ℚ) X_ne_zero 2 X

example := scaleMod_scaleMod (K := ℚ) X_ne_zero two_ne_zero X

example := degree_scaleMod_sub_lt (K := ℚ) X_ne_zero X_ne_zero le_rfl 2

example := monic_scaleMod (K := ℚ) X_ne_zero monic_X le_rfl 2

example := natDegree_scaleMod (K := ℚ) X_ne_zero X_ne_zero le_rfl 2

example := resultant_scaleMod (K := ℚ) X_ne_zero 2 (g := X) natDegree_X.le natDegree_X.le

/-- **`L` is a polynomial of degree `< m`** for `χ(f(x))`: if `χ(t)^m ≠ 1` for some `t ≠ 0`,
`m = deg f`, then `c_n = 0` for every `n ≥ m`. -/
theorem lCoeff_mulWeight_eq_zero [Fintype K] {χ : MulChar K ℂ} {f : K[X]} {t : K} (ht0 : t ≠ 0)
    (ht : χ t ^ f.natDegree ≠ 1) {n : ℕ} (hn : f.natDegree ≤ n) :
    lCoeff (mulWeight χ f) n = 0 := by
  have hf : f ≠ 0 := by
    rintro rfl
    exact ht (by rw [natDegree_zero, pow_zero])
  have hmem (s : K) (g : K[X]) (hg : g ∈ monics K n) : scaleMod f s g ∈ monics K n := by
    obtain ⟨hgm, hgn⟩ := mem_monics.mp hg
    exact mem_monics.mpr ⟨monic_scaleMod hf hgm (hgn ▸ hn) s,
      (natDegree_scaleMod hf hgm.ne_zero (hgn ▸ hn) s).trans hgn⟩
  have hw (g : K[X]) (hg : g ∈ monics K n) :
      mulWeight χ f (scaleMod f t g) = χ t ^ f.natDegree * mulWeight χ f g := by
    obtain ⟨hgm, hgn⟩ := mem_monics.mp hg
    rw [mulWeight, mulWeight, natDegree_scaleMod hf hgm.ne_zero (hgn ▸ hn) t,
      resultant_scaleMod hf t le_rfl (hgn ▸ hn), map_mul, map_pow]
  have key : lCoeff (mulWeight χ f) n = χ t ^ f.natDegree * lCoeff (mulWeight χ f) n := by
    rw [lCoeff, mul_sum, ← sum_congr rfl hw]
    exact (sum_nbij' (scaleMod f t) (scaleMod f t⁻¹) (hmem t) (hmem t⁻¹)
      (fun g _ => scaleMod_scaleMod hf ht0 g)
      (fun g _ => by simpa only [inv_inv] using scaleMod_scaleMod hf (inv_ne_zero ht0) g)
      fun _ _ => rfl).symm
  have h1 : (1 - χ t ^ f.natDegree) * lCoeff (mulWeight χ f) n = 0 := by
    rw [sub_mul, one_mul, ← key, sub_self]
  exact (mul_eq_zero.mp h1).resolve_left (sub_ne_zero.mpr (Ne.symm ht))

/-- For the Legendre symbol of `𝔽₃` and `f = X`, `c_1 = 0`. -/
example : lCoeff (mulWeight ((quadraticChar (ZMod 3)).ringHomComp (Int.castRingHom ℂ))
    (X : (ZMod 3)[X])) 1 = 0 := by
  obtain ⟨t, ht⟩ := quadraticChar_exists_neg_one (F := ZMod 3)
    (by rw [ZMod.ringChar_zmod_n]; norm_num)
  refine lCoeff_mulWeight_eq_zero (t := t) (fun h => ?_) ?_ (by rw [natDegree_X])
  · rw [h, quadraticChar_zero] at ht
    exact absurd ht (by norm_num)
  · rw [natDegree_X, pow_one, MulChar.ringHomComp_apply, ht]
    norm_num

end Transformer.Weil

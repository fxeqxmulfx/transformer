module

public import Mathlib.Algebra.Group.AddChar
public import Mathlib.Analysis.Complex.Basic
public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Transformer.Weil.RootSum

/-!
# The coefficients of the `L`-function of `y^p − y = f(x)`

Over a finite field `K`, every monic `g` of degree `n` is `X^n + ∑_{i < n} b_i X^i` for exactly
one `b ∈ K^n`. For an additive character `ψ` of `K` and `f ∈ K[X]`, the character
`λ(g) = ψ(∑_i f(β_i))` of the monic `g` is multiplicative, and the coefficients of
`L(T) = ∑_g λ(g) T^{deg g}` are `c_n = ∑_{deg g = n} λ(g)`, with `c_0 = 1`.

For `ψ ≠ 1` and `m = deg f ≥ 1` prime to the characteristic, `c_n = 0` for every `n ≥ m`: the
translation `b_{n − m} ↦ b_{n − m} + t` permutes the monics of degree `n` and multiplies `λ` by
`ψ(−m f_m t)`, which is not `1` for a suitable `t`. So `L` is a polynomial of degree `< m`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K]

/-- The monic polynomial `X^n + ∑_{i < n} b_i X^i`. -/
noncomputable def monicOfCoeffs (n : ℕ) (b : Fin n → K) : K[X] :=
  X ^ n + ∑ i : Fin n, C (b i) * X ^ (i : ℕ)

theorem degree_sum_lt_degree_X_pow (n : ℕ) (b : Fin n → K) :
    degree (∑ i : Fin n, C (b i) * X ^ (i : ℕ)) < degree (X ^ n : K[X]) :=
  (degree_sum_fin_lt b).trans_le (degree_X_pow n).ge

theorem monicOfCoeffs_monic (n : ℕ) (b : Fin n → K) : (monicOfCoeffs n b).Monic :=
  (monic_X_pow n).add_of_left (degree_sum_lt_degree_X_pow n b)

theorem natDegree_monicOfCoeffs (n : ℕ) (b : Fin n → K) : (monicOfCoeffs n b).natDegree = n := by
  rw [monicOfCoeffs, natDegree_add_eq_left_of_degree_lt (degree_sum_lt_degree_X_pow n b),
    natDegree_X_pow]

theorem coeff_monicOfCoeffs (n : ℕ) (b : Fin n → K) (i : Fin n) :
    (monicOfCoeffs n b).coeff i = b i := by
  simp [monicOfCoeffs, coeff_X_pow, i.2.ne, Fin.val_inj]

theorem monicOfCoeffs_injective (n : ℕ) : Function.Injective (monicOfCoeffs (K := K) n) :=
  fun b b' h => funext fun i => by rw [← coeff_monicOfCoeffs n b, h, coeff_monicOfCoeffs]

/-- Every monic `g` of degree `n` is `X^n + ∑_{i < n} g_i X^i`. -/
theorem monicOfCoeffs_coeff {g : K[X]} (hg : g.Monic) {n : ℕ} (hn : g.natDegree = n) :
    monicOfCoeffs n (fun i => g.coeff i) = g := by
  conv_rhs => rw [hg.as_sum, hn]
  rw [monicOfCoeffs, Fin.sum_univ_eq_sum_range (fun i => C (g.coeff i) * X ^ i)]

example := monicOfCoeffs_coeff (K := ℚ) monic_X natDegree_X

variable [Fintype K]

/-- The monic polynomials of degree `n` over the finite field `K`. -/
noncomputable def monics (K : Type*) [Field K] [Fintype K] (n : ℕ) : Finset K[X] :=
  univ.map ⟨monicOfCoeffs n, monicOfCoeffs_injective n⟩

theorem mem_monics {n : ℕ} {g : K[X]} : g ∈ monics K n ↔ g.Monic ∧ g.natDegree = n := by
  simp only [monics, mem_map, mem_univ, true_and, Function.Embedding.coeFn_mk]
  refine ⟨?_, fun h => ⟨_, monicOfCoeffs_coeff h.1 h.2⟩⟩
  rintro ⟨b, rfl⟩
  exact ⟨monicOfCoeffs_monic n b, natDegree_monicOfCoeffs n b⟩

/-- The coefficients `c_n = ∑_{g monic, deg g = n} ψ(∑_i f(β_i))` of the `L`-function. -/
noncomputable def lCoeff (ψ : AddChar K ℂ) (f : K[X]) (n : ℕ) : ℂ :=
  ∑ g ∈ monics K n, ψ (rootSum f g)

theorem lCoeff_eq_sum (ψ : AddChar K ℂ) (f : K[X]) (n : ℕ) :
    lCoeff ψ f n = ∑ b : Fin n → K, ψ (rootSum f (monicOfCoeffs n b)) := by
  rw [lCoeff, monics, sum_map]
  rfl

theorem lCoeff_zero (ψ : AddChar K ℂ) (f : K[X]) : lCoeff ψ f 0 = 1 := by
  rw [lCoeff_eq_sum, Fintype.sum_unique]
  simp [monicOfCoeffs, rootSum_one]

/-- **`L` is a polynomial of degree `< m`**: for `ψ ≠ 1` and `m = deg f ≥ 1` with `m ≠ 0` in
`K`, `c_n = 0` for every `n ≥ m`. -/
theorem lCoeff_eq_zero {ψ : AddChar K ℂ} (hψ : ψ ≠ 1) {f : K[X]} (hm0 : f.natDegree ≠ 0)
    (hm : (f.natDegree : K) ≠ 0) {n : ℕ} (hn : f.natDegree ≤ n) : lCoeff ψ f n = 0 := by
  set m := f.natDegree with hmdef
  obtain ⟨a, ha⟩ := AddChar.ne_one_iff.mp hψ
  have hlc : f.coeff m ≠ 0 := leadingCoeff_ne_zero.mpr fun h0 => by simp [h0, m] at hm0
  have hu : -(m * f.coeff m) ≠ 0 := neg_ne_zero.mpr (mul_ne_zero hm hlc)
  set t := a / -(m * f.coeff m)
  have hut : -(m * f.coeff m) * t = a := mul_div_cancel₀ a hu
  set s : Fin n → K := Pi.single ⟨n - m, by omega⟩ t with hsdef
  have hs : ∑ i : Fin n, C (s i) * X ^ (i : ℕ) = C t * X ^ (n - m) := by
    rw [hsdef, sum_eq_single (⟨n - m, by omega⟩ : Fin n)
      (fun i _ hi => by rw [Pi.single_eq_of_ne hi, C_0, zero_mul]) (fun h => absurd (mem_univ _) h),
      Pi.single_eq_same]
  have hshift (b : Fin n → K) :
      ψ (rootSum f (monicOfCoeffs n (b + s))) = ψ a * ψ (rootSum f (monicOfCoeffs n b)) := by
    have hb : monicOfCoeffs n (b + s) = monicOfCoeffs n b + C t * X ^ (n - m) := by
      simp only [monicOfCoeffs, Pi.add_apply, C_add, add_mul, sum_add_distrib, hs]
      ring
    rw [hb, rootSum_add_C_mul_X_pow (monicOfCoeffs_monic n b) hmdef.symm hm0 hn
      (natDegree_monicOfCoeffs n b) t, mul_comm (ψ a), ← AddChar.map_add_eq_mul]
    congr 1
    rw [← hut]
    ring
  have key : lCoeff ψ f n = ψ a * lCoeff ψ f n := by
    rw [lCoeff_eq_sum, mul_sum]
    calc ∑ b : Fin n → K, ψ (rootSum f (monicOfCoeffs n b))
        = ∑ b : Fin n → K, ψ (rootSum f (monicOfCoeffs n (b + s))) :=
          (Fintype.sum_equiv (Equiv.addRight s) _ _ fun b => rfl).symm
      _ = _ := sum_congr rfl fun b _ => hshift b
  have h1 : (1 - ψ a) * lCoeff ψ f n = 0 := by rw [sub_mul, one_mul, ← key, sub_self]
  exact (mul_eq_zero.mp h1).resolve_left (sub_ne_zero.mpr (Ne.symm ha))

example := lCoeff_eq_zero (K := ZMod 2) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero
    (ZMod.injective_stdAddChar (h.trans (AddChar.map_zero_eq_one _).symm))⟩)
  (f := X) (by rw [natDegree_X]; exact one_ne_zero)
  (by rw [natDegree_X, Nat.cast_one]; exact one_ne_zero) (n := 1) (by rw [natDegree_X])

end Transformer.Weil

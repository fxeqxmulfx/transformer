module

public import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic
public import Mathlib.NumberTheory.MulChar.Lemmas
public import Mathlib.RingTheory.RootsOfUnity.Complex
public import Transformer.Weil.PowBound

/-!
# The character sum `∑_x χ(f(x))` for `y^d = f(x)`

Let `χ` be a multiplicative character of `F`, `|F| = q`, with `χ^d = 1` for some `d ≥ 2` and
`χ(u) = 1 ⇒ u^c = 1`, `d c + 1 = q`. For `f(x) ≠ 0`, `χ(f(x))` is a power `μ^k`, `k < d`, of
`μ = e^{2πi/d}`; and `χ(f(x)) = χ(f(x₀))` gives `χ(f(x)/f(x₀)) = 1`, so `f(x)^c = f(x₀)^c`.
So each fiber of `x ↦ χ(f(x))` lies in a level set `{f^c = a}`, `a ≠ 0`, of at most
`E = q/d + 4 m √q` points (`PowBound`). As `∑_{k < d} μ^k = 0`,
`∑_{f(x) ≠ 0} χ(f(x)) = ∑_k (n_k − E) μ^k` with `0 ≤ E − n_k`, so
`‖∑_x χ(f(x))‖ ≤ d E − (q − m) ≤ 5 d m √q`: the at most `m` roots of `f` give `χ(0) = 0`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

open Classical in
/-- **From fibers to the sum**: if `T` maps `s` into `Z ⊆ ℂ` with `∑_{z ∈ Z} z = 0` and
`‖z‖ ≤ 1` on `Z`, and every fiber `{x ∈ s : T x = z}`, `z ∈ Z`, has at most `E` points, then
`‖∑_{x ∈ s} T x‖ ≤ |Z| E − |s|`, as `∑_{x ∈ s} T x = ∑_{z ∈ Z} (n_z − E) z`. -/
theorem norm_sum_le_of_card_fiber_le {α : Type*} {s : Finset α} {T : α → ℂ} {Z : Finset ℂ}
    (hT : ∀ x ∈ s, T x ∈ Z) (hZ : ∑ z ∈ Z, z = 0) (hZ1 : ∀ z ∈ Z, ‖z‖ ≤ 1) {E : ℝ}
    (hE : ∀ z ∈ Z, (#{x ∈ s | T x = z} : ℝ) ≤ E) :
    ‖∑ x ∈ s, T x‖ ≤ #Z * E - #s := by
  have h1 : ∑ x ∈ s, T x = ∑ z ∈ Z, ((((#{x ∈ s | T x = z} : ℕ) : ℝ) - E : ℝ) : ℂ) * z := by
    rw [← sum_fiberwise_of_maps_to' hT fun z => z]
    simp only [sum_const, nsmul_eq_mul, Complex.ofReal_sub, Complex.ofReal_natCast, sub_mul,
      sum_sub_distrib]
    rw [← mul_sum, hZ, mul_zero, sub_zero]
  have h2 : ∑ z ∈ Z, ((#{x ∈ s | T x = z} : ℕ) : ℝ) = #s := by
    rw [card_eq_sum_card_fiberwise fun x hx => hT x hx, Nat.cast_sum]
  rw [h1]
  calc ‖∑ z ∈ Z, ((((#{x ∈ s | T x = z} : ℕ) : ℝ) - E : ℝ) : ℂ) * z‖
      ≤ ∑ z ∈ Z, ‖((((#{x ∈ s | T x = z} : ℕ) : ℝ) - E : ℝ) : ℂ) * z‖ := norm_sum_le _ _
    _ ≤ ∑ z ∈ Z, (E - ((#{x ∈ s | T x = z} : ℕ) : ℝ)) := by
      refine sum_le_sum fun z hz => ?_
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonpos (sub_nonpos.mpr (hE z hz)), neg_sub]
      exact mul_le_of_le_one_right (sub_nonneg.mpr (hE z hz)) (hZ1 z hz)
    _ = #Z * E - #s := by rw [sum_sub_distrib, sum_const, nsmul_eq_mul, h2]

example := norm_sum_le_of_card_fiber_le (s := (∅ : Finset ℕ)) (T := fun _ => 0) (Z := ∅)
  (by simp) (by simp) (by simp) (E := 0) (by simp)

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- **The character sum for `y^d = f(x)`**: for a multiplicative character `χ` of `F` with
`χ^d = 1`, `d ≥ 2`, and `χ(u) = 1 ⇒ u^c = 1`, `d c + 1 = q = |F|`, and `f` of degree `m ≥ 1`
prime to `d`, with `q ≥ 64`, `8 d m ≤ q` and `d² ≤ q`: `‖∑_x χ(f(x))‖ ≤ 5 d m √q`. -/
theorem norm_sum_mulChar_eval_le_of_coprime {χ : MulChar F ℂ} {d c : ℕ} (hd : 2 ≤ d)
    (hχ : χ ^ d = 1) (hker : ∀ u, χ u = 1 → u ^ c = 1) (hq : d * c + 1 = Fintype.card F)
    {f : F[X]} (hm : 1 ≤ f.natDegree) (hdm : Nat.Coprime d f.natDegree)
    (h64 : 64 ≤ Fintype.card F) (h8 : 8 * d * f.natDegree ≤ Fintype.card F)
    (hd2 : d * d ≤ Fintype.card F) :
    ‖∑ x, χ (f.eval x)‖ ≤ 5 * d * f.natDegree * √(Fintype.card F) := by
  classical
  have : NeZero d := ⟨by omega⟩
  have hf0 : f ≠ 0 := ne_zero_of_natDegree_gt hm
  obtain ⟨μ, hμ⟩ : ∃ μ : ℂ, IsPrimitiveRoot μ d := ⟨_, Complex.isPrimitiveRoot_exp d (by omega)⟩
  have hsum : ∑ x, χ (f.eval x) = ∑ x with f.eval x ≠ 0, χ (f.eval x) := by
    rw [sum_filter]
    refine sum_congr rfl fun x _ => ?_
    split_ifs with h
    · rfl
    · rw [not_not.mp h, MulChar.map_zero]
  have hT : ∀ x ∈ ({x | f.eval x ≠ 0} : Finset F), χ (f.eval x) ∈ (range d).image (μ ^ ·) :=
    fun x hx => by
      obtain ⟨k, hk, hχk⟩ := MulChar.exists_apply_eq_pow hχ hμ (mem_filter.mp hx).2
      exact mem_image.mpr ⟨k, mem_range.mpr hk, hχk.symm⟩
  have hZ : ∑ z ∈ (range d).image (μ ^ ·), z = 0 := by
    rw [sum_image fun i hi j hj h => hμ.pow_inj (mem_range.mp hi) (mem_range.mp hj) h]
    exact hμ.geom_sum_eq_zero (by omega)
  have hZ1 : ∀ z ∈ (range d).image (μ ^ ·), ‖z‖ ≤ 1 := fun z hz => by
    obtain ⟨k, _, rfl⟩ := mem_image.mp hz
    rw [norm_pow, hμ.norm'_eq_one (by omega), one_pow]
  have hb := norm_sum_le_of_card_fiber_le hT hZ hZ1
    (E := Fintype.card F / d + 4 * f.natDegree * √(Fintype.card F)) fun z _ => by
      by_cases hne : ∃ x₀ ∈ ({x | f.eval x ≠ 0} : Finset F), χ (f.eval x₀) = z
      · obtain ⟨x₀, hx₀, rfl⟩ := hne
        have hv : f.eval x₀ ≠ 0 := (mem_filter.mp hx₀).2
        refine le_trans (Nat.cast_le.mpr (card_le_card fun x hx => ?_))
          (card_pow_eq_le_of_coprime hm hq hdm h64 h8 hd2 (pow_ne_zero c hv))
        obtain ⟨-, hxz⟩ := mem_filter.mp hx
        refine mem_filter.mpr ⟨mem_univ _, ?_⟩
        have h1 : χ (f.eval x * (f.eval x₀)⁻¹) = 1 := by
          rw [map_mul, hxz, ← map_mul, mul_inv_cancel₀ hv, map_one]
        have h2 := hker _ h1
        rwa [mul_pow, inv_pow, mul_inv_eq_one₀ (pow_ne_zero c hv)] at h2
      · push Not at hne
        rw [filter_false_of_mem hne, card_empty, Nat.cast_zero]
        positivity
  have hroots : #{x | ¬f.eval x ≠ 0} ≤ f.natDegree :=
    card_le_degree_of_subset_roots fun x hx =>
      (mem_roots hf0).mpr (not_not.mp (mem_filter.mp hx).2)
  have hpart := card_filter_add_card_filter_not (s := (univ : Finset F)) fun x => f.eval x ≠ 0
  rw [card_univ] at hpart
  have hZd : (#((range d).image (μ ^ ·)) : ℝ) ≤ d := by
    exact_mod_cast card_image_le.trans (card_range d).le
  have hpartR : (#{x | f.eval x ≠ 0} : ℝ) + #{x | ¬f.eval x ≠ 0} = Fintype.card F := by
    exact_mod_cast hpart
  have hrootsR : (#{x | ¬f.eval x ≠ 0} : ℝ) ≤ f.natDegree := by exact_mod_cast hroots
  have hd1 : (1 : ℝ) ≤ d := by exact_mod_cast (by omega : 1 ≤ d)
  have hs1 : (1 : ℝ) ≤ √(Fintype.card F) :=
    Real.one_le_sqrt.mpr (by exact_mod_cast (by omega : 1 ≤ Fintype.card F))
  have hE0 : (0 : ℝ) ≤ Fintype.card F / d + 4 * f.natDegree * √(Fintype.card F) := by positivity
  have hdE : (d : ℝ) * (Fintype.card F / d + 4 * f.natDegree * √(Fintype.card F)) =
      Fintype.card F + 4 * d * f.natDegree * √(Fintype.card F) := by
    field_simp
  have P1 := mul_le_mul_of_nonneg_right hZd hE0
  have P2 := mul_le_mul_of_nonneg_left (one_le_mul_of_one_le_of_one_le hd1 hs1)
    (Nat.cast_nonneg f.natDegree)
  rw [hsum]
  linarith

section Examples

attribute [local instance] fact_prime_71

example := norm_sum_mulChar_eval_le_of_coprime (F := ZMod 71)
  (χ := (quadraticChar (ZMod 71)).ringHomComp (Int.castRingHom ℂ)) (d := 2) (c := 35) le_rfl
  ((quadraticChar_isQuadratic _).comp _).sq_eq_one
  (fun u hu => by
    rw [MulChar.ringHomComp_apply, eq_intCast, Int.cast_eq_one] at hu
    have h := quadraticChar_eq_pow_of_char_ne_two' (F := ZMod 71)
      (by rw [ZMod.ringChar_zmod_n]; decide) u
    rw [hu, ZMod.card] at h
    simpa using h.symm)
  (by rw [ZMod.card]) (f := X) (by simp) (by simp) (by simp) (by simp) (by simp)

end Examples

end Transformer.Weil

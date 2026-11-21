module

public import Transformer.Grokking.FlatSpectrum.FlatSupport
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Div

/-!
# PT6: M does not return the flat tables `(a / b)^k`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. For `k ≠ 0` coprime
to `p − 1`, the map `x ↦ x^k` is a bijection of `ℤ/p` (`pow_injective_of_coprime`). So two columns
`b ≠ b'` of the kernel of `(a / b)^k` have the inner product `∑_x ψ(x (b'^{−k} − b^{−k})) = 0`, and
the kernel is flat with `c = p` although the table is outside the class (`isFlat_divKernel`). M
returns it from no proper part (`not_returned_div`). Seven tasks of the arithmetic data are of this
kind: `k = 1, 3, 5` at `p = 17` and `k = 1, 5` at `p = 97` and `p = 193` (`not_returned_div17`,
`not_returned_div97`, `not_returned_div193`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97 fact_prime_17 fact_prime_193

variable {p : ℕ} [Fact p.Prime]

/-- The kernel of `(a / b)^k` on the table without the column `b = 0`. -/
noncomputable def divKernel (p : ℕ) [Fact p.Prime] (k : ℕ) :
    Matrix (Fin 1 → ZMod p) {v : Fin 1 → ZMod p // v ≠ 0} ℂ :=
  Matrix.of fun u v => (ZMod.stdAddChar (divTable p k (u, v.1)) : ℂ)

/-- The cells of the kernel in the part with the mask `T`. -/
def divSeen (p T : ℕ) : Set ((Fin 1 → ZMod p) × {v : Fin 1 → ZMod p // v ≠ 0}) :=
  {c | (c.1, c.2.1) ∈ divPart p T}

/-- For `k ≠ 0` coprime to `p − 1`, `x ↦ x^k` is injective on `ℤ/p`. -/
lemma pow_injective_of_coprime {k : ℕ} (hk : k ≠ 0) (hcop : k.Coprime (p - 1)) :
    Function.Injective fun x : ZMod p => x ^ k := by
  intro x y hxy
  simp only at hxy
  by_cases hy : y = 0
  · subst hy
    rw [zero_pow hk] at hxy
    exact (pow_eq_zero_iff hk).1 hxy
  · have hx : x ≠ 0 := by
      rintro rfl
      rw [zero_pow hk] at hxy
      exact hy ((pow_eq_zero_iff hk).1 hxy.symm)
    have h1 : (x / y) ^ k = 1 := by rw [div_pow, hxy, div_self (pow_ne_zero _ hy)]
    have h2 : (x / y) ^ (p - 1) = 1 := ZMod.pow_card_sub_one_eq_one (div_ne_zero hx hy)
    have h3 : (x / y) ^ Nat.gcd k (p - 1) = 1 := pow_gcd_eq_one.2 ⟨h1, h2⟩
    rw [Nat.Coprime.gcd_eq_one hcop, pow_one] at h3
    exact (div_eq_one_iff_eq hy).1 h3

example : Function.Injective fun x : ZMod 17 => x ^ 3 :=
  pow_injective_of_coprime (by norm_num) (by norm_num)

/-- A sum over the lines is a sum over `ℤ/p`. -/
lemma sum_line_apply (F : ZMod p → ℂ) : ∑ u : Fin 1 → ZMod p, F (u 0) = ∑ x, F x :=
  (Equiv.funUnique (Fin 1) (ZMod p)).sum_comp F

/-- **The columns of the kernel of `(a / b)^k` are orthogonal** for `k ≠ 0` coprime to `p − 1`. -/
theorem divKernel_conjTranspose_mul {k : ℕ} (hk : k ≠ 0) (hcop : k.Coprime (p - 1)) :
    (divKernel p k)ᴴ * divKernel p k = (p : ℂ) • 1 := by
  have hinj := pow_injective_of_coprime (p := p) hk hcop
  have hbij : Function.Bijective fun x : ZMod p => x ^ k := Finite.injective_iff_bijective.1 hinj
  ext v v'
  simp only [Matrix.mul_apply, conjTranspose_apply, divKernel, of_apply, divTable,
    star_stdAddChar_mul, Matrix.smul_apply, smul_eq_mul]
  set d : ZMod p := (v'.1 0)⁻¹ ^ k - (v.1 0)⁻¹ ^ k with hd
  have hterm (u : Fin 1 → ZMod p) :
      (u 0 / v'.1 0) ^ k - (u 0 / v.1 0) ^ k = u 0 ^ k * d := by
    rw [hd, div_eq_mul_inv, div_eq_mul_inv, mul_pow, mul_pow]
    ring
  simp only [hterm]
  rw [sum_line_apply (fun x => (ZMod.stdAddChar (x ^ k * d) : ℂ)),
    hbij.sum_comp (fun y => (ZMod.stdAddChar (y * d) : ℂ)), sum_stdAddChar_mul]
  by_cases hvv : v = v'
  · subst hvv
    simp [hd]
  · have hne : v.1 0 ≠ v'.1 0 := fun h => hvv (Subtype.ext (funext fun i => by
      rwa [Subsingleton.elim i 0]))
    have hd0 : d ≠ 0 := by
      rw [hd, sub_ne_zero]
      intro h
      exact hne (inv_injective (hinj h)).symm
    simp [hd0, Matrix.one_apply_ne hvv]

/-- **The kernel of `(a / b)^k` is flat** for `k ≠ 0` coprime to `p − 1`, with `c = p`. -/
theorem isFlat_divKernel {k : ℕ} (hk : k ≠ 0) (hcop : k.Coprime (p - 1)) :
    IsFlat (divKernel p k) p :=
  Or.inl (by rw [divKernel_conjTranspose_mul hk hcop]; simp)

example : IsFlat (divKernel 17 3) 17 := by
  exact_mod_cast isFlat_divKernel (p := 17) (k := 3) (by norm_num) (by norm_num)

/-- **M does not return the kernel of `(a / b)^k` from a proper part**, for `k ≠ 0` coprime to
`p − 1` (`review/flat_spectrum.md`, PT6). -/
theorem not_returned_div {T k : ℕ} (hk : k ≠ 0) (hcop : k.Coprime (p - 1))
    (hΩ : divSeen p T ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen p T, X ij.1 ij.2 = divKernel p k ij.1 ij.2}
      (divKernel p k) :=
  (isFlat_divKernel hk hcop).not_isMinOn (Nat.cast_pos.2 (Fact.out : p.Prime).pos)
    (fun i j => norm_ne_zero_iff.mp (by simp [divKernel])) hΩ

omit [Fact p.Prime] in
/-- A clear bit of the mask is a cell outside the part. -/
lemma notMem_divPart {T a b : ℕ} (hp : 1 < p) (ha : a < p) (hb : b < p - 1)
    (h : T.testBit ((p - 1) * a + b) = false) : (line p a, line p (b + 1)) ∉ divPart p T := by
  intro hmem
  obtain ⟨c, hc, hcell⟩ := mem_image.1 hmem
  obtain ⟨hcr, hcT⟩ := mem_filter.1 hc
  have hq : 0 < p - 1 := by omega
  have hdiv : c / (p - 1) < p := Nat.div_lt_of_lt_mul (mem_range.1 hcr |>.trans_eq (Nat.mul_comm _ _))
  have hmod : c % (p - 1) < p - 1 := Nat.mod_lt _ hq
  have h1 : c / (p - 1) = a := by
    by_contra hne
    exact line_ne_line hdiv ha hne (congrArg Prod.fst hcell)
  have h2 : c % (p - 1) = b := by
    by_contra hne
    exact line_ne_line (a := c % (p - 1) + 1) (b := b + 1) (by omega) (by omega) (by omega)
      (congrArg Prod.snd hcell)
  have hcv : c = (p - 1) * a + b := by
    rw [← h1, ← h2, Nat.div_add_mod]
  rw [hcv, h] at hcT
  exact Bool.false_ne_true hcT

example : (line 3 0, line 3 (0 + 1)) ∉ divPart 3 0 :=
  notMem_divPart (by norm_num) (by norm_num) (by norm_num) (by simp)

/-- A clear bit of the mask leaves a cell of the kernel unseen. -/
lemma divSeen_ne_univ {T a b : ℕ} (hp : 1 < p) (ha : a < p) (hb : b < p - 1)
    (h : T.testBit ((p - 1) * a + b) = false) : divSeen p T ≠ Set.univ := by
  have hv : line p (b + 1) ≠ 0 := by
    have := line_ne_line (p := p) (a := b + 1) (b := 0) (by omega) (by omega) (by omega)
    simpa [line, funext_iff] using this
  intro hu
  have hmem : ((line p a, ⟨line p (b + 1), hv⟩) :
      (Fin 1 → ZMod p) × {v : Fin 1 → ZMod p // v ≠ 0}) ∈ divSeen p T := by
    rw [hu]
    trivial
  exact notMem_divPart hp ha hb h hmem

/-- **M does not return `(a / b)^k`, `k = 1, 3, 5`, from the training half at `p = 17`**
(`review/flat_spectrum.md`, PT6). -/
theorem not_returned_div17 {k : ℕ} (hk : k ∈ [1, 3, 5]) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 17 mask17, X ij.1 ij.2 = divKernel 17 k ij.1 ij.2}
      (divKernel 17 k) := by
  have hΩ : divSeen 17 mask17 ≠ Set.univ :=
    divSeen_ne_univ (a := 0) (b := 0) (by norm_num) (by norm_num) (by norm_num) (by decide +kernel)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl <;> exact not_returned_div (by norm_num) (by norm_num) hΩ

example := not_returned_div17 (k := 5) (by simp)

/-- **M does not return `(a / b)^k`, `k = 1, 5`, from the training half at `p = 97`**
(`review/flat_spectrum.md`, PT6). -/
theorem not_returned_div97 {k : ℕ} (hk : k ∈ [1, 5]) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 97 mask97, X ij.1 ij.2 = divKernel 97 k ij.1 ij.2}
      (divKernel 97 k) := by
  have hΩ : divSeen 97 mask97 ≠ Set.univ :=
    divSeen_ne_univ (a := 0) (b := 5) (by norm_num) (by norm_num) (by norm_num) (by decide +kernel)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl <;> exact not_returned_div (by norm_num) (by norm_num) hΩ

example := not_returned_div97 (k := 1) (by simp)

/-- **M does not return `(a / b)^k`, `k = 1, 5`, from the training half at `p = 193`**
(`review/flat_spectrum.md`, PT6). -/
theorem not_returned_div193 {k : ℕ} (hk : k ∈ [1, 5]) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 193 mask193, X ij.1 ij.2 = divKernel 193 k ij.1 ij.2}
      (divKernel 193 k) := by
  have hΩ : divSeen 193 mask193 ≠ Set.univ :=
    divSeen_ne_univ (a := 0) (b := 0) (by norm_num) (by norm_num) (by norm_num) (by decide +kernel)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl <;> exact not_returned_div (by norm_num) (by norm_num) hΩ

example := not_returned_div193 (k := 5) (by simp)

end Transformer.Grokking.FlatSpectrum.TablePart.E6

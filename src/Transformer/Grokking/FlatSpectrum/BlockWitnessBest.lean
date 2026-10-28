module

public import Transformer.Grokking.FlatSpectrum.BlockWitnessNorm

/-!
# The best block witness

Source: `review/flat_spectrum.md`, *Patterns*, for `X = J − t 1_R 1_Cᵀ`: "The best `t` gives
`‖X‖²_S1 = N² − (k l − g)²/(k l) < N²`."

Generalized to `m × n`. With `P = k l` and `g = √(k l (m − k)(n − l))`, the radicand of
`nuclearNorm_onesSubBlock` is `f(t) = m n − 2tP + t²P + 2|t| g`. For `t ≥ 0`,
`f(t) = m n − (P − g)²/P + P (t − t*)²` with `t* = 1 − g/P`, and `f(t) ≥ m n` for `t < 0`
(`radicand_ge`). `m n ≤ k n + l m` is `(m − k)(n − l) ≤ k l`, that is `g ≤ P`, so `t* ≥ 0`
reaches the bound (`radicand_best`), and `m n − (P − g)²/P = (√(k (n − l)) + √(l (m − k)))²`
(`sq_sqrt_add_sqrt`). So for nonempty `R`, `C` the least `‖X‖_S1` over real `t` is
`√(k (n − l)) + √(l (m − k))` (`isLeast_nuclearNorm_onesSubBlock`). When `k n + l m ≤ m n`,
`g ≥ P` and `f(t) ≥ m n`: the least is `‖J‖_S1 = √(m n)`, at `t = 0`
(`isLeast_nuclearNorm_onesSubBlock_of_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- For `0 < k l` and `0 ≤ g`: `A − 2t k l + t² k l + 2|t| g ≥ A − (k l − g)²/(k l)`. -/
lemma radicand_ge {A k l g : ℝ} (hP : 0 < k * l) (hg : 0 ≤ g) (t : ℝ) :
    A - (k * l - g) ^ 2 / (k * l) ≤ A - 2 * t * k * l + t ^ 2 * k * l + 2 * |t| * g := by
  have hq : 0 ≤ (k * l - g) ^ 2 / (k * l) := by positivity
  obtain ⟨hk, hl⟩ := mul_ne_zero_iff.mp hP.ne'
  rcases le_or_gt 0 t with ht | ht
  · rw [abs_of_nonneg ht]
    have h : A - 2 * t * k * l + t ^ 2 * k * l + 2 * t * g - (A - (k * l - g) ^ 2 / (k * l)) =
        k * l * (t - (1 - g / (k * l))) ^ 2 := by
      field_simp
      ring
    nlinarith [mul_nonneg hP.le (sq_nonneg (t - (1 - g / (k * l))))]
  · rw [abs_of_neg ht]
    nlinarith [mul_nonneg (neg_nonneg.mpr ht.le) hP.le, mul_nonneg (neg_nonneg.mpr ht.le) hg,
      mul_nonneg (sq_nonneg t) hP.le]

example := radicand_ge (A := 0) (k := 1) (l := 1) (g := 0) (by norm_num) le_rfl 0

/-- The bound of `radicand_ge` is reached at `t = 1 − g/(k l)`. -/
lemma radicand_best {A k l g : ℝ} (hP : 0 < k * l) (hgP : g ≤ k * l) :
    A - 2 * (1 - g / (k * l)) * k * l + (1 - g / (k * l)) ^ 2 * k * l +
      2 * |1 - g / (k * l)| * g = A - (k * l - g) ^ 2 / (k * l) := by
  obtain ⟨hk, hl⟩ := mul_ne_zero_iff.mp hP.ne'
  rw [abs_of_nonneg (sub_nonneg.mpr ((div_le_one hP).mpr hgP))]
  field_simp
  ring

example := radicand_best (A := 0) (k := 1) (l := 1) (g := 0) (by norm_num) (by norm_num)

/-- For `0 ≤ k l ≤ g`: `A − 2t k l + t² k l + 2|t| g ≥ A`. -/
lemma radicand_ge_of_le {A k l g : ℝ} (hP : 0 ≤ k * l) (hgP : k * l ≤ g) (t : ℝ) :
    A ≤ A - 2 * t * k * l + t ^ 2 * k * l + 2 * |t| * g := by
  rcases le_or_gt 0 t with ht | ht
  · rw [abs_of_nonneg ht]
    nlinarith [mul_nonneg ht (sub_nonneg.mpr hgP), mul_nonneg (sq_nonneg t) hP]
  · rw [abs_of_neg ht]
    nlinarith [mul_nonneg (neg_nonneg.mpr ht.le) hP, mul_nonneg (neg_nonneg.mpr ht.le)
      (hP.trans hgP), mul_nonneg (sq_nonneg t) hP]

example := radicand_ge_of_le (A := 0) (k := 1) (l := 1) (g := 1) (by norm_num) (by norm_num) 0

/-- `(√(k (N − l)) + √(l (M − k)))² = M N − (k l − g)²/(k l)` for `g = √(k l (M − k)(N − l))`,
`0 < k ≤ M`, `0 < l ≤ N`. -/
lemma sq_sqrt_add_sqrt {M N k l : ℝ} (hk : 0 < k) (hl : 0 < l) (hkM : k ≤ M) (hlN : l ≤ N) :
    (√(k * (N - l)) + √(l * (M - k))) ^ 2 =
      M * N - (k * l - √(k * l * (M - k) * (N - l))) ^ 2 / (k * l) := by
  have ha := Real.sq_sqrt (mul_nonneg hk.le (sub_nonneg.mpr hlN))
  have hb := Real.sq_sqrt (mul_nonneg hl.le (sub_nonneg.mpr hkM))
  have hg : √(k * l * (M - k) * (N - l)) = √(k * (N - l)) * √(l * (M - k)) := by
    rw [← Real.sqrt_mul (mul_nonneg hk.le (sub_nonneg.mpr hlN))]
    congr 1
    ring
  have key : (k * l - √(k * (N - l)) * √(l * (M - k))) ^ 2 =
      (M * N - (√(k * (N - l)) + √(l * (M - k))) ^ 2) * (k * l) := by
    linear_combination (k * l + l * (M - k)) * ha + (k * l + √(k * (N - l)) ^ 2) * hb
  rw [hg, key, mul_div_cancel_right₀ _ (mul_pos hk hl).ne']
  ring

example := sq_sqrt_add_sqrt (M := 1) (N := 1) (k := 1) (l := 1) one_pos one_pos le_rfl le_rfl

/-- `m n ≤ k n + l m` gives `g ≤ k l`, for `g = √(k l (m − k)(n − l))`. -/
lemma sqrt_le_mul_of_le {M N k l : ℝ} (hk : 0 ≤ k) (hl : 0 ≤ l)
    (h : M * N ≤ k * N + l * M) : √(k * l * (M - k) * (N - l)) ≤ k * l := by
  rw [Real.sqrt_le_left (mul_nonneg hk hl)]
  have h' : (M - k) * (N - l) ≤ k * l := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_left h' (mul_nonneg hk hl)]

example := sqrt_le_mul_of_le (M := 1) (N := 1) (k := 1) (l := 1) zero_le_one zero_le_one
  (by norm_num)

/-- `m n < k n + l m` gives `g < k l` for `0 < k`, `0 < l`. -/
lemma sqrt_lt_mul_of_lt {M N k l : ℝ} (hk : 0 < k) (hl : 0 < l)
    (h : M * N < k * N + l * M) : √(k * l * (M - k) * (N - l)) < k * l := by
  rw [Real.sqrt_lt' (mul_pos hk hl)]
  have h' : (M - k) * (N - l) < k * l := by nlinarith
  nlinarith [mul_lt_mul_of_pos_left h' (mul_pos hk hl)]

example := sqrt_lt_mul_of_lt (M := 1) (N := 1) (k := 1) (l := 1) one_pos one_pos (by norm_num)

/-- `k n + l m ≤ m n` gives `k l ≤ g`. -/
lemma mul_le_sqrt_of_le {M N k l : ℝ} (hk : 0 ≤ k) (hl : 0 ≤ l)
    (h : k * N + l * M ≤ M * N) : k * l ≤ √(k * l * (M - k) * (N - l)) := by
  refine Real.le_sqrt_of_sq_le ?_
  have h' : k * l ≤ (M - k) * (N - l) := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_left h' (mul_nonneg hk hl)]

example := mul_le_sqrt_of_le (M := 2) (N := 2) (k := 1) (l := 1) zero_le_one zero_le_one
  (by norm_num)

/-- **The best block witness**, `review/flat_spectrum.md`, *Patterns*, generalized to `m × n`:
for nonempty `R`, `C` with `m n ≤ k n + l m`, the least `‖J − t 1_R 1_Cᵀ‖_S1` over real `t`
is `√(k (n − l)) + √(l (m − k))`, reached at `t = 1 − g/(k l)`, `g = √(k l (m − k)(n − l))`. -/
theorem isLeast_nuclearNorm_onesSubBlock {R : Finset m} {C : Finset n} (hR : R.Nonempty)
    (hC : C.Nonempty)
    (h : (Fintype.card m * Fintype.card n : ℝ) ≤ #R * Fintype.card n + #C * Fintype.card m) :
    IsLeast (Set.range fun t => nuclearNorm (onesSubBlock R C t))
      (√(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card m - #R))) := by
  have hk : (0 : ℝ) < #R := by exact_mod_cast hR.card_pos
  have hl : (0 : ℝ) < #C := by exact_mod_cast hC.card_pos
  have hkM : (#R : ℝ) ≤ Fintype.card m := by exact_mod_cast card_le_univ R
  have hlN : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  have hgP := sqrt_le_mul_of_le hk.le hl.le h
  have hab : 0 ≤ √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card m - #R)) := by positivity
  rw [← Real.sqrt_sq hab, sq_sqrt_add_sqrt hk hl hkM hlN]
  refine ⟨⟨1 - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C)) / (#R * #C), ?_⟩, ?_⟩
  · dsimp only
    rw [nuclearNorm_onesSubBlock, radicand_best (mul_pos hk hl) hgP]
  · rintro _ ⟨t, rfl⟩
    dsimp only
    rw [nuclearNorm_onesSubBlock]
    exact Real.sqrt_le_sqrt (radicand_ge (mul_pos hk hl) (Real.sqrt_nonneg _) t)

example := isLeast_nuclearNorm_onesSubBlock (R := ({0, 1} : Finset (Fin 2)))
  (C := ({0} : Finset (Fin 2))) ⟨0, by decide⟩ ⟨0, by decide⟩ (by norm_num)

/-- **A small block never beats `J`**: if `k n + l m ≤ m n`, the least `‖J − t 1_R 1_Cᵀ‖_S1`
over real `t` is `√(m n) = ‖J‖_S1`, at `t = 0`. -/
theorem isLeast_nuclearNorm_onesSubBlock_of_le {R : Finset m} {C : Finset n}
    (h : (#R * Fintype.card n + #C * Fintype.card m : ℝ) ≤ Fintype.card m * Fintype.card n) :
    IsLeast (Set.range fun t => nuclearNorm (onesSubBlock R C t))
      (√(Fintype.card m * Fintype.card n)) := by
  have hk : (0 : ℝ) ≤ #R := Nat.cast_nonneg _
  have hl : (0 : ℝ) ≤ #C := Nat.cast_nonneg _
  refine ⟨⟨0, ?_⟩, ?_⟩
  · dsimp only
    rw [nuclearNorm_onesSubBlock]
    simp
  · rintro _ ⟨t, rfl⟩
    dsimp only
    rw [nuclearNorm_onesSubBlock]
    exact Real.sqrt_le_sqrt (radicand_ge_of_le (mul_nonneg hk hl) (mul_le_sqrt_of_le hk hl h) t)

example := isLeast_nuclearNorm_onesSubBlock_of_le (R := ({0} : Finset (Fin 2)))
  (C := ({0} : Finset (Fin 2))) (by norm_num)

end Transformer.Grokking.FlatSpectrum

module

public import Transformer.Grokking.FlatSpectrum.MarginSharp

/-!
# The largest failure margin, for every `N`

Source: `review/flat_spectrum.md`, *The failure margin*, item 3 (added). For `k + l = N + 1`
the value `√(k (N − l)) + √(l (N − k))` is `√(k (k − 1)) + √(l (l − 1))`, whose square is
`N (N + 1) − 2p + 2 √(p (p − N))` with `p = k l ≥ N`. It grows with `p`
(`sqrt_mul_sub_sub_le`, `sqrt_add_sqrt_le_of_mul_le`), and `p` is largest when `k` and `l` differ
by at most one (`mul_le_mul_of_add_eq`). So for `a + b = N + 1` with `a ≤ b ≤ a + 1`, the
largest least completion norm of `J` over the patterns without a permutation graph is
`√(a (a − 1)) + √(b (b − 1))` (`failureMargin`). For odd `N` this is `failureMargin_odd`'s
`√(N² − 1)`; for `N = 2m` it is `√(m (m − 1)) + √((m + 1) m)`, `failureMargin_four`'s
`√2 + √6` at `N = 4`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `√(p (p − c)) − p` grows with `p` once `p ≥ c ≥ 0`. -/
lemma sqrt_mul_sub_sub_le {c p q : ℝ} (hc : 0 ≤ c) (hp : c ≤ p) (hq : p ≤ q) :
    √(p * (p - c)) - p ≤ √(q * (q - c)) - q := by
  have hA := Real.sq_sqrt (mul_nonneg (hc.trans hp) (sub_nonneg.mpr hp))
  have hA0 := Real.sqrt_nonneg (p * (p - c))
  have h2 : √(p * (p - c)) ≤ p - c / 2 :=
    calc √(p * (p - c)) ≤ √((p - c / 2) ^ 2) := Real.sqrt_le_sqrt (by nlinarith [sq_nonneg c])
      _ = p - c / 2 := Real.sqrt_sq (by linarith)
  have hB : √(p * (p - c)) + (q - p) ≤ √(q * (q - c)) := by
    rw [Real.le_sqrt (by linarith) (mul_nonneg (by linarith) (by linarith))]
    nlinarith [mul_nonneg (sub_nonneg.mpr hq) (by linarith : (0 : ℝ) ≤ 2 * p - c - 2 * √(p * (p - c)))]
  linarith

example := sqrt_mul_sub_sub_le (c := 1) (p := 1) (q := 2) zero_le_one le_rfl one_le_two

/-- The square of `√(x (x − 1)) + √(y (y − 1))` through `x + y` and `x y`. -/
lemma sq_sqrt_add_sqrt_mul_sub_one {x y : ℝ} (hx : 1 ≤ x) (hy : 1 ≤ y) :
    (√(x * (x - 1)) + √(y * (y - 1))) ^ 2 =
      (x + y) * (x + y - 1) - 2 * (x * y) + 2 * √(x * y * (x * y - (x + y - 1))) := by
  have hx' : 0 ≤ x * (x - 1) := mul_nonneg (by linarith) (sub_nonneg.mpr hx)
  rw [add_sq, Real.sq_sqrt hx', Real.sq_sqrt (mul_nonneg (by linarith) (sub_nonneg.mpr hy)),
    mul_assoc 2, ← Real.sqrt_mul hx', show x * (x - 1) * (y * (y - 1)) =
      x * y * (x * y - (x + y - 1)) by ring]
  ring

example := sq_sqrt_add_sqrt_mul_sub_one (x := 1) (y := 1) le_rfl le_rfl

/-- **With `k + l` fixed, `√(k (k − 1)) + √(l (l − 1))` grows with `k l`**, for `k, l ≥ 1`. -/
lemma sqrt_add_sqrt_le_of_mul_le {k l a b : ℝ} (hk : 1 ≤ k) (hl : 1 ≤ l) (ha : 1 ≤ a)
    (hb : 1 ≤ b) (hs : k + l = a + b) (hp : k * l ≤ a * b) :
    √(k * (k - 1)) + √(l * (l - 1)) ≤ √(a * (a - 1)) + √(b * (b - 1)) := by
  rw [← pow_le_pow_iff_left₀ (by positivity) (by positivity) two_ne_zero,
    sq_sqrt_add_sqrt_mul_sub_one hk hl, sq_sqrt_add_sqrt_mul_sub_one ha hb, hs]
  have := sqrt_mul_sub_sub_le (c := a + b - 1) (p := k * l) (q := a * b) (by linarith)
    (by nlinarith [mul_nonneg (sub_nonneg.mpr hk) (sub_nonneg.mpr hl)]) hp
  linarith

example := sqrt_add_sqrt_le_of_mul_le (k := 1) (l := 3) (a := 2) (b := 2) le_rfl (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- With `k + l = a + b` and `k` not strictly between `a ≤ b`, `k l ≤ a b`: the difference is
`(a − k)(b − k)`. -/
lemma mul_le_mul_of_add_eq {k l a b : ℝ} (hs : k + l = a + b) (hab : a ≤ b)
    (hk : k ≤ a ∨ b ≤ k) : k * l ≤ a * b := by
  obtain rfl : l = a + b - k := by linarith
  rcases hk with h | h
  · nlinarith [mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr (h.trans hab))]
  · nlinarith [mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr (hab.trans h))]

example := mul_le_mul_of_add_eq (k := 1) (l := 3) (a := 2) (b := 2) (by norm_num) le_rfl
  (Or.inl one_le_two)

/-- **The failure margin for every `N`**, `review/flat_spectrum.md`, *The failure margin*,
item 3: for `N ≥ 1` and `a + b = N + 1` with `a ≤ b ≤ a + 1`, every pattern without a
permutation graph has a completion of `J` of norm at most `√(a (a − 1)) + √(b (b − 1))`, and
some pattern without a permutation graph, the cells off an `a × b` block, has this least
completion norm. -/
theorem failureMargin (hN : 1 ≤ Fintype.card n) {a b : ℕ} (hab : a + b = Fintype.card n + 1)
    (hba : a ≤ b) (hb : b ≤ a + 1) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
        nuclearNorm X ≤ √((a : ℝ) * (a - 1)) + √((b : ℝ) * (b - 1))) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2})
        (√((a : ℝ) * (a - 1)) + √((b : ℝ) * (b - 1))) := by
  have hab' : (a : ℝ) + b = Fintype.card n + 1 := by exact_mod_cast hab
  refine ⟨fun Ω hΩ => ?_, ?_⟩
  · obtain ⟨R, C, -, hcard, X, hX, hnorm⟩ := exists_nuclearNorm_eq_of_not_perm hΩ
    refine ⟨X, hX, ?_⟩
    have hk := card_le_univ R
    have hl := card_le_univ C
    have hcard' : (#R : ℝ) + #C = Fintype.card n + 1 := by exact_mod_cast hcard
    rw [hnorm, show (Fintype.card n : ℝ) - #C = #R - 1 by linarith,
      show (Fintype.card n : ℝ) - #R = #C - 1 by linarith]
    refine sqrt_add_sqrt_le_of_mul_le (by exact_mod_cast (show 1 ≤ #R by omega))
      (by exact_mod_cast (show 1 ≤ #C by omega)) (by exact_mod_cast (show 1 ≤ a by omega))
      (by exact_mod_cast (show 1 ≤ b by omega)) (by linarith)
      (mul_le_mul_of_add_eq (by linarith) (by exact_mod_cast hba) ?_)
    rcases le_or_gt #R a with h | h
    · exact Or.inl (by exact_mod_cast h)
    · exact Or.inr (by exact_mod_cast (show b ≤ #R by omega))
  · obtain ⟨R, C, -, -, hperm, hleast⟩ :=
      exists_compl_block_isLeast (n := n) (k := a) (l := b) (by omega) (by omega) (by omega)
    rw [show (Fintype.card n : ℝ) - b = a - 1 by linarith,
      show (Fintype.card n : ℝ) - a = b - 1 by linarith] at hleast
    exact ⟨_, hperm, hleast⟩

example := failureMargin (n := Fin 4) (by simp) (a := 2) (b := 3) (by simp) (by norm_num)
  (by norm_num)

example := failureMargin (n := Fin 1) (by simp) (a := 1) (b := 1) (by simp) le_rfl
  (by norm_num)

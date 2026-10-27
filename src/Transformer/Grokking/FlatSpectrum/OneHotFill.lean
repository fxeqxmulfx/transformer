module

public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.Intermediate

/-!
# One-hot rows are never completed

Source: `review/flat_spectrum.md`, Torch check: "One-hot class kernels [f = c], including the
indicator of 0: not a single hidden 1 is recovered, at any fraction up to 99%. Each row holds a
single 1, which is the worst case for completion." Zeroing row `i` of a least-cost
factorization `X = U V*` (Lemma 1) zeroes row `i` of `X` and lowers `‖U‖²`, strictly when row
`i` of `X` is nonzero. So zeroing a nonzero row lowers `‖·‖_S1` strictly, and every
minimum-nuclear-norm completion is zero on each row whose seen entries are all zero. In a row
`a ≠ 0` of `[a³ + a b = c]` the only 1 is at `b = a⁻¹ (c − a³)`. Once it is hidden, every
completion fills it with `0`, from every training set.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The one-hot class kernel `[f(a, b) = c]`: the task kernel of the indicator of answer `c`,
`review/flat_spectrum.md`, Torch check. -/
def classKernel {G : Type*} [DecidableEq G] (f : G → G → G) (c : G) : Matrix G G ℂ :=
  taskKernel f fun x => if x = c then 1 else 0

/-- Zeroing row `i` removes its squared entries from `‖U‖²`. -/
lemma frobeniusSq_updateRow_zero {k : Type*} [Fintype k] (U : Matrix m k ℂ) (i : m) :
    frobeniusSq (U.updateRow i 0) + ∑ s, ‖U i s‖ ^ 2 = frobeniusSq U := by
  have h (l : m) : ∑ s, ‖U.updateRow i 0 l s‖ ^ 2 + (if l = i then ∑ s, ‖U i s‖ ^ 2 else 0) =
      ∑ s, ‖U l s‖ ^ 2 := by
    rcases eq_or_ne l i with rfl | hl
    · simp
    · simp [hl]
  simp only [frobeniusSq]
  rw [← Finset.sum_congr rfl fun l _ => h l, Finset.sum_add_distrib, Finset.sum_ite_eq']
  simp

/-- Zeroing a nonzero row lowers `‖·‖_S1` strictly. -/
theorem nuclearNorm_updateRow_zero_lt (X : Matrix m n ℂ) {i : m} {j : n} (hij : X i j ≠ 0) :
    nuclearNorm (X.updateRow i 0) < nuclearNorm X := by
  obtain ⟨r, U, V, hX, hc⟩ := (nuclearNorm_isLeast X).1
  have hUV : X.updateRow i 0 = U.updateRow i 0 * Vᴴ := by
    rw [updateRow_mul, zero_vecMul, hX]
  have hrow : 0 < ∑ s, ‖U i s‖ ^ 2 := by
    obtain ⟨s, hs⟩ : ∃ s, U i s ≠ 0 := by
      by_contra h
      push Not at h
      apply hij
      rw [hX, mul_apply]
      simp [h]
    exact Finset.sum_pos' (fun s _ => sq_nonneg _) ⟨s, mem_univ _, pow_pos (norm_pos_iff.mpr hs) 2⟩
  have h1 := nuclearNorm_le_of_eq_mul (U.updateRow i 0) V hUV
  have h2 := frobeniusSq_updateRow_zero U i
  rw [hc]
  linarith

example := nuclearNorm_updateRow_zero_lt (1 : Matrix (Fin 2) (Fin 2) ℂ) (i := 0) (j := 0)
  (by simp)

/-- Zeroing a row does not raise `‖·‖_S1`. -/
theorem nuclearNorm_updateRow_zero_le (X : Matrix m n ℂ) (i : m) :
    nuclearNorm (X.updateRow i 0) ≤ nuclearNorm X := by
  by_cases h : ∃ j, X i j ≠ 0
  · obtain ⟨j, hj⟩ := h
    exact (nuclearNorm_updateRow_zero_lt X hj).le
  · push Not at h
    have : X.updateRow i 0 = X := by
      ext l s
      rcases eq_or_ne l i with rfl | hl
      · simp [h]
      · simp [hl]
    rw [this]

/-- Every minimum-nuclear-norm completion of `K` from `Ω` is zero on a row whose seen entries
are all zero. -/
theorem eq_zero_of_isMinOn_of_row {K X : Matrix m n ℂ} {Ω : Set (m × n)}
    (hX : ∀ kl ∈ Ω, X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = K kl.1 kl.2} X)
    {i : m} (hK : ∀ j, (i, j) ∈ Ω → K i j = 0) (j : n) : X i j = 0 := by
  by_contra hij
  refine not_le.mpr (nuclearNorm_updateRow_zero_lt X hij) (isMinOn_iff.mp hmin _ ?_)
  rintro ⟨l, s⟩ hls
  rcases eq_or_ne l i with rfl | hl
  · simpa using (hK s hls).symm
  · simpa [hl] using hX _ hls

/-- With the single cell `(i, j)` missing, zeroing row `i` gives a fill of least `‖·‖_S1`. -/
theorem isMinOn_updateRow_zero (K : Matrix m n ℂ) (i : m) (j : n) :
    IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2}
      (K.updateRow i 0) := by
  rw [isMinOn_iff]
  intro Y hY
  have h : K.updateRow i 0 = Y.updateRow i 0 := by
    ext l s
    rcases eq_or_ne l i with rfl | hl
    · simp
    · simpa [hl] using (hY (l, s) (by simp [hl])).symm
  rw [h]
  exact nuclearNorm_updateRow_zero_le Y i

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Zeroing row `i` keeps every seen entry when row `i` of `K` vanishes off `j`. -/
lemma updateRow_zero_mem {K : Matrix m n ℂ} {i : m} {j : n} (hK : ∀ l, l ≠ j → K i l = 0) :
    ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), K.updateRow i 0 kl.1 kl.2 = K kl.1 kl.2 := by
  rintro ⟨l, s⟩ hls
  rcases eq_or_ne l i with rfl | hl
  · have hs : s ≠ j := fun h => hls (by simp [h])
    simp [hK s hs]
  · simp [hl]

example := updateRow_zero_mem (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (i := 0) (j := 0)
  fun _ hl => one_apply_ne hl.symm

example := eq_zero_of_isMinOn_of_row (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := {(0, 0)}ᶜ)
  (updateRow_zero_mem fun _ hl => one_apply_ne hl.symm) (isMinOn_updateRow_zero 1 0 0)
  (i := 0) (fun _ hj => one_apply_ne fun h => hj (by subst h; simp)) 0

/-- Row `a` of `[f = c]` vanishes off `b` when `b` is its only answer `c`. -/
lemma classKernel_eq_zero {G : Type*} [DecidableEq G] (f : G → G → G) (c : G) {a b : G}
    (huniq : ∀ b', f a b' = c → b' = b) {l : G} (hl : l ≠ b) : classKernel f c a l = 0 := by
  have h : ¬f a l = c := fun h => hl (huniq l h)
  simp [classKernel, taskKernel, h]

example := classKernel_eq_zero (· + · : ZMod 3 → ZMod 3 → ZMod 3) 0 (a := 1) (b := 2)
  (by decide) (l := 0) (by decide)

/-- **One-hot kernels are never completed**, `review/flat_spectrum.md`, Torch check: if `b` is
the only answer `c` in row `a` and `(a, b)` is hidden, every minimum-nuclear-norm completion of
`[f = c]` from `Ω` is `0` at `(a, b)`. -/
theorem eq_zero_of_isMinOn_classKernel {G : Type*} [Fintype G] [DecidableEq G]
    (f : G → G → G) (c : G) {Ω : Set (G × G)} {X : Matrix G G ℂ}
    (hX : ∀ kl ∈ Ω, X kl.1 kl.2 = classKernel f c kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = classKernel f c kl.1 kl.2} X)
    {a b : G} (huniq : ∀ b', f a b' = c → b' = b) (hΩ : (a, b) ∉ Ω) : X a b = 0 :=
  eq_zero_of_isMinOn_of_row hX hmin (fun _ hj => classKernel_eq_zero f c huniq
    fun h => hΩ (h ▸ hj)) b

example := eq_zero_of_isMinOn_classKernel (· + · : ZMod 3 → ZMod 3 → ZMod 3) 0 (Ω := {(1, 2)}ᶜ)
  (updateRow_zero_mem fun _ hl => classKernel_eq_zero _ _ (a := 1) (b := 2) (by decide) hl)
  (isMinOn_updateRow_zero _ 1 2) (a := 1) (b := 2) (by decide) (by simp)

variable {p : ℕ} [Fact p.Prime]

/-- For `a ≠ 0`, row `a` of `a³ + a b` takes the value `c` only once. -/
lemma cube_add_mul_unique {a b b' c : ZMod p} (ha : a ≠ 0) (hab : a ^ 3 + a * b = c)
    (hb' : a ^ 3 + a * b' = c) : b' = b :=
  ((cube_add_mul_eq_iff ha).mp hb').trans ((cube_add_mul_eq_iff ha).mp hab).symm

example := @cube_add_mul_unique 5 ⟨Nat.prime_five⟩ 1 4 4 0 (by decide) (by decide) (by decide)

/-- The torch check of `review/flat_spectrum.md` for `x³ + x y`: for prime `p` and `a ≠ 0`, the
hidden 1 of `[a³ + a b = c]` at `(a, b)` is `0` in every minimum-nuclear-norm completion, from
every training set `Ω` that hides it. -/
theorem eq_zero_of_isMinOn_classKernel_cube_add_mul {c : ZMod p} {Ω : Set (ZMod p × ZMod p)}
    {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ Ω, X kl.1 kl.2 = classKernel (fun a b => a ^ 3 + a * b) c kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = classKernel (fun a b => a ^ 3 + a * b) c kl.1 kl.2} X)
    {a b : ZMod p} (ha : a ≠ 0) (hab : a ^ 3 + a * b = c) (hΩ : (a, b) ∉ Ω) : X a b = 0 :=
  eq_zero_of_isMinOn_classKernel _ c hX hmin (fun _ hb' => cube_add_mul_unique ha hab hb') hΩ

example := @eq_zero_of_isMinOn_classKernel_cube_add_mul 5 ⟨Nat.prime_five⟩ 0 {(1, 4)}ᶜ _
  (updateRow_zero_mem fun _ hl => classKernel_eq_zero _ _
    (fun _ hb' => @cube_add_mul_unique 5 ⟨Nat.prime_five⟩ 1 4 _ 0 (by decide) (by decide) hb') hl)
  (isMinOn_updateRow_zero _ 1 4) 1 4 (by decide) (by decide) (by simp)

end Transformer.Grokking.FlatSpectrum

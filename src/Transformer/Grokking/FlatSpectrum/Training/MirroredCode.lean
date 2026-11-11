module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Training.SeenMargin

/-!
# A mirrored answer code: with every cell seen a flat kernel has the largest margin

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11.
An answer code `φ` is mirrored at the gap `g` (`IsMirroredCode`) when it is unimodular, every
other codeword is at squared distance at least `2 g` from `φ t`, and two of them have
`φ c + φ c' = 2 (1 − g) φ t`: the nearest codewords of `φ t`, mirror images in its line. For the
probes of such a code (`codeProbe`, item XE8):

* the gaps of a sample against the two neighbours of its label add up to
  `2 g Re (conj φ(y s) X_{cell s})` (`gap_codeProbe_add_gap`), which bounds the margin
  (`margin_codeProbe_le_mul_re`); a kernel that shows the labels has the margin `g`
  (`margin_codeProbe_eq`);
* with every cell seen, `|m| |n| μ(X) ≤ g ⟨K, X⟩` (`card_mul_margin_codeProbe_le`). A flat `K`
  has `|m| |n| = √c ‖K‖_S1` (`card_mul_card_eq_of_isFlat`) and `⟨K, X⟩ ≤ √c ‖X‖_S1`, so
  `μ(X) ‖K‖_S1 ≤ g ‖X‖_S1` (`margin_codeProbe_mul_nuclearNorm_le`): no product has a larger
  margin for its norm than `K`.

The witness is the code `±1`, mirrored at `g = 2`, with the `1 × 1` kernel `1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexConjugate

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
  {φ : C → ℂ} {cell : S → m × n} {y : S → C} {K : Matrix m n ℂ} {g c₀ : ℝ}

/-- The answer code `φ` is **mirrored at the gap `g`**: it is unimodular, every other codeword
is at squared distance at least `2 g` from `φ t`, and two of them have
`φ c + φ c' = 2 (1 − g) φ t` (`review/flat_spectrum.md`, item XE11). -/
def IsMirroredCode (φ : C → ℂ) (g : ℝ) : Prop :=
  (∀ c, ‖φ c‖ = 1) ∧ (∀ t c, c ≠ t → 2 * g ≤ ‖φ t - φ c‖ ^ 2) ∧
    ∀ t, ∃ c c', c ≠ t ∧ c' ≠ t ∧ φ c + φ c' = ((2 * (1 - g) : ℝ) : ℂ) * φ t

/-- A mirrored code with a positive gap is injective. -/
lemma IsMirroredCode.injective (h : IsMirroredCode φ g) (hg : 0 < g) : Function.Injective φ := by
  intro a b hab
  by_contra hne
  have h1 := h.2.1 a b (Ne.symm hne)
  rw [hab, sub_self, norm_zero] at h1
  nlinarith

/-- The gaps of a sample against two classes with `φ c + φ c' = 2 (1 − g) φ(y s)` add up to
`2 g Re (conj φ(y s) X_{cell s})`. -/
lemma gap_codeProbe_add_gap {c c' : C} (X : Matrix m n ℂ) (s : S)
    (h : φ c + φ c' = ((2 * (1 - g) : ℝ) : ℂ) * φ (y s)) :
    gap (codeProbe φ cell) y X s c + gap (codeProbe φ cell) y X s c'
      = 2 * g * (conj (φ (y s)) * X (cell s).1 (cell s).2).re := by
  have h' : (conj (φ c) * X (cell s).1 (cell s).2).re
      + (conj (φ c') * X (cell s).1 (cell s).2).re
      = 2 * (1 - g) * (conj (φ (y s)) * X (cell s).1 (cell s).2).re := by
    rw [← Complex.add_re, ← add_mul, ← map_add, h, map_mul, Complex.conj_ofReal, mul_assoc,
      Complex.re_ofReal_mul]
  simp only [gap, pairing_codeProbe]
  linarith

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- With every cell seen, a kernel that shows the labels of a unimodular code has unimodular
entries. -/
lemma norm_apply_eq_one_of_surjective (hφ : ∀ c, ‖φ c‖ = 1)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell) (i : m)
    (j : n) : ‖K i j‖ = 1 := by
  obtain ⟨s, hs⟩ := hsurj (i, j)
  have h3 := hK s
  rw [hs] at h3
  exact h3 ▸ hφ (y s)

variable [Fintype S] [Fintype C]

/-- **With a mirrored code the margin is at most `g Re (conj φ(y s) X_{cell s})`**
(`review/flat_spectrum.md`, item XE11), at every sample `s`. -/
theorem margin_codeProbe_le_mul_re (h : IsMirroredCode φ g) (X : Matrix m n ℂ) (s : S) :
    margin (codeProbe φ cell) y X ≤ g * (conj (φ (y s)) * X (cell s).1 (cell s).2).re := by
  obtain ⟨c, c', hc, hc', hsum⟩ := h.2.2 (y s)
  have h1 := margin_le_gap (A := codeProbe φ cell) (y := y) X hc
  have h2 := margin_le_gap (A := codeProbe φ cell) (y := y) X hc'
  have h3 := gap_codeProbe_add_gap (cell := cell) X s hsum
  linarith

/-- **A kernel that shows the labels with a mirrored code has the margin `g`**
(`review/flat_spectrum.md`, item XE11). -/
theorem margin_codeProbe_eq [Nonempty S] [Nontrivial C] (h : IsMirroredCode φ g)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) : margin (codeProbe φ cell) y K = g := by
  refine le_antisymm ?_ (le_margin fun s c hc => ?_)
  · obtain ⟨s⟩ := ‹Nonempty S›
    have h1 := margin_codeProbe_le_mul_re (cell := cell) (y := y) h K s
    rwa [hK s, Complex.conj_mul', h.1, ← Complex.ofReal_pow, Complex.ofReal_re, one_pow,
      mul_one] at h1
  · rw [gap_codeProbe h.1 hK]
    linarith [h.2.1 (y s) c hc]

/-- With every cell seen, `μ(X) ≤ g Re (conj K_ij X_ij)` at every cell. -/
lemma margin_codeProbe_le_mul_re_apply (h : IsMirroredCode φ g)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell)
    (X : Matrix m n ℂ) (i : m) (j : n) :
    margin (codeProbe φ cell) y X ≤ g * (conj (K i j) * X i j).re := by
  obtain ⟨s, hs⟩ := hsurj (i, j)
  have h1 := margin_codeProbe_le_mul_re (cell := cell) (y := y) h X s
  rw [← hK s, hs] at h1
  exact h1

/-- With every cell seen: `|m| |n| μ(X) ≤ g ⟨K, X⟩`. -/
theorem card_mul_margin_codeProbe_le (h : IsMirroredCode φ g)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell)
    (X : Matrix m n ℂ) :
    Fintype.card m * Fintype.card n * margin (codeProbe φ cell) y X ≤ g * pairing K X := by
  rw [pairing_eq_sum_re, Finset.mul_sum]
  calc (Fintype.card m : ℝ) * Fintype.card n * margin (codeProbe φ cell) y X
      = ∑ _i : m, ∑ _j : n, margin (codeProbe φ cell) y X := by
        simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        ring
    _ ≤ ∑ i, g * ∑ j, (conj (K i j) * X i j).re := Finset.sum_le_sum fun i _ => by
        rw [Finset.mul_sum]
        exact Finset.sum_le_sum fun j _ => margin_codeProbe_le_mul_re_apply h hK hsurj X i j

omit [Fintype S] [Fintype C] in
/-- A flat kernel with unimodular entries has `|m| |n| = √c ‖K‖_S1`. -/
lemma card_mul_card_eq_of_isFlat (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hK : ∀ i j, ‖K i j‖ = 1) :
    (Fintype.card m : ℝ) * Fintype.card n = √c₀ * nuclearNorm K := by
  have h1 : frobeniusSq K = Fintype.card m * Fintype.card n := by
    simp [frobeniusSq, hK]
  rw [← h1, hflat.frobeniusSq_eq hc.ne', hflat.nuclearNorm_eq hc.ne', mul_left_comm,
    Real.mul_self_sqrt hc.le]

/-- **No product has a larger margin for its norm than a flat kernel with every cell seen**
(`review/flat_spectrum.md`, item XE11): `μ(X) ‖K‖_S1 ≤ g ‖X‖_S1` for a mirrored code. -/
theorem margin_codeProbe_mul_nuclearNorm_le (h : IsMirroredCode φ g) (hg : 0 ≤ g)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell)
    (hc : 0 < c₀) (hflat : IsFlat K c₀) (X : Matrix m n ℂ) :
    margin (codeProbe φ cell) y X * nuclearNorm K ≤ g * nuclearNorm X := by
  have h1 := card_mul_margin_codeProbe_le (y := y) h hK hsurj X
  have h2 := mul_le_mul_of_nonneg_left (hflat.pairing_le hc X) hg
  rw [card_mul_card_eq_of_isFlat hc hflat (norm_apply_eq_one_of_surjective h.1 hK hsurj)] at h1
  refine le_of_mul_le_mul_left ?_ (Real.sqrt_pos.2 hc)
  linarith

/-! The witness: the code `±1`, mirrored at `g = 2`, with the `1 × 1` kernel `1` on its cell. -/

lemma isMirroredCode_signCode : IsMirroredCode signCode 2 := by
  refine ⟨norm_signCode, fun t c hc => ?_, fun t => ⟨!t, !t, by simp, by simp, ?_⟩⟩
  · cases t <;> cases c <;> simp_all [signCode] <;> norm_num
  · cases t <;> simp [signCode] <;> norm_num

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))

example := isMirroredCode_signCode.injective two_pos

example := gap_codeProbe_add_gap (φ := signCode) (g := 2) (cell := cell₀) (y := fun _ => true)
  (c := false) (c' := false) 1 0 (by simp [signCode]; norm_num)

example := margin_codeProbe_le_mul_re (cell := cell₀) (y := fun _ => true)
  isMirroredCode_signCode 1 0

example := margin_codeProbe_eq (cell := cell₀) (y := fun _ => true) (K := 1)
  isMirroredCode_signCode fun _ => by simp [signCode]

example := card_mul_card_eq_of_isFlat (K := (1 : Matrix (Fin 1) (Fin 1) ℂ)) one_pos isFlat_one
  fun i j => by rw [Subsingleton.elim i j, Matrix.one_apply_eq, norm_one]

example := norm_apply_eq_one_of_surjective (cell := cell₀) (y := fun _ => true) (K := 1)
  norm_signCode (fun _ => by simp [signCode]) (fun _ => ⟨0, Subsingleton.elim _ _⟩) 0 0

example (X : Matrix (Fin 1) (Fin 1) ℂ) := card_mul_margin_codeProbe_le (cell := cell₀)
  (y := fun _ => true) (K := 1) isMirroredCode_signCode (fun _ => by simp [signCode])
  (fun _ => ⟨0, Subsingleton.elim _ _⟩) X

/-- The `1 × 1` kernel `1` with the code `±1`: `μ(X) ‖1‖_S1 ≤ 2 ‖X‖_S1`. -/
example (X : Matrix (Fin 1) (Fin 1) ℂ) := margin_codeProbe_mul_nuclearNorm_le (cell := cell₀)
  (y := fun _ => true) (K := 1) isMirroredCode_signCode zero_le_two
  (fun _ => by simp [signCode]) (fun _ => ⟨0, Subsingleton.elim _ _⟩) one_pos isFlat_one X

end Transformer.Grokking.FlatSpectrum.Training

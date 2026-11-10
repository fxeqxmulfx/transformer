module

public import Transformer.Grokking.FlatSpectrum.Recall.Basic

/-!
# The parities of the bindings form a flat kernel

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR1, for
the task of Arora et al., arXiv:2312.04927v1, §3, with two values.

Read a value as `±1`. A model that recalls every binding also knows every product of bindings:
for a set `S` of keys, the parity `∏_{q ∈ S} ±1`. The **parity kernel** has a row for each of
the `2^m` sets of keys and a column for each of the `2^m` dictionaries with two values,

    K(S, β) = ∏_{q ∈ S} (±1 of β q)                                  (`parityKernel`).

Its rows are the characters of `(ℤ/2)^m`: orthogonal, of squared length `2^m`
(`sum_prod_mul_prod`, `parityKernel_mul_conjTranspose`). So it is flat with full rank `2^m`
(`isFlat_parityKernel`, `rank_parityKernel`), and by Theorem A it is a minimum of M from no
proper set of its entries (`not_isMinOn_parityKernel`). The recall kernel of AR1 is its `m` rows
of the single keys.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- **The parity kernel** `K(S, β) = ∏_{q ∈ S} (±1 of β q)` of the sets of keys and the
dictionaries with two values (`review/flat_spectrum.md`, PR1). -/
noncomputable def parityKernel (κ : Type*) [Fintype κ] : Matrix (Finset κ) (κ → Bool) ℂ :=
  Matrix.of fun S β => ((∏ q ∈ S, (if β q then 1 else -1 : ℝ) : ℝ) : ℂ)

/-- **Different parities are orthogonal** (`review/flat_spectrum.md`, PR1): the sum over the
dictionaries of the product of two parities is `2^m` for equal sets and `0` otherwise. -/
theorem sum_prod_mul_prod (S T : Finset κ) :
    ∑ β : κ → Bool, (∏ q ∈ S, (if β q then (1 : ℝ) else -1)) *
      ∏ q ∈ T, (if β q then (1 : ℝ) else -1) = if S = T then 2 ^ Fintype.card κ else 0 := by
  have h (U : Finset κ) (β : κ → Bool) : ∏ q ∈ U, (if β q then (1 : ℝ) else -1) =
      ∏ q, if q ∈ U then (if β q then (1 : ℝ) else -1) else 1 := by
    rw [Finset.prod_ite_mem, Finset.univ_inter]
  simp only [h, ← Finset.prod_mul_distrib]
  refine (Fintype.prod_sum fun q (b : Bool) => (if q ∈ S then (if b then (1 : ℝ) else -1) else 1) *
    (if q ∈ T then (if b then (1 : ℝ) else -1) else 1)).symm.trans ?_
  split_ifs with hST
  · subst hST
    have h2 (q : κ) : ∑ b : Bool, (if q ∈ S then (if b then (1 : ℝ) else -1) else 1) *
        (if q ∈ S then (if b then (1 : ℝ) else -1) else 1) = 2 := by
      by_cases hq : q ∈ S <;> simp [hq]
      norm_num
    simp only [h2, Finset.prod_const, Finset.card_univ]
  · obtain ⟨q, hq⟩ : ∃ q, ¬ (q ∈ S ↔ q ∈ T) := by
      by_contra hall
      push Not at hall
      exact hST (Finset.ext hall)
    refine Finset.prod_eq_zero (Finset.mem_univ q) ?_
    by_cases h1 : q ∈ S <;> by_cases h2 : q ∈ T <;> simp_all

/-- Two keys: the parities of `{0}` and of `{0, 1}` are orthogonal. -/
example : ∑ β : Fin 2 → Bool, (∏ q ∈ ({0} : Finset (Fin 2)), (if β q then (1 : ℝ) else -1)) *
    ∏ q ∈ ({0, 1} : Finset (Fin 2)), (if β q then (1 : ℝ) else -1) = 0 := by
  rw [sum_prod_mul_prod, ite_eq_right (by decide)]

/-- **The rows of the parity kernel are orthogonal of squared length `2^m`**
(`review/flat_spectrum.md`, PR1). -/
theorem parityKernel_mul_conjTranspose :
    parityKernel κ * (parityKernel κ)ᴴ =
      (((2 : ℝ) ^ Fintype.card κ : ℝ) : ℂ) • (1 : Matrix (Finset κ) (Finset κ) ℂ) := by
  ext S T
  simp only [mul_apply, conjTranspose_apply, parityKernel, of_apply, Matrix.smul_apply,
    smul_eq_mul, one_apply, RCLike.star_def, Complex.conj_ofReal]
  rw [← Complex.ofReal_one, ← Complex.ofReal_zero]
  simp only [← Complex.ofReal_mul, ← Complex.ofReal_sum]
  rw [sum_prod_mul_prod]
  split_ifs <;> simp

/-- **The parity kernel is flat** (`review/flat_spectrum.md`, PR1). -/
theorem isFlat_parityKernel : IsFlat (parityKernel κ) ((2 : ℝ) ^ Fintype.card κ) :=
  Or.inr parityKernel_mul_conjTranspose

/-- **The parity kernel has rank `2^m`** (`review/flat_spectrum.md`, PR1). -/
theorem rank_parityKernel : (parityKernel κ).rank = 2 ^ Fintype.card κ := by
  rw [isFlat_parityKernel.rank_eq (by positivity), Fintype.card_finset, Fintype.card_fun,
    Fintype.card_bool, min_self]

/-- **The parity kernel is a minimum of M from no proper set of its entries**
(`review/flat_spectrum.md`, PR1): its entries are `±1`. -/
theorem not_isMinOn_parityKernel {Ω : Set (Finset κ × (κ → Bool))} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = parityKernel κ ij.1 ij.2}
      (parityKernel κ) := by
  obtain ⟨⟨S, β⟩, hSβ⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  refine isFlat_parityKernel.not_isMinOn_of_ne_zero (by positivity) hSβ ?_
  rw [parityKernel, of_apply, Complex.ofReal_ne_zero]
  exact Finset.prod_ne_zero_iff.2 fun q _ => by split_ifs <;> norm_num

/-- Three keys: the `8 × 8` parity kernel is flat and has rank `8`. -/
example : (parityKernel (Fin 3)).rank = 2 ^ 3 := by simpa using rank_parityKernel (κ := Fin 3)

example {Ω : Set (Finset (Fin 3) × (Fin 3 → Bool))} (hΩ : Ω ≠ Set.univ) :=
  not_isMinOn_parityKernel hΩ

end Transformer.Grokking.FlatSpectrum.Recall

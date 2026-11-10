module

public import Transformer.Grokking.FlatSpectrum.FlatSupport
public import Transformer.Grokking.FlatSpectrum.Recall.Defs

/-!
# The recall kernel is flat

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), items AR2
and AR3. Write `m = |κ|` for the number of keys and `c = |ν|` for the number of values.

AR2. A dictionary is a key's value and the values of the other keys, independently, so

    ∑_π f(π q) = c^{m−1} ∑_v f(v),      ∑_π f(π q) g(π q') = (∑_v f(v)) · ∑_ρ g(ρ q')  (q' ≠ q)

(`sum_apply`, `sum_apply_mul_apply`, with `ρ` over the dictionaries of the other keys). For a
centered readout, `∑_v φ(v) = 0`, the rows of `K(q, π) = φ(π q)` are orthogonal:

    K Kᴴ = (c^{m−1} ∑_v |φ(v)|²) I          (`recallKernel_mul_conjTranspose`).

So the recall kernel is flat (`isFlat_recallKernel`): rank `m` and `r_eff K = m`
(`rank_recallKernel`, `effectiveRank_recallKernel`), every key a direction of the same weight.

AR3. Theorem A for flat kernels of any shape: the recall kernel is a minimum of M from `Ω`
exactly when every unseen pair `(q, π)` has `φ(π q) = 0` (`isMinOn_recallKernel_iff`). For a
readout without zeros it is a minimum from no proper `Ω` (`not_isMinOn_recallKernel`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix
open scoped ComplexOrder

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- Summing a function of one binding over the dictionaries: `∑_π f(π q) = c^{m−1} ∑_v f(v)`
(`review/flat_spectrum.md`, AR2). -/
lemma sum_apply {R : Type*} [CommSemiring R] (q : κ) (f : ν → R) :
    ∑ π : κ → ν, f (π q) = (Fintype.card ν : R) ^ (Fintype.card κ - 1) * ∑ v, f v := by
  rw [Fintype.sum_equiv (Equiv.funSplitAt q ν) (fun π : κ → ν => f (π q))
    (fun x : ν × ({j // j ≠ q} → ν) => f x.1) fun _ => rfl, Fintype.sum_prod_type]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← Finset.mul_sum]
  rw [Fintype.card_fun, Fintype.card_subtype_compl, Fintype.card_subtype_eq, Nat.cast_pow]

/-- Summing a product over two bindings of different keys: the sum over the first binding
factors out (`review/flat_spectrum.md`, AR2). -/
lemma sum_apply_mul_apply {R : Type*} [CommSemiring R] {q q' : κ} (hq : q' ≠ q) (f g : ν → R) :
    ∑ π : κ → ν, f (π q) * g (π q') = (∑ v, f v) * ∑ ρ : {j // j ≠ q} → ν, g (ρ ⟨q', hq⟩) := by
  rw [Fintype.sum_equiv (Equiv.funSplitAt q ν) (fun π : κ → ν => f (π q) * g (π q'))
    (fun x : ν × ({j // j ≠ q} → ν) => f x.1 * g (x.2 ⟨q', hq⟩)) fun _ => rfl,
    Fintype.sum_prod_type, Finset.sum_mul_sum]

/-- Two keys and two values: the four dictionaries bind the key `0` to each value twice. -/
example (f : Fin 2 → ℝ) : ∑ π : Fin 2 → Fin 2, f (π 0) = 2 * (f 0 + f 1) := by
  rw [sum_apply, Fin.sum_univ_two]
  norm_num

example (f g : Fin 2 → ℝ) := sum_apply_mul_apply (κ := Fin 2) (q := 0) (q' := 1) (by decide) f g

/-- **The rows of the recall kernel of a centered readout are orthogonal**:
`K Kᴴ = (c^{m−1} ∑_v |φ(v)|²) I` (`review/flat_spectrum.md`, AR2). -/
theorem recallKernel_mul_conjTranspose {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) :
    recallKernel (κ := κ) φ * (recallKernel φ)ᴴ =
      (((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2 : ℝ) : ℂ) •
        (1 : Matrix κ κ ℂ) := by
  ext q q'
  simp only [mul_apply, conjTranspose_apply, recallKernel, of_apply, Matrix.smul_apply,
    smul_eq_mul]
  by_cases h : q = q'
  · subst h
    rw [one_apply_eq, mul_one, sum_apply q fun v => φ v * star (φ v)]
    push_cast
    simp only [RCLike.star_def, Complex.mul_conj']
  · rw [one_apply_ne h, mul_zero, sum_apply_mul_apply (Ne.symm h) φ fun v => star (φ v), hφ,
      zero_mul]

variable [DecidableEq ν]

/-- **The recall kernel of a centered readout is flat** (`review/flat_spectrum.md`, AR2). -/
theorem isFlat_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) :
    IsFlat (recallKernel (κ := κ) φ)
      ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2) :=
  Or.inr (recallKernel_mul_conjTranspose hφ)

omit [DecidableEq κ] [DecidableEq ν] in
/-- The squared singular value `c^{m−1} ∑_v |φ(v)|²` is positive for a readout that is not
zero. -/
lemma scale_pos {φ : ν → ℂ} {v : ν} (h0 : φ v ≠ 0) :
    0 < (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2 :=
  mul_pos (pow_pos (Nat.cast_pos.2 (Fintype.card_pos_iff.2 ⟨v⟩)) _)
    (Finset.sum_pos' (fun _ _ => sq_nonneg _) ⟨v, Finset.mem_univ v, by positivity⟩)

/-- The readout `±1` of two values is centered and not zero. -/
example : ∑ v : Fin 2, (![1, -1] : Fin 2 → ℂ) v = 0 ∧ (![1, -1] : Fin 2 → ℂ) 0 ≠ 0 := by
  simp [Fin.sum_univ_two]

example := recallKernel_mul_conjTranspose (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ))
  (by simp [Fin.sum_univ_two])

example := isFlat_recallKernel (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ))
  (by simp [Fin.sum_univ_two])

example := scale_pos (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ)) (v := 0) (by simp)

omit [DecidableEq ν] in
/-- **The recall kernel of a centered readout has rank `m`**, the number of keys
(`review/flat_spectrum.md`, AR2). -/
theorem rank_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) {v : ν} (h0 : φ v ≠ 0) :
    (recallKernel (κ := κ) φ).rank = Fintype.card κ :=
  rank_eq_card_of_mul_conjTranspose_self (scale_pos h0).ne' (recallKernel_mul_conjTranspose hφ)

/-- **The effective rank of recall is the number of keys**: `r_eff K = m`
(`review/flat_spectrum.md`, AR2). -/
theorem effectiveRank_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) {v : ν} (h0 : φ v ≠ 0) :
    effectiveRank (recallKernel (κ := κ) φ) = Fintype.card κ := by
  rw [effectiveRank_eq_rank (scale_pos h0) (isFlat_recallKernel hφ).gram_mul_gram,
    rank_recallKernel hφ h0]

example := rank_recallKernel (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ))
  (by simp [Fin.sum_univ_two]) (v := 0) (by simp)

example := effectiveRank_recallKernel (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ))
  (by simp [Fin.sum_univ_two]) (v := 0) (by simp)

/-- **Theorem A for recall** (`review/flat_spectrum.md`, AR3): the recall kernel of a centered
readout is a minimum of the nuclear norm among the matrices equal to it on `Ω` exactly when the
readout vanishes at every unseen pair of a query and a dictionary. -/
theorem isMinOn_recallKernel_iff {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) {v : ν} (h0 : φ v ≠ 0)
    {Ω : Set (κ × (κ → ν))} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = recallKernel φ ij.1 ij.2}
      (recallKernel φ) ↔ ∀ q π, (q, π) ∉ Ω → φ (π q) = 0 :=
  (isFlat_recallKernel hφ).isMinOn_iff (scale_pos h0)

/-- **Recall with a readout without zeros is never a minimum of M**
(`review/flat_spectrum.md`, AR3): from every proper set of pairs of a query and a dictionary,
some matrix equal to the recall kernel on the seen pairs has a smaller nuclear norm. -/
theorem not_isMinOn_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0) (h0 : ∀ v, φ v ≠ 0)
    {Ω : Set (κ × (κ → ν))} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = recallKernel φ ij.1 ij.2}
      (recallKernel φ) := by
  obtain ⟨⟨q, π⟩, hqπ⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  exact (isFlat_recallKernel hφ).not_isMinOn_of_ne_zero (scale_pos (h0 (π q))) hqπ (h0 (π q))

/-- Three keys with values `±1`: recall is a minimum of M from no proper set of pairs. -/
example {Ω : Set (Fin 3 × (Fin 3 → Fin 2))} (hΩ : Ω ≠ Set.univ) :=
  not_isMinOn_recallKernel (φ := (![1, -1] : Fin 2 → ℂ)) (by simp [Fin.sum_univ_two])
    (fun v => by fin_cases v <;> simp) hΩ

example {Ω : Set (Fin 3 × (Fin 3 → Fin 2))} :=
  isMinOn_recallKernel_iff (φ := (![1, -1] : Fin 2 → ℂ)) (by simp [Fin.sum_univ_two])
    (v := 0) (by simp) (Ω := Ω)

end Transformer.Grokking.FlatSpectrum.Recall

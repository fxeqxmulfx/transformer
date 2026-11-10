module

public import Transformer.Grokking.FlatSpectrum.Recall.Subkernel

/-!
# A pairwise uniform family of dictionaries is as hard as all of them

Source: `review/flat_spectrum.md`, *Families of dictionaries and the multi-query task* (added),
items FD1 to FD3, for the task of Arora et al., arXiv:2312.04927v1, §3, and the scope item of
AR10 on the family of contexts.

FD1. A family `D : ι → κ → ν` of dictionaries is **pairwise uniform** of index `λ` when every
key takes every value `λ c` times in it and every two keys take every pair of values `λ` times
(`IsPairwiseUniform`): an orthogonal array of strength two. Then `∑_i f(D_i q) = λ c ∑_v f(v)`
and, for `q ≠ q'`, `∑_i f(D_i q) g(D_i q') = λ (∑ f) (∑ g)` (`IsPairwiseUniform.sum_comp`,
`IsPairwiseUniform.sum_comp_mul_comp`).

FD2. So for a centered readout the kernel `K(q, i) = φ(D_i q)` of the family has orthogonal
rows of squared length `λ c ∑_v φ(v)²` (`familyKernel`, `isFlat_familyKernel`). By Theorem A it
is a minimum of M from no proper set of its entries when `φ` has no zeros
(`not_isMinOn_familyKernel`).

FD3. A readout of cut width `d` misses recall on the family by at least
`λ c ∑_v (φ(v) − μ)² (min(m, |ι|) − d)` (`HasCutWidth.mul_le_sum_sq_of_isPairwiseUniform`), and
exact recall on it has `min(m, |ι|) ≤ d` (`HasCutWidth.card_le_of_recall_of_isPairwiseUniform`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix

variable {ι κ ν : Type*} [Fintype ι] [Fintype ν] [DecidableEq ν]

/-- **A pairwise uniform family of dictionaries** of index `lam`
(`review/flat_spectrum.md`, FD1): an orthogonal array of strength two. -/
structure IsPairwiseUniform (D : ι → κ → ν) (lam : ℕ) : Prop where
  /-- Every key takes every value `λ c` times. -/
  single : ∀ q v, (Finset.univ.filter fun i => D i q = v).card = lam * Fintype.card ν
  /-- Two different keys take every pair of values `λ` times. -/
  pair : ∀ q q', q ≠ q' → ∀ v v',
    (Finset.univ.filter fun i => D i q = v ∧ D i q' = v').card = lam

variable {D : ι → κ → ν} {lam : ℕ}

/-- A sum over the family of a function of one binding (`review/flat_spectrum.md`, FD1). -/
lemma IsPairwiseUniform.sum_comp (hD : IsPairwiseUniform D lam) (q : κ) (f : ν → ℝ) :
    ∑ i, f (D i q) = lam * Fintype.card ν * ∑ v, f v := by
  rw [← Finset.sum_fiberwise Finset.univ (fun i => D i q) fun i => f (D i q), Finset.mul_sum]
  refine Finset.sum_congr rfl fun v _ => ?_
  have h1 : ∑ i ∈ Finset.univ.filter (fun i => D i q = v), f (D i q) =
      ∑ i ∈ Finset.univ.filter (fun i => D i q = v), f v :=
    Finset.sum_congr rfl fun i hi => by rw [(Finset.mem_filter.1 hi).2]
  rw [h1, Finset.sum_const, hD.single q v, nsmul_eq_mul, Nat.cast_mul]

/-- A sum over the family of a product of functions of two bindings
(`review/flat_spectrum.md`, FD1). -/
lemma IsPairwiseUniform.sum_comp_mul_comp (hD : IsPairwiseUniform D lam) {q q' : κ}
    (hqq' : q ≠ q') (f g : ν → ℝ) :
    ∑ i, f (D i q) * g (D i q') = lam * (∑ v, f v) * ∑ v, g v := by
  rw [← Finset.sum_fiberwise Finset.univ (fun i => (D i q, D i q'))
    fun i => f (D i q) * g (D i q'), Fintype.sum_prod_type, mul_assoc, Finset.sum_mul_sum,
    Finset.mul_sum]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun v' _ => ?_
  have h1 : ∑ i ∈ Finset.univ.filter (fun i => (D i q, D i q') = (v, v')),
      f (D i q) * g (D i q') =
        ∑ i ∈ Finset.univ.filter (fun i => (D i q, D i q') = (v, v')), f v * g v' :=
    Finset.sum_congr rfl fun i hi => by
      have h := Prod.mk.inj (Finset.mem_filter.1 hi).2
      rw [h.1, h.2]
  have h2 : (Finset.univ.filter fun i => (D i q, D i q') = (v, v')) =
      Finset.univ.filter fun i => D i q = v ∧ D i q' = v' := by
    ext i
    simp
  rw [h1, Finset.sum_const, h2, hD.pair q q' hqq' v v', nsmul_eq_mul]

/-- The kernel of recall on a family of dictionaries: `K(q, i) = φ(D_i q)`
(`review/flat_spectrum.md`, FD2). -/
def familyKernel (φ : ν → ℝ) (D : ι → κ → ν) : Matrix κ ι ℝ :=
  Matrix.of fun q i => φ (D i q)

variable [DecidableEq κ] {φ : ν → ℝ}

/-- The rows of the kernel of a pairwise uniform family are orthogonal, for a centered
readout. -/
lemma IsPairwiseUniform.sum_mul (hD : IsPairwiseUniform D lam) (hφ : ∑ v, φ v = 0)
    (q q' : κ) : ∑ i, φ (D i q) * φ (D i q') =
      if q = q' then lam * Fintype.card ν * ∑ v, φ v ^ 2 else 0 := by
  split_ifs with h
  · subst h
    have h1 := hD.sum_comp q fun v => φ v * φ v
    simp only [sq]
    exact h1
  · rw [hD.sum_comp_mul_comp h φ φ, hφ, mul_zero]

variable [Fintype κ] [DecidableEq ι]

/-- **The kernel of a pairwise uniform family is flat** (`review/flat_spectrum.md`, FD2), for a
centered readout, with squared singular value `λ c ∑_v φ(v)²`. -/
theorem isFlat_familyKernel (hD : IsPairwiseUniform D lam) (hφ : ∑ v, φ v = 0) :
    IsFlat ((familyKernel φ D).map ((↑) : ℝ → ℂ)) (lam * Fintype.card ν * ∑ v, φ v ^ 2) := by
  refine Or.inr ?_
  ext q q'
  simp only [mul_apply, conjTranspose_apply, map_apply, familyKernel, of_apply,
    Matrix.smul_apply, smul_eq_mul, one_apply, RCLike.star_def, Complex.conj_ofReal,
    ← Complex.ofReal_mul, ← Complex.ofReal_sum]
  rw [hD.sum_mul hφ]
  split_ifs <;> simp

/-- **The kernel of a pairwise uniform family is a minimum of M from no proper set of its
entries** (`review/flat_spectrum.md`, FD2), for a centered readout without zeros. -/
theorem not_isMinOn_familyKernel (hD : IsPairwiseUniform D lam) (hlam : 0 < lam)
    (hφ : ∑ v, φ v = 0) (hφ0 : ∀ v, φ v ≠ 0) {Ω : Set (κ × ι)} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm
      {X | ∀ qi ∈ Ω, X qi.1 qi.2 = (familyKernel φ D).map ((↑) : ℝ → ℂ) qi.1 qi.2}
      ((familyKernel φ D).map ((↑) : ℝ → ℂ)) := by
  obtain ⟨⟨q, i⟩, hqi⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  have hsq : 0 < ∑ v, φ v ^ 2 := Finset.sum_pos' (fun _ _ => sq_nonneg _)
    ⟨D i q, Finset.mem_univ _, sq_pos_iff.2 (hφ0 _)⟩
  have hν : 0 < Fintype.card ν := Fintype.card_pos_iff.2 ⟨D i q⟩
  refine (isFlat_familyKernel hD hφ).not_isMinOn_of_ne_zero (by positivity) hqi ?_
  simpa [familyKernel] using hφ0 (D i q)

/-- **The error of a readout of cut width `d` on a pairwise uniform family**
(`review/flat_spectrum.md`, FD3): at least `λ c ∑_v (φ(v) − μ)² (min(m, |ι|) − d)`, with `μ`
the mean of `φ`. -/
theorem HasCutWidth.mul_le_sum_sq_of_isPairwiseUniform {d : ℕ} {F : κ → (κ → ν) → ℝ}
    (hF : HasCutWidth d F) (hD : IsPairwiseUniform D lam) {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    lam * Fintype.card ν * (∑ v, (φ v - μ) ^ 2) *
      ((min (Fintype.card κ) (Fintype.card ι) : ℕ) - (d : ℕ)) ≤
        ∑ q, ∑ i, (φ (D i q) - F q (D i)) ^ 2 := by
  rcases (by positivity : (0 : ℝ) ≤ lam * Fintype.card ν * ∑ v, (φ v - μ) ^ 2).eq_or_lt
    with h0 | hpos
  · rw [← h0, zero_mul]
    positivity
  have hF' : HasCutWidth d fun q x => F q x - μ := by
    obtain ⟨g, a, b, h⟩ := hF
    exact ⟨fun q => g q - μ, a, b, fun q x => by simp only [h]; ring⟩
  refine (hF'.mul_le_sum_sq_of_isFlat_of_sum_eq_zero (fun q : κ => q) D hpos
    (isFlat_familyKernel hD hμ) fun q => ?_).trans_eq
      (Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun i _ => ?_)
  · exact (hD.sum_comp q fun v => φ v - μ).trans (by simp only [hμ, mul_zero])
  · simp only [familyKernel, of_apply]
    ring

/-- **Exact recall on a pairwise uniform family takes cut width `min(m, |ι|)`**
(`review/flat_spectrum.md`, FD3), for a readout `φ` that is not constant. -/
theorem HasCutWidth.card_le_of_recall_of_isPairwiseUniform {d : ℕ} {F : κ → (κ → ν) → ℝ}
    (hF : HasCutWidth d F) (hD : IsPairwiseUniform D lam) (hlam : 0 < lam) {v v' : ν}
    (hφ : φ v ≠ φ v') (h : ∀ q i, F q (D i) = φ (D i q)) :
    min (Fintype.card κ) (Fintype.card ι) ≤ d := by
  have hν : 0 < Fintype.card ν := Fintype.card_pos_iff.2 ⟨v⟩
  have hν' : (Fintype.card ν : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hν.ne'
  have hμ : ∑ w, (φ w - (∑ w, φ w) / Fintype.card ν) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_div_cancel₀ _ hν', sub_self]
  obtain ⟨w, hw⟩ : ∃ w, φ w ≠ (∑ w, φ w) / Fintype.card ν := by
    by_contra hall
    push Not at hall
    exact hφ ((hall v).trans (hall v').symm)
  have hsq : 0 < ∑ w, (φ w - (∑ w, φ w) / Fintype.card ν) ^ 2 :=
    Finset.sum_pos' (fun _ _ => sq_nonneg _)
      ⟨w, Finset.mem_univ w, sq_pos_iff.2 (sub_ne_zero.2 hw)⟩
  have hb := hF.mul_le_sum_sq_of_isPairwiseUniform hD hμ
  rw [show ∑ q, ∑ i, (φ (D i q) - F q (D i)) ^ 2 = 0 by simp [h]] at hb
  have hsub : ((min (Fintype.card κ) (Fintype.card ι) : ℕ) : ℝ) - (d : ℕ) ≤ 0 :=
    nonpos_of_mul_nonpos_right hb (by positivity)
  exact_mod_cast sub_nonpos.1 hsub

/-- All four dictionaries of two keys with two values are pairwise uniform of index 1. -/
lemma isPairwiseUniform_two : IsPairwiseUniform (fun (π : Fin 2 → Bool) => π) 1 :=
  ⟨by decide, by decide⟩

/-- The readout `±1` of a value in `Bool`. -/
def signValue (b : Bool) : ℝ := if b then 1 else -1

example := isPairwiseUniform_two.sum_comp 0 signValue
example := isPairwiseUniform_two.sum_comp_mul_comp (q := 0) (q' := 1) (by decide) signValue
  fun _ => 1
example := isPairwiseUniform_two.sum_mul (φ := signValue) (by simp [signValue]) 0 1
example := isFlat_familyKernel isPairwiseUniform_two (φ := signValue) (by simp [signValue])
example {Ω : Set (Fin 2 × (Fin 2 → Bool))} (hΩ : Ω ≠ Set.univ) :=
  not_isMinOn_familyKernel isPairwiseUniform_two one_pos (φ := signValue)
    (by simp [signValue]) (fun v => by cases v <;> norm_num [signValue]) hΩ

/-- Recall itself, at cut width two, on the four dictionaries. -/
example := (hasCutWidth_recall (κ := Fin 2) (d := 2) (by simp)
  signValue).mul_le_sum_sq_of_isPairwiseUniform isPairwiseUniform_two (φ := signValue)
    (μ := 0) (by simp [signValue])

example : min (Fintype.card (Fin 2)) (Fintype.card (Fin 2 → Bool)) ≤ 2 :=
  (hasCutWidth_recall (κ := Fin 2) (d := 2) (by simp)
    signValue).card_le_of_recall_of_isPairwiseUniform isPairwiseUniform_two one_pos
      (φ := signValue) (v := true) (v' := false) (by norm_num [signValue]) fun _ _ => rfl

end Transformer.Grokking.FlatSpectrum.Recall

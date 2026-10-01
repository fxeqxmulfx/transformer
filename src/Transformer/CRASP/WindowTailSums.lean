/-
# Exact tail corrections from finite recent-state profiles

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
Long prefixes have an all-ordinary recent window. Short prefixes add the
known BOS correction at their last available distance.
-/

import Transformer.CRASP.FiniteTailSums

namespace Transformer.CRASP.WindowTailSums

universe u
variable {α : Type u} {m Δ : ℕ}

/-- The finite integer correction associated with a recent-state profile (F). -/
def correction (f : ℕ → α → ℤ) (B : α → ℤ) (P : Fin m → α) : ℤ :=
  ∑ δ : Fin m, (f δ.val (P δ) - B (P δ))

/-- A long prefix's profile gives its exact correction to the stable count.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem sum_long (f : ℕ → α → ℤ) (B : α → ℤ)
    (hB : ∀ δ, Δ ≤ δ → ∀ a, f δ a = B a) (q : ℕ → α) (b : α) (hb : q 0 = b)
    (i : ℕ) (hi : Δ ≤ i) (P : Fin Δ → α) (hP : ∀ δ, q (i - δ.val) = P δ) :
    (∑ j ∈ Finset.Icc 0 i, f (i - j) (q j)) =
      B b + (∑ j ∈ Finset.Icc 1 i, B (q j)) + correction f B P := by
  rw [sum_stable_tail f B Δ hB q i, hb]
  congr 1
  unfold correction
  symm
  refine Finset.sum_bij (fun δ _ => δ.val) ?_ ?_ ?_ ?_
  · intro δ hδ
    exact Finset.mem_range.mpr δ.isLt
  · intro δ hδ ε hε heq
    exact Fin.ext heq
  · intro δ hδ
    exact ⟨⟨δ, Finset.mem_range.mp hδ⟩, Finset.mem_univ _, rfl⟩
  · intro δ hδ
    rw [ite_eq_left (le_trans δ.isLt.le hi), hP δ]

/-- Truncating a finite correction sum to the available prefix (Appendix F). -/
theorem sum_range_available (g : ℕ → ℤ) (m Δ : ℕ) (hm : m < Δ) :
    (∑ δ ∈ Finset.range Δ, if δ ≤ m then g δ else 0) =
      ∑ δ ∈ Finset.range (m + 1), g δ := by
  have hs : Finset.range (m + 1) ⊆ Finset.range Δ := Finset.range_mono (by omega)
  have hz : ∀ δ ∈ Finset.range Δ, δ ∉ Finset.range (m + 1) →
      (if δ ≤ m then g δ else 0) = 0 := by
    intro δ hδ hn
    have hd : ¬ δ ≤ m := by
      simp only [Finset.mem_range] at hn
      omega
    simp [hd]
  rw [← Finset.sum_subset hs hz]
  apply Finset.sum_congr rfl
  intro δ hδ
  have hd := Finset.mem_range.mp hδ
  rw [ite_eq_left (by omega)]

/-- A short prefix's profile plus its BOS correction gives the exact sum.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem sum_short (f : ℕ → α → ℤ) (B : α → ℤ)
    (hB : ∀ δ, Δ ≤ δ → ∀ a, f δ a = B a) (q : ℕ → α) (b : α) (hb : q 0 = b)
    (hm : m < Δ) (P : Fin m → α) (hP : ∀ δ, q (m - δ.val) = P δ) :
    (∑ j ∈ Finset.Icc 0 m, f (m - j) (q j)) =
      B b + (∑ j ∈ Finset.Icc 1 m, B (q j)) +
        (correction f B P + (f m b - B b)) := by
  rw [sum_stable_tail f B Δ hB q m, hb, sum_range_available _ m Δ hm,
    Finset.sum_range_succ, Nat.sub_self, hb]
  congr 2
  unfold correction
  symm
  refine Finset.sum_bij (fun δ _ => δ.val) ?_ ?_ ?_ ?_
  · intro δ hδ
    exact Finset.mem_range.mpr δ.isLt
  · intro δ hδ ε hε heq
    exact Fin.ext heq
  · intro δ hδ
    exact ⟨⟨δ, Finset.mem_range.mp hδ⟩, Finset.mem_univ _, rfl⟩
  · intro δ hδ
    rw [hP δ]

/-- Stable coefficients, profiles and both boundary regimes have a witness (F). -/
example : (∀ δ : ℕ, 1 ≤ δ → ∀ _ : Unit, (1 : ℤ) = 1) ∧
    (fun _ : ℕ => ()) 0 = () ∧ 1 ≤ 2 ∧ 1 < 2 ∧
    (∀ δ : Fin 1, (fun _ : ℕ => ()) (2 - δ.val) = ()) := by simp

end Transformer.CRASP.WindowTailSums

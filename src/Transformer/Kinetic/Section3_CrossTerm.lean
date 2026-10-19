module

public import Transformer.Kinetic.Section3_InnerCovariance

/-!
# How the cross term moves

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
cross term `δ_l G · Δ_l H` of the swap identity moves with the coordinate `m`
of `x` by at most `s²_{l,m}(G) s_l(H) + s_l(G) s²_{l,m}(H)`
(`abs_crossTerm_sub_le`). Here `s_l` bounds the first differences, `s²_{l,m}`
for `l ≠ m` the mixed second differences, and `s²_{l,l} ≥ s_l`. This is the
deterministic content of the source's second-order chain rule
`eq:chain-rule-Glaub-2nd` applied to a product.
-/

@[expose] public section

namespace Transformer
namespace Kinetic

variable {N : ℕ}

/-- A bound on mixed second differences is nonnegative. -/
theorem second_diff_bound_nonneg {G : (Idx N → ℝ) → ℝ} {s2G : Idx N → Idx N → ℝ}
    (hs2G : ∀ x l m z z', m ≠ l → |G x - G (Function.update x l z) - G (Function.update x m z') +
      G (Function.update (Function.update x l z) m z')| ≤ s2G l m)
    {l m : Idx N} (hml : m ≠ l) : 0 ≤ s2G l m := by
  have := hs2G 0 l m ((0 : Idx N → ℝ) l) ((0 : Idx N → ℝ) m) hml
  simpa using this

/-- **The cross term moves little.** -/
theorem abs_crossTerm_sub_le {G H : (Idx N → ℝ) → ℝ} {sG sH : Idx N → ℝ}
    {s2G s2H : Idx N → Idx N → ℝ}
    (hsG : ∀ x l z, |G x - G (Function.update x l z)| ≤ sG l)
    (hsH : ∀ x l z, |H x - H (Function.update x l z)| ≤ sH l)
    (hs2G : ∀ x l m z z', m ≠ l → |G x - G (Function.update x l z) - G (Function.update x m z') +
      G (Function.update (Function.update x l z) m z')| ≤ s2G l m)
    (hs2H : ∀ x l m z z', m ≠ l → |H x - H (Function.update x l z) - H (Function.update x m z') +
      H (Function.update (Function.update x l z) m z')| ≤ s2H l m)
    (hdG : ∀ l, sG l ≤ s2G l l) (hdH : ∀ l, sH l ≤ s2H l l)
    (x x' : Idx N → ℝ) (l m : Idx N) (z : ℝ) :
    |crossTerm G H l (x, x') - crossTerm G H l (Function.update x m z, x')| ≤
      s2G l m * sH l + sG l * s2H l m := by
  set a := G x - G (Function.update x l (x' l))
  set b := H (hybrid l x x') - H (hybrid ((l : ℕ) + 1) x x')
  set a' := G (Function.update x m z) - G (Function.update (Function.update x m z) l (x' l))
  set b' := H (hybrid l (Function.update x m z) x') -
    H (hybrid ((l : ℕ) + 1) (Function.update x m z) x')
  have hcross : crossTerm G H l (x, x') - crossTerm G H l (Function.update x m z, x') =
      (a - a') * b + a' * (b - b') := by
    simp only [crossTerm, a, b, a', b']
    ring
  have hb : |b| ≤ sH l := by
    simp only [b]
    rw [hybrid_succ]
    exact hsH _ _ _
  have ha' : |a'| ≤ sG l := hsG _ _ _
  have hsH0 : 0 ≤ sH l := (abs_nonneg _).trans hb
  have hsG0 : 0 ≤ sG l := (abs_nonneg _).trans ha'
  have ha : |a - a'| ≤ s2G l m := by
    by_cases hml : m = l
    · subst hml
      simp only [a, a', Function.update_idem]
      rw [show G x - G (Function.update x m (x' m)) -
          (G (Function.update x m z) - G (Function.update x m (x' m))) =
          G x - G (Function.update x m z) by ring]
      exact (hsG _ _ _).trans (hdG m)
    · simp only [a, a']
      rw [Function.update_comm hml]
      rw [show G x - G (Function.update x l (x' l)) - (G (Function.update x m z) -
          G (Function.update (Function.update x l (x' l)) m z)) =
          G x - G (Function.update x l (x' l)) - G (Function.update x m z) +
          G (Function.update (Function.update x l (x' l)) m z) by ring]
      exact hs2G _ _ _ _ _ hml
  have hbb : |b - b'| ≤ s2H l m := by
    rcases lt_trichotomy (m : ℕ) l with hlt | heq | hgt
    · simp only [b, b']
      rw [hybrid_update_of_lt hlt, hybrid_update_of_lt (by omega : (m : ℕ) < (l : ℕ) + 1), sub_self,
        abs_zero]
      exact second_diff_bound_nonneg hs2H fun h => by subst h; omega
    · have hml : m = l := Fin.ext heq
      subst hml
      simp only [b, b']
      rw [hybrid_update_of_le le_rfl, hybrid_update_of_lt (by omega : (m : ℕ) < (m : ℕ) + 1),
        show H (hybrid m x x') - H (hybrid ((m : ℕ) + 1) x x') -
          (H (Function.update (hybrid m x x') m z) - H (hybrid ((m : ℕ) + 1) x x')) =
          H (hybrid m x x') - H (Function.update (hybrid m x x') m z) by ring]
      exact (hsH _ _ _).trans (hdH m)
    · have hml : m ≠ l := fun h => by subst h; omega
      simp only [b, b']
      rw [hybrid_update_of_le hgt.le, hybrid_update_of_le (by omega : (l : ℕ) + 1 ≤ m),
        hybrid_succ,
        show H (hybrid l x x') - H (Function.update (hybrid l x x') l (x' l)) -
          (H (Function.update (hybrid l x x') m z) -
            H (Function.update (Function.update (hybrid l x x') l (x' l)) m z)) =
          H (hybrid l x x') - H (Function.update (hybrid l x x') l (x' l)) -
            H (Function.update (hybrid l x x') m z) +
            H (Function.update (Function.update (hybrid l x x') l (x' l)) m z) by ring]
      exact hs2H _ _ _ _ _ hml
  rw [hcross]
  calc |(a - a') * b + a' * (b - b')| ≤ |a - a'| * |b| + |a'| * |b - b'| := by
        rw [← abs_mul, ← abs_mul]
        exact abs_add_le _ _
    _ ≤ s2G l m * sH l + sG l * s2H l m :=
        add_le_add (mul_le_mul ha hb (abs_nonneg _) ((abs_nonneg _).trans ha))
          (mul_le_mul ha' hbb (abs_nonneg _) hsG0)

/-- The hypotheses of `second_diff_bound_nonneg` hold for a constant observable with zero
bound. -/
example := second_diff_bound_nonneg (G := fun _ : Idx 2 → ℝ => 0) (s2G := fun _ _ => 0)
  (fun _ _ _ _ _ _ => by simp) (l := 0) (m := 1) (by decide)

/-- The hypotheses of `abs_crossTerm_sub_le` hold for constant observables with zero
bounds. -/
example (x x' : Idx 2 → ℝ) := abs_crossTerm_sub_le (G := fun _ => 0) (H := fun _ => 0)
  (sG := fun _ => 0) (sH := fun _ => 0) (s2G := fun _ _ => 0) (s2H := fun _ _ => 0)
  (fun _ _ _ => by simp) (fun _ _ _ => by simp) (fun _ _ _ _ _ _ => by simp)
  (fun _ _ _ _ _ _ => by simp) (fun _ => le_rfl) (fun _ => le_rfl) x x' 0 1 0

end Kinetic
end Transformer

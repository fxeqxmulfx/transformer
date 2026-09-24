/-
# Wendel's deterministic sign-pattern count

In linear general position, the number of strict sign patterns of `n` vectors
in `d` dimensions is `2 * Σ_{k<d} C(n-1,k)`. The proof uses the hyperplane
slice recurrence, the dimension-reduced projected family, and Pascal's rule.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCountBinomial

namespace Transformer.Perspective

/-- Deterministic form of Wendel's sign-count formula for a linearly
general-position vector family.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem strictSignCount_generalPosition :
    ∀ (n d : ℕ), 1 ≤ d → d ≤ n →
      ∀ (v : Idx n → EucSpace d),
      (∀ I : Finset (Idx n), I.card ≤ d →
        LinearIndependent ℝ (fun i : I => v i)) →
      strictSignCount d n v =
        2 * ∑ k ∈ Finset.range d, (n - 1).choose k := by
  intro n
  induction n with
  | zero =>
    intro d hd hdn v hgen
    omega
  | succ n ih =>
    intro d hd hdn v hgen
    by_cases hfull : d = n + 1
    · subst d
      have hli : LinearIndependent ℝ v := by
        let g : Idx (n + 1) → (Finset.univ : Finset (Idx (n + 1))) :=
          fun i => ⟨i, Finset.mem_univ i⟩
        have hg : Function.Injective g := by
          intro a b hab
          exact congrArg Subtype.val hab
        have h := (hgen Finset.univ (by simp)).comp g hg
        convert h using 1
        funext i
        rfl
      rw [strictSignCount_eq_pow_of_linearIndependent (n + 1) (n + 1)
        (by omega) v hli]
      exact (wendel_binomial_fullDim (n + 1) (by omega)).symm
    · have hdle : d ≤ n := by omega
      have hinit : ∀ I : Finset (Idx n), I.card ≤ d →
          LinearIndependent ℝ (fun i : I => Fin.init v i) :=
        linearGeneralPosition_init d n v hgen
      have hx : v (Fin.last n) ≠ 0 :=
        last_ne_zero_of_linearGeneralPosition d n hd v hgen
      cases d with
      | zero => omega
      | succ k =>
        have hgenSnoc : ∀ J : Finset (Idx (n + 1)), J.card ≤ k + 1 →
            LinearIndependent ℝ (fun j : J =>
              (Fin.snoc (Fin.init v) (v (Fin.last n)) :
                Idx (n + 1) → EucSpace (k + 1)) j.1) := by
          simpa only [Fin.snoc_init_self] using hgen
        have hrec := strictSignCount_snoc_dimRecurrence k n (Fin.init v)
          (v (Fin.last n)) hx
        rw [Fin.snoc_init_self] at hrec
        have hfirst := ih (k + 1) (by omega) (by omega) (Fin.init v) hinit
        by_cases hk : k = 0
        · subst k
          have hsecond := strictSignCount_zero n (by omega)
            (fun i => euclideanIsometryOfFinrank ((ℝ ∙ v (Fin.last n))ᗮ) 0
              (finrank_orthogonal_singleton_eucSpace 0 (v (Fin.last n)) hx)
              (((ℝ ∙ v (Fin.last n))ᗮ).orthogonalProjectionOnto (Fin.init v i)))
          rw [hrec, hfirst, hsecond]
          simp
        · have hkpos : 1 ≤ k := by omega
          have hproj := linearGeneralPosition_projectedEuclidean k n (Fin.init v)
            (v (Fin.last n)) hx hgenSnoc
          have hsecond := ih k hkpos (by omega)
            (fun i => euclideanIsometryOfFinrank ((ℝ ∙ v (Fin.last n))ᗮ) k
              (finrank_orthogonal_singleton_eucSpace k (v (Fin.last n)) hx)
              (((ℝ ∙ v (Fin.last n))ᗮ).orthogonalProjectionOnto (Fin.init v i)))
            hproj
          rw [hrec, hfirst, hsecond]
          have hbin := wendel_binomial_recurrence (n - 1) k
          have hn : n - 1 + 1 = n := by omega
          rw [hn] at hbin
          simp only [Nat.add_sub_cancel_right]
          omega

/-- The general-position hypotheses are satisfiable for the one-vector
standard-basis sample in one dimension. -/
example : ∀ I : Finset (Idx 1), I.card ≤ 1 →
    LinearIndependent ℝ (fun _ : I => (eOne : EucSpace 1)) := by
  intro I _
  have hli : LinearIndependent ℝ (fun _ : Idx 1 => (eOne : EucSpace 1)) := by
    rw [linearIndependent_unique_iff]
    intro h
    have hh := inner_eOne_eOne
    simp [h] at hh
  exact hli.comp (fun i : I => (i : Idx 1)) Subtype.val_injective

end Transformer.Perspective

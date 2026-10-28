module

public import Transformer.Grokking.FlatSpectrum.AffineEnergy
public import Mathlib.Algebra.Polynomial.Degree.Lemmas
public import Mathlib.Algebra.Polynomial.Roots

/-!
# Dilations of a polynomial: the off-diagonal affine energy of `P(a) + a b`

Source: `review/flat_spectrum.md`, *Readout-free bound for `P(a) + a b`* (added). For rows
`a = λ a'` and `a'` of `P(a) + a b`, the map between their fibers is `x ↦ λ x + μ_λ(a')` with
slope `μ_λ(a') = (P(λ a') − P(a')) / a'`. Call `λ` degenerate when `P(λ t) − P(t)` is linear
in `t`. For a nondegenerate `λ`, `P(λ t) − P(t) − μ t` is a nonzero polynomial of degree at
most `m = deg P` with the root `0`, so each slope value is taken by at most `m − 1` nonzero
`a'`. For `m ≥ 2` a degenerate `λ ≠ 0` has `λ^m = 1`, from the coefficient of `t^m`, so at most
`m − 1` of them differ from `1`. With the affine energy `p ‖h‖⁴` of a centered `h`, the
off-diagonal sum `∑_{a' ≠ 0} ∑_{λ ∉ {0, 1}} |E_h(λ, μ_λ(a'))|²` is at most
`(m − 1) p ‖h‖⁴ + (m − 1)(p − 1) ‖h‖⁴ = (m − 1)(2p − 1) ‖h‖⁴`
(`sum_sum_norm_sq_affineCorr_slope_le`, in `FiberTransport`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- The dilation difference `D_λ(t) = P(λ t) − P(t)`. -/
noncomputable def dilationDiff (P : (ZMod p)[X]) (l : ZMod p) : (ZMod p)[X] :=
  P.comp (C l * X) - P

lemma eval_dilationDiff (P : (ZMod p)[X]) (l t : ZMod p) :
    (dilationDiff P l).eval t = P.eval (l * t) - P.eval t := by
  simp [dilationDiff, eval_comp]

example := @eval_dilationDiff 5 ⟨Nat.prime_five⟩ X 2 1

/-- The slope `μ_λ(a) = (P(λ a) − P(a)) / a` of the map between the fibers of rows `λ a`
and `a`. -/
noncomputable def dilationSlope (P : (ZMod p)[X]) (l a : ZMod p) : ZMod p :=
  (P.eval (l * a) - P.eval a) * a⁻¹

/-- `λ` is degenerate for `P` when `P(λ t) − P(t)` is linear in `t`. -/
def IsDegenerate (P : (ZMod p)[X]) (l : ZMod p) : Prop :=
  ∃ c, dilationDiff P l = C c * X

/-- A sum over `s` of a nonnegative `F ∘ φ` is at most `k ∑ F` when every fiber of `φ` in `s`
has at most `k` points. -/
lemma sum_comp_le_of_card_fiber_le {ι κ : Type*} [Fintype κ] [DecidableEq κ] (s : Finset ι)
    (φ : ι → κ) {F : κ → ℝ} (hF : ∀ j, 0 ≤ F j) {k : ℕ} (hk : ∀ j, #{i ∈ s | φ i = j} ≤ k) :
    ∑ i ∈ s, F (φ i) ≤ k * ∑ j, F j := by
  rw [← sum_fiberwise s φ fun i => F (φ i), mul_sum]
  refine sum_le_sum fun j _ => ?_
  rw [sum_congr rfl fun i hi => by rw [(mem_filter.mp hi).2], sum_const, nsmul_eq_mul]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hk j) (hF j)

example := sum_comp_le_of_card_fiber_le (univ : Finset (Fin 2)) id (F := fun _ => (1 : ℝ))
  (fun _ => zero_le_one) (k := 1) fun j => by
    calc #{i ∈ (univ : Finset (Fin 2)) | id i = j} ≤ #({j} : Finset (Fin 2)) :=
          card_le_card fun i hi => by simpa using (mem_filter.mp hi).2
      _ = 1 := card_singleton j

/-- For `deg P = m ≥ 2`, a degenerate `λ ≠ 0` is an `m`-th root of unity: the coefficient of
`t^m` in `P(λ t) − P(t)` is `lc(P) (λ^m − 1)`. -/
lemma pow_eq_one_of_isDegenerate {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) {l : ZMod p}
    (hl0 : l ≠ 0) (hl : IsDegenerate P l) : l ^ P.natDegree = 1 := by
  obtain ⟨c, hc⟩ := hl
  have hq : (C l * X).natDegree = 1 := natDegree_C_mul_X l hl0
  have hnd : (P.comp (C l * X)).natDegree = P.natDegree := by rw [natDegree_comp, hq, mul_one]
  have hcomp : (P.comp (C l * X)).coeff P.natDegree = P.leadingCoeff * l ^ P.natDegree := by
    conv_lhs => rw [← hnd]
    rw [coeff_natDegree, leadingCoeff_comp (by rw [hq]; exact one_ne_zero),
      leadingCoeff_C_mul_X]
  have hcoeff := congrArg (fun q => q.coeff P.natDegree) hc
  have hm1 : P.natDegree ≠ 1 := by omega
  simp only [dilationDiff, coeff_sub, hcomp, coeff_natDegree, coeff_C_mul_X, hm1,
    ↓reduceIte] at hcoeff
  have hlc : P.leadingCoeff ≠ 0 :=
    leadingCoeff_ne_zero.mpr fun h => by simp [h] at hm
  have : P.leadingCoeff * (l ^ P.natDegree - 1) = 0 := by rw [mul_sub, mul_one]; exact hcoeff
  exact sub_eq_zero.mp ((mul_eq_zero.mp this).resolve_left hlc)

/-- For a nondegenerate `λ`, each slope value is taken by at most `deg P − 1` nonzero `a`. -/
lemma card_slope_fiber_le {P : (ZMod p)[X]} (hm : 1 ≤ P.natDegree)
    {l : ZMod p} (hl : ¬IsDegenerate P l) (μ : ZMod p) :
    #{a ∈ univ.erase (0 : ZMod p) | dilationSlope P l a = μ} ≤ P.natDegree - 1 := by
  set Q := dilationDiff P l - C μ * X with hQdef
  have hQ : Q ≠ 0 := fun h => hl ⟨μ, sub_eq_zero.mp h⟩
  have hdeg : Q.natDegree ≤ P.natDegree := by
    have hc : (P.comp (C l * X)).natDegree ≤ P.natDegree :=
      natDegree_comp_le.trans ((Nat.mul_le_mul_left _
        ((natDegree_C_mul_le l X).trans natDegree_X_le)).trans (mul_one _).le)
    have hl1 : (C μ * X).natDegree ≤ P.natDegree :=
      ((natDegree_C_mul_le μ X).trans natDegree_X_le).trans hm
    refine (natDegree_sub_le _ _).trans (max_le ((natDegree_sub_le _ _).trans ?_) hl1)
    exact max_le hc le_rfl
  have h0 : (0 : ZMod p) ∈ Q.roots.toFinset := by
    rw [Multiset.mem_toFinset, mem_roots hQ, IsRoot.def, hQdef, eval_sub, eval_dilationDiff]
    simp
  have hsub : {a ∈ univ.erase (0 : ZMod p) | dilationSlope P l a = μ} ⊆ Q.roots.toFinset.erase 0 := by
    intro a ha
    obtain ⟨ha0, hμ⟩ := mem_filter.mp ha
    refine mem_erase.mpr ⟨ne_of_mem_erase ha0, ?_⟩
    rw [Multiset.mem_toFinset, mem_roots hQ, IsRoot.def, hQdef, eval_sub, eval_dilationDiff,
      eval_mul, eval_C, eval_X, ← hμ, dilationSlope]
    field_simp [ne_of_mem_erase ha0]
    ring
  calc #{a ∈ univ.erase (0 : ZMod p) | dilationSlope P l a = μ} ≤ #(Q.roots.toFinset.erase 0) :=
        card_le_card hsub
    _ = #Q.roots.toFinset - 1 := card_erase_of_mem h0
    _ ≤ Multiset.card Q.roots - 1 := Nat.sub_le_sub_right (Multiset.toFinset_card_le _) 1
    _ ≤ Q.natDegree - 1 := Nat.sub_le_sub_right (card_roots' Q) 1
    _ ≤ P.natDegree - 1 := Nat.sub_le_sub_right hdeg 1

/-- At most `deg P − 1` degenerate `λ` lie outside `{0, 1}`. -/
lemma card_le_of_forall_isDegenerate {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    {S : Finset (ZMod p)} (hS : ∀ l ∈ S, l ≠ 0 ∧ l ≠ 1 ∧ IsDegenerate P l) :
    #S ≤ P.natDegree - 1 := by
  classical
  have hpos : 0 < P.natDegree := by omega
  have h1 : (1 : ZMod p) ∈ (nthRoots P.natDegree (1 : ZMod p)).toFinset := by
    rw [Multiset.mem_toFinset, mem_nthRoots hpos, one_pow]
  have hsub : S ⊆ (nthRoots P.natDegree (1 : ZMod p)).toFinset.erase 1 := by
    intro l hl
    obtain ⟨hl0, hl1, hdeg⟩ := hS l hl
    rw [mem_erase, Multiset.mem_toFinset, mem_nthRoots hpos]
    exact ⟨hl1, pow_eq_one_of_isDegenerate hm hl0 hdeg⟩
  calc #S ≤ #((nthRoots P.natDegree (1 : ZMod p)).toFinset.erase 1) := card_le_card hsub
    _ = #(nthRoots P.natDegree (1 : ZMod p)).toFinset - 1 := card_erase_of_mem h1
    _ ≤ P.natDegree - 1 :=
        Nat.sub_le_sub_right ((Multiset.toFinset_card_le _).trans (card_nthRoots _ _)) 1

section Examples

local instance fact_prime_5 : Fact (Nat.Prime 5) := ⟨Nat.prime_five⟩

example := pow_eq_one_of_isDegenerate (p := 5) (P := X ^ 2) (by rw [natDegree_X_pow])
  (l := 1) one_ne_zero ⟨0, by simp [dilationDiff]⟩

example := card_slope_fiber_le (p := 5) (P := X ^ 3) (by rw [natDegree_X_pow]; omega) (l := 2)
  (fun hd => by
    have h := pow_eq_one_of_isDegenerate (P := X ^ 3) (by rw [natDegree_X_pow]; omega)
      (show (2 : ZMod 5) ≠ 0 by decide) hd
    rw [natDegree_X_pow] at h
    exact absurd h (by decide)) 1

example := card_le_of_forall_isDegenerate (p := 5) (P := X ^ 2) (by rw [natDegree_X_pow])
  (S := {4}) fun l hl => by
    rw [mem_singleton.mp hl]
    refine ⟨by decide, by decide, 0, ?_⟩
    rw [dilationDiff, X_pow_comp, mul_pow, ← C_pow, show (4 : ZMod 5) ^ 2 = 1 by decide, C_1,
      one_mul, sub_self, C_0, zero_mul]

end Examples

end Transformer.Grokking.FlatSpectrum

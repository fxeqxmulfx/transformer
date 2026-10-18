/-
# Local uniformization: principalizing `(y, t^p)` by monomial maps

On `ℝⁿ⁺¹` with coordinates `(t, y)`, finitely many monomial maps with compact
domains cover the region `|t^p| < c |y|` (with `|t i| ≤ 1` and `|y| ≤ 1/c`)
so that in each chart `t^p = y · r` for a monomial `r` with `|r| ≤ c`. By
induction on `|p|`: either `|t j| ≤ c |y|` and `t j = y · s j`, or
`y = t j · y'` and the region for `p - e j` contains `(t, y')`. Auxiliary
for the Newton step of local uniformization for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Landscape.Uniformization.Snoc
import Transformer.Landscape.Uniformization.Factor

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

variable {n : ℕ}

theorem abs_monomial_le {p : Fin n → ℕ} {x b : Base n} (h : ∀ i, |x i| ≤ b i) :
    |monomial p x| ≤ monomial p b := by
  rw [monomial, Finset.abs_prod]
  exact Finset.prod_le_prod₀ (fun i _ => abs_nonneg _) fun i _ => by
    rw [abs_pow]
    exact pow_le_pow_left₀ (abs_nonneg _) (h i) _

theorem monomial_update_one (p : Fin n → ℕ) (j : Fin n) (a : ℝ) :
    monomial p (Function.update (fun _ => 1) j a) = a ^ p j := by
  classical
  simp [monomial, Function.update_apply, ite_pow]

/-- The chart `t j = s j · y`. -/
def blowTwo (j : Fin n) : Fin (n + 1) → Fin (n + 1) → ℕ :=
  Function.update (fun l => Pi.single l 1) j.castSucc
    (Pi.single j.castSucc 1 + Pi.single (Fin.last n) 1)

/-- Composition with the chart `y = t j · y'`. -/
def blowOne (j : Fin n) (E : Fin (n + 1) → Fin (n + 1) → ℕ) :
    Fin (n + 1) → Fin (n + 1) → ℕ :=
  Function.update E (Fin.last n) (E j.castSucc + E (Fin.last n))

theorem monomialMap_blowTwo (j : Fin n) (s : Base (n + 1)) :
    monomialMap (blowTwo j) s =
      Function.update s j.castSucc (s j.castSucc * s (Fin.last n)) := by
  funext l
  by_cases hl : l = j.castSucc
  · subst hl
    simp [monomialMap, blowTwo, monomial_add_exp, monomial_single]
  · simp [monomialMap, blowTwo, hl, monomial_single]

theorem monomialMap_blowOne (j : Fin n) (E : Fin (n + 1) → Fin (n + 1) → ℕ)
    (s : Base (n + 1)) :
    monomialMap (blowOne j E) s = Function.update (monomialMap E s) (Fin.last n)
      (monomialMap E s j.castSucc * monomialMap E s (Fin.last n)) := by
  funext l
  by_cases hl : l = Fin.last n
  · subst hl
    simp [monomialMap, blowOne, monomial_add_exp]
  · simp [monomialMap, blowOne, hl]

/-- The statement of the principalization, for one exponent `p`. -/
def PrincipalCharts (c : ℝ) (p : Fin n → ℕ)
    (C : Set ((Fin (n + 1) → Fin (n + 1) → ℕ) × (Fin (n + 1) → ℕ) × Set (Base (n + 1)))) :
    Prop :=
  C.Finite ∧
    (∀ e ∈ C, IsCompact e.2.2 ∧
      (∀ s, monomial p (Fin.init (monomialMap e.1 s)) =
        monomialMap e.1 s (Fin.last n) * monomial e.2.1 s) ∧
      ∀ s ∈ e.2.2, |monomial e.2.1 s| ≤ c) ∧
    ∀ (t : Base n) (y : ℝ), (∀ j, |t j| ≤ 1) → |y| ≤ c⁻¹ → |monomial p t| < c * |y| →
      ∃ e ∈ C, ∃ s ∈ e.2.2, monomialMap e.1 s = Fin.snoc t y

/-- The bounds of the chart `t j = s j · y`: `1` on `t l` for `l ≠ j`, `c` on `s j`
and `c⁻¹` on `y`. -/
noncomputable def blowTwoBound (c : ℝ) (j : Fin n) : Base (n + 1) :=
  Fin.snoc (Function.update (fun _ => 1) j c) c⁻¹

theorem init_monomialMap_blowTwo (j : Fin n) (s : Base (n + 1)) :
    Fin.init (monomialMap (blowTwo j) s) =
      Fin.init s * Function.update (fun _ => 1) j (s (Fin.last n)) := by
  rw [monomialMap_blowTwo, Fin.init_update_castSucc]
  funext l
  by_cases hl : l = j
  · subst hl
    simp [Fin.init]
  · simp [hl]

theorem monomial_blowTwo {p : Fin n → ℕ} {j : Fin n} (hj : p j ≠ 0) (s : Base (n + 1)) :
    monomial p (Fin.init (monomialMap (blowTwo j) s)) =
      monomialMap (blowTwo j) s (Fin.last n) * monomial (Fin.snoc p (p j - 1)) s := by
  have hlast : monomialMap (blowTwo j) s (Fin.last n) = s (Fin.last n) := by
    rw [monomialMap_blowTwo, Function.update_of_ne (Fin.castSucc_ne_last j).symm]
  rw [init_monomialMap_blowTwo, monomial_mul, monomial_update_one, hlast, monomial_snoc]
  obtain ⟨r, hr⟩ := Nat.exists_eq_succ_of_ne_zero hj
  rw [hr, Nat.succ_sub_one, pow_succ]
  ring

theorem monomial_blowTwoBound {c : ℝ} (hc : 0 < c) {p : Fin n → ℕ} {j : Fin n}
    (hj : p j ≠ 0) : monomial (Fin.snoc p (p j - 1)) (blowTwoBound c j) = c := by
  rw [blowTwoBound, monomial_snoc, Fin.init_snoc, Fin.snoc_last, monomial_update_one]
  obtain ⟨r, hr⟩ := Nat.exists_eq_succ_of_ne_zero hj
  rw [hr, Nat.succ_sub_one, pow_succ, inv_pow, mul_right_comm,
    mul_inv_cancel₀ (pow_ne_zero _ hc.ne'), one_mul]

theorem exists_principalCharts_aux {c : ℝ} (hc : 0 < c) (k : ℕ) :
    ∀ p : Fin n → ℕ, ∑ j, p j = k → ∃ C, PrincipalCharts c p C := by
  induction k with
  | zero =>
    intro p hp
    have hp0 : p = 0 := funext fun j => Finset.sum_eq_zero_iff.1 hp j (Finset.mem_univ j)
    subst hp0
    refine ⟨∅, Set.finite_empty, fun e he => he.elim, fun t y _ hy h => ?_⟩
    rw [monomial_zero_exp, abs_one] at h
    have : c * |y| ≤ c * c⁻¹ := by gcongr
    rw [mul_inv_cancel₀ hc.ne'] at this
    linarith
  | succ k ih =>
    intro p hp
    obtain ⟨j, hj⟩ : ∃ j, p j ≠ 0 := by
      by_contra hne
      push Not at hne
      simp [hne] at hp
    obtain ⟨C, hCfin, hC, hcov⟩ :=
      ih (p - Pi.single j 1) (by have := sum_sub_single hj; omega)
    refine ⟨insert (blowTwo j, Fin.snoc p (p j - 1),
        univ.pi fun l => Icc (-blowTwoBound c j l) (blowTwoBound c j l))
      ((fun e => (blowOne j e.1, e.2)) '' C), (hCfin.image _).insert _, ?_, ?_⟩
    · rintro e (rfl | ⟨e', he', rfl⟩)
      · refine ⟨isCompact_univ_pi fun l => isCompact_Icc, monomial_blowTwo hj, fun s hs => ?_⟩
        rw [← monomial_blowTwoBound hc hj]
        exact abs_monomial_le fun l => abs_le.2 (Set.mem_univ_pi.1 hs l)
      · obtain ⟨hK, hmon, hbd⟩ := hC e' he'
        refine ⟨hK, fun s => ?_, hbd⟩
        rw [monomialMap_blowOne, Fin.init_update_last, Function.update_self,
          monomial_eq_coord_mul hj, hmon s]
        simp only [Fin.init]
        ring
    · intro t y ht hy h
      by_cases hty : |t j| ≤ c * |y|
      · have hdiv : |t j / y| ≤ c := by
          rcases eq_or_ne y 0 with rfl | hy0
          · simp [hc.le]
          · rw [abs_div, div_le_iff₀ (abs_pos.2 hy0)]
            exact hty
        have hmul : t j / y * y = t j := by
          rcases eq_or_ne y 0 with rfl | hy0
          · simp only [abs_zero, mul_zero, abs_nonpos_iff] at hty
            simp [hty]
          · exact div_mul_cancel₀ _ hy0
        refine ⟨_, mem_insert _ _, Fin.snoc (Function.update t j (t j / y)) y, ?_, ?_⟩
        · refine Set.mem_univ_pi.2 fun l => ?_
          induction l using Fin.lastCases with
          | last => simpa [blowTwoBound] using abs_le.1 hy
          | cast l =>
            by_cases hl : l = j
            · subst hl
              simpa [blowTwoBound] using abs_le.1 hdiv
            · simpa [blowTwoBound, hl] using abs_le.1 (ht l)
        · rw [monomialMap_blowTwo, Fin.snoc_castSucc, Function.update_self, Fin.snoc_last, hmul,
            Fin.snoc_update, Function.update_idem, Function.update_eq_self_iff, Fin.snoc_castSucc]
      · push Not at hty
        have htj : 0 < |t j| := lt_of_le_of_lt (by positivity) hty
        have h1 : |y / t j| ≤ c⁻¹ := by
          rw [abs_div, div_le_iff₀ htj, inv_mul_eq_div, le_div_iff₀ hc]
          linarith [mul_comm |y| c]
        have h2 : |monomial (p - Pi.single j 1) t| < c * |y / t j| := by
          rw [abs_div, mul_div_assoc', lt_div_iff₀ htj]
          calc |monomial (p - Pi.single j 1) t| * |t j| = |monomial p t| := by
                rw [monomial_eq_coord_mul hj t, abs_mul, mul_comm]
            _ < c * |y| := h
        obtain ⟨e, he, s, hs, hes⟩ := hcov t (y / t j) ht h1 h2
        refine ⟨_, mem_insert_of_mem _ ⟨e, he, rfl⟩, s, hs, ?_⟩
        rw [monomialMap_blowOne, hes, Fin.snoc_castSucc, Fin.snoc_last,
          mul_div_cancel₀ _ (abs_pos.1 htj), Fin.update_snoc_last]

/-- Principalization of `(y, t^p)`: finitely many monomial maps with compact domains, in
each of which `t^p = y · r` with `|r| ≤ c`, cover the region `|t^p| < c |y|` of
`|t i| ≤ 1`, `|y| ≤ c⁻¹`. Auxiliary for the Newton step of local uniformization for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_principalCharts {c : ℝ} (hc : 0 < c) (p : Fin n → ℕ) :
    ∃ C, PrincipalCharts c p C :=
  exists_principalCharts_aux hc _ p rfl

end Transformer.Landscape.Uniformization

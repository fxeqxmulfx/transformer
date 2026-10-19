module

public import Transformer.Landscape.Uniformization.Factor

/-!
# Local uniformization: comparable exponents

If a difference `U (x - w)^A - V (x - w)^B` of two units times monomials has
normal crossings at `w`, then `A ≤ B` or `B ≤ A`. Dividing by the monomial
of `A ⊓ B` leaves a function with normal crossings in which the two
monomials have disjoint supports; if both were nontrivial, it would vanish at
`w` and hence on a coordinate hyperplane, where exactly one of its two terms
vanishes. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

@[expose] public section

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

variable {n : ℕ}

/-- The vector with zero coordinates on `T` and `s` elsewhere. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def testVec (T : Finset (Fin n)) (s : ℝ) : Base n := fun l => if l ∈ T then 0 else s

theorem monomial_testVec_of_mem {P : Fin n → ℕ} {T : Finset (Fin n)} {l : Fin n}
    (hl : l ∈ T) (hP : P l ≠ 0) (s : ℝ) : monomial P (testVec T s) = 0 := by
  by_contra h
  exact (monomial_ne_zero_iff P _).1 h l hP (by simp [testVec, hl])

theorem monomial_testVec_of_forall {P : Fin n → ℕ} {T : Finset (Fin n)}
    (hP : ∀ l ∈ T, P l = 0) (s : ℝ) : monomial P (testVec T s) = s ^ ∑ l, P l := by
  rw [← monomial_const]
  refine Finset.prod_congr rfl fun l _ => ?_
  by_cases hl : l ∈ T
  · simp [testVec, hl, hP l hl]
  · simp [testVec, hl]

/-- A property holding near `w` holds at some `w + testVec T s` with `s ≠ 0`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_testVec (T : Finset (Fin n)) {p : Base n → Prop} {w : Base n}
    (h : ∀ᶠ x in 𝓝 w, p x) : ∃ s : ℝ, s ≠ 0 ∧ p (w + testVec T s) := by
  have hc : Continuous (fun s : ℝ => w + testVec T s) := by
    refine continuous_const.add (continuous_pi fun l => ?_)
    by_cases hl : l ∈ T
    · simp only [testVec, hl, ite_true]
      exact continuous_const
    · simp only [testVec, hl, ite_false]
      exact continuous_id'
  have h0 : w + testVec T 0 = w := by
    ext l
    by_cases hl : l ∈ T <;> simp [testVec, hl]
  have ht : Tendsto (fun s : ℝ => w + testVec T s) (𝓝[≠] 0) (𝓝 w) :=
    (hc.tendsto' 0 w h0).mono_left nhdsWithin_le_nhds
  obtain ⟨s, hs, hs0⟩ := ((ht.eventually h).and self_mem_nhdsWithin).exists
  exact ⟨s, hs0, hs⟩

/-- If `U (x - w)^A - V (x - w)^B` has normal crossings at `w`, with units
`U` and `V`, then the exponents `A` and `B` are comparable. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem le_or_le_of_normalCrossingAt_sub {U V : Base n → ℝ} {w : Base n}
    {A B : Fin n → ℕ} (hU : AnalyticAt ℝ U w) (hUw : U w ≠ 0) (hV : AnalyticAt ℝ V w)
    (hVw : V w ≠ 0)
    (h : NormalCrossingAt (fun x => U x * monomial A (x - w) - V x * monomial B (x - w)) w) :
    A ≤ B ∨ B ≤ A := by
  classical
  by_contra hcon
  rw [not_or, Pi.le_def, Pi.le_def] at hcon
  push Not at hcon
  obtain ⟨⟨j, hj⟩, ⟨j', hj'⟩⟩ := hcon
  let C : Fin n → ℕ := fun l => min (A l) (B l)
  let A' : Fin n → ℕ := fun l => A l - C l
  let B' : Fin n → ℕ := fun l => B l - C l
  have hA : ∀ y, monomial A y = monomial C y * monomial A' y := fun y => by
    rw [← monomial_add_exp]
    congr 1
    funext l
    simp only [Pi.add_apply, C, A']
    omega
  have hB : ∀ y, monomial B y = monomial C y * monomial B' y := fun y => by
    rw [← monomial_add_exp]
    congr 1
    funext l
    simp only [Pi.add_apply, C, B']
    omega
  have hdisj : ∀ l, A' l = 0 ∨ B' l = 0 := fun l => by
    simp only [A', B', C]
    omega
  have hAj : A' j ≠ 0 := by
    simp only [A', C]
    omega
  have hBj : B' j' ≠ 0 := by
    simp only [B', C]
    omega
  let D : Base n → ℝ := fun x => U x * monomial A' (x - w) - V x * monomial B' (x - w)
  have hD : AnalyticAt ℝ D w :=
    (hU.fun_mul (analyticAt_monomial_sub _ w w)).fun_sub
      (hV.fun_mul (analyticAt_monomial_sub _ w w))
  have hND : NormalCrossingAt D w := by
    refine NormalCrossingAt.of_mul hD (analyticAt_monomial_sub C w w) (h.congr ?_)
    refine Eventually.of_forall fun x => ?_
    simp only [D, hA, hB]
    ring
  obtain ⟨E, W, hW, hWw, hDW⟩ := hND
  have hDw : D w = 0 := by
    have hA0 : A' ≠ 0 := fun h0 => hAj (congrFun h0 j)
    have hB0 : B' ≠ 0 := fun h0 => hBj (congrFun h0 j')
    simp only [D, sub_self, monomial_zero_point _ hA0, monomial_zero_point _ hB0, mul_zero]
  obtain ⟨k, hk⟩ : ∃ k, E k ≠ 0 := by
    by_contra hne
    push Not at hne
    have hE : E = 0 := funext hne
    have := hDW.self_of_nhds
    rw [hDw, hE, monomial_zero_exp, mul_one] at this
    exact hWw this.symm
  have hvanish : ∀ T : Finset (Fin n), k ∈ T → ∃ s : ℝ, s ≠ 0 ∧
      D (w + testVec T s) = 0 ∧ U (w + testVec T s) ≠ 0 ∧ V (w + testVec T s) ≠ 0 := by
    intro T hkT
    obtain ⟨s, hs, hDs, hUs, hVs⟩ := exists_testVec T
      (hDW.and ((hU.continuousAt.eventually_ne hUw).and (hV.continuousAt.eventually_ne hVw)))
    refine ⟨s, hs, ?_, hUs, hVs⟩
    rw [hDs, add_sub_cancel_left, monomial_testVec_of_mem hkT hk, mul_zero]
  by_cases hBk : B' k = 0
  · obtain ⟨s, hs, hDs, -, hVs⟩ := hvanish {k, j} (by simp)
    have hBT : ∀ l ∈ ({k, j} : Finset (Fin n)), B' l = 0 := by
      intro l hl
      rcases Finset.mem_insert.1 hl with rfl | hl
      · exact hBk
      · rw [Finset.mem_singleton.1 hl]
        exact (hdisj j).resolve_left hAj
    have h1 : monomial A' (testVec {k, j} s) = 0 := monomial_testVec_of_mem (by simp) hAj s
    simp only [D, add_sub_cancel_left, h1, monomial_testVec_of_forall hBT, mul_zero, zero_sub,
      neg_eq_zero] at hDs
    exact mul_ne_zero hVs (pow_ne_zero _ hs) hDs
  · obtain ⟨s, hs, hDs, hUs, -⟩ := hvanish {k} (by simp)
    have hAT : ∀ l ∈ ({k} : Finset (Fin n)), A' l = 0 := by
      intro l hl
      rw [Finset.mem_singleton.1 hl]
      exact (hdisj k).resolve_right hBk
    have h1 : monomial B' (testVec {k} s) = 0 := monomial_testVec_of_mem (by simp) hBk s
    simp only [D, add_sub_cancel_left, h1, monomial_testVec_of_forall hAT, mul_zero,
      sub_zero] at hDs
    exact mul_ne_zero hUs (pow_ne_zero _ hs) hDs

/-- Among exponents `α i` with comparable multiples `w i' • α i` and `w i • α i'`, one,
`α k`, has the least ratio: `w i • α k ≤ w k • α i` for every `i`. It minimizes
`|α i| / w i`, and comparable vectors with equal sums are equal. Auxiliary for the
Newton step of local uniformization for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem exists_min_ratio {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (w : ι → ℕ)
    (hw : ∀ i ∈ S, w i ≠ 0) (α : ι → Fin n → ℕ)
    (hcomp : ∀ i ∈ S, ∀ i' ∈ S, w i' • α i ≤ w i • α i' ∨ w i • α i' ≤ w i' • α i) :
    ∃ k ∈ S, ∀ i ∈ S, w i • α k ≤ w k • α i := by
  obtain ⟨k, hk, hmin⟩ := S.exists_min_image (fun i => (∑ j, (α i j : ℝ)) / w i) hS
  refine ⟨k, hk, fun i hi => ?_⟩
  rcases hcomp i hi k hk with h | h
  · have hle : (∑ j, (α k j : ℝ)) / w k ≤ (∑ j, (α i j : ℝ)) / w i := hmin i hi
    rw [div_le_div_iff₀ (by exact_mod_cast Nat.pos_of_ne_zero (hw k hk))
      (by exact_mod_cast Nat.pos_of_ne_zero (hw i hi))] at hle
    have hsum : ∑ j, (w k • α i) j = ∑ j, (w i • α k) j := by
      refine le_antisymm (Finset.sum_le_sum fun j _ => h j) ?_
      have : ((∑ j, (w i • α k) j : ℕ) : ℝ) ≤ ((∑ j, (w k • α i) j : ℕ) : ℝ) := by
        simp only [Pi.smul_apply, smul_eq_mul, Nat.cast_sum, Nat.cast_mul]
        rw [← Finset.mul_sum, ← Finset.mul_sum]
        linarith
      exact_mod_cast this
    have heq := (Finset.sum_eq_sum_iff_of_le fun j _ => h j).1 hsum
    exact le_of_eq (funext fun j => (heq j (Finset.mem_univ j)).symm)
  · exact h

end Transformer.Landscape.Uniformization

/-
# Parallel lookup at a finite set of input-dependent distances

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`,
generalization following equation `eq: output-t1`. The theorem here covers
the functional sum of the selected-distance channels. Its hypotheses make
explicit that the selected shifts include every required prior interaction
distance and exclude the zero shift. The paper's autocorrelation procedure
is not proved to provide that coverage by this result.
-/

import Transformer.Zoology.Appendix_DependentDistance
import Transformer.Zoology.Section4_PairedRecall

open scoped BigOperators

namespace Transformer.Zoology

/-- Sum of the parallel selected-distance channels at a query coordinate.
Source: Appendix `sec: data-dep-ar`, final output `Linear_sum(z')`. -/
def distanceLookupMany {n c : ℕ} (x : MQARInstance n c)
    (shifts : Finset (Fin n)) : RealSequence n c :=
  fun i q => ∑ s ∈ shifts, distanceLookup x s i q

/-- The selected distances cover every earlier matching key. This is the
functional condition required by the appendix's input-dependent lookup.
Source: Appendix `sec: data-dep-ar`, definition of interaction distances. -/
def CoversPriorDistances {n c : ℕ} (x : MQARInstance n c)
    (shifts : Finset (Fin n)) : Prop :=
  ∀ i j : Fin n, j < i → x.key j = x.query i → i - j ∈ shifts

/-- All selected distances point strictly backward. Source: Appendix
`sec: data-dep-ar`, interaction distances `i-j` for `j<i`. -/
def PositiveDistances {n : ℕ} (shifts : Finset (Fin n)) : Prop :=
  ∀ s ∈ shifts, 0 < s.val

/-- A positive selected distance always points to a strictly earlier pair.
Source: Appendix `sec: data-dep-ar`, interaction distance `i-j` with `j<i`. -/
theorem distance_shift_strict {n : ℕ} (i s : Fin n)
    (hs : s ≤ i) (hpos : 0 < s.val) : i - s < i := by
  apply Fin.lt_def.mpr
  rw [Fin.sub_val_of_le hs]
  omega

/-- Subtracting the interaction distance retrieves its earlier index.
Source: Appendix `sec: data-dep-ar`, interaction distance `i-j`. -/
theorem distance_sub_interaction {n : ℕ} (i j : Fin n) (hj : j < i) :
    i - (i - j) = j := by
  have hle : i - j ≤ i := by
    apply Fin.le_def.mpr
    rw [Fin.sub_val_of_le (le_of_lt hj)]
    omega
  apply Fin.ext
  rw [Fin.sub_val_of_le hle, Fin.sub_val_of_le (le_of_lt hj)]
  omega

/-- At a fixed query, two valid causal shifts identifying the same earlier
index are equal. Source: Appendix `sec: data-dep-ar`, distance `i-j`. -/
theorem distance_shift_injective {n : ℕ} (i s t : Fin n)
    (hs : s ≤ i) (ht : t ≤ i) (h : i - s = i - t) : s = t := by
  apply Fin.ext
  have hv := congrArg Fin.val h
  rw [Fin.sub_val_of_le hs, Fin.sub_val_of_le ht] at hv
  omega

/-- Under unique keys, one selected channel contributes the value and all
other selected channels contribute zero. Source: Appendix
`sec: data-dep-ar`, final sum of the parallel distance channels, with
the unique-key condition used in the appendix's synthetic setup. -/
theorem distanceLookupMany_match {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (shifts : Finset (Fin n))
    (s i : Fin n) (hmem : s ∈ shifts) (hs : s ≤ i)
    (hmatch : x.key (i - s) = x.query i) (q : Fin c) :
    distanceLookupMany x shifts i q = oneHot (x.value (i - s)) q := by
  classical
  unfold distanceLookupMany
  rw [Finset.sum_eq_single s]
  · exact distanceLookup_match x s i hs hmatch q
  · intro t ht hts
    apply distanceLookup_no_match
    intro hti hkey
    have hindex : i - t = i - s := hx (hkey.trans hmatch.symm)
    exact hts (distance_shift_injective i t s hti hs hindex)
  · intro h
    exact (h hmem).elim

/-- With no earlier matching key, every positive selected channel is zero.
Source: Appendix `sec: data-dep-ar`, final zero-output case. -/
theorem distanceLookupMany_no_match {n c : ℕ} (x : MQARInstance n c)
    (shifts : Finset (Fin n)) (hpositive : PositiveDistances shifts)
    (i : Fin n) (hnone : ∀ j : Fin n, j < i → x.key j ≠ x.query i)
    (q : Fin c) : distanceLookupMany x shifts i q = 0 := by
  classical
  unfold distanceLookupMany
  apply Finset.sum_eq_zero
  intro s hs
  apply distanceLookup_no_match
  intro hsi
  exact hnone (i - s) (distance_shift_strict i s hsi (hpositive s hs))

/-- The finite sum computes every coordinate of causal MQAR when the
selected positive distances cover all earlier matches. This proves the
functional part of the appendix's data-dependent construction under an
explicit coverage hypothesis; it does not establish the paper's
autocorrelation selection or its Coyote layer and parameter bounds.
Source: Appendix Theorem `thm: input-dep-genar`, corrected hypotheses. -/
theorem distanceLookupMany_solves_mqar {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (shifts : Finset (Fin n))
    (hpositive : PositiveDistances shifts)
    (hcover : CoversPriorDistances x shifts) (i : Fin n) (q : Fin c) :
    distanceLookupMany x shifts i q = expectedPairedAnswer x i q := by
  classical
  by_cases hm : ∃ j : Fin n, j < i ∧ x.key j = x.query i
  · obtain ⟨j, hj, hkey⟩ := hm
    have hshift : i - j ∈ shifts := hcover i j hj hkey
    have hle : i - j ≤ i := by
      apply Fin.le_def.mpr
      rw [Fin.sub_val_of_le (le_of_lt hj)]
      omega
    rw [distanceLookupMany_match x hx shifts (i - j) i hshift hle]
    · rw [distance_sub_interaction i j hj]
      have hiff : PriorAnswer x i q ↔ x.value j = q := by
        constructor
        · rintro ⟨k, _, hk, hv⟩
          have hkj : k = j := hx (hk.trans hkey.symm)
          simpa [hkj] using hv
        · intro hv
          exact ⟨j, hj, hkey, hv⟩
      by_cases hv : x.value j = q
      · simp [oneHot, hv, expectedPairedAnswer, hiff.mpr hv]
      · have hno : ¬ PriorAnswer x i q := hiff.not.mpr hv
        simp [oneHot, expectedPairedAnswer, hno, Ne.symm hv]
    · simpa [distance_sub_interaction i j hj] using hkey
  · have hnone : ∀ j : Fin n, j < i → x.key j ≠ x.query i := by
      intro j hj hkey
      exact hm ⟨j, hj, hkey⟩
    rw [distanceLookupMany_no_match x shifts hpositive i hnone q]
    have hno : ¬ PriorAnswer x i q := by
      rintro ⟨j, hj, hkey, _⟩
      exact hnone j hj hkey
    simp [expectedPairedAnswer, hno]

/-- The unique-key, nonzero-distance, and coverage hypotheses are jointly
satisfiable with an actual prior-key match. -/
example : ∃ x : MQARInstance 2 2, ∃ shifts : Finset (Fin 2),
    UniqueKeys x ∧ PositiveDistances shifts ∧
      CoversPriorDistances x shifts ∧ PriorAnswer x 1 0 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, {1}, ?_, ?_, ?_, ?_⟩
  · intro a b h
    exact h
  · intro s hs
    simp only [Finset.mem_singleton] at hs
    subst s
    decide
  · intro i j hj hkey
    fin_cases i <;> fin_cases j <;> simp_all [x]
  · exact ⟨0, by decide, rfl, rfl⟩

end Transformer.Zoology

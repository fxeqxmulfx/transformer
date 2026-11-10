module

public import Transformer.CRASP.Transformers
public import Transformer.Grokking.FlatSpectrum.BlockCut.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Subkernel

/-!
# The indicator of `L_k` holds a flat kernel of size `k / 2`

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), items BL4 and
BL5, for the languages `L_k` and the depth hierarchy of arXiv:2506.16055v3, §4, Theorem
`thm:rtfr_depth_hierarchy`: "A depth-`(k+1)` fixed-precision transformer can recognize `L_{k+1}`,
but no depth-`k` fixed-precision transformer can" (`Transformer.CRASP.rtfr_depth_hierarchy`).

BL4. The prefix `u_i = (ab)^i a` has `2 i + 1` blocks and ends with `a` (`oddPrefix`). The
suffix `v_j` starts with `b` and has `k − (2 j + 1)` blocks (`evenSuffix`), for `2 j + 1 < k`.
No blocks join at the cut, so `u_i v_j ∈ L_k` exactly when `i = j`
(`oddPrefix_append_mem_iff`): on these `k / 2` prefixes and suffixes the indicator of `L_k` is
the identity, a flat kernel. So a readout of cut width `d` misses the indicator by at least
`k / 2 − (d + 1)` in squared error on these pairs (`div_two_sub_le_sum_sq_altPlus`),
and one that equals the indicator has `k / 2 ≤ d + 1` (`div_two_le_of_altPlus`).

BL5. By a sign five numbers suffice for every `k` (`BlockCut.Sign`), while no transformer of
depth `k`, of any width and precision, recognizes `L_{k+1}`: the depth hierarchy of the paper is
not a hierarchy of cut width (`not_recognizes_and_hasCutWidth_five`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.BlockCut

open Transformer.CRASP Recall

/-- The prefix `(ab)^i a`: `2 i + 1` blocks, the first and the last of the letter `a`. -/
def oddPrefix (i : ℕ) : List Bool := altList false (2 * i + 1)

/-- The suffix that starts with `b` and has `k − (2 j + 1)` blocks. -/
def evenSuffix (k j : ℕ) : List Bool := altList true (k - (2 * j + 1))

/-- **The indicator of `L_k` is the identity on `k / 2` prefixes and suffixes**
(`review/flat_spectrum.md`, BL4): `(ab)^i a` before the suffix of `k − (2 j + 1)` blocks that
starts with `b` is in `L_k` exactly when `i = j`. -/
theorem oddPrefix_append_mem_iff {k i j : ℕ} (hj : 2 * j + 1 < k) :
    oddPrefix i ++ evenSuffix k j ∈ altPlus false k ↔ i = j := by
  obtain ⟨n, hn⟩ : ∃ n, k - (2 * j + 1) = n + 1 := ⟨k - (2 * j + 1) - 1, by omega⟩
  have hu : oddPrefix i ≠ [] := by simp [oddPrefix, altList_succ]
  have hv : evenSuffix k j ≠ [] := by simp [evenSuffix, hn, altList_succ]
  have hlast : (oddPrefix i).getLast? = some false := getLast?_altList_odd false i
  have hhead : (evenSuffix k j).head? = some true := by rw [evenSuffix, hn]; rfl
  have hb := blocks_append (oddPrefix i) hv
  rw [hlast, hhead, ite_eq_right (by simp)] at hb
  have hbu : blocks (oddPrefix i) = 2 * i + 1 := blocks_altList _ _
  have hbv : blocks (evenSuffix k j) = k - (2 * j + 1) := blocks_altList _ _
  rw [mem_altPlus_iff (by simp [hu]), List.head?_append_of_ne_nil _ hu]
  exact ⟨fun h => by omega, fun h => ⟨by simp [oddPrefix, altList_succ], by omega⟩⟩

/-- `a` before `ba` is `aba`, in `L_3`. -/
example : oddPrefix 0 ++ evenSuffix 3 0 ∈ altPlus false 3 :=
  (oddPrefix_append_mem_iff (by norm_num)).2 rfl

/-- **A readout of cut width `d` misses the indicator of `L_k`** (`review/flat_spectrum.md`,
BL4): on the `k / 2` prefixes `(ab)^i a` and the `k / 2` suffixes that start with `b`, its
squared error is at least `k / 2 − (d + 1)`. -/
theorem div_two_sub_le_sum_sq_altPlus {d : ℕ} {F : List Bool → List Bool → ℝ}
    (hF : HasCutWidth d F) (k : ℕ) :
    ((k / 2 : ℕ) : ℝ) - (d + 1 : ℕ) ≤ ∑ j : Fin (k / 2), ∑ i : Fin (k / 2),
      (Set.indicator (altPlus false k) (fun _ => (1 : ℝ)) (oddPrefix i ++ evenSuffix k j) -
        F (evenSuffix k j) (oddPrefix i)) ^ 2 := by
  classical
  have h := hF.mul_le_sum_sq_of_isFlat (fun j : Fin (k / 2) => evenSuffix k j)
    (fun i : Fin (k / 2) => oddPrefix i) (K := 1) one_pos
    (by simpa using isFlat_one (n := Fin (k / 2)))
  rw [one_mul, Fintype.card_fin, min_self] at h
  refine h.trans (le_of_eq ?_)
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
  have hj : 2 * (j : ℕ) + 1 < k := by have := j.isLt; omega
  rw [Set.indicator_apply, Matrix.one_apply]
  simp only [oddPrefix_append_mem_iff hj, Fin.ext_iff, eq_comm]

/-- **A readout that equals the indicator of `L_k` has cut width at least `k / 2 − 1`**
(`review/flat_spectrum.md`, BL4). -/
theorem div_two_le_of_altPlus {d : ℕ} {F : List Bool → List Bool → ℝ}
    (hF : HasCutWidth d F) {k : ℕ}
    (h : ∀ u v, F v u = Set.indicator (altPlus false k) (fun _ => (1 : ℝ)) (u ++ v)) :
    k / 2 ≤ d + 1 := by
  have hb := div_two_sub_le_sum_sq_altPlus hF k
  rw [show ∑ j : Fin (k / 2), ∑ i : Fin (k / 2),
      (Set.indicator (altPlus false k) (fun _ => (1 : ℝ)) (oddPrefix i ++ evenSuffix k j) -
        F (evenSuffix k j) (oddPrefix i)) ^ 2 = 0 by simp [h]] at hb
  exact_mod_cast sub_nonpos.1 hb

/-- `L_0` holds the empty word alone: its indicator is `[v = []] · [u = []]`, of cut width
one. -/
example : 0 / 2 ≤ 1 + 1 :=
  div_two_le_of_altPlus (k := 0)
    (F := fun v u => (if v = [] then 1 else 0) * (if u = [] then 1 else 0))
    ⟨fun _ => 0, fun v _ => if v = [] then 1 else 0, fun u _ => if u = [] then 1 else 0,
      fun v u => by simp⟩
    fun u v => by
      classical
      rw [Set.indicator_apply]
      by_cases hu : u = [] <;> by_cases hv : v = [] <;> simp [hu, hv, altPlus]

example := div_two_sub_le_sum_sq_altPlus (hasCutWidth_blockReadout 4) 4

/-- **The depth hierarchy is not a hierarchy of cut width** (`review/flat_spectrum.md`, BL5):
no transformer of depth `k`, of any width and precision, recognizes `L_{k+1}`
(arXiv:2506.16055v3, Theorem `thm:rtfr_depth_hierarchy`), while five numbers across every cut
decide `L_{k+1}` by a sign. -/
theorem not_recognizes_and_hasCutWidth_five (k : ℕ) :
    (∀ (p s d : ℕ) (T : RTfr (Option Bool) p s d k),
        ¬ T.Recognizes (altPlus false (k + 1))) ∧
      ∃ F : List Bool → List Bool → ℝ, HasCutWidth 5 F ∧
        ∀ u v, 0 < F v u ↔ u ++ v ∈ altPlus false (k + 1) :=
  ⟨(rtfr_depth_hierarchy k).2, exists_hasCutWidth_altPlus_sign (k + 1)⟩

end Transformer.Grokking.FlatSpectrum.BlockCut

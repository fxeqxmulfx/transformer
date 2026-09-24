/-
# Slicing a strict sign cone by a hyperplane

For Wendel's sign-pattern recurrence, a pattern on the first `n` vectors has
both extensions to a new nonzero vector exactly when its strict cone meets
the new vector's orthogonal hyperplane. The forward implication interpolates
two witnesses; the reverse implication perturbs a witness in both directions.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelSignAverage
import Mathlib.Analysis.InnerProductSpace.Basic

namespace Transformer.Perspective

/-- Both strict signs for a new nonzero vector are feasible precisely when
the old strict cone intersects its orthogonal hyperplane.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictCone_two_sides_iff_hyperplane (d n : ℕ)
    (v : Idx n → EucSpace d) (x : EucSpace d) (hx : x ≠ 0) :
    ((∃ w : EucSpace d, (∀ i, 0 < inner (𝕜 := ℝ) (v i) w) ∧
        0 < inner (𝕜 := ℝ) x w) ∧
      (∃ w : EucSpace d, (∀ i, 0 < inner (𝕜 := ℝ) (v i) w) ∧
        inner (𝕜 := ℝ) x w < 0)) ↔
      ∃ w : EucSpace d, (∀ i, 0 < inner (𝕜 := ℝ) (v i) w) ∧
        inner (𝕜 := ℝ) x w = 0 := by
  constructor
  · rintro ⟨⟨wp, hwp, hxp⟩, ⟨wm, hwm, hxm⟩⟩
    let A := inner (𝕜 := ℝ) x wp
    let B := inner (𝕜 := ℝ) x wm
    refine ⟨(-B) • wp + A • wm, ?_, ?_⟩
    · intro i
      rw [inner_add_right, real_inner_smul_right, real_inner_smul_right]
      exact add_pos (mul_pos (by dsimp [B]; linarith) (hwp i))
        (mul_pos hxp (hwm i))
    · rw [inner_add_right, real_inner_smul_right, real_inner_smul_right]
      dsimp [A, B]
      ring
  · rintro ⟨w₀, hw₀, hx₀⟩
    let C : Set (EucSpace d) := {w | ∀ i : Idx n, 0 < inner (𝕜 := ℝ) (v i) w}
    have hC : IsOpen C := by
      have heq : C = ⋂ i ∈ (Finset.univ : Finset (Idx n)),
          {w : EucSpace d | 0 < inner (𝕜 := ℝ) (v i) w} := by
        ext w
        simp [C]
      rw [heq]
      apply isOpen_biInter_finset
      intro i _
      have hc : Continuous (fun w : EucSpace d => inner (𝕜 := ℝ) (v i) w) := by
        fun_prop
      exact isOpen_Ioi.preimage hc
    have hwC : w₀ ∈ C := hw₀
    have hplus : (fun ε : ℝ => w₀ + ε • x) ⁻¹' C ∈ nhds (0 : ℝ) := by
      have hc : ContinuousAt (fun ε : ℝ => w₀ + ε • x) 0 := by fun_prop
      exact hc.preimage_mem_nhds (hC.mem_nhds (by simpa using hwC))
    have hminus : (fun ε : ℝ => w₀ - ε • x) ⁻¹' C ∈ nhds (0 : ℝ) := by
      have hc : ContinuousAt (fun ε : ℝ => w₀ - ε • x) 0 := by fun_prop
      exact hc.preimage_mem_nhds (hC.mem_nhds (by simpa using hwC))
    have hε : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0),
        0 < ε ∧ w₀ + ε • x ∈ C ∧ w₀ - ε • x ∈ C := by
      filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds hplus,
        nhdsWithin_le_nhds hminus] with ε hε hp hm
      exact ⟨hε, hp, hm⟩
    obtain ⟨ε, hεpos, hp, hm⟩ := hε.exists
    have hxx : 0 < inner (𝕜 := ℝ) x x := real_inner_self_pos.mpr hx
    have hxp : 0 < inner (𝕜 := ℝ) x (w₀ + ε • x) := by
      rw [inner_add_right, real_inner_smul_right, hx₀, zero_add]
      exact mul_pos hεpos hxx
    have hxm : inner (𝕜 := ℝ) x (w₀ - ε • x) < 0 := by
      rw [inner_sub_right, real_inner_smul_right, hx₀, zero_sub]
      exact neg_neg_of_pos (mul_pos hεpos hxx)
    exact ⟨⟨w₀ + ε • x, hp, hxp⟩, ⟨w₀ - ε • x, hm, hxm⟩⟩

/-- A strict sign pattern is one realized by an (unnormalized) linear
functional. Strict inequalities make normalizing the witness unnecessary.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
def StrictSignPattern (d n : ℕ) (v : Idx n → EucSpace d)
    (mask : Idx n → Bool) : Prop :=
  ∃ w : EucSpace d, ∀ i : Idx n,
    0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w

/-- The two extensions of a sign pattern to one more vector are both
realizable exactly when the old sign cone meets its orthogonal hyperplane.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignPattern_snoc_both_iff_hyperplane (d n : ℕ)
    (v : Idx n → EucSpace d) (mask : Idx n → Bool)
    (x : EucSpace d) (hx : x ≠ 0) :
    (StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask false) ∧
      StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask true)) ↔
      ∃ w : EucSpace d,
        (∀ i : Idx n, 0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w) ∧
        inner (𝕜 := ℝ) x w = 0 := by
  simpa [StrictSignPattern, Fin.forall_fin_succ', Fin.snoc_castSucc,
    Fin.snoc_last, inner_neg_left] using
    strictCone_two_sides_iff_hyperplane d n
      (fun i => if mask i then -v i else v i) x hx

/-- A strict sign pattern on the old vectors has at least one extension to a
new nonzero vector. If its witness is orthogonal to that vector, both
extensions exist by `strictSignPattern_snoc_both_iff_hyperplane`.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignPattern_snoc_exists_iff (d n : ℕ)
    (v : Idx n → EucSpace d) (mask : Idx n → Bool)
    (x : EucSpace d) (hx : x ≠ 0) :
    (StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask false) ∨
      StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask true)) ↔
        StrictSignPattern d n v mask := by
  constructor
  · rintro (⟨w, hw⟩ | ⟨w, hw⟩)
    · refine ⟨w, fun i => ?_⟩
      have hi := hw i.castSucc
      simpa [StrictSignPattern, Fin.snoc_castSucc] using hi
    · refine ⟨w, fun i => ?_⟩
      have hi := hw i.castSucc
      simpa [StrictSignPattern, Fin.snoc_castSucc] using hi
  · rintro ⟨w, hw⟩
    rcases lt_trichotomy (inner (𝕜 := ℝ) x w) 0 with hneg | hzero | hpos
    · right
      refine ⟨w, ?_⟩
      rw [Fin.forall_fin_succ']
      constructor
      · intro i
        simpa [Fin.snoc_castSucc] using hw i
      · simpa [Fin.snoc_last, inner_neg_left] using hneg
    · exact Or.inl ((strictSignPattern_snoc_both_iff_hyperplane d n v mask x hx).2
        ⟨w, hw, hzero⟩).1
    · left
      refine ⟨w, ?_⟩
      rw [Fin.forall_fin_succ']
      constructor
      · intro i
        simpa [Fin.snoc_castSucc] using hw i
      · simpa [Fin.snoc_last] using hpos

/-- The slice condition is nonempty: the first coordinate vector is positive
on itself and orthogonal to the second coordinate vector. -/
example :
    (EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2) ≠ 0 ∧
      ∃ w : EucSpace 2,
        (∀ i : Idx 1,
          0 < inner (𝕜 := ℝ)
            (if (fun _ : Idx 1 => false) i then
              -(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
             else EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) w) ∧
        inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) w = 0 := by
  refine ⟨by simp, ⟨EuclideanSpace.single (0 : Fin 2) (1 : ℝ), ?_, ?_⟩⟩
  · intro i
    fin_cases i
    norm_num [EuclideanSpace.inner_single_left]
  · norm_num [EuclideanSpace.inner_single_left]

end Transformer.Perspective

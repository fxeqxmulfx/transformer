/-
# Finite original quadrant families in generic plane position

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the finite point matching construction. Choose new positive unit
points successively outside all planes of the prescribed old vectors
and outside every plane of an old vector with an earlier new point.
There are only finitely many constraints at each step.

Reverse plane exclusion supplies the symmetric constraint between
an earlier and a later new point. Thus every two distinct new points
and any fixed old vector satisfy the plane conditions needed by a
protected original transfer. A nonempty pool of old vectors also
forces the entire new unit family to be injective.
-/

import Transformer.Interpolation.Section4_GenericQuadrantPoints

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ} {κ : Type*}

/-- Any finite number of actual positive unit points can simultaneously
avoid all old planes and every mixed old-new plane. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, finite protected point matching. -/
theorem exists_generic_quadrant_family [Fintype κ] (hd : 3 ≤ d)
    (A : κ → EucSpace d) (N : ℕ) :
    ∃ Z : Idx N → SSphere d, (∀ k, Z k ∈ positiveQuadrant d) ∧
      (∀ k, ∀ i j, ∀ α β : ℝ, (Z k : EucSpace d) ≠ α • A i + β • A j) ∧
      ∀ k l, k ≠ l → ∀ i, ∀ α β : ℝ,
        (Z k : EucSpace d) ≠ α • A i + β • (Z l : EucSpace d) := by
  induction N with
  | zero =>
    refine ⟨Fin.elim0, ?_, ?_, ?_⟩
    · intro k; exact Fin.elim0 k
    · intro k; exact Fin.elim0 k
    · intro k; exact Fin.elim0 k
  | succ N ih =>
    obtain ⟨Z, hQ, hplanes, hpairs⟩ := ih
    let J := (κ × κ) ⊕ (κ × Idx N)
    let p : J → EucSpace d := Sum.elim (fun ij => A ij.1) (fun ij => A ij.1)
    let q : J → EucSpace d := Sum.elim (fun ij => A ij.2) (fun ij => (Z ij.2 : EucSpace d))
    obtain ⟨z, hzQ, hz⟩ := exists_quadrant_point_avoiding_planes hd p q
    let Z' : Idx (N + 1) → SSphere d := Fin.snoc Z z
    have hnoncol (k : Idx N) (i : κ) (α : ℝ) : (Z k : EucSpace d) ≠ α • A i := by
      simpa only [zero_smul, add_zero] using hplanes k i i α 0
    refine ⟨Z', ?_, ?_, ?_⟩
    · intro k
      refine Fin.lastCases ?_ (fun k => ?_) k
      · simpa only [Z', Fin.snoc_last] using hzQ
      · simpa only [Z', Fin.snoc_castSucc] using hQ k
    · intro k
      refine Fin.lastCases ?_ (fun k => ?_) k
      · intro i j α β
        simp only [Z', Fin.snoc_last]
        exact hz (Sum.inl (i, j)) α β
      · intro i j α β
        simp only [Z', Fin.snoc_castSucc]
        exact hplanes k i j α β
    · intro k
      refine Fin.lastCases ?_ (fun k => ?_) k
      · intro l
        refine Fin.lastCases ?_ (fun l => ?_) l
        · intro hkl
          exact False.elim (hkl rfl)
        · intro _ i α β
          simp only [Z', Fin.snoc_last, Fin.snoc_castSucc]
          exact hz (Sum.inr (i, l)) α β
      · intro l
        refine Fin.lastCases ?_ (fun l => ?_) l
        · intro _ i α β
          simp only [Z', Fin.snoc_castSucc, Fin.snoc_last]
          exact plane_exclusion_reversed (A i) (z : EucSpace d) (Z k : EucSpace d)
            (hnoncol k i) (fun α β => hz (Sum.inr (i, k)) α β) α β
        · intro hkl i α β
          have hne : k ≠ l := fun he => hkl (congrArg Fin.castSucc he)
          simp only [Z', Fin.snoc_castSucc]
          exact hpairs k l hne i α β

/-- A nonempty original reference pool makes the generic unit family
injective as well. Source auxiliary: arXiv:2411.04551v3, §1,
the distinct intermediate atoms for protected interpolation. -/
theorem exists_injective_generic_quadrant_family [Fintype κ] [Nonempty κ]
    (hd : 3 ≤ d) (A : κ → EucSpace d) (N : ℕ) :
    ∃ Z : Idx N → SSphere d, Function.Injective Z ∧ (∀ k, Z k ∈ positiveQuadrant d) ∧
      (∀ k, ∀ i j, ∀ α β : ℝ, (Z k : EucSpace d) ≠ α • A i + β • A j) ∧
      ∀ k l, k ≠ l → ∀ i, ∀ α β : ℝ,
        (Z k : EucSpace d) ≠ α • A i + β • (Z l : EucSpace d) := by
  obtain ⟨Z, hQ, hplanes, hpairs⟩ := exists_generic_quadrant_family hd A N
  refine ⟨Z, ?_, hQ, hplanes, hpairs⟩
  intro k l he
  by_contra hkl
  apply hpairs k l hkl (Classical.arbitrary κ) 0 1
  simp only [zero_smul, one_smul, zero_add]
  exact congrArg Subtype.val he

/-- The old-plane exclusion forbids every new point from being a
scalar multiple of any old vector. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, protected plane matching. -/
theorem generic_quadrant_family_noncollinear {N : ℕ} (A : κ → EucSpace d)
    (Z : Idx N → SSphere d)
    (hplanes : ∀ k, ∀ i j, ∀ α β : ℝ, (Z k : EucSpace d) ≠ α • A i + β • A j) :
    ∀ k i, ∀ α : ℝ, (Z k : EucSpace d) ≠ α • A i := by
  intro k i α
  simpa only [zero_smul, add_zero] using hplanes k i i α 0

/-- Every non-collinear old pair gives the reverse exclusion required
to protect an old atom during a transfer to a generic new one. Source
auxiliary: arXiv:2411.04551v3, §1 and §3.2, finite point matching. -/
theorem generic_old_outside_transfer_plane {N : ℕ} (A : κ → EucSpace d)
    (Z : Idx N → SSphere d)
    (hplanes : ∀ k, ∀ i j, ∀ α β : ℝ, (Z k : EucSpace d) ≠ α • A i + β • A j)
    (k : Idx N) (i j : κ) (hij : ∀ γ : ℝ, A j ≠ γ • A i) :
    ∀ α β : ℝ, A j ≠ α • A i + β • (Z k : EucSpace d) :=
  plane_exclusion_reversed (A i) (Z k : EucSpace d) (A j) hij (hplanes k i j)

/-- Two genuine distinct original positive unit references and a
positive three-dimensional ambient space satisfy all input dimension
and nonempty-reference hypotheses of the construction. -/
example : ∃ A : Fin 2 → EucSpace 3, 3 ≤ 3 ∧ Function.Injective A ∧
    (∀ i, ‖A i‖ = 1) ∧ ∀ i j, 0 < A i j := by
  let A (i : Fin 2) : EucSpace 3 := quadrantThreePoint i.castSucc
  refine ⟨A, le_rfl, ?_, ?_, ?_⟩
  · intro i j he
    have hh : quadrantThreePoint i.castSucc = quadrantThreePoint j.castSucc := Subtype.ext he
    have hij := quadrantThreePoint_injective hh
    exact Fin.castSucc_injective 2 hij
  · intro i
    exact mem_sphere_zero_iff_norm.mp (quadrantThreePoint i.castSucc).2
  · intro i j
    exact quadrantThreePoint_mem i.castSucc j

/-- A genuine positive unit new point outside an actual old coordinate
plane realizes the exclusion hypotheses with two non-collinear old
vectors and a nonempty new family. -/
example : ∃ (A : Fin 2 → EucSpace 3) (Z : Fin 1 → SSphere 3),
    (∀ k, ∀ i j, ∀ α β : ℝ, (Z k : EucSpace 3) ≠ α • A i + β • A j) ∧
    ∀ γ : ℝ, A 1 ≠ γ • A 0 := by
  let e₀ : EucSpace 3 := PiLp.single 2 (0 : Fin 3) 1
  let e₁ : EucSpace 3 := PiLp.single 2 (1 : Fin 3) 1
  let A (i : Fin 2) := if i = 0 then e₀ else e₁
  let Z : Fin 1 → SSphere 3 := fun _ => quadrantThreePoint 2
  refine ⟨A, Z, ?_, ?_⟩
  · intro k i j α β he
    have hh := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
    fin_cases i <;> fin_cases j <;> norm_num [A, Z, e₀, e₁, quadrantThreePoint] at hh
  · intro γ he
    have hh := congrArg (fun x : EucSpace 3 => x (1 : Fin 3)) he
    norm_num [A, e₀, e₁] at hh

end Transformer.Interpolation

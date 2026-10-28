module

public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery

/-!
# The fewest cells: `2N − 1`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*. Its Scope: "A rigid
pattern is connected, so it has at least `2p − 1` cells"; its Numerical check lists
"diagonals `{0, 1}` minus one cell | `2p − 1`" among the patterns that recover `ψ(a + b)`.

Added, for every `N = k + 1`:

* The diagonal and the cells just right of it, `{(i, i)} ∪ {(j, j + 1)}`, are the diagonals of
  offsets `0` and `1` without the cell `(N − 1, 0)`: `2k + 1 = 2N − 1` cells
  (`ncard_staircase`). They connect the rows with the columns along the path
  `c₀ r₀ c₁ r₁ … c_k r_k` (`staircase_connected`) and contain the diagonal, so M recovers from
  them every `N × N` matrix `u vᵀ` with `|u_i v_j| = 1` (`forall_isMinOn_vecMulVec_staircase`).
* A set from which M recovers one such matrix has at least `2N − 1` cells
  (`two_mul_card_le_ncard_add_one_of_forall_isMinOn`): its graph is connected, and a connected
  graph on `2N` vertices has at least `2N − 1` edges.

So `2N − 1` is the least number of cells from which M recovers every such matrix
(`isLeast_ncard_forall_isMinOn_vecMulVec`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix SimpleGraph

/-- The diagonal and the cells just right of it, `{(i, i)} ∪ {(j, j + 1)}` in `N = k + 1` rows:
the diagonals of offsets `0` and `1` of `review/flat_spectrum.md` without the cell
`(N − 1, 0)`. -/
def staircase (k : ℕ) : Set (Fin (k + 1) × Fin (k + 1)) :=
  Set.range (fun i => (i, i)) ∪ Set.range fun j : Fin k => (j.castSucc, j.succ)

/-- The staircase has `2k + 1 = 2N − 1` cells. -/
theorem ncard_staircase (k : ℕ) : (staircase k).ncard = 2 * k + 1 := by
  have hD : Disjoint (Set.range fun i : Fin (k + 1) => (i, i))
      (Set.range fun j : Fin k => (j.castSucc, j.succ)) := by
    rw [Set.disjoint_left]
    rintro _ ⟨i, rfl⟩ ⟨j, hj⟩
    simp only [Prod.mk.injEq] at hj
    exact (show j.castSucc < j.succ from Fin.castSucc_lt_succ).ne (hj.1.trans hj.2.symm)
  rw [staircase, Set.ncard_union_eq hD (Set.toFinite _) (Set.toFinite _),
    Set.ncard_range_of_injective fun i i' h => congrArg Prod.fst h,
    Set.ncard_range_of_injective fun j j' h =>
      Fin.castSucc_injective k (congrArg Prod.fst h),
    Nat.card_eq_fintype_card, Nat.card_eq_fintype_card, Fintype.card_fin, Fintype.card_fin]
  ring

/-- The staircase connects the rows with the columns. -/
theorem staircase_connected (k : ℕ) : (patternGraph (staircase k)).Connected := by
  have hrow (i : Fin (k + 1)) : (patternGraph (staircase k)).Reachable (.inl 0) (.inl i) := by
    induction i using Fin.induction with
    | zero => rfl
    | succ j ih =>
      have h1 : (patternGraph (staircase k)).Adj (.inl j.castSucc) (.inr j.succ) :=
        patternGraph_adj_inl_inr.mpr (Or.inr ⟨j, rfl⟩)
      have h2 : (patternGraph (staircase k)).Adj (.inr j.succ) (.inl j.succ) :=
        patternGraph_adj_inr_inl.mpr (Or.inl ⟨j.succ, rfl⟩)
      exact ih.trans (h1.reachable.trans h2.reachable)
  have hall : ∀ x, (patternGraph (staircase k)).Reachable (.inl 0) x
    | .inl i => hrow i
    | .inr j => (hrow j).trans (Adj.reachable (patternGraph_adj_inl_inr.mpr (Or.inl ⟨j, rfl⟩)))
  exact (connected_iff _).mpr ⟨fun u v => (hall u).symm.trans (hall v), ⟨.inl 0⟩⟩

/-- **M recovers every rank-one matrix with entries of modulus one from the staircase**, `2N − 1`
cells. -/
theorem forall_isMinOn_vecMulVec_staircase {k : ℕ} {u v : Fin (k + 1) → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) :
    ∀ X ∈ {X' : Matrix (Fin (k + 1)) (Fin (k + 1)) ℂ |
        ∀ ij ∈ staircase k, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (Fin (k + 1)) (Fin (k + 1)) ℂ |
        ∀ ij ∈ staircase k, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔ X = vecMulVec u v) :=
  (forall_isMinOn_vecMulVec_iff_connected_of_entries h).mpr
    ⟨staircase_connected k, Equiv.refl _, fun i => Or.inl ⟨i, rfl⟩⟩

/-- The all-ones `3 × 3` matrix from the five cells of the staircase. -/
example := forall_isMinOn_vecMulVec_staircase (k := 2) (u := fun _ => 1) (v := fun _ => 1)
  fun i j => by simp [vecMulVec_apply]

/-- **A set from which M recovers a rank-one matrix with entries of modulus one has at least
`2N − 1` cells** (`review/flat_spectrum.md`, Scope, for these matrices): `2N ≤ #Ω + 1`. -/
theorem two_mul_card_le_ncard_add_one_of_forall_isMinOn {n : Type*} [Fintype n] [DecidableEq n]
    [Nonempty n] {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) {Ω : Set (n × n)}
    (hΩ : ∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔
        X = vecMulVec u v)) :
    2 * Fintype.card n ≤ Ω.ncard + 1 := by
  have := card_add_card_le_ncard_add_one_of_rigid
    ((rigid_iff_connected Ω).mpr ((forall_isMinOn_vecMulVec_iff_connected_of_entries h).mp hΩ).1)
  rwa [Nat.card_eq_fintype_card, ← two_mul] at this

example : 2 * Fintype.card (Fin 3) ≤ (staircase 2).ncard + 1 :=
  two_mul_card_le_ncard_add_one_of_forall_isMinOn (u := fun _ => 1) (v := fun _ => 1)
    (fun i j => by simp [vecMulVec_apply])
    (forall_isMinOn_vecMulVec_staircase fun i j => by simp [vecMulVec_apply])

/-- **`2N − 1` is the least number of cells from which M recovers every `N × N` rank-one matrix
with entries of modulus one** (added; `N = k + 1`). -/
theorem isLeast_ncard_forall_isMinOn_vecMulVec (k : ℕ) :
    IsLeast {c | ∃ Ω : Set (Fin (k + 1) × Fin (k + 1)), Ω.ncard = c ∧
      ∀ u v : Fin (k + 1) → ℂ, (∀ i j, ‖vecMulVec u v i j‖ = 1) →
        ∀ X ∈ {X' : Matrix (Fin (k + 1)) (Fin (k + 1)) ℂ |
            ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
          (IsMinOn nuclearNorm {X' : Matrix (Fin (k + 1)) (Fin (k + 1)) ℂ |
            ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔ X = vecMulVec u v)}
      (2 * k + 1) := by
  refine ⟨⟨staircase k, ncard_staircase k, fun _ _ h => forall_isMinOn_vecMulVec_staircase h⟩, ?_⟩
  rintro c ⟨Ω, rfl, hΩ⟩
  have h1 : ∀ i j, ‖vecMulVec (fun _ : Fin (k + 1) => (1 : ℂ)) (fun _ : Fin (k + 1) => 1) i j‖ =
      1 := by
    simp [vecMulVec_apply]
  have := two_mul_card_le_ncard_add_one_of_forall_isMinOn h1 (hΩ _ _ h1)
  rw [Fintype.card_fin] at this
  omega

end Transformer.Grokking.FlatSpectrum

module

public import Mathlib.Algebra.Module.Pi
public import Mathlib.Tactic.FinCases
public import Transformer.GainMatroid.Section2_K4Flats

/-!
# The edges of `K₄` on the rows of a split vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. The generalized parallel
connection of `M*` and `M(K₄)` across a triad `Y = {a, b, c}` splits the vertex of the triad into a
triangle on three new rows `x', y', z'` (`Section2_CoconnectionMatrix`). On these rows, the columns
of the edges of `K₄` are `k4NewCol`: `a, b, c` are `-x'`, `-y'`, `-z'` and `a', b', c'` are
`y' − z'`, `x' − z'`, `x' − y'`.

* Each triangle of `M(K₄)` has a functional on the new rows vanishing on exactly the columns of the
  other edges (`exists_k4_functional`).
* `thirdIndex i j` is the index other than `i ≠ j`. The sets `{inl i, inr j, inr k}` with `k` the
  third index are triangles of `M(K₄)` (`thirdIndex_mem_k4Triangles`), and the difference of the
  new rows `i` and `j` is the new column of the third index, up to sign (`k4NewCol_thirdIndex`).
* The second new column is the sum of the other two (`k4NewCol_inr_one`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {F : Type*} [Field F]

/-- The columns of the edges of `K₄` on the new rows `x', y', z'`: `a, b, c` are `-x'`, `-y'`, `-z'`
and `a', b', c'` are `y' − z'`, `x' − z'`, `x' − y'`. -/
def k4NewCol : Fin 3 ⊕ Fin 3 → Fin 3 → F
  | .inl i => -Pi.single i 1
  | .inr k => Pi.single (![1, 0, 0] k) 1 - Pi.single (![2, 2, 1] k) 1

/-- The index other than `i` and `j`, when `i ≠ j`. -/
def thirdIndex (i j : Fin 3) : Fin 3 := -(i + j)

theorem thirdIndex_ne_left : ∀ i j : Fin 3, i ≠ j → thirdIndex i j ≠ i := by
  decide

theorem thirdIndex_ne_right : ∀ i j : Fin 3, i ≠ j → thirdIndex i j ≠ j := by
  decide

theorem thirdIndex_thirdIndex : ∀ j k : Fin 3, thirdIndex (thirdIndex j k) j = k := by
  decide

theorem thirdIndex_eq_iff : ∀ l j k : Fin 3, thirdIndex l j = k ↔ l = thirdIndex j k := by
  decide

/-- The sets `{inl i, inr j, inr k}` with `k` the third index are triangles of `M(K₄)`. -/
theorem thirdIndex_mem_k4Triangles : ∀ i j : Fin 3, i ≠ j →
    ({.inl i, .inr j, .inr (thirdIndex i j)} : Finset (Fin 3 ⊕ Fin 3)) ∈ k4Triangles := by
  decide

/-- The new column of the third index is the difference of the other two new rows, up to sign. -/
theorem k4NewCol_thirdIndex (i j : Fin 3) (hij : i ≠ j) :
    k4NewCol (.inr (thirdIndex i j)) = (Pi.single i 1 - Pi.single j 1 : Fin 3 → F) ∨
      k4NewCol (.inr (thirdIndex i j)) = -(Pi.single i 1 - Pi.single j 1 : Fin 3 → F) := by
  fin_cases i <;> fin_cases j <;> simp_all [k4NewCol, thirdIndex]

/-- On the new rows, the second new column is the sum of the other two. -/
theorem k4NewCol_inr_one :
    (k4NewCol (.inr 1) : Fin 3 → F) = k4NewCol (.inr 0) + k4NewCol (.inr 2) := by
  funext k
  fin_cases k <;> simp [k4NewCol]

/-- Each triangle of `M(K₄)` has a functional on the new rows that vanishes on the column of an
edge exactly when the edge is outside the triangle. -/
theorem exists_k4_functional {C : Finset (Fin 3 ⊕ Fin 3)} (hC : C ∈ k4Triangles) :
    ∃ μ : (Fin 3 → F) →ₗ[F] F, ∀ g, μ (k4NewCol g) ≠ 0 ↔ g ∈ C := by
  simp only [k4Triangles, Finset.mem_insert, Finset.mem_singleton] at hC
  rcases hC with rfl | rfl | rfl | rfl
  · refine ⟨(LinearMap.proj 0 : (Fin 3 → F) →ₗ[F] F) + LinearMap.proj 1 + LinearMap.proj 2,
      fun g => ?_⟩
    rcases g with i | i <;> fin_cases i <;> simp [k4NewCol]
  · refine ⟨(LinearMap.proj 0 : (Fin 3 → F) →ₗ[F] F), fun g => ?_⟩
    rcases g with i | i <;> fin_cases i <;> simp [k4NewCol]
  · refine ⟨(LinearMap.proj 1 : (Fin 3 → F) →ₗ[F] F), fun g => ?_⟩
    rcases g with i | i <;> fin_cases i <;> simp [k4NewCol]
  · refine ⟨(LinearMap.proj 2 : (Fin 3 → F) →ₗ[F] F), fun g => ?_⟩
    rcases g with i | i <;> fin_cases i <;> simp [k4NewCol]

example := thirdIndex_ne_left 0 1 (by decide)

example := thirdIndex_ne_right 0 1 (by decide)

example := thirdIndex_mem_k4Triangles 0 1 (by decide)

example := k4NewCol_thirdIndex (F := ℚ) 0 1 (by decide)

example := exists_k4_functional (F := ℚ) (C := {.inl 0, .inl 1, .inl 2}) (by decide)

end Transformer.GainMatroid

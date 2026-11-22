module

public import Transformer.Phasor.InclusiveTransitions
public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-!
# Multiplicative interaction between two categorical reads

Source: the modulated transition-table trial in `experiments/phasor_value_transport/README.md`.
Each hidden coordinate adds its count entry, its current-token entry, and
their product. Both tables are learned across positions. We characterize
the resulting read before normalization and its transpose on a finite batch.
-/

@[expose] public section

namespace Transformer.Phasor

/-- The contribution from one coordinate of each selected embedding. -/
noncomputable def modulatedRead (x y : ℝ) : ℝ := x + y + x * y

/-- Zero initialization of the token table preserves every count embedding. -/
theorem modulatedRead_zero_token (x : ℝ) : modulatedRead x 0 = x := by
  simp [modulatedRead]

/-- The four-entry contrast measures the interaction of the two keys.
Its value can be nonzero even for a single coordinate. -/
theorem modulatedRead_contrast (x₁ x₂ y₁ y₂ : ℝ) :
    modulatedRead x₁ y₁ - modulatedRead x₂ y₁ - modulatedRead x₁ y₂ + modulatedRead x₂ y₂ =
      (x₁ - x₂) * (y₁ - y₂) := by
  unfold modulatedRead
  ring

/-- An exact expansion under a simultaneous perturbation of both reads.
The linear part weights each table's perturbation by the other table. -/
theorem modulatedRead_variation (x y dx dy t : ℝ) :
    modulatedRead (x + t * dx) (y + t * dy) =
      modulatedRead x y + t * ((1 + y) * dx + (1 + x) * dy) + t^2 * dx * dy := by
  unfold modulatedRead
  ring

/-- With one coordinate per second key, the sum of the modulated coordinates
can represent any table on the two keys, for arbitrary first-key type. -/
theorem exists_modulated_categorical_read {α : Type*} {p : ℕ} (target : α → Fin p → ℝ) :
    ∃ count : α → Fin p → ℝ, ∃ token : Fin p → Fin p → ℝ, ∀ a b,
      ∑ j, modulatedRead (count a j) (token b j) = target a b := by
  refine ⟨fun a j => target a j + p - 1,
    fun b j => (if j = b then 1 else 0) - 1, ?_⟩
  intro a b
  have coordinate (j : Fin p) :
      modulatedRead (target a j + p - 1) ((if j = b then 1 else 0) - 1) =
        (if j = b then target a j + p else 0) - 1 := by
    by_cases h : j = b <;> simp [modulatedRead, h]
    ring
  simp_rw [coordinate]
  rw [Finset.sum_sub_distrib]
  simp

/-- Grouping repeated addresses before multiplying by their common
perturbation gives the same scalar pairing as summing over active rows. -/
private theorem group_modulated_rows {ι κ : Type*} [DecidableEq κ] [Fintype κ]
    (s : Finset ι) (key : ι → κ) (weight : ι → ℝ) (change : κ → ℝ) :
    ∑ k, (∑ i ∈ s with key i = k, weight i) * change k =
      ∑ i ∈ s, weight i * change (key i) := by
  rw [← Finset.sum_fiberwise s key (fun i => weight i * change (key i))]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  rw [(Finset.mem_filter.mp hi).2]

/-- The transpose can accumulate into each parameter exactly once, including
when several active rows share either key. Rows outside the mask contribute
nothing and the two key spaces can have different sizes. -/
theorem modulatedRead_batch_adjoint {ι α β : Type*}
    [DecidableEq α] [Fintype α] [DecidableEq β] [Fintype β]
    (active : Finset ι) (a : ι → α) (b : ι → β)
    (x dx : α → ℝ) (y dy : β → ℝ) (g : ι → ℝ) :
    ∑ i ∈ active, g i * ((1 + y (b i)) * dx (a i) + (1 + x (a i)) * dy (b i)) =
      (∑ k, (∑ i ∈ active with a i = k, g i * (1 + y (b i))) * dx k) +
       ∑ k, (∑ i ∈ active with b i = k, g i * (1 + x (a i))) * dy k := by
  rw [group_modulated_rows, group_modulated_rows, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

example : modulatedRead 1 1 - modulatedRead 0 1 - modulatedRead 1 0 + modulatedRead 0 0 = 1 := by
  rw [modulatedRead_contrast]
  norm_num

example (x dx t : ℝ) : modulatedRead (x + t * dx) (-1) = -1 := by
  unfold modulatedRead
  ring

end Transformer.Phasor

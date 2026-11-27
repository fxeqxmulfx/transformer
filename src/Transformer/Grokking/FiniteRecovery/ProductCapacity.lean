module

public import Transformer.Grokking.FlatSpectrum.ProductDepth
public import Transformer.Grokking.FlatSpectrum.ProductLoopTree

/-! Capacities of the product constructions in Flat Spectrum, NV7 and LP1–LP2.
These constructions start with supplied local features or pair phases. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Transformer.Grokking.FlatSpectrum
open scoped BigOperators

variable {k : ℕ}

/-- One product layer per pass gives depth `layers * repeats`. On token-local
one-hot features, the mixed difference of the paired phase forces at least
`1 + ceil(log₂ k)` such layers, even with arbitrary width (NV7, LP1). -/
theorem depth_of_paired_product_readout {S : Type} [Fintype S] {m layers repeats : ℕ}
    (hk : 0 < k) (P : ZMod 2 → ZMod 2)
    {F : (((Fin k × ZMod 2) ⊕ (Fin k × ZMod 2)) ⊕ S → ℂ) →
      ((Fin k × ZMod 2) ⊕ (Fin k × ZMod 2)) ⊕ S → ℂ}
    (hF : IsProductNet m (layers * repeats) F)
    (read : (((Fin k × ZMod 2) ⊕ (Fin k × ZMod 2)) ⊕ S → ℂ) →ₗ[ℂ] ℂ)
    (hread : ∀ u v, read (F (Sum.elim (pairFeatures u v) 0)) =
      ZMod.stdAddChar (∑ i, (P (u i) + u i * v i))) :
    1 + Nat.clog 2 k ≤ layers * repeats :=
  le_depth_of_productNet_copies hk (fun a b => P a + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by decide) P⟩ hF read hread

/-- For a power of two pairs the product tree has depth `1 + log₂ k`,
`2k` hidden units per layer and fewer than `6k` complex stream coordinates.
The bound includes the `4k` input features and fewer than `2k` scratch
coordinates. As in NV7, the features are supplied to a single stream. -/
theorem exists_binary_paired_product_tree (depth : ℕ) (P : ZMod 2 → ZMod 2) :
    let k := 2 ^ depth
    ∃ (S : Type) (inst : Fintype S) (root : S)
      (F : (((Fin k × ZMod 2) ⊕ (Fin k × ZMod 2)) ⊕ S → ℂ) →
        ((Fin k × ZMod 2) ⊕ (Fin k × ZMod 2)) ⊕ S → ℂ),
      @Fintype.card S inst < 2 * k ∧ IsProductNet (2 * k) (1 + depth) F ∧
      ∀ u v, F (Sum.elim (pairFeatures u v) 0) (Sum.inr root) =
        ZMod.stdAddChar (∑ i, (P (u i) + u i * v i)) := by
  dsimp only
  classical
  obtain ⟨S, _, root, F, hS, hF, hval⟩ := exists_productNet depth (ZMod 2)
    (id : Fin (2 ^ depth) → Fin (2 ^ depth))
    (fun _ b a => ZMod.stdAddChar (P a + a * b))
    (fun _ b y => if y = b then 1 else 0)
  refine ⟨S, inferInstance, root, F, ?_, ?_, fun u v => ?_⟩
  · simpa only [pow_succ, mul_comm] using hS
  · simpa only [ZMod.card, mul_comm] using hF
  · rw [hval, stdAddChar_sum]
    exact Finset.prod_congr rfl fun i _ => by simp

example := exists_binary_paired_product_tree 2 (fun _ => 0)
example := exists_binary_paired_product_tree 3 (fun _ => 0)
example := exists_binary_paired_product_tree 4 (fun _ => 0)

example : 1 + Nat.clog 2 4 = 3 ∧ 1 + Nat.clog 2 8 = 4 ∧ 1 + Nat.clog 2 16 = 5 := by decide

example : 3 ≤ 2 * 2 ∧ ¬ 3 ≤ 2 * 1 ∧ 4 ≤ 2 * 2 ∧ ¬ 4 ≤ 2 * 1 ∧
    5 ≤ 2 * 3 ∧ ¬ 5 ≤ 2 * 2 := by decide

/-- In the LP2 loop, each pass uses the same product core. With all pair
phases supplied as leaves, pass `n + 1` computes their product if
`k ≤ 2^(n+1)`. The core has `h` product units and room for `2h` leaves. -/
theorem exists_binary_paired_loop_readout {h n : ℕ} (hh : 0 < h) (hk : k ≤ 2 * h)
    (hn : k ≤ 2 ^ (n + 1)) (P : ZMod 2 → ZMod 2) (u v : Fin k → ZMod 2) :
    ∃ z : Empty ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ, TreeStart z ∧
      loopState treeCore treeRead z n (prodSlot ⟨0, by omega⟩) =
        ZMod.stdAddChar (∑ i, (P (u i) + u i * v i)) := by
  let phase (i : Fin k) := ZMod.stdAddChar (P (u i) + u i * v i)
  let z : Empty ⊕ (Fin (2 * h) ⊕ Fin (2 * h)) → ℂ :=
    Sum.elim (fun _ => 0) (Sum.elim
      (fun j => if hj : (j : ℕ) < k then phase ⟨j, hj⟩ else 1)
      (fun j => if (j : ℕ) < h then 0 else 1))
  have hz : TreeStart z := fun _ => rfl
  have hpad (j : ℕ) (hj : k ≤ j) : treeLeaves z j = 1 := by
    by_cases hj' : j < 2 * h
    · simp [treeLeaves, z, hj', not_lt.mpr hj]
    · simp [treeLeaves, hj']
  refine ⟨z, hz, ?_⟩
  rw [loopState_treeCore_root hz hh hpad hn, stdAddChar_sum,
    ← Fin.prod_univ_eq_prod_range (treeLeaves z)]
  exact Finset.prod_congr rfl fun j _ => by
    simp [treeLeaves, z, j.isLt.trans_le hk, j.isLt, phase]

example (u v : Fin 4 → ZMod 2) :=
  exists_binary_paired_loop_readout (h := 2) (n := 1) (by decide) (by decide) (by decide) 0 u v

example (u v : Fin 8 → ZMod 2) :=
  exists_binary_paired_loop_readout (h := 4) (n := 2) (by decide) (by decide) (by decide) 0 u v

example (u v : Fin 16 → ZMod 2) :=
  exists_binary_paired_loop_readout (h := 8) (n := 3) (by decide) (by decide) (by decide) 0 u v

end Transformer.Grokking.FiniteRecovery

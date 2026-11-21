module

public import Transformer.Grokking.FlatSpectrum.SharedEmbedding
public import Transformer.Grokking.FlatSpectrum.ProductLayer
public import Transformer.Grokking.FlatSpectrum.BilinearKernel

/-!
# One product layer writes a shared table

Source: `review/flat_spectrum.md`, NV9. Let a state read its operands through one embedding `E`,
each side through its own linear map: `A (E a) + B (E b) + z`. Positional heads over complex-linear
value maps give this form once `R_D` and the routers are set aside. Then every complex-linear
reading of one product layer of that state is a shared table, at any number of hidden units
(`IsProductLayer.isSharedTable`). For `k` copies the operands of one side of the cut are read
through `copiesEmbedding`, of width `k D`, so a layer that writes `ψ(Σᵢ xᵢ³ + xᵢ yᵢ)` has
`k D + 3 ≥ p^k` (`IsProductLayer.ne_copies`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Module

section Closure

variable {K α ι ζ : Type*} [Field K] [Fintype ι] {E : Matrix α ι K}

/-- Shared tables add. -/
lemma IsSharedTable.add {X Y : Matrix α α K} (hX : IsSharedTable E X) (hY : IsSharedTable E Y) :
    IsSharedTable E (X + Y) := by
  obtain ⟨C, f, g, c, hX⟩ := hX
  obtain ⟨C', f', g', c', hY⟩ := hY
  refine ⟨C + C', f + f', g + g', c + c', fun a b => ?_⟩
  simp only [Matrix.add_apply, hX, hY, Matrix.mul_add, Matrix.add_mul, Pi.add_apply]
  ring

/-- A multiple of a shared table is one. -/
lemma IsSharedTable.smul (s : K) {X : Matrix α α K} (hX : IsSharedTable E X) :
    IsSharedTable E (s • X) := by
  obtain ⟨C, f, g, c, hX⟩ := hX
  refine ⟨s • C, s • f, s • g, s * c, fun a b => ?_⟩
  simp only [Matrix.smul_apply, hX, Matrix.mul_smul, Matrix.smul_mul, Pi.smul_apply, smul_eq_mul]
  ring

lemma isSharedTable_left (f : α → K) : IsSharedTable E (of fun a _ => f a) :=
  ⟨0, f, 0, 0, fun a b => by simp⟩

lemma isSharedTable_right (g : α → K) : IsSharedTable E (of fun _ b => g b) :=
  ⟨0, 0, g, 0, fun a b => by simp⟩

lemma isSharedTable_const (c : K) : IsSharedTable E (of fun _ _ => c) :=
  ⟨0, 0, 0, c, fun a b => by simp⟩

lemma isSharedTable_zero : IsSharedTable E (0 : Matrix α α K) :=
  ⟨0, 0, 0, 0, fun a b => by simp⟩

example : IsSharedTable (1 : Matrix (Fin 2) (Fin 2) ℚ)
    ((2 : ℚ) • ((0 : Matrix (Fin 2) (Fin 2) ℚ) + 0)) :=
  (isSharedTable_zero.add isSharedTable_zero).smul 2

variable [DecidableEq ι]

/-- The product of a linear reading of the first operand and one of the second. -/
lemma isSharedTable_bilinear (L M : (ι → K) →ₗ[K] K) :
    IsSharedTable E (of fun a b => L (E a) * M (E b)) := by
  set x : ι → K := fun i => L fun j => if i = j then 1 else 0
  set y : ι → K := fun i => M fun j => if i = j then 1 else 0
  refine ⟨vecMulVec x y, 0, 0, 0, fun a b => ?_⟩
  simp only [of_apply, LinearMap.pi_apply_eq_sum_univ L (E a),
    LinearMap.pi_apply_eq_sum_univ M (E b), smul_eq_mul, Matrix.mul_apply, vecMulVec_apply,
    transpose_apply, Pi.zero_apply, add_zero]
  rw [Finset.sum_mul_sum, Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [x, y]
  ring

/-- The product of two linear readings of a state `A (E a) + B (E b) + z`. -/
lemma isSharedTable_product
    (φ ψ : (ζ → K) →ₗ[K] K) (A B : (ι → K) →ₗ[K] ζ → K) (z : ζ → K) :
    IsSharedTable E
      (of fun a b => φ (A (E a) + B (E b) + z) * ψ (A (E a) + B (E b) + z)) := by
  have hsplit : (of fun a b => φ (A (E a) + B (E b) + z) * ψ (A (E a) + B (E b) + z)) =
      (of fun a b => (φ ∘ₗ A) (E a) * (ψ ∘ₗ B) (E b)) +
        (of fun a b => (ψ ∘ₗ A) (E a) * (φ ∘ₗ B) (E b)) +
        (of fun a _ => φ (A (E a)) * ψ (A (E a)) + φ (A (E a)) * ψ z + φ z * ψ (A (E a))) +
        (of fun _ b => φ (B (E b)) * ψ (B (E b)) + φ (B (E b)) * ψ z + φ z * ψ (B (E b))) +
        (of fun _ _ => φ z * ψ z) := by
    ext a b
    simp only [of_apply, Matrix.add_apply, map_add, LinearMap.comp_apply]
    ring
  rw [hsplit]
  exact (((((isSharedTable_bilinear _ _).add (isSharedTable_bilinear _ _)).add
    (isSharedTable_left _)).add (isSharedTable_right _)).add (isSharedTable_const _))

end Closure

section Layer

variable {α ι ζ : Type*} [Fintype ι] [DecidableEq ι]

/-- **One product layer writes a shared table**, at any number of hidden units: every
complex-linear reading of the layer at a state `A (E a) + B (E b) + z`. -/
theorem IsProductLayer.isSharedTable {m : ℕ} {F : (ζ → ℂ) → ζ → ℂ} (hF : IsProductLayer m F)
    (E : Matrix α ι ℂ) (A B : (ι → ℂ) →ₗ[ℂ] ζ → ℂ) (z : ζ → ℂ) (r : (ζ → ℂ) →ₗ[ℂ] ℂ) :
    IsSharedTable E (of fun a b => r (F (A (E a) + B (E b) + z))) := by
  obtain ⟨η, _, -, gate, up, out, hF⟩ := hF
  have hsplit : (of fun a b => r (F (A (E a) + B (E b) + z))) =
      (of fun a _ => r (A (E a))) + (of fun _ b => r (B (E b))) + (of fun _ _ => r z) +
        ∑ u, r (out u) • (of fun a b =>
          gate u (A (E a) + B (E b) + z) * up u (A (E a) + B (E b) + z)) := by
    ext a b
    rw [of_apply, hF, map_add, map_sum]
    simp only [Matrix.add_apply, of_apply, Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
      map_add r, map_smul]
    congr 1
    exact Finset.sum_congr rfl fun u _ => by ring
  rw [hsplit]
  exact ((isSharedTable_left _).add (isSharedTable_right _)).add (isSharedTable_const _) |>.add
    (Finset.sum_induction _ (IsSharedTable E) (fun _ _ => IsSharedTable.add) isSharedTable_zero
      fun u _ => (isSharedTable_product _ _ _ _ _).smul _)

example := (isProductLayer_id (ι := Fin 2) 0).isSharedTable (1 : Matrix (Fin 2) (Fin 2) ℂ)
  LinearMap.id LinearMap.id 0 (LinearMap.proj 0)

end Layer

section Copies

variable {ι ζ : Type*} [Fintype ι] [DecidableEq ι]

/-- The operands of one side of the cut between `k` copies, read through one embedding: the
coordinate `(i, l)` holds coordinate `l` of the embedding of operand `i`. -/
def copiesEmbedding {K α : Type*} (k : ℕ) (E : Matrix α ι K) : Matrix (Fin k → α) (Fin k × ι) K :=
  of fun x j => E (x j.1) j.2

/-- **One layer pays the cut price in the width of its state.** A complex-linear reading of one
product layer, at any number of hidden units, is not a flat table of `k` copies across the cut
unless `k D + 3 ≥ p^k`. -/
theorem IsProductLayer.ne_bilinearKernel {p k m : ℕ} [NeZero p] {F : (ζ → ℂ) → ζ → ℂ}
    (hF : IsProductLayer m F) (E : Matrix (ZMod p) ι ℂ) (A B : (Fin k × ι → ℂ) →ₗ[ℂ] ζ → ℂ)
    (z : ζ → ℂ) (r : (ζ → ℂ) →ₗ[ℂ] ℂ) (g h : (Fin k → ZMod p) → ZMod p)
    {M : Matrix (Fin k) (Fin k) (ZMod p)} (hM : IsUnit M)
    (hD : k * Fintype.card ι + 3 < p ^ k) :
    (of fun x y => r (F (A (copiesEmbedding k E x) + B (copiesEmbedding k E y) + z))) ≠
      bilinearKernel p g h M := by
  intro heq
  have hs := hF.isSharedTable (copiesEmbedding k E) A B z r
  rw [heq] at hs
  have hle := hs.finrank_col_sup_row_le
  have hge : (bilinearKernel p g h M).rank ≤
      finrank ℂ ↥(LinearMap.range (bilinearKernel p g h M).mulVecLin ⊔
        LinearMap.range (bilinearKernel p g h M)ᵀ.mulVecLin) := Submodule.finrank_mono le_sup_left
  rw [rank_bilinearKernel g h hM, Fintype.card_fin] at hge
  rw [Fintype.card_prod, Fintype.card_fin] at hle
  exact absurd (hge.trans hle) (not_le.2 hD)

example := (isProductLayer_id (ι := Fin 1) 0).ne_bilinearKernel (p := 5) (k := 1)
  (0 : Matrix (ZMod 5) (Fin 1) ℂ) 0 0 0 0 0 0 (M := 1) isUnit_one (by simp)

/-- **No product layer writes the additive code of `k` copies of `x³ + x y`** through an
embedding with `k D + 3 < p^k`. -/
theorem IsProductLayer.ne_copies {p k m : ℕ} [NeZero p] {F : (ζ → ℂ) → ζ → ℂ}
    (hF : IsProductLayer m F) (E : Matrix (ZMod p) ι ℂ) (A B : (Fin k × ι → ℂ) →ₗ[ℂ] ζ → ℂ)
    (z : ζ → ℂ) (r : (ζ → ℂ) →ₗ[ℂ] ℂ) (hD : k * Fintype.card ι + 3 < p ^ k) :
    (of fun x y => r (F (A (copiesEmbedding k E x) + B (copiesEmbedding k E y) + z))) ≠
      of fun x y : Fin k → ZMod p => ZMod.stdAddChar (∑ i, (x i ^ 3 + x i * y i)) := by
  rw [← bilinearKernel_cube_add_mul]
  exact hF.ne_bilinearKernel E A B z r _ _ isUnit_one hD

/-- At main's width, `D_ℂ = 64`, two copies at `p = 13`: `2 · 64 + 3 < 169`. -/
example := (isProductLayer_id (ι := Fin 64) 0).ne_copies (p := 13) (k := 2)
  (0 : Matrix (ZMod 13) (Fin 64) ℂ) 0 0 0 0 (by simp)

end Copies

end Transformer.Grokking.FlatSpectrum

module

public import Transformer.Grokking.FlatSpectrum.ProductCopies

/-!
# One product layer pays the price of the cut

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. Feed a product layer a
state `e(u) + f(v)`, a part from each side of a cut, and read its output by any linear
functional `ℓ`. Each reading of the state is a sum `g(e u) + g(f v)`, so the product of two
readings is a function of `u`, plus a function of `v`, plus two terms (function of `u`) ·
(function of `v`). Across the cut, the readout is a matrix of rank at most `2m + 2` for `m`
hidden units (`rank_productLayer_le`). A table `ψ(g(x) + h(y) + x ⬝ M y)` with `M` invertible
has rank `p^k` (NV1), so one layer that reads it out exactly, from any encodings of `x` and of
`y`, has `2m + 2 ≥ p^k` (`pow_le_of_productLayer_bilinear`; for `k` copies of `x³ + x y`,
`pow_le_of_productLayer_copies`). The one-hot features of the pairs are such encodings
(`pairFeatures_eq_add`), and on them `1 + ⌈log₂ k⌉` layers need fewer than `2kp` units
(`exists_productNet_mulChar`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Matrix

/-- **One product layer across a cut**: on states `e(u) + f(v)`, every linear readout `ℓ` of a
product layer of at most `m` hidden units is a matrix of rank at most `2m + 2` in `(u, v)`. -/
theorem rank_productLayer_le {A B ι : Type*} [Fintype A] [Fintype B] {m : ℕ}
    {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F) (e : A → ι → ℂ) (f : B → ι → ℂ)
    (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ) : (Matrix.of fun u v => ℓ (F (e u + f v))).rank ≤ 2 * m + 2 := by
  classical
  obtain ⟨η, _, hm, g, w, o, hF⟩ := hF
  -- what the columns combine: the part of `u` alone, the constant, and both readings of `e u`
  let fam : Bool × Option η → A → ℂ := fun bj u => bj.2.elim
    (cond bj.1 (ℓ (e u) + ∑ j, ℓ (o j) * (g j (e u) * w j (e u))) 1)
    fun j => cond bj.1 (g j (e u)) (w j (e u))
  have hmem (bj : Bool × Option η) : fam bj ∈ Submodule.span ℂ (Set.range fam) :=
    Submodule.subset_span ⟨bj, rfl⟩
  have hcol (v : B) : (Matrix.of fun u v => ℓ (F (e u + f v))).col v ∈
      Submodule.span ℂ (Set.range fam) := by
    have key : (Matrix.of fun u v => ℓ (F (e u + f v))).col v = fam (true, none) +
        (ℓ (f v) + ∑ j, ℓ (o j) * (g j (f v) * w j (f v))) • fam (false, none) +
        ∑ j, ((ℓ (o j) * w j (f v)) • fam (true, some j) +
          (ℓ (o j) * g j (f v)) • fam (false, some j)) := by
      funext u
      have hsum : ∑ j, (g j (e u) + g j (f v)) * (w j (e u) + w j (f v)) * ℓ (o j) =
          ∑ j, ℓ (o j) * (g j (e u) * w j (e u)) + ∑ j, ℓ (o j) * (g j (f v) * w j (f v)) +
            ∑ j, (ℓ (o j) * w j (f v) * g j (e u) + ℓ (o j) * g j (f v) * w j (e u)) := by
        rw [← sum_add_distrib, ← sum_add_distrib]
        exact sum_congr rfl fun j _ => by ring
      simp only [col_apply, of_apply, hF, map_add, map_sum, map_smul, smul_eq_mul, hsum, fam,
        Pi.add_apply, Finset.sum_apply, Pi.smul_apply, Option.elim_none, Option.elim_some,
        Bool.cond_true, Bool.cond_false, mul_one]
      ring
    rw [key]
    exact Submodule.add_mem _ (Submodule.add_mem _ (hmem _) (Submodule.smul_mem _ _ (hmem _)))
      (Submodule.sum_mem _ fun j _ => Submodule.add_mem _ (Submodule.smul_mem _ _ (hmem _))
        (Submodule.smul_mem _ _ (hmem _)))
  rw [rank_eq_finrank_span_cols]
  calc Module.finrank ℂ (Submodule.span ℂ (Set.range (Matrix.of fun u v => ℓ (F (e u + f v))).col))
      ≤ Module.finrank ℂ (Submodule.span ℂ (Set.range fam)) :=
        Submodule.finrank_mono (Submodule.span_le.2 (Set.range_subset_iff.2 hcol))
    _ ≤ Fintype.card (Bool × Option η) := finrank_range_le_card fam
    _ ≤ 2 * m + 2 := by
      rw [Fintype.card_prod, Fintype.card_bool, Fintype.card_option]
      omega

/-- The identity on one coordinate, read on `1 + 0`: rank at most `2`. -/
example := rank_productLayer_le (isProductLayer_id (ι := Unit) 0) (fun _ : Unit => fun _ => 1)
  (fun _ : Unit => 0) (LinearMap.proj ())

/-- **One layer pays the cut price**: if a linear readout of one product layer of at most `m`
hidden units is `ψ(g(x) + h(y) + x ⬝ M y)` with `M` invertible on states `e(x) + f(y)`, for any
encodings `e` of `x` and `f` of `y`, then `p^|κ| ≤ 2m + 2`. -/
theorem pow_le_of_productLayer_bilinear {p : ℕ} [NeZero p] {κ ι : Type*} [Fintype κ]
    [DecidableEq κ] {m : ℕ} {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F)
    (e f : (κ → ZMod p) → ι → ℂ) (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ) (g h : (κ → ZMod p) → ZMod p)
    {M : Matrix κ κ (ZMod p)} (hM : IsUnit M)
    (hval : ∀ x y, ℓ (F (e x + f y)) = bilinearKernel p g h M x y) :
    p ^ Fintype.card κ ≤ 2 * m + 2 := by
  have hK : (Matrix.of fun x y => ℓ (F (e x + f y))) = bilinearKernel p g h M := by
    ext x y
    rw [of_apply, hval]
  have := rank_productLayer_le hF e f ℓ
  rwa [hK, rank_bilinearKernel g h hM] at this

/-- **`k` copies of `x³ + x y` in one layer**: if a linear readout of one product layer of at
most `m` hidden units is `ψ(∑_i (x_i³ + x_i y_i))` on states `e(x) + f(y)`, then
`p^k ≤ 2m + 2`. -/
theorem pow_le_of_productLayer_copies {p k : ℕ} [NeZero p] {ι : Type*} {m : ℕ}
    {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F) (e f : (Fin k → ZMod p) → ι → ℂ)
    (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ)
    (hval : ∀ x y, ℓ (F (e x + f y)) = ZMod.stdAddChar (∑ i, (x i ^ 3 + x i * y i))) :
    p ^ k ≤ 2 * m + 2 := by
  have := pow_le_of_productLayer_bilinear hF e f ℓ (fun x => ∑ i, x i ^ 3) (fun _ => 0)
    isUnit_one fun x y => by rw [hval, bilinearKernel_cube_add_mul, of_apply]
  rwa [Fintype.card_fin] at this

variable {κ α β : Type} [DecidableEq α] [DecidableEq β]

/-- The one-hot features of the pairs are a sum of an encoding of the `x`'s and one of the
`y`'s, so the bounds above apply to them. -/
lemma pairFeatures_eq_add {T : Type} (x : κ → α) (y : κ → β) :
    (Sum.elim (pairFeatures x y) 0 : ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) =
      Sum.elim (Sum.elim (oneHot x) 0) 0 + Sum.elim (Sum.elim (0 : κ × α → ℂ) (oneHot y)) 0 := by
  funext n
  rcases n with (n | n) | n <;> simp [pairFeatures]

/-- At `k = 1` the network of `exists_productNet_copies` is one layer of `2 · 7` units that
writes `ψ(x³ + x y)` over `ℤ/7`; the bound asks for at least `(7 − 2)/2`. -/
example : 7 ^ Fintype.card (Fin 1) ≤ 2 * (2 * 1 * Fintype.card (ZMod 7)) + 2 := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 1) one_pos fun a b => a ^ 3 + a * b
  have hF1 : IsProductNet (2 * 1 * Fintype.card (ZMod 7)) 1 F := by simpa using hF
  exact pow_le_of_productLayer_bilinear hF1.isProductLayer
    (fun x => Sum.elim (Sum.elim (oneHot x) 0) 0) (fun y => Sum.elim (Sum.elim 0 (oneHot y)) 0)
    (LinearMap.proj (Sum.inr root)) (fun x => ∑ i, x i ^ 3) (fun _ => 0) isUnit_one
    fun x y => by
      rw [LinearMap.proj_apply, ← pairFeatures_eq_add, hval, bilinearKernel_cube_add_mul,
        of_apply]

example : 7 ^ 1 ≤ 2 * (2 * 1 * Fintype.card (ZMod 7)) + 2 := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 1) one_pos fun a b => a ^ 3 + a * b
  have hF1 : IsProductNet (2 * 1 * Fintype.card (ZMod 7)) 1 F := by simpa using hF
  exact pow_le_of_productLayer_copies hF1.isProductLayer
    (fun x => Sum.elim (Sum.elim (oneHot x) 0) 0) (fun y => Sum.elim (Sum.elim 0 (oneHot y)) 0)
    (LinearMap.proj (Sum.inr root)) fun x y => by
      rw [LinearMap.proj_apply, ← pairFeatures_eq_add, hval]

end Transformer.Grokking.FlatSpectrum

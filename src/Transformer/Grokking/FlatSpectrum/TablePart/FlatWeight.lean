module

public import Mathlib.FieldTheory.Finiteness
public import Mathlib.LinearAlgebra.Dual.Lemmas
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatClass

/-!
# PT4: counting the nonzero cells

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT4, *The worst*. The
counts behind `δ(H) = min(p^k, p^l)`. A nonzero linear form on `F^{l+1}`, shifted by any point, is
nonzero at `q^{l+1} − q^l` points (`card_form_ne_zero`). So a function `(u − u₀)ᵀ M (v − v₀)` with
`M ≠ 0` has at least `q^{k+l} (q − 1)²` nonzero cells on `F^{k+1} × F^{l+1}`
(`mul_le_hammingNorm_form`), and this is at least `min(q^{k+1}, q^{l+1})` unless `q = 2`,
`k = l = 0` (`min_pow_succ_le_mul`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Module

/-- A map that sends `X` injectively into the nonzero cells of `f` bounds its weight. -/
lemma card_le_hammingNorm {B X G : Type*} [Fintype B] [Fintype X] [DecidableEq G] [Zero G]
    {f : B → G} (e : X → B) (he : Function.Injective e) (h : ∀ x, f (e x) ≠ 0) :
    Fintype.card X ≤ hammingNorm f := by
  have hle := card_le_card_of_injOn (s := (univ : Finset X)) (t := univ.filter fun b => f b ≠ 0) e
    (fun x _ => mem_coe.2 (mem_filter.2 ⟨mem_univ _, h x⟩)) he.injOn
  rwa [card_univ] at hle

example : Fintype.card (Fin 1) ≤ hammingNorm (fun _ : Fin 1 => (1 : ℚ)) :=
  card_le_hammingNorm id Function.injective_id fun _ => one_ne_zero

/-- A function whose nonzero cells are exactly the image of an injection from `X` has weight
`|X|`. -/
lemma hammingNorm_eq_card {B X G : Type*} [Fintype B] [DecidableEq B] [Fintype X] [DecidableEq G]
    [Zero G] {f : B → G} (e : X → B) (he : Function.Injective e)
    (h : ∀ b, f b ≠ 0 ↔ b ∈ Set.range e) : hammingNorm f = Fintype.card X := by
  have hset : (univ.filter fun b => f b ≠ 0) = univ.image e := by
    ext b
    simp [h b]
  unfold hammingNorm
  rw [hset, card_image_of_injective _ he, card_univ]

example : hammingNorm (fun _ : Fin 1 => (1 : ℚ)) = Fintype.card (Fin 1) :=
  hammingNorm_eq_card id Function.injective_id fun b => by simp

/-- From `a + q^l = q^{l+1}`, the value `a = q^l (q − 1)`. -/
lemma eq_pow_mul_sub_one {q a l : ℕ} (hq : 1 ≤ q) (h : a + q ^ l = q ^ (l + 1)) :
    a = q ^ l * (q - 1) := by
  obtain ⟨d, rfl⟩ : ∃ d, q = d + 1 := ⟨q - 1, by omega⟩
  rw [pow_succ, Nat.mul_add, Nat.mul_one] at h
  simpa using h

example : 2 = 2 ^ 0 * (3 - 1) := eq_pow_mul_sub_one (q := 3) (a := 2) (l := 0) (by norm_num) (by norm_num)

/-- **The numbers of PT4**: unless `q = 2` and `k = l = 0`, the count of the nonzero cells is at
least the shorter line (`review/flat_spectrum.md`, PT4). -/
lemma min_pow_succ_le_mul {q k l : ℕ} (hq : 2 ≤ q) (hne : ¬ (q = 2 ∧ k = 0 ∧ l = 0)) :
    min (q ^ (k + 1)) (q ^ (l + 1)) ≤ (q ^ k * (q - 1)) * (q ^ l * (q - 1)) := by
  obtain ⟨d, rfl⟩ : ∃ d, q = d + 1 := ⟨q - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  have hd : 1 ≤ d := by omega
  have hdd : 1 ≤ d * d := Nat.mul_pos hd hd
  have hbase : (d + 1) ^ k * (d + 1) ^ l ≤ (d + 1) ^ k * d * ((d + 1) ^ l * d) := by
    calc (d + 1) ^ k * (d + 1) ^ l = (d + 1) ^ k * (d + 1) ^ l * 1 := (Nat.mul_one _).symm
      _ ≤ (d + 1) ^ k * (d + 1) ^ l * (d * d) := Nat.mul_le_mul_left _ hdd
      _ = (d + 1) ^ k * d * ((d + 1) ^ l * d) := by ring
  by_cases hl : l = 0
  · by_cases hk : k = 0
    · subst hl hk
      have hd2 : 2 ≤ d := by
        by_contra h
        exact hne ⟨by omega, rfl, rfl⟩
      simp only [pow_zero, one_mul, zero_add, pow_one]
      exact (min_le_left _ _).trans (by nlinarith)
    · refine (min_le_right _ _).trans (le_trans ?_ hbase)
      rw [pow_succ, Nat.mul_comm]
      exact Nat.mul_le_mul_right _ (Nat.le_self_pow hk _)
  · refine (min_le_left _ _).trans (le_trans ?_ hbase)
    rw [pow_succ]
    exact Nat.mul_le_mul_left _ (Nat.le_self_pow hl _)

example : min (3 ^ 1) (3 ^ 1) ≤ (3 ^ 0 * (3 - 1)) * (3 ^ 0 * (3 - 1)) :=
  min_pow_succ_le_mul (by norm_num) (by norm_num)

variable {F ι κ : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- A nonzero linear form on `F^{l+1}`, shifted by `z₀`, is nonzero at `q^{l+1} − q^l` points;
written without subtraction. -/
lemma card_form_ne_zero [Fintype κ] [DecidableEq κ] {l : ℕ} (hl : Fintype.card κ = l + 1)
    (φ : (κ → F) →ₗ[F] F) (hφ : φ ≠ 0) (z₀ : κ → F) :
    #{z : κ → F | φ (z - z₀) ≠ 0} + Fintype.card F ^ l = Fintype.card F ^ (l + 1) := by
  have hker := Module.Dual.finrank_ker_add_one_of_ne_zero hφ
  rw [finrank_fintype_fun_eq_card, hl] at hker
  have hzero : #{z : κ → F | φ z = 0} = Fintype.card F ^ l := by
    have h := Module.natCard_eq_pow_finrank (K := F) (V := LinearMap.ker φ)
    rw [Nat.card_eq_fintype_card (α := F), show finrank F (LinearMap.ker φ) = l by omega] at h
    rw [← h, ← Fintype.card_subtype, ← Nat.card_eq_fintype_card]
    exact Nat.card_congr (Equiv.subtypeEquivRight fun z => LinearMap.mem_ker.symm)
  have hshift : #{z : κ → F | φ (z - z₀) = 0} = #{z : κ → F | φ z = 0} :=
    card_equiv (Equiv.subRight z₀) fun z => by simp
  have hsplit := card_filter_add_card_filter_not (s := (univ : Finset (κ → F)))
    (fun z => φ (z - z₀) = 0)
  rw [hshift, hzero, card_univ, Fintype.card_fun, hl] at hsplit
  have hne : #{z : κ → F | φ (z - z₀) ≠ 0} = #{z : κ → F | ¬ φ (z - z₀) = 0} := rfl
  omega

example := card_form_ne_zero (F := ZMod 2) (κ := Fin 1) (l := 0) (by simp)
  (LinearMap.proj 0) (fun h => one_ne_zero (LinearMap.congr_fun h fun _ => (1 : ZMod 2))) 0

/-- The form `z ↦ wᵀ M z`. -/
def rightForm [Fintype κ] [Fintype ι] (w : ι → F) (M : Matrix ι κ F) : (κ → F) →ₗ[F] F where
  toFun z := w ⬝ᵥ M *ᵥ z
  map_add' a b := by simp [mulVec_add, dotProduct_add]
  map_smul' t a := by simp [mulVec_smul, dotProduct_smul]

/-- The form `w ↦ wᵀ M z`. -/
def leftForm [Fintype κ] [Fintype ι] (M : Matrix ι κ F) (z : κ → F) : (ι → F) →ₗ[F] F where
  toFun w := w ⬝ᵥ M *ᵥ z
  map_add' a b := by simp [add_dotProduct]
  map_smul' t a := by simp [smul_dotProduct]

variable [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

/-- **The nonzero cells of `(u − u₀)ᵀ M (v − v₀)`** (`review/flat_spectrum.md`, PT4): if
`M_{i₀ j₀} ≠ 0`, at least `q^k (q − 1)` rows are nonzero at the column `e_{j₀}`, and each of them
is a nonconstant affine function with `q^l (q − 1)` nonzero cells. -/
lemma mul_le_hammingNorm_form {M : Matrix ι κ F} {i₀ : ι} {j₀ : κ} (hij : M i₀ j₀ ≠ 0)
    (u₀ : ι → F) (v₀ : κ → F) {k l : ℕ} (hk : Fintype.card ι = k + 1)
    (hl : Fintype.card κ = l + 1) :
    (Fintype.card F ^ k * (Fintype.card F - 1)) * (Fintype.card F ^ l * (Fintype.card F - 1)) ≤
      hammingNorm fun c : (ι → F) × (κ → F) => (c.1 - u₀) ⬝ᵥ M *ᵥ (c.2 - v₀) := by
  have hq : 1 ≤ Fintype.card F := Fintype.card_pos
  have hψ : leftForm M (Pi.single j₀ 1) ≠ 0 := fun h => hij (by
    have h' := LinearMap.congr_fun h (Pi.single i₀ 1)
    simpa [leftForm, mulVec_single_one] using h')
  let R : Finset (ι → F) := {u | leftForm M (Pi.single j₀ 1) (u - u₀) ≠ 0}
  have hR : #R = Fintype.card F ^ k * (Fintype.card F - 1) :=
    eq_pow_mul_sub_one hq (card_form_ne_zero hk _ hψ u₀)
  let T (u : ι → F) : Finset (κ → F) := {v | rightForm (u - u₀) M (v - v₀) ≠ 0}
  have hT (u : ι → F) (hu : u ∈ R) :
      #(T u) = Fintype.card F ^ l * (Fintype.card F - 1) := by
    refine eq_pow_mul_sub_one hq (card_form_ne_zero hl _ (fun h => (mem_filter.1 hu).2 ?_) v₀)
    exact LinearMap.congr_fun h (Pi.single j₀ 1)
  have hsum : ∑ u ∈ R, #(T u) =
      #R * (Fintype.card F ^ l * (Fintype.card F - 1)) := by
    rw [sum_congr rfl hT, sum_const, smul_eq_mul]
  rw [← hR, ← hsum, ← card_sigma]
  refine card_le_card_of_injOn (fun x => (x.1, x.2)) (fun x hx => ?_) ?_
  · have hx' := mem_sigma.1 (mem_coe.1 hx)
    exact mem_coe.2 (mem_filter.2 ⟨mem_univ _, (mem_filter.1 hx'.2).2⟩)
  · rintro ⟨u, v⟩ _ ⟨u', v'⟩ _ h
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    rfl

example := mul_le_hammingNorm_form (F := ZMod 2) (M := (1 : Matrix (Fin 1) (Fin 1) (ZMod 2)))
  (i₀ := 0) (j₀ := 0) (by simp) 0 0 (k := 0) (l := 0) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum.TablePart

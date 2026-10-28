module

public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery
public import Transformer.Grokking.FlatSpectrum.ReadoutChoice

/-!
# The table of `a b` from `2p − 3` cells

Source: `review/flat_spectrum.md`, *Theorem G for moduli one*: M recovers `[χ(a b)]` from `Ω`
exactly when the cells of `Ω` with `a, b ≠ 0` are rigid on the nonzero residues and contain the
graph of a permutation of them; and its Scope: "A rigid pattern is connected, so it has at least
`2p − 1` cells", here for the `p − 1` nonzero rows and columns.

Added:

* The diagonal and row `i₀`, the cells `(i, i)` and `(i₀, j)`, connect the rows with the columns:
  every column meets row `i₀`, and every row meets its own column (`diagRow_connected`). They are
  `2N − 1` cells (`ncard_diagRow`).
* For prime `p`, the cells `(a, a)` and `(1, b)` with `a, b ≠ 0` (`productCells`, `2p − 3` cells)
  recover `[χ(a b)]` for every multiplicative character `χ`
  (`forall_isMinOn_taskKernel_mul_mulChar_productCells`), and no set of fewer cells recovers it
  (`two_mul_sub_three_le_ncard_of_forall_isMinOn`): its cells with `a, b ≠ 0` connect the
  `2p − 2` nonzero rows and columns. So `2p − 3` is the least number of cells from which M
  recovers `[χ(a b)]` (`isLeast_ncard_forall_isMinOn_taskKernel_mul_mulChar`), and some such `χ`
  is injective (`exists_injective_mulChar_isLeast`), so that the table gives back `a b`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix SimpleGraph

/-- The diagonal and row `i₀`: the cells `(i, i)` and `(i₀, j)`. -/
def diagRow {n : Type*} (i₀ : n) : Set (n × n) :=
  Set.range (fun i => (i, i)) ∪ Set.range fun j => (i₀, j)

/-- The diagonal and a row connect the rows with the columns. -/
theorem diagRow_connected {n : Type*} (i₀ : n) : (patternGraph (diagRow i₀)).Connected := by
  have hcol (j : n) : (patternGraph (diagRow i₀)).Reachable (.inl i₀) (.inr j) :=
    (patternGraph_adj_inl_inr.mpr (Or.inr ⟨j, rfl⟩)).reachable
  have hall : ∀ x, (patternGraph (diagRow i₀)).Reachable (.inl i₀) x
    | .inl i => (hcol i).trans (patternGraph_adj_inr_inl.mpr (Or.inl ⟨i, rfl⟩)).reachable
    | .inr j => hcol j
  exact (connected_iff _).mpr ⟨fun u v => (hall u).symm.trans (hall v), ⟨.inl i₀⟩⟩

/-- The diagonal and a row are `2N − 1` cells. -/
theorem ncard_diagRow {n : Type*} [Fintype n] (i₀ : n) :
    (diagRow i₀).ncard = 2 * Fintype.card n - 1 := by
  have hinter : (Set.range fun i : n => (i, i)) ∩ (Set.range fun j => (i₀, j)) = {(i₀, i₀)} := by
    ext ⟨a, b⟩
    simp only [Set.mem_inter_iff, Set.mem_range, Prod.mk.injEq, Set.mem_singleton_iff]
    constructor
    · rintro ⟨⟨i, rfl, rfl⟩, ⟨j, rfl, -⟩⟩
      exact ⟨rfl, rfl⟩
    · rintro ⟨rfl, rfl⟩
      exact ⟨⟨_, rfl, rfl⟩, ⟨_, rfl, rfl⟩⟩
  have h := Set.ncard_union_add_ncard_inter (Set.range fun i : n => (i, i))
    (Set.range fun j => (i₀, j)) (Set.toFinite _) (Set.toFinite _)
  rw [hinter, Set.ncard_singleton,
    Set.ncard_range_of_injective fun i i' h => congrArg Prod.fst h,
    Set.ncard_range_of_injective fun j j' h => congrArg Prod.snd h, Nat.card_eq_fintype_card] at h
  rw [diagRow]
  omega

example : (diagRow (0 : Fin 3)).ncard = 5 := by rw [ncard_diagRow, Fintype.card_fin]

/-- The pairs `(a, b)` of nonzero residues, as cells of the table. -/
abbrev nonzeroCell {p : ℕ} (ij : {x : ZMod p // x ≠ 0} × {x : ZMod p // x ≠ 0}) :
    ZMod p × ZMod p :=
  ((ij.1 : ZMod p), (ij.2 : ZMod p))

lemma nonzeroCell_injective {p : ℕ} : Function.Injective (nonzeroCell (p := p)) :=
  fun _ _ h => Prod.ext (Subtype.ext (congrArg Prod.fst h)) (Subtype.ext (congrArg Prod.snd h))

variable {p : ℕ} [Fact p.Prime]

/-- The cells `(a, a)` and `(1, b)` with `a, b ≠ 0`. -/
def productCells (p : ℕ) [Fact p.Prime] : Set (ZMod p × ZMod p) :=
  nonzeroCell '' diagRow ⟨1, one_ne_zero⟩

lemma mem_productCells {a b : ZMod p} :
    (a, b) ∈ productCells p ↔ a ≠ 0 ∧ b ≠ 0 ∧ (a = b ∨ a = 1) := by
  constructor
  · rintro ⟨⟨i, j⟩, ⟨k, hk⟩ | ⟨k, hk⟩, h⟩ <;> simp only [Prod.mk.injEq] at hk h <;>
      obtain ⟨rfl, rfl⟩ := hk <;> obtain ⟨rfl, rfl⟩ := h
    · exact ⟨k.2, k.2, Or.inl rfl⟩
    · exact ⟨one_ne_zero, k.2, Or.inr rfl⟩
  · rintro ⟨ha, hb, rfl | rfl⟩
    · exact ⟨(⟨a, ha⟩, ⟨a, ha⟩), Or.inl ⟨⟨a, ha⟩, rfl⟩, rfl⟩
    · exact ⟨(⟨1, ha⟩, ⟨b, hb⟩), Or.inr ⟨⟨b, hb⟩, rfl⟩, rfl⟩

/-- `productCells p` has `2p − 3` cells. -/
theorem ncard_productCells : (productCells p).ncard = 2 * p - 3 := by
  have hcard : Fintype.card {x : ZMod p // x ≠ 0} = p - 1 := by
    rw [Fintype.card_subtype_compl, Fintype.card_subtype_eq, ZMod.card]
  rw [productCells, Set.ncard_image_of_injective _ nonzeroCell_injective, ncard_diagRow, hcard]
  have := (Fact.out : p.Prime).two_le
  omega

example := @ncard_productCells 3 ⟨Nat.prime_three⟩

/-- **M recovers `[χ(a b)]` from the `2p − 3` cells `(a, a)` and `(1, b)`, `a, b ≠ 0`.** -/
theorem forall_isMinOn_taskKernel_mul_mulChar_productCells (χ : MulChar (ZMod p) ℂ) :
    ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ productCells p,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ productCells p,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ) := by
  refine (forall_isMinOn_taskKernel_mul_mulChar_iff_connected χ _).mpr
    ⟨?_, Equiv.refl _, fun i => ⟨(i, i), Or.inl ⟨i, rfl⟩, rfl⟩⟩
  have h : nonzeroCell ⁻¹' productCells p = diagRow ⟨(1 : ZMod p), one_ne_zero⟩ :=
    Set.preimage_image_eq _ nonzeroCell_injective
  show (patternGraph (nonzeroCell ⁻¹' productCells p)).Connected
  rw [h]
  exact diagRow_connected _

example := @forall_isMinOn_taskKernel_mul_mulChar_productCells 3 ⟨Nat.prime_three⟩ 1

/-- **No set of fewer than `2p − 3` cells recovers `[χ(a b)]`**: its cells with `a, b ≠ 0`
connect the `p − 1` nonzero rows with the `p − 1` nonzero columns. -/
theorem two_mul_sub_three_le_ncard_of_forall_isMinOn (χ : MulChar (ZMod p) ℂ)
    {Ω : Set (ZMod p × ZMod p)}
    (hΩ : ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ)) :
    2 * p - 3 ≤ Ω.ncard := by
  have : Nonempty {x : ZMod p // x ≠ 0} := ⟨⟨1, one_ne_zero⟩⟩
  have h1 := card_add_card_le_ncard_add_one_of_rigid ((rigid_iff_connected _).mpr
    ((forall_isMinOn_taskKernel_mul_mulChar_iff_connected χ Ω).mp hΩ).1)
  have h2 : (nonzeroCell ⁻¹' Ω).ncard ≤ Ω.ncard :=
    Set.ncard_le_ncard_of_injOn nonzeroCell (fun _ h => h) nonzeroCell_injective.injOn
  have hcard : Nat.card {x : ZMod p // x ≠ 0} = p - 1 := by
    rw [Nat.card_eq_fintype_card, Fintype.card_subtype_compl, Fintype.card_subtype_eq, ZMod.card]
  have h3 : ({ij : {x : ZMod p // x ≠ 0} × {x : ZMod p // x ≠ 0} |
      ((ij.1 : ZMod p), (ij.2 : ZMod p)) ∈ Ω}).ncard = (nonzeroCell ⁻¹' Ω).ncard := rfl
  rw [hcard, h3] at h1
  have := (Fact.out : p.Prime).two_le
  omega

example : 2 * 3 - 3 ≤ (productCells 3).ncard :=
  @two_mul_sub_three_le_ncard_of_forall_isMinOn 3 ⟨Nat.prime_three⟩ 1 _
    (@forall_isMinOn_taskKernel_mul_mulChar_productCells 3 ⟨Nat.prime_three⟩ 1)

/-- **`2p − 3` is the least number of cells from which M recovers `[χ(a b)]`.** -/
theorem isLeast_ncard_forall_isMinOn_taskKernel_mul_mulChar (χ : MulChar (ZMod p) ℂ) :
    IsLeast {c | ∃ Ω : Set (ZMod p × ZMod p), Ω.ncard = c ∧
      ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
          X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
        (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
          X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ)}
      (2 * p - 3) :=
  ⟨⟨productCells p, ncard_productCells, forall_isMinOn_taskKernel_mul_mulChar_productCells χ⟩,
    fun _ ⟨_, hc, hΩ⟩ => hc ▸ two_mul_sub_three_le_ncard_of_forall_isMinOn χ hΩ⟩

/-- **Some injective readout `χ` makes `2p − 3` the least number of cells from which M recovers
the table of `a b`**; through it the table gives back `a b` (`eq_of_taskKernel_eq`). -/
theorem exists_injective_mulChar_isLeast :
    ∃ χ : MulChar (ZMod p) ℂ, Function.Injective χ ∧
      IsLeast {c | ∃ Ω : Set (ZMod p × ZMod p), Ω.ncard = c ∧
        ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
            X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
          (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
            X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ)}
        (2 * p - 3) :=
  exists_injective_mulChar.elim fun χ hχ =>
    ⟨χ, hχ, isLeast_ncard_forall_isMinOn_taskKernel_mul_mulChar χ⟩

end Transformer.Grokking.FlatSpectrum

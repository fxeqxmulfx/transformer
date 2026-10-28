module

public import Transformer.Grokking.FlatSpectrum.CharacterRecovery
public import Transformer.Grokking.FlatSpectrum.PatternGraph
public import Transformer.Grokking.FlatSpectrum.PatternRecovery

/-!
# Theorems E and G with graphs

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorems E and G and
the corollaries for characters, with rigidity read as connectivity (`rigid_iff_connected`).

* For `|a_i| = |b_j| = 1`, M recovers `a bᵀ` from `Ω` exactly when the cells of `Ω` connect the
  rows with the columns and contain the graph of a permutation
  (`forall_isMinOn_vecMulVec_iff_connected`); for `ψ(a + b)`,
  `forall_isMinOn_addKernel_iff_connected`.
* Rigidity on the supports of `a ≠ 0` and `b ≠ 0` is connectivity of the cells inside them
  (`rigid_support_iff_connected`).
* M recovers `[χ(a b)]` exactly when the cells with `a, b ≠ 0` connect the nonzero rows and
  columns and contain the graph of a permutation of them
  (`forall_isMinOn_taskKernel_mul_mulChar_iff_connected`). For `[χ(a + a b)]` the columns are
  those with `b ≠ −1` (`forall_isMinOn_taskKernel_add_mul_mulChar_iff_connected`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix SimpleGraph

variable {m n : Type*}

/-- **Theorem E with graphs** (`review/flat_spectrum.md`, Theorem E, rigidity read as
connectivity): for `|a i| = |b j| = 1`, M recovers `a bᵀ` from `Ω` exactly when the cells of `Ω`
connect the rows with the columns and contain the graph of a permutation. -/
theorem forall_isMinOn_vecMulVec_iff_connected [Fintype n] [DecidableEq n] [Nonempty n]
    {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1) (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)} :
    (∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec a b ij.1 ij.2} X ↔
        X = vecMulVec a b)) ↔
      (patternGraph Ω).Connected ∧ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω :=
  (forall_isMinOn_vecMulVec_iff ha hb).trans (and_congr_left' (rigid_iff_connected Ω))

example := (forall_isMinOn_vecMulVec_iff_connected (n := Fin 2) (a := fun _ => 1)
  (b := fun _ => 1) (by simp) (by simp) (Ω := {ij | ij ≠ (1, 0)})).mpr
  ⟨(rigid_iff_connected _).mp rigid_ne_one_zero, 1, Fin.forall_fin_two.mpr ⟨by simp, by simp⟩⟩

/-- **`ψ(a + b)` with graphs** (`review/flat_spectrum.md`, Theorem E): M recovers `[ψ(a + b)]`
from `Ω` exactly when the cells of `Ω` connect the rows with the columns and contain the graph
of a permutation. -/
theorem forall_isMinOn_addKernel_iff_connected {p : ℕ} [NeZero p] {Ω : Set (ZMod p × ZMod p)} :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
        X = addKernel p)) ↔
      (patternGraph Ω).Connected ∧ ∃ σ : Equiv.Perm (ZMod p), ∀ i, (i, σ i) ∈ Ω :=
  forall_isMinOn_addKernel_iff.trans (and_congr_left' (rigid_iff_connected Ω))

example := (forall_isMinOn_addKernel_iff_connected (p := 3) (Ω := Set.univ)).mpr
  ⟨(rigid_iff_connected _).mp fun _ _ h i j => h (i, j) (Set.mem_univ _), 1,
    fun _ => Set.mem_univ _⟩

/-- **Rigidity on the supports is connectivity**: if `a ≠ 0` and `b ≠ 0`, `Ω` is rigid on their
supports exactly when the cells of `Ω` inside the supports connect them. -/
theorem rigid_support_iff_connected {a : m → ℂ} {b : n → ℂ} (ha : ∃ i, a i ≠ 0)
    (hb : ∃ j, b j ≠ 0) (Ω : Set (m × n)) :
    (∀ (r : m → ℂ) (s : n → ℂ), (∀ ij ∈ Ω, a ij.1 ≠ 0 → b ij.2 ≠ 0 → r ij.1 + s ij.2 = 0) →
      ∀ i j, a i ≠ 0 → b j ≠ 0 → r i + s j = 0) ↔
      (patternGraph {ij : {i // a i ≠ 0} × {j // b j ≠ 0} |
        ((ij.1 : m), (ij.2 : n)) ∈ Ω}).Connected := by
  have : Nonempty {i // a i ≠ 0} := ha.elim fun i hi => ⟨⟨i, hi⟩⟩
  have : Nonempty {j // b j ≠ 0} := hb.elim fun j hj => ⟨⟨j, hj⟩⟩
  exact (rigid_support_iff_subtype a b Ω).trans (rigid_iff_connected' _)

example := (rigid_support_iff_connected (a := ![(1 : ℂ), 0]) (b := ![(1 : ℂ)])
  ⟨0, by simp⟩ ⟨0, by simp⟩ {(0, 0)})

variable {p : ℕ} [Fact p.Prime]

/-- **`χ(a b)` from Theorem G, with graphs** (`review/flat_spectrum.md`): M recovers `[χ(a b)]`
from `Ω` exactly when the cells of `Ω` with `a, b ≠ 0` connect the nonzero rows and columns and
contain the graph of a permutation of them. -/
theorem forall_isMinOn_taskKernel_mul_mulChar_iff_connected (χ : MulChar (ZMod p) ℂ)
    (Ω : Set (ZMod p × ZMod p)) :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) χ ij.1 ij.2} X ↔ X = taskKernel (· * ·) χ)) ↔
      (patternGraph {ij : {x : ZMod p // x ≠ 0} × {x : ZMod p // x ≠ 0} |
        ((ij.1 : ZMod p), (ij.2 : ZMod p)) ∈ Ω}).Connected ∧
        ∃ σ : Equiv.Perm {x : ZMod p // x ≠ 0}, ∀ i : {x : ZMod p // x ≠ 0},
          ((i : ZMod p), (σ i : ZMod p)) ∈ Ω := by
  have : Nonempty {x : ZMod p // x ≠ 0} := ⟨⟨1, one_ne_zero⟩⟩
  exact (forall_isMinOn_taskKernel_mul_mulChar_iff χ Ω).trans
    (and_congr_left' (rigid_iff_connected' _))

/-- For `p = 3` the nonzero block recovers `[χ(a b)]` for the principal character. -/
example :=
  have : Nonempty {x : ZMod 3 // x ≠ 0} := ⟨⟨1, by decide⟩⟩
  (forall_isMinOn_taskKernel_mul_mulChar_iff_connected (p := 3) 1
    {ij | ij.1 ≠ 0 ∧ ij.2 ≠ 0}).mpr
    ⟨(rigid_iff_connected' fun (i j : {x : ZMod 3 // x ≠ 0}) =>
        ((i : ZMod 3), (j : ZMod 3)) ∈ {ij : ZMod 3 × ZMod 3 | ij.1 ≠ 0 ∧ ij.2 ≠ 0}).mp
      fun _ _ h i j => h i j ⟨i.2, j.2⟩, Equiv.refl _, fun i => ⟨i.2, i.2⟩⟩

/-- **`χ(a + a b)` from Theorem G, with graphs** (`review/flat_spectrum.md`): M recovers
`[χ(a + a b)] = χ ⊗ χ(1 + ·)` from `Ω` exactly when the cells of `Ω` with `a ≠ 0` and `b ≠ −1`
connect these rows and columns and contain the graph of a bijection between them. -/
theorem forall_isMinOn_taskKernel_add_mul_mulChar_iff_connected (χ : MulChar (ZMod p) ℂ)
    (Ω : Set (ZMod p × ZMod p)) :
    (∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (fun a b => a + a * b) χ ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (fun a b => a + a * b) χ ij.1 ij.2} X ↔
          X = taskKernel (fun a b => a + a * b) χ)) ↔
      (patternGraph {ij : {x : ZMod p // x ≠ 0} × {x : ZMod p // x ≠ -1} |
        ((ij.1 : ZMod p), (ij.2 : ZMod p)) ∈ Ω}).Connected ∧
        ∃ σ : {x : ZMod p // x ≠ 0} ≃ {x : ZMod p // x ≠ -1}, ∀ i : {x : ZMod p // x ≠ 0},
          ((i : ZMod p), (σ i : ZMod p)) ∈ Ω := by
  have : Nonempty {x : ZMod p // x ≠ 0} := ⟨⟨1, one_ne_zero⟩⟩
  have : Nonempty {x : ZMod p // x ≠ -1} :=
    ⟨⟨0, fun h => one_ne_zero (neg_eq_zero.mp h.symm)⟩⟩
  exact (forall_isMinOn_taskKernel_add_mul_mulChar_iff χ Ω).trans
    (and_congr_left' (rigid_iff_connected' _))

/-- For `p = 3` the cells with `a ≠ 0` and `b ≠ −1` recover `[χ(a + a b)]` for the principal
character. -/
example :=
  have : Nonempty {x : ZMod 3 // x ≠ 0} := ⟨⟨1, by decide⟩⟩
  have : Nonempty {x : ZMod 3 // x ≠ -1} := ⟨⟨0, by decide⟩⟩
  let σ : {x : ZMod 3 // x ≠ 0} ≃ {x : ZMod 3 // x ≠ -1} :=
    (Equiv.subRight 1).subtypeEquiv fun x => by rw [Equiv.subRight_apply, Ne, Ne, sub_eq_neg_self]
  (forall_isMinOn_taskKernel_add_mul_mulChar_iff_connected (p := 3) 1
    {ij | ij.1 ≠ 0 ∧ ij.2 ≠ -1}).mpr
    ⟨(rigid_iff_connected' fun (i : {x : ZMod 3 // x ≠ 0}) (j : {x : ZMod 3 // x ≠ -1}) =>
        ((i : ZMod 3), (j : ZMod 3)) ∈ {ij : ZMod 3 × ZMod 3 | ij.1 ≠ 0 ∧ ij.2 ≠ -1}).mp
      fun _ _ h i j => h i j ⟨i.2, j.2⟩, σ, fun i => ⟨i.2, (σ i).2⟩⟩

end Transformer.Grokking.FlatSpectrum

/-
# `TL[◁#]^pos` reduces to `TL[◁#]`: the translation

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, the proof of `lem:tlclpos_reduction`.

The translation `T_ρ⟦·⟧` takes a `TL[◁#]^pos` formula `φ` over `Σ ∪ {e}` in
`Y`-normal form to a `TL[◁#]` formula over `Σ` that, at position `i` of `w`,
simulates `φ` at position `r·i + ρ` of `f(w) = e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}`.
The `Y`-atoms go through `FormP.atomTr` of
`Transformer.CRASP.PositionalReductionAtom`, whose docstring also has the block
size; the rest follows the paper except in two places.

* **Counts.**  The paper translates `◁#[φ]` to
  `C_φ + Σ_{ρ' ∈ [r]} ◁#strict[T_ρ'⟦φ⟧] + Σ_{ρ' ∈ [ρ]} (T_ρ'⟦φ⟧ ? 1 : 0)`,
  which is not a term of `TL[◁#]`: the strict count and the indicator are not
  in its syntax.  Here the count is `C_φ + Σ_{ρ' ∈ [r]} ◁#[T_ρ'⟦φ⟧]` less the
  indicators of the `ρ' ∈ (ρ, r]`, and a comparison moves those indicators to
  its other side, where `Form.ltSum` removes them at no cost in depth.  A term
  is translated to `Summands`: its constant, its counts and the formulas whose
  indicators it subtracts.
* **The first block.**  The paper proves its invariant for `i ∈ [|w|]` only,
  so its final step `f(w) ⊨ φ ⟺ w ⊨ T_r⟦φ⟧` is not established for `w = ε`,
  and at `i = 0` its count would take the first block in twice.  Here the
  invariant also covers `i = 0` with `ρ = r`, the end of the first block.
-/

import Transformer.CRASP.Indicator
import Transformer.CRASP.PositionalReductionAtom

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A translated term `const + Σ terms − Σ_{ψ ∈ negs} (ψ ? 1 : 0)`.  The
subtracted indicators are added on the other side of a comparison. -/
structure Summands (σ : Type u) where
  /-- The constant. -/
  const : ℕ
  /-- The terms added. -/
  terms : List (Term σ)
  /-- The formulas whose indicators are subtracted. -/
  negs : List (Form σ)


variable [DecidableEq σ]

mutual

/-- **The translation `T_ρ⟦φ⟧`** of `lem:tlclpos_reduction` (Appendix E), with
the count and comparison rules described in the module docstring. -/
def FormP.tr (r ρ : ℕ) : FormP (Option σ) → Form σ
  | .sym a => FormP.atomTr r ρ 0 (.sym a)
  | .mod m s => FormP.atomTr r ρ 0 (.mod m s)
  | .prev φ => FormP.atomTr r ρ 1 φ
  | .lt t₁ t₂ =>
      Form.ltSum (t₁.tr r ρ).const (t₁.tr r ρ).terms (t₂.tr r ρ).negs
        (t₂.tr r ρ).const (t₂.tr r ρ).terms (t₁.tr r ρ).negs
  | .neg φ => .neg (φ.tr r ρ)
  | .and φ₁ φ₂ => .and (φ₁.tr r ρ) (φ₂.tr r ρ)

/-- `T_ρ⟦t⟧` on terms.  `◁#[φ]` becomes the count `C_φ` of the first block,
which every formula sees as `e^r`, plus `◁#[T_ρ'⟦φ⟧]` for every `ρ' ∈ [r]`,
less the indicators of `T_ρ'⟦φ⟧` for the `ρ' ∈ (ρ, r]` of the current block
that `r·i + ρ` has not reached. -/
def TermP.tr (r ρ : ℕ) : TermP (Option σ) → Summands σ
  | .countL φ =>
      ⟨((List.range' 1 r).filter fun j => φ.sat (List.replicate r none) j).length,
        (List.range' 1 r).map fun ρ' => .countL (φ.tr r ρ'),
        (List.range' (ρ + 1) (r - ρ)).map fun ρ' => φ.tr r ρ'⟩
  | .add t₁ t₂ =>
      ⟨(t₁.tr r ρ).const + (t₂.tr r ρ).const, (t₁.tr r ρ).terms ++ (t₂.tr r ρ).terms,
        (t₁.tr r ρ).negs ++ (t₂.tr r ρ).negs⟩
  | .one => ⟨1, [], []⟩

end

mutual

/-- **`T_ρ⟦φ⟧` lies in `TL[◁#]` at the depth of `φ`** (Appendix E, proof of
`lem:tlclpos_reduction`: "by construction, `T_r⟦φ⟧` has the same depth as
`φ`"). -/
theorem FormP.tr_mem_TLCl (r ρ : ℕ) :
    ∀ (φ : FormP (Option σ)) {d : ℕ}, φ.depth ≤ d → φ.tr r ρ ∈ TLCl σ d
  | .sym a, d, _ => by rw [FormP.tr]; exact FormP.atomTr_mem_TLCl r ρ d 0 _
  | .mod m s, d, _ => by rw [FormP.tr]; exact FormP.atomTr_mem_TLCl r ρ d 0 _
  | .prev φ, d, _ => by rw [FormP.tr]; exact FormP.atomTr_mem_TLCl r ρ d 1 φ
  | .lt t₁ t₂, d, h => by
      rw [FormP.depth, max_le_iff] at h
      rw [FormP.tr]
      exact Form.ltSum_mem_TLCl _ _ (t₁.tr_mem r ρ h.1).1 (t₂.tr_mem r ρ h.2).2
        (t₂.tr_mem r ρ h.2).1 (t₁.tr_mem r ρ h.1).2
  | .neg φ, d, h => by
      rw [FormP.depth] at h
      obtain ⟨hp, hn, hd⟩ := φ.tr_mem_TLCl r ρ h
      rw [FormP.tr]
      exact ⟨hp, hn, hd⟩
  | .and φ₁ φ₂, d, h => by
      rw [FormP.depth, max_le_iff] at h
      obtain ⟨h₁p, h₁n, h₁d⟩ := φ₁.tr_mem_TLCl r ρ h.1
      obtain ⟨h₂p, h₂n, h₂d⟩ := φ₂.tr_mem_TLCl r ρ h.2
      rw [FormP.tr]
      refine ⟨?_, ?_, max_le h₁d h₂d⟩
      · rw [Form.past, h₁p, h₂p]; rfl
      · rw [Form.pnpFree, h₁n, h₂n]; rfl

/-- The parts of `T_ρ⟦t⟧` lie in `TL[◁#]` at the depth of `t`. -/
theorem TermP.tr_mem (r ρ : ℕ) :
    ∀ (t : TermP (Option σ)) {d : ℕ}, t.depth ≤ d →
      (∀ s ∈ (t.tr r ρ).terms, s.past = true ∧ s.pnpFree = true ∧ s.depth ≤ d) ∧
        ∀ ψ ∈ (t.tr r ρ).negs, ψ ∈ TLCl σ d
  | .countL φ, d, h => by
      rw [TermP.depth] at h
      rw [TermP.tr]
      refine ⟨fun s hs => ?_, fun ψ hψ => ?_⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hs
        obtain ⟨hp, hn, hd⟩ := φ.tr_mem_TLCl r ρ' le_rfl
        exact ⟨hp, hn, by rw [Term.depth]; omega⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hψ
        exact φ.tr_mem_TLCl r ρ' (Nat.le_of_succ_le h)
  | .add t₁ t₂, d, h => by
      rw [TermP.depth, max_le_iff] at h
      rw [TermP.tr]
      refine ⟨fun s hs => ?_, fun ψ hψ => ?_⟩
      · rcases List.mem_append.mp hs with hs | hs
        · exact (t₁.tr_mem r ρ h.1).1 s hs
        · exact (t₂.tr_mem r ρ h.2).1 s hs
      · rcases List.mem_append.mp hψ with hψ | hψ
        · exact (t₁.tr_mem r ρ h.1).2 ψ hψ
        · exact (t₂.tr_mem r ρ h.2).2 ψ hψ
  | .one, _, _ => by
      rw [TermP.tr]
      exact ⟨fun s hs => absurd hs List.not_mem_nil, fun ψ hψ => absurd hψ List.not_mem_nil⟩

end

/-- The hypotheses of `FormP.tr_mem_TLCl` and `TermP.tr_mem` are satisfiable:
`Q_e` and `1` have depth `0`. -/
example (d : ℕ) :
    (FormP.sym none : FormP (Option σ)).depth ≤ d ∧ (TermP.one : TermP (Option σ)).depth ≤ d :=
  ⟨Nat.zero_le d, Nat.zero_le d⟩

end CRASP
end Transformer

module

public import Transformer.CRASP.ModBlocks

/-!
# Blocks of one period turn `TL[◁#, MOD]` into `TL[◁#]`: the depth

The translations of `Transformer.CRASP.ModBlocks` lie in `TL[◁#]` one level deeper than the
formula they translate (`FormP.blockTr_mem`, `FormP.lastTr_mem`). The extra level is the count of
`MOD_0^r`, which finds the block of one position; `◁#[φ]` adds a level on both sides, as in
Appendix F of arXiv:2506.16055v3 (`FormP.tr_mem_TLCl`). `Y` needs no level: it is translated by
`⊥` and kept out by the hypotheses of `Transformer.CRASP.ModBlocksCorrect`.
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

mutual

/-- `T_ρ⟦φ⟧` lies in `TL[◁#]` one level deeper than `φ`. -/
theorem FormP.blockTr_mem (g : Bool → List σ) (M ρ : ℕ) :
    ∀ (φ : FormP σ) {d : ℕ}, φ.depth ≤ d → φ.blockTr g M ρ ∈ TLCl Bool (d + 1)
  | .sym _, _, _ => by
      rw [FormP.blockTr]
      exact Form.ite_mem_TLCl ⟨rfl, rfl, Nat.zero_le _⟩ (Form.ofBool_mem_TLCl _ _)
        (Form.ofBool_mem_TLCl _ _)
  | .mod m r, _, _ => by
      rw [FormP.blockTr]
      exact Form.mem_mono (modTr_mem M ρ m r) (by omega)
  | .prev _, _, _ => by rw [FormP.blockTr]; exact Form.ofBool_mem_TLCl _ _
  | .lt t₁ t₂, _, h => by
      rw [FormP.depth, max_le_iff] at h
      rw [FormP.blockTr]
      exact (t₁.blockTr_mem g M ρ h.1).lt (t₂.blockTr_mem g M ρ h.2)
  | .neg φ, _, h => by
      rw [FormP.depth] at h
      obtain ⟨hp, hn, hd⟩ := φ.blockTr_mem g M ρ h
      rw [FormP.blockTr]
      exact ⟨hp, hn, hd⟩
  | .and φ₁ φ₂, _, h => by
      rw [FormP.depth, max_le_iff] at h
      rw [FormP.blockTr]
      exact Form.and_mem (φ₁.blockTr_mem g M ρ h.1) (φ₂.blockTr_mem g M ρ h.2)

/-- The parts of `T_ρ⟦t⟧` lie in `TL[◁#]` one level deeper than `t`. -/
theorem TermP.blockTr_mem (g : Bool → List σ) (M ρ : ℕ) :
    ∀ (t : TermP σ) {d : ℕ}, t.depth ≤ d → (t.blockTr g M ρ).InTLCl (d + 1)
  | .countL φ, _, h => by
      rw [TermP.depth] at h
      rw [TermP.blockTr]
      refine ⟨fun u hu => ?_, fun ψ hψ => ?_, fun ψ hψ => absurd hψ List.not_mem_nil⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hu
        obtain ⟨hp, hn, hd⟩ := φ.blockTr_mem g M ρ' le_rfl
        exact ⟨hp, hn, by rw [Term.depth]; omega⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hψ
        exact φ.blockTr_mem g M ρ' (by omega)
  | .add t₁ t₂, _, h => by
      rw [TermP.depth, max_le_iff] at h
      rw [TermP.blockTr]
      exact (t₁.blockTr_mem g M ρ h.1).add (t₂.blockTr_mem g M ρ h.2)
  | .one, _, _ => by
      rw [TermP.blockTr]
      exact ⟨fun _ h => absurd h List.not_mem_nil, fun _ h => absurd h List.not_mem_nil,
        fun _ h => absurd h List.not_mem_nil⟩

end

mutual

/-- `φ` at the position `ρ` of the last block, `s` in its place, lies in `TL[◁#]` one level
deeper than `φ`. -/
theorem FormP.lastTr_mem (g : Bool → List σ) (M : ℕ) (s : List σ) (ρ : ℕ) :
    ∀ (φ : FormP σ) {d : ℕ}, φ.depth ≤ d → φ.lastTr g M s ρ ∈ TLCl Bool (d + 1)
  | .sym _, _, _ => by rw [FormP.lastTr]; exact Form.ofBool_mem_TLCl _ _
  | .mod m r, _, _ => by
      rw [FormP.lastTr]
      exact Form.mem_mono (modTr_mem M ρ m r) (by omega)
  | .prev _, _, _ => by rw [FormP.lastTr]; exact Form.ofBool_mem_TLCl _ _
  | .lt t₁ t₂, _, h => by
      rw [FormP.depth, max_le_iff] at h
      rw [FormP.lastTr]
      exact (t₁.lastTr_mem g M s ρ h.1).lt (t₂.lastTr_mem g M s ρ h.2)
  | .neg φ, _, h => by
      rw [FormP.depth] at h
      obtain ⟨hp, hn, hd⟩ := φ.lastTr_mem g M s ρ h
      rw [FormP.lastTr]
      exact ⟨hp, hn, hd⟩
  | .and φ₁ φ₂, _, h => by
      rw [FormP.depth, max_le_iff] at h
      rw [FormP.lastTr]
      exact Form.and_mem (φ₁.lastTr_mem g M s ρ h.1) (φ₂.lastTr_mem g M s ρ h.2)

/-- The parts of a term at the last block lie in `TL[◁#]` one level deeper than the term. -/
theorem TermP.lastTr_mem (g : Bool → List σ) (M : ℕ) (s : List σ) (ρ : ℕ) :
    ∀ (t : TermP σ) {d : ℕ}, t.depth ≤ d → (t.lastTr g M s ρ).InTLCl (d + 1)
  | .countL φ, _, h => by
      rw [TermP.depth] at h
      rw [TermP.lastTr]
      refine ⟨fun u hu => ?_, fun ψ hψ => ?_, fun ψ hψ => ?_⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hu
        obtain ⟨hp, hn, hd⟩ := φ.blockTr_mem g M ρ' le_rfl
        exact ⟨hp, hn, by rw [Term.depth]; omega⟩
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hψ
        exact φ.blockTr_mem g M ρ' (by omega)
      · obtain ⟨ρ', -, rfl⟩ := List.mem_map.mp hψ
        exact φ.lastTr_mem g M s ρ' (by omega)
  | .add t₁ t₂, _, h => by
      rw [TermP.depth, max_le_iff] at h
      rw [TermP.lastTr]
      exact (t₁.lastTr_mem g M s ρ h.1).add (t₂.lastTr_mem g M s ρ h.2)
  | .one, _, _ => by
      rw [TermP.lastTr]
      exact ⟨fun _ h => absurd h List.not_mem_nil, fun _ h => absurd h List.not_mem_nil,
        fun _ h => absurd h List.not_mem_nil⟩

end

/-- The hypotheses of the four theorems above are satisfiable: `MOD_0^3` and `◁#[Q_⊤]` have
depths `0` and `1`. -/
example : (FormP.mod 0 3 : FormP Bool).blockTr (fun c => [c]) 1 1 ∈ TLCl Bool 1 ∧
    ((TermP.countL (.sym true) : TermP Bool).blockTr (fun c => [c]) 1 1).InTLCl 2 ∧
    (FormP.mod 0 3 : FormP Bool).lastTr (fun c => [c]) 1 [true] 1 ∈ TLCl Bool 1 ∧
    ((TermP.countL (.sym true) : TermP Bool).lastTr (fun c => [c]) 1 [true] 1).InTLCl 2 :=
  ⟨FormP.blockTr_mem _ 1 1 _ (d := 0) le_rfl, TermP.blockTr_mem _ 1 1 _ (d := 1) le_rfl,
    FormP.lastTr_mem _ 1 _ 1 _ (d := 0) le_rfl, TermP.lastTr_mem _ 1 _ 1 _ (d := 1) le_rfl⟩

end CRASP
end Transformer

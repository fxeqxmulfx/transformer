module

public import Transformer.CRASP.FormulaBounds
public import Transformer.CRASP.Indicator
public import Transformer.CRASP.PositionalReductionAtom

/-!
# Blocks of one period turn `TL[◁#, MOD]` into `TL[◁#]`: the translation

In no paper. It is the reduction of `lem:tlclpos_reduction` of arXiv:2506.16055v3, Appendix F
(`Transformer.CRASP.PositionalReduction`), without `Y` and with a block of any content in place
of `wᵢ e^{r−1}`.

Let `M` be a multiple of every positive modulus of a formula `φ` of `TL[◁#, MOD]`
(`FormP.modLcm`), and give each bit `c` a block `g c` of `M` symbols. In `g y₁ ⋯ g yₙ` the
position `(i − 1) M + ρ`, `1 ≤ ρ ≤ M`, is the `ρ`-th of block `i`; there `MOD_m^r` is a constant
when `m ∣ M`, and `MOD_0^r`, which tests the position itself, tests the block against one number
(`modTr`). So `φ` at the `ρ`-th position of block `i` is a formula `T_ρ⟦φ⟧` of `TL[◁#]` over the
bits at `i` (`FormP.blockTr`). A count is split as in Appendix F: the full blocks up to `i`,
less the positions of block `i` after `ρ` (`TermP.blockTr`); the comparisons that result are
those of `Transformer.CRASP.Indicator`. The last block may be replaced by any string `s`
(`FormP.lastTr`): a count there is the full blocks before it and the positions of `s` up to `ρ`.

A translation is one deeper than `φ` at most, for `MOD_0^r` (`FormP.blockTr_mem`,
`FormP.lastTr_mem`). `Transformer.CRASP.ModBlocksCorrect` proves that it simulates `φ`.
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- The least common multiple of the positive moduli of `φ`, and `1` without any. -/
def FormP.modLcm : FormP σ → ℕ
  | .sym _ => 1
  | .mod m _ => if m = 0 then 1 else m
  | .prev φ => φ.modLcm
  | .lt t₁ t₂ => Nat.lcm t₁.modLcm t₂.modLcm
  | .neg φ => φ.modLcm
  | .and φ₁ φ₂ => Nat.lcm φ₁.modLcm φ₂.modLcm

/-- The least common multiple of the positive moduli of a term. -/
def TermP.modLcm : TermP σ → ℕ
  | .countL φ => φ.modLcm
  | .add t₁ t₂ => Nat.lcm t₁.modLcm t₂.modLcm
  | .one => 1

end

mutual

theorem FormP.modLcm_pos : ∀ φ : FormP σ, 0 < φ.modLcm
  | .sym _ => Nat.one_pos
  | .mod m _ => by rw [FormP.modLcm]; split <;> omega
  | .prev φ => φ.modLcm_pos
  | .lt t₁ t₂ => Nat.lcm_pos t₁.modLcm_pos t₂.modLcm_pos
  | .neg φ => φ.modLcm_pos
  | .and φ₁ φ₂ => Nat.lcm_pos φ₁.modLcm_pos φ₂.modLcm_pos

theorem TermP.modLcm_pos : ∀ t : TermP σ, 0 < t.modLcm
  | .countL φ => φ.modLcm_pos
  | .add t₁ t₂ => Nat.lcm_pos t₁.modLcm_pos t₂.modLcm_pos
  | .one => Nat.one_pos

end

/-- `MOD_m^r` at the position `ρ` of block `i`, that is at `(i − 1) M + ρ`: for `m ∣ M` the
constant `ρ ≡ r (mod m)`; for `m = 0` the test `#[⊤] = (r − ρ)/M + 1` of the block when
`M ∣ r − ρ`, and `⊥` otherwise. -/
def modTr (M ρ m r : ℕ) : Form Bool :=
  if m = 0 then
    if ρ ≤ r ∧ (r - ρ) % M = 0 then
      Form.eq (.countL (Form.ofBool true)) (Term.ofPos ((r - ρ) / M))
    else Form.ofBool false
  else Form.ofBool (decide (ρ % m = r % m))

theorem modTr_mem (M ρ m r : ℕ) : modTr M ρ m r ∈ TLCl Bool 1 := by
  unfold modTr
  split_ifs
  · refine ⟨?_, ?_, ?_⟩ <;>
      simp [Form.eq, Form.past, Form.pnpFree, Form.depth, Term.past, Term.pnpFree, Term.depth,
        Form.ofBool, Term.pnpFree_ofPos]
  · exact Form.ofBool_mem_TLCl _ 1
  · exact Form.ofBool_mem_TLCl _ 1

/-- A translated term `const + Σ terms − #negs + #poss`, where `#L` is the number of formulas of
`L` that hold. -/
structure BlockSum where
  /-- The constant. -/
  const : ℕ
  /-- The counts added. -/
  terms : List (Term Bool)
  /-- The formulas whose indicators are subtracted. -/
  negs : List (Form Bool)
  /-- The formulas whose indicators are added. -/
  poss : List (Form Bool)

namespace BlockSum

/-- The sum of two translated terms. -/
def add (b₁ b₂ : BlockSum) : BlockSum :=
  ⟨b₁.const + b₂.const, b₁.terms ++ b₂.terms, b₁.negs ++ b₂.negs, b₁.poss ++ b₂.poss⟩

/-- `b₁ < b₂`, with the subtracted indicators of each side added to the other. -/
def lt (b₁ b₂ : BlockSum) : Form Bool :=
  Form.ltSum b₁.const b₁.terms (b₁.poss ++ b₂.negs) b₂.const b₂.terms (b₂.poss ++ b₁.negs)

/-- The parts of a translated term lie in `TL[◁#]_d`. -/
def InTLCl (b : BlockSum) (d : ℕ) : Prop :=
  (∀ t ∈ b.terms, t.past = true ∧ t.pnpFree = true ∧ t.depth ≤ d) ∧
    (∀ ψ ∈ b.negs, ψ ∈ TLCl Bool d) ∧ ∀ ψ ∈ b.poss, ψ ∈ TLCl Bool d

theorem InTLCl.add {b₁ b₂ : BlockSum} {d : ℕ} (h₁ : b₁.InTLCl d) (h₂ : b₂.InTLCl d) :
    (b₁.add b₂).InTLCl d :=
  ⟨fun t ht => (List.mem_append.mp ht).elim (h₁.1 t) (h₂.1 t),
    fun ψ hψ => (List.mem_append.mp hψ).elim (h₁.2.1 ψ) (h₂.2.1 ψ),
    fun ψ hψ => (List.mem_append.mp hψ).elim (h₁.2.2 ψ) (h₂.2.2 ψ)⟩

theorem InTLCl.lt {b₁ b₂ : BlockSum} {d : ℕ} (h₁ : b₁.InTLCl d) (h₂ : b₂.InTLCl d) :
    b₁.lt b₂ ∈ TLCl Bool d :=
  Form.ltSum_mem_TLCl _ _ h₁.1 (fun ψ hψ => (List.mem_append.mp hψ).elim (h₁.2.2 ψ) (h₂.2.1 ψ))
    h₂.1 (fun ψ hψ => (List.mem_append.mp hψ).elim (h₂.2.2 ψ) (h₁.2.1 ψ))

/-- The hypotheses of `InTLCl.add` and `InTLCl.lt` are satisfiable: the constant `1`. -/
example : (BlockSum.mk 1 [] [] []).InTLCl 0 :=
  ⟨fun _ h => absurd h List.not_mem_nil, fun _ h => absurd h List.not_mem_nil,
    fun _ h => absurd h List.not_mem_nil⟩

end BlockSum

variable [DecidableEq σ]

mutual

/-- **`T_ρ⟦φ⟧`**: `φ` at the position `ρ` of the current block, as a formula over the bits.
`Y` has no translation; the simulation assumes `φ` free of it. -/
def FormP.blockTr (g : Bool → List σ) (M ρ : ℕ) : FormP σ → Form Bool
  | .sym a => Form.ite (.sym true) (Form.ofBool (decide ((g true)[ρ - 1]? = some a)))
      (Form.ofBool (decide ((g false)[ρ - 1]? = some a)))
  | .mod m r => modTr M ρ m r
  | .prev _ => Form.ofBool false
  | .lt t₁ t₂ => (t₁.blockTr g M ρ).lt (t₂.blockTr g M ρ)
  | .neg φ => .neg (φ.blockTr g M ρ)
  | .and φ₁ φ₂ => .and (φ₁.blockTr g M ρ) (φ₂.blockTr g M ρ)

/-- `T_ρ⟦t⟧`: `◁#[φ]` counts every `T_ρ'⟦φ⟧` up to the current block, less those of the
`ρ' ∈ (ρ, M]` of the current block. -/
def TermP.blockTr (g : Bool → List σ) (M ρ : ℕ) : TermP σ → BlockSum
  | .countL φ => ⟨0, (List.range' 1 M).map fun ρ' => .countL (φ.blockTr g M ρ'),
      (List.range' (ρ + 1) (M - ρ)).map fun ρ' => φ.blockTr g M ρ', []⟩
  | .add t₁ t₂ => (t₁.blockTr g M ρ).add (t₂.blockTr g M ρ)
  | .one => ⟨1, [], [], []⟩

end

mutual

/-- `φ` at the position `ρ` of the last block, when `s` stands in place of that block. -/
def FormP.lastTr (g : Bool → List σ) (M : ℕ) (s : List σ) (ρ : ℕ) : FormP σ → Form Bool
  | .sym a => Form.ofBool (decide (s[ρ - 1]? = some a))
  | .mod m r => modTr M ρ m r
  | .prev _ => Form.ofBool false
  | .lt t₁ t₂ => (t₁.lastTr g M s ρ).lt (t₂.lastTr g M s ρ)
  | .neg φ => .neg (φ.lastTr g M s ρ)
  | .and φ₁ φ₂ => .and (φ₁.lastTr g M s ρ) (φ₂.lastTr g M s ρ)

/-- On terms: `◁#[φ]` counts the blocks before the last, as every `T_ρ'⟦φ⟧` up to the last block
less its own indicators there, and adds the positions of `s` up to `ρ`. -/
def TermP.lastTr (g : Bool → List σ) (M : ℕ) (s : List σ) (ρ : ℕ) : TermP σ → BlockSum
  | .countL φ => ⟨0, (List.range' 1 M).map fun ρ' => .countL (φ.blockTr g M ρ'),
      (List.range' 1 M).map fun ρ' => φ.blockTr g M ρ',
      (List.range' 1 ρ).map fun ρ' => φ.lastTr g M s ρ'⟩
  | .add t₁ t₂ => (t₁.lastTr g M s ρ).add (t₂.lastTr g M s ρ)
  | .one => ⟨1, [], [], []⟩

end

end CRASP
end Transformer

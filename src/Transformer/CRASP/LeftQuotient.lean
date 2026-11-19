module

public import Transformer.CRASP.Positional

/-!
# A fixed prefix moves into a formula of `TL[◁#]^pos`

In no paper. For a string `p` and a formula `φ` of `TL[◁#]^pos` (arXiv:2506.16055v3,
Appendix F, `app:tlclpos`), the formula `p⁻¹φ` (`FormP.leftQuot`) is meant to hold at the
position `i ≥ 1` of `w` iff `φ` holds at the position `|p| + i` of `p ++ w`, and so to accept a
nonempty `w` iff `φ` accepts `p ++ w`; `Transformer.CRASP.LeftQuotientCorrect` proves both.
`Q_a` stays; `MOD_m^r` becomes `MOD_m^{r + (m − 1)|p|}`, and `MOD_0^r` becomes `MOD_0^{r − |p|}`;
`Y φ` at the first position of `w` is the value of `φ` at the end of `p`, a constant, and `¬Y⊤`
tests the first position; a count adds the constant count over `p` (`TermP.addConst`).

The translation keeps the depth, and the absence of `Y` and of `MOD` (`FormP.depth_leftQuot`,
`FormP.prevFree_leftQuot`, `FormP.modFree_leftQuot`), so it acts on `TL[◁#, MOD]` and on
`TL[◁#, Y]` at every depth. `Transformer.CRASP.Suffix` moves a suffix into a formula of
`TL[◁#]`.
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- `n + t`, the constant written `1 + ⋯ + 1`: `TL[◁#]^pos` has no `0`. -/
def TermP.addConst : ℕ → TermP σ → TermP σ
  | 0, t => t
  | n + 1, t => .add .one (TermP.addConst n t)

theorem TermP.depth_addConst (t : TermP σ) : ∀ n, (TermP.addConst n t).depth = t.depth
  | 0 => rfl
  | n + 1 => by rw [TermP.addConst, TermP.depth, TermP.depth_addConst t n, TermP.depth]; simp

theorem TermP.prevFree_addConst (t : TermP σ) : ∀ n, (TermP.addConst n t).prevFree = t.prevFree
  | 0 => rfl
  | n + 1 => by rw [TermP.addConst, TermP.prevFree, TermP.prevFree_addConst t n]; rfl

theorem TermP.modFree_addConst (t : TermP σ) : ∀ n, (TermP.addConst n t).modFree = t.modFree
  | 0 => rfl
  | n + 1 => by rw [TermP.addConst, TermP.modFree, TermP.modFree_addConst t n]; rfl

variable [DecidableEq σ]

mutual

/-- **`p⁻¹φ`**: `φ` read after the prefix `p`. -/
def FormP.leftQuot (p : List σ) : FormP σ → FormP σ
  | .sym a => .sym a
  | .mod m r => .mod m (r + m * p.length - p.length)
  | .prev φ =>
      if (FormP.prev φ).sat p (p.length + 1) then
        .neg (.and (.prev (.neg (.lt .one .one))) (.neg (.prev (φ.leftQuot p))))
      else .prev (φ.leftQuot p)
  | .lt t₁ t₂ => .lt (t₁.leftQuot p) (t₂.leftQuot p)
  | .neg φ => .neg (φ.leftQuot p)
  | .and φ₁ φ₂ => .and (φ₁.leftQuot p) (φ₂.leftQuot p)

/-- `p⁻¹t`: a count adds its value over `p`. -/
def TermP.leftQuot (p : List σ) : TermP σ → TermP σ
  | .countL φ => TermP.addConst ((TermP.countL φ).val p p.length) (.countL (φ.leftQuot p))
  | .add t₁ t₂ => .add (t₁.leftQuot p) (t₂.leftQuot p)
  | .one => .one

end

mutual

/-- `p⁻¹φ` has the depth of `φ`. -/
theorem FormP.depth_leftQuot (p : List σ) : ∀ φ : FormP σ, (φ.leftQuot p).depth = φ.depth
  | .sym _ => rfl
  | .mod _ _ => rfl
  | .prev φ => by
      rw [FormP.leftQuot]
      split <;> simp [FormP.depth, TermP.depth, φ.depth_leftQuot p]
  | .lt t₁ t₂ => by
      rw [FormP.leftQuot, FormP.depth, FormP.depth, t₁.depth_leftQuot p, t₂.depth_leftQuot p]
  | .neg φ => by rw [FormP.leftQuot, FormP.depth, FormP.depth, φ.depth_leftQuot p]
  | .and φ₁ φ₂ => by
      rw [FormP.leftQuot, FormP.depth, FormP.depth, φ₁.depth_leftQuot p, φ₂.depth_leftQuot p]

theorem TermP.depth_leftQuot (p : List σ) : ∀ t : TermP σ, (t.leftQuot p).depth = t.depth
  | .countL φ => by
      rw [TermP.leftQuot, TermP.depth_addConst, TermP.depth, TermP.depth, φ.depth_leftQuot p]
  | .add t₁ t₂ => by
      rw [TermP.leftQuot, TermP.depth, TermP.depth, t₁.depth_leftQuot p, t₂.depth_leftQuot p]
  | .one => rfl

end

mutual

/-- `p⁻¹φ` is free of `Y` iff `φ` is: `p⁻¹` maps `TL[◁#, MOD]` into itself. -/
theorem FormP.prevFree_leftQuot (p : List σ) :
    ∀ φ : FormP σ, (φ.leftQuot p).prevFree = φ.prevFree
  | .sym _ => rfl
  | .mod _ _ => rfl
  | .prev φ => by
      rw [FormP.leftQuot]
      split <;> rfl
  | .lt t₁ t₂ => by
      rw [FormP.leftQuot, FormP.prevFree, FormP.prevFree, t₁.prevFree_leftQuot p,
        t₂.prevFree_leftQuot p]
  | .neg φ => by rw [FormP.leftQuot, FormP.prevFree, FormP.prevFree, φ.prevFree_leftQuot p]
  | .and φ₁ φ₂ => by
      rw [FormP.leftQuot, FormP.prevFree, FormP.prevFree, φ₁.prevFree_leftQuot p,
        φ₂.prevFree_leftQuot p]

theorem TermP.prevFree_leftQuot (p : List σ) :
    ∀ t : TermP σ, (t.leftQuot p).prevFree = t.prevFree
  | .countL φ => by
      rw [TermP.leftQuot, TermP.prevFree_addConst, TermP.prevFree, TermP.prevFree,
        φ.prevFree_leftQuot p]
  | .add t₁ t₂ => by
      rw [TermP.leftQuot, TermP.prevFree, TermP.prevFree, t₁.prevFree_leftQuot p,
        t₂.prevFree_leftQuot p]
  | .one => rfl

end

mutual

/-- `p⁻¹φ` is free of `MOD` iff `φ` is: `p⁻¹` maps `TL[◁#, Y]` into itself. -/
theorem FormP.modFree_leftQuot (p : List σ) : ∀ φ : FormP σ, (φ.leftQuot p).modFree = φ.modFree
  | .sym _ => rfl
  | .mod _ _ => rfl
  | .prev φ => by
      rw [FormP.leftQuot]
      split <;> simp [FormP.modFree, TermP.modFree, φ.modFree_leftQuot p]
  | .lt t₁ t₂ => by
      rw [FormP.leftQuot, FormP.modFree, FormP.modFree, t₁.modFree_leftQuot p,
        t₂.modFree_leftQuot p]
  | .neg φ => by rw [FormP.leftQuot, FormP.modFree, FormP.modFree, φ.modFree_leftQuot p]
  | .and φ₁ φ₂ => by
      rw [FormP.leftQuot, FormP.modFree, FormP.modFree, φ₁.modFree_leftQuot p,
        φ₂.modFree_leftQuot p]

theorem TermP.modFree_leftQuot (p : List σ) : ∀ t : TermP σ, (t.leftQuot p).modFree = t.modFree
  | .countL φ => by
      rw [TermP.leftQuot, TermP.modFree_addConst, TermP.modFree, TermP.modFree,
        φ.modFree_leftQuot p]
  | .add t₁ t₂ => by
      rw [TermP.leftQuot, TermP.modFree, TermP.modFree, t₁.modFree_leftQuot p,
        t₂.modFree_leftQuot p]
  | .one => rfl

end

end CRASP
end Transformer

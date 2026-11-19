module

public import Transformer.CRASP.Indicator
public import Transformer.CRASP.Locality

/-!
# A fixed suffix costs no depth

In no paper. A formula of `TL[◁#]_k` judged at the end of `w ++ s`, for a fixed
string `s`, is a formula of `TL[◁#]_k` judged at the end of `w`
(`Form.models_snocs`): the right quotient `{w | w ++ s ∈ L}` of a language of
`TL[◁#]_k` by a word is again one (`DefinableL.append`). A reduction may thus
close its instances by a fixed suffix, such as the query of a recall task.

One symbol at a time (`Form.snoc`). At the new last position `|w| + 1` of
`w ++ [a]` the symbol is `a`, known in advance, and a count `◁#[ψ]` is its
value at `|w|` on `w` (`Form.sat_append`) plus the indicator of `ψ` at the new
position, which is by induction a formula judged at the end of `w`
(`Term.val_snoc`). A comparison of such sums is a plain formula of the same
depth (`Form.ltInd`, arXiv:2506.16055v3, Appendix A.3).
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

mutual

/-- `φ` judged at the end of `w ++ [a]`, as a formula judged at the end of `w`:
`Q_b` becomes `⊤` or `⊥` as `a = b`, and a comparison gains on each side the
indicators of `Term.snocInds`. -/
def Form.snoc (a : σ) : Form σ → Form σ
  | .sym b => if a = b then .neg (.lt .one .one) else .lt .one .one
  | .lt t₁ t₂ => Form.ltInd t₁ (t₁.snocInds a) t₂ (t₂.snocInds a)
  | .neg φ => .neg (φ.snoc a)
  | .and φ₁ φ₂ => .and (φ₁.snoc a) (φ₂.snoc a)
  | .pnp π => .pnp π

/-- What a term gains at the new last position of `w ++ [a]`: the indicator of
`ψ.snoc a` for each count `◁#[ψ]` of the term. -/
def Term.snocInds (a : σ) : Term σ → List (Form σ)
  | .countL φ => [φ.snoc a]
  | .countR φ => [φ.snoc a]
  | .add t₁ t₂ => t₁.snocInds a ++ t₂.snocInds a
  | .one => []

end

mutual

/-- **A formula judged after one more symbol.**  `φ.snoc a` holds at the end of
`w` iff `φ` holds at the end of `w ++ [a]`. -/
theorem Form.sat_snoc (a : σ) (w : List σ) :
    ∀ φ : Form σ, φ.past = true → φ.pnpFree = true →
      (φ.snoc a).sat w w.length = φ.sat (w ++ [a]) (w.length + 1)
  | .sym b, _, _ => by
      by_cases h : a = b
      · subst h
        simp [Form.snoc, Form.sat, Term.val]
      · simp [Form.snoc, Form.sat, Term.val, h]
  | .lt t₁ t₂, hp, hf => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
      rw [Form.snoc, Form.sat_ltInd, Form.sat, t₁.val_snoc a w hp.1 hf.1,
        t₂.val_snoc a w hp.2 hf.2]
  | .neg φ, hp, hf => by
      simp only [Form.past, Form.pnpFree] at hp hf
      rw [Form.snoc, Form.sat, Form.sat, φ.sat_snoc a w hp hf]
  | .and φ₁ φ₂, hp, hf => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
      rw [Form.snoc, Form.sat, Form.sat, φ₁.sat_snoc a w hp.1 hf.1, φ₂.sat_snoc a w hp.2 hf.2]
  | .pnp _, _, hf => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- A term after one more symbol: its value at the end of `w ++ [a]` is its
value at the end of `w` plus the number of its indicators that hold there. -/
theorem Term.val_snoc (a : σ) (w : List σ) :
    ∀ t : Term σ, t.past = true → t.pnpFree = true →
      t.val (w ++ [a]) (w.length + 1) =
        t.val w w.length + (t.snocInds a).countP (·.sat w w.length)
  | .countL φ, hp, hf => by
      simp only [Term.past, Term.pnpFree] at hp hf
      have h₁ : (List.range' 1 w.length).filter (fun j => φ.sat (w ++ [a]) j) =
          (List.range' 1 w.length).filter (fun j => φ.sat w j) :=
        List.filter_congr fun j hj => φ.sat_append w [a] hp hf j (List.mem_range'_1.mp hj).1
          (by have := (List.mem_range'_1.mp hj).2; omega)
      rw [Term.val, Term.val, Term.snocInds, List.range'_concat, List.filter_append,
        List.length_append, h₁, List.countP_singleton, φ.sat_snoc a w hp hf,
        Nat.one_mul, Nat.add_comm 1 w.length]
      cases h : φ.sat (w ++ [a]) (w.length + 1) <;> simp [h]
  | .countR _, hp, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf => by
      simp only [Term.past, Term.pnpFree, Bool.and_eq_true] at hp hf
      rw [Term.val, Term.val, t₁.val_snoc a w hp.1 hf.1, t₂.val_snoc a w hp.2 hf.2,
        Term.snocInds, List.countP_append]
      omega
  | .one, _, _ => by simp [Term.val, Term.snocInds]

end

mutual

/-- `Form.snoc` keeps a formula in `TL[◁#]_d`. -/
theorem Form.snoc_mem_TLCl (a : σ) {d : ℕ} : ∀ φ : Form σ, φ ∈ TLCl σ d → φ.snoc a ∈ TLCl σ d
  | .sym b, _ => by rw [Form.snoc]; split <;> exact ⟨rfl, rfl, Nat.zero_le d⟩
  | .lt t₁ t₂, h => by
      obtain ⟨hp, hf, hd⟩ := id h
      simp only [Form.past, Form.pnpFree, Form.depth, Bool.and_eq_true, max_le_iff] at hp hf hd
      exact Form.ltInd_mem_TLCl t₁ _ t₂ _ h (t₁.snocInds_mem_TLCl a hp.1 hf.1 hd.1)
        (t₂.snocInds_mem_TLCl a hp.2 hf.2 hd.2)
  | .neg φ, h => φ.snoc_mem_TLCl a h
  | .and φ₁ φ₂, ⟨hp, hf, hd⟩ => by
      simp only [Form.past, Form.pnpFree, Form.depth, Bool.and_eq_true, max_le_iff] at hp hf hd
      obtain ⟨hp₁, hf₁, hd₁⟩ := φ₁.snoc_mem_TLCl a ⟨hp.1, hf.1, hd.1⟩
      obtain ⟨hp₂, hf₂, hd₂⟩ := φ₂.snoc_mem_TLCl a ⟨hp.2, hf.2, hd.2⟩
      exact ⟨by simp [Form.snoc, Form.past, hp₁, hp₂], by simp [Form.snoc, Form.pnpFree, hf₁, hf₂],
        max_le hd₁ hd₂⟩
  | .pnp _, ⟨_, hf, _⟩ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The indicators a past-only term without PNPs of depth at most `d` gains lie
in `TL[◁#]_d`. -/
theorem Term.snocInds_mem_TLCl (a : σ) {d : ℕ} :
    ∀ t : Term σ, t.past = true → t.pnpFree = true → t.depth ≤ d →
      ∀ ψ ∈ t.snocInds a, ψ ∈ TLCl σ d
  | .countL φ, hp, hf, hd, ψ, h => by
      rw [Term.snocInds, List.mem_singleton] at h
      simp only [Term.past, Term.pnpFree, Term.depth] at hp hf hd
      exact h ▸ φ.snoc_mem_TLCl a ⟨hp, hf, by omega⟩
  | .countR _, hp, _, _, _, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf, hd, ψ, h => by
      simp only [Term.past, Term.pnpFree, Term.depth, Bool.and_eq_true, max_le_iff] at hp hf hd
      rw [Term.snocInds, List.mem_append] at h
      exact h.elim (t₁.snocInds_mem_TLCl a hp.1 hf.1 hd.1 ψ)
        (t₂.snocInds_mem_TLCl a hp.2 hf.2 hd.2 ψ)
  | .one, _, _, _, _, h => by rw [Term.snocInds] at h; exact absurd h List.not_mem_nil

end

/-- The hypotheses of `Form.sat_snoc`, `Term.val_snoc`, `Form.snoc_mem_TLCl`
and `Term.snocInds_mem_TLCl` are satisfiable: `◁#[Q_b] < 1`, a formula of
`TL[◁#]_1`. -/
example (w : List Bool) :
    ((Form.lt (.countL (.sym true)) .one).snoc true).sat w w.length =
        (Form.lt (.countL (.sym true)) .one).sat (w ++ [true]) (w.length + 1) ∧
      (Term.countL (.sym true)).val (w ++ [true]) (w.length + 1) =
        (Term.countL (.sym true)).val w w.length +
          ((Term.countL (.sym true)).snocInds true).countP (·.sat w w.length) ∧
      (Form.lt (.countL (.sym true)) .one).snoc true ∈ TLCl Bool 1 ∧
      ∀ ψ ∈ (Term.countL (.sym true)).snocInds true, ψ ∈ TLCl Bool 1 :=
  ⟨Form.sat_snoc true w _ rfl rfl, Term.val_snoc true w _ rfl rfl,
    Form.snoc_mem_TLCl true _ ⟨rfl, rfl, le_rfl⟩, Term.snocInds_mem_TLCl true _ rfl rfl le_rfl⟩

/-- `φ` judged at the end of `w ++ s`, as a formula judged at the end of `w`. -/
def Form.snocs (s : List σ) (φ : Form σ) : Form σ := s.foldr Form.snoc φ

/-- `Form.snocs` keeps a formula in `TL[◁#]_d`. -/
theorem Form.snocs_mem_TLCl {d : ℕ} (s : List σ) {φ : Form σ} (hφ : φ ∈ TLCl σ d) :
    φ.snocs s ∈ TLCl σ d := by
  induction s with
  | nil => exact hφ
  | cons a s ih => exact Form.snoc_mem_TLCl a _ ih

/-- **A fixed suffix costs no depth.**  `φ.snocs s` holds at the end of `w`
iff `φ` holds at the end of `w ++ s`. -/
theorem Form.models_snocs {d : ℕ} (s : List σ) {φ : Form σ} (hφ : φ ∈ TLCl σ d)
    (w : List σ) : (φ.snocs s).models w ↔ φ.models (w ++ s) := by
  induction s generalizing w with
  | nil => rw [List.append_nil]; rfl
  | cons a s ih =>
      obtain ⟨hp, hf, -⟩ := Form.snocs_mem_TLCl s hφ
      have e := Form.sat_snoc a w (φ.snocs s) hp hf
      rw [show w ++ a :: s = (w ++ [a]) ++ s by simp, ← ih]
      change (Form.snoc a (φ.snocs s)).sat w w.length = true ↔
        (φ.snocs s).sat (w ++ [a]) (w ++ [a]).length = true
      rw [e, List.length_append, List.length_singleton]

/-- The hypothesis of `Form.snocs_mem_TLCl` and `Form.models_snocs` is
satisfiable: `Q_b` holds at the end of `w · ab`. -/
example (w : List Bool) : (Form.sym true).snocs [false, true] ∈ TLCl Bool 0 ∧
    (((Form.sym true).snocs [false, true]).models w ↔
      (Form.sym true).models (w ++ [false, true])) :=
  ⟨Form.snocs_mem_TLCl _ ⟨rfl, rfl, le_rfl⟩, Form.models_snocs _ ⟨rfl, rfl, le_rfl⟩ w⟩

/-- **The right quotient by a word costs no depth.**  If `L` is definable at
depth `k`, so is `{w | w ++ s ∈ L}`. -/
theorem DefinableL.append (s : List σ) {L : Set (List σ)} {k : ℕ} (h : DefinableL L k) :
    DefinableL {w | w ++ s ∈ L} k := by
  obtain ⟨φ, hφ, rfl⟩ := h
  exact ⟨φ.snocs s, Form.snocs_mem_TLCl s hφ, Set.ext fun w => Form.models_snocs s hφ w⟩

/-- The hypothesis of `DefinableL.append` is satisfiable: the strings ending in
`b` are definable at depth `0`, and so are those `w` with `w · a` among them. -/
example : DefinableL {w : List Bool | w ++ [false] ∈ (Form.sym true).lang} 0 :=
  DefinableL.append [false] ⟨_, ⟨rfl, rfl, le_rfl⟩, rfl⟩

end CRASP
end Transformer

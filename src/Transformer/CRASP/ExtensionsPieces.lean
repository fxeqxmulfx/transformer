/-
# Eliminating the sugar of Appendix A.3: terms as guarded pieces

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A.3: the `?`-elimination lemma of Yang & Chiang and
`thm:strict`.

The rewriting rules of the appendix, `◁#_<[φ] ≡ ◁#[φ] − (φ ? 1 : 0)` and its
mirror images, subtract; the plain syntax cannot.  So an extended term is
represented by *guarded pieces* `(g, p, k)`: some guard holds at every position,
and wherever `g` holds the term plus `k` equals the plain term `p`
(`Covers`).  A comparison of two represented terms is then a disjunction over
pairs of pieces of `g₁ ∧ g₂ ∧ p₁ + k₂ < p₂ + k₁` (`ltPieces`), and `?` and `+`
combine pieces (`condPieces`, `addPieces`).  None of it adds depth.

One caveat the rules hide: at position `0`, where the empty word is judged,
`◁#_<[φ]` is `0` and not `◁#[φ] − (φ ? 1 : 0)`; the pieces of `◁#_<` guard
against it with `pos`.
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- `⊤`, of depth `d`: `¬(◁#[⊤_{d-1}] < ◁#[⊤_{d-1}])` (Appendix A.3, used to
pad a translation to the depth of its source). -/
def Form.topAt : ℕ → Form σ
  | 0 => .neg (.lt .one .one)
  | d + 1 => .neg (.lt (.countL (topAt d)) (.countL (topAt d)))

@[simp] theorem Form.sat_topAt (w : List σ) (i d : ℕ) : (Form.topAt d : Form σ).sat w i = true := by
  cases d <;> simp [Form.topAt, Form.sat]

omit [DecidableEq σ] in
@[simp] theorem Form.depth_topAt (d : ℕ) : (Form.topAt d : Form σ).depth = d := by
  induction d with
  | zero => rfl
  | succ d ih => simp [Form.topAt, Form.depth, Term.depth, ih]

/-- `i > 0`, as `¬(◁#[⊤] < 1)` (Appendix A.3). -/
def Form.pos : Form σ := .neg (.lt (.countL (Form.topAt 0)) .one)

@[simp] theorem Form.sat_pos (w : List σ) (i : ℕ) : (Form.pos : Form σ).sat w i = decide (0 < i) := by
  simp [Form.pos, Form.sat, Term.val, List.filter_true, Nat.pos_iff_ne_zero]

omit [DecidableEq σ] in
@[simp] theorem Form.depth_pos : (Form.pos : Form σ).depth = 1 := rfl

/-- The disjunction of a list of formulas (Appendix A.3). -/
def Form.any (L : List (Form σ)) : Form σ := L.foldr Form.or (.lt .one .one)

theorem Form.sat_any (w : List σ) (i : ℕ) (L : List (Form σ)) :
    (Form.any L).sat w i = true ↔ ∃ φ ∈ L, φ.sat w i = true := by
  induction L with
  | nil => simp [Form.any, Form.sat, Term.val]
  | cons φ L ih => simp [Form.any, ← ih]

omit [DecidableEq σ] in
theorem Form.depth_any_le {d : ℕ} (L : List (Form σ)) (h : ∀ φ ∈ L, φ.depth ≤ d) :
    (Form.any L).depth ≤ d := by
  induction L with
  | nil => simp [Form.any, Form.depth, Term.depth]
  | cons φ L ih =>
      simp only [Form.any, List.foldr_cons, Form.depth_or, max_le_iff]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

/-- The plain term `t + k`, as `t + 1 + ⋯ + 1` (§2.2). -/
def Term.addNat (t : Term σ) : ℕ → Term σ
  | 0 => t
  | k + 1 => .add (t.addNat k) .one

@[simp] theorem Term.val_addNat (w : List σ) (i : ℕ) (t : Term σ) (k : ℕ) :
    (t.addNat k).val w i = t.val w i + k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Term.addNat, Term.val, ih, Term.val, Nat.add_assoc]

omit [DecidableEq σ] in
@[simp] theorem Term.depth_addNat (t : Term σ) (k : ℕ) : (t.addNat k).depth = t.depth := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Term.addNat, Term.depth, ih, Term.depth]; omega

/-- A term of value `n` at `w, i` is represented by the guarded pieces `P`:
some guard holds, and each guard that holds gives `n + k = p` (Appendix A.3,
the rules `◁#_<[φ] ≡ ◁#[φ] − (φ ? 1 : 0)` read without subtraction). -/
def Covers (P : List (Form σ × Term σ × ℕ)) (w : List σ) (i n : ℕ) : Prop :=
  (∃ x ∈ P, x.1.sat w i = true) ∧ ∀ x ∈ P, x.1.sat w i = true → n + x.2.2 = x.2.1.val w i

/-- `t₁ < t₂`, over the pieces of both (Appendix A.3). -/
def ltPieces (P Q : List (Form σ × Term σ × ℕ)) : Form σ :=
  Form.any (P.flatMap fun x => Q.map fun y =>
    .and (.and x.1 y.1) (.lt (x.2.1.addNat y.2.2) (y.2.1.addNat x.2.2)))

theorem sat_ltPieces {P Q : List (Form σ × Term σ × ℕ)} {w : List σ} {i n₁ n₂ : ℕ}
    (hP : Covers P w i n₁) (hQ : Covers Q w i n₂) :
    (ltPieces P Q).sat w i = decide (n₁ < n₂) := by
  obtain ⟨⟨x, hx, hgx⟩, hP⟩ := hP
  obtain ⟨⟨y, hy, hgy⟩, hQ⟩ := hQ
  apply Bool.eq_iff_iff.2
  rw [ltPieces, Form.sat_any, decide_eq_true_iff]
  constructor
  · rintro ⟨φ, hφ, h⟩
    simp only [List.mem_flatMap, List.mem_map] at hφ
    obtain ⟨x', hx', y', hy', rfl⟩ := hφ
    simp only [Form.sat, Bool.and_eq_true, Term.val_addNat, decide_eq_true_eq] at h
    have := hP x' hx' h.1.1
    have := hQ y' hy' h.1.2
    omega
  · intro h
    refine ⟨_, List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨y, hy, rfl⟩⟩, ?_⟩
    have := hP x hx hgx
    have := hQ y hy hgy
    simp only [Form.sat, hgx, hgy, Bool.and_self, Term.val_addNat, Bool.true_and,
      decide_eq_true_eq]
    omega

omit [DecidableEq σ] in
theorem depth_ltPieces_le {d : ℕ} {P Q : List (Form σ × Term σ × ℕ)}
    (hP : ∀ x ∈ P, x.1.depth ≤ d ∧ x.2.1.depth ≤ d) (hQ : ∀ x ∈ Q, x.1.depth ≤ d ∧ x.2.1.depth ≤ d) :
    (ltPieces P Q).depth ≤ d := by
  refine Form.depth_any_le _ fun φ hφ => ?_
  simp only [List.mem_flatMap, List.mem_map] at hφ
  obtain ⟨x, hx, y, hy, rfl⟩ := hφ
  have := hP x hx
  have := hQ y hy
  simp only [Form.depth, Term.depth_addNat]
  omega

/-- The hypotheses of `depth_ltPieces_le` are satisfiable. -/
example : ∀ x ∈ [((Form.topAt 0 : Form Bool), (.one : Term Bool), 0)], x.1.depth ≤ 0 ∧ x.2.1.depth ≤ 0 := by
  simp [Term.depth]

/-- `c ? t₁ : t₂`, over the pieces of both branches (Appendix A.3). -/
def condPieces (c : Form σ) (P Q : List (Form σ × Term σ × ℕ)) : List (Form σ × Term σ × ℕ) :=
  P.map (fun x => (.and c x.1, x.2)) ++ Q.map (fun x => (.and (.neg c) x.1, x.2))

theorem covers_condPieces {c : Form σ} {P Q : List (Form σ × Term σ × ℕ)} {w : List σ}
    {i n₁ n₂ : ℕ} (hP : Covers P w i n₁) (hQ : Covers Q w i n₂) :
    Covers (condPieces c P Q) w i (if c.sat w i then n₁ else n₂) := by
  obtain ⟨⟨x, hx, hgx⟩, hP⟩ := hP
  obtain ⟨⟨y, hy, hgy⟩, hQ⟩ := hQ
  simp only [Covers, condPieces, List.mem_append, List.mem_map]
  cases hc : c.sat w i
  · refine ⟨⟨_, Or.inr ⟨y, hy, rfl⟩, by simp [Form.sat, hc, hgy]⟩, ?_⟩
    rintro _ (⟨z, -, rfl⟩ | ⟨z, hz, rfl⟩) h <;> simp only [Form.sat, hc, Bool.false_and,
      Bool.not_false, Bool.true_and, Bool.false_eq_true] at h
    exact hQ z hz h
  · refine ⟨⟨_, Or.inl ⟨x, hx, rfl⟩, by simp [Form.sat, hc, hgx]⟩, ?_⟩
    rintro _ (⟨z, hz, rfl⟩ | ⟨z, -, rfl⟩) h <;> simp only [Form.sat, hc, Bool.true_and,
      Bool.not_true, Bool.false_and, Bool.false_eq_true] at h
    exact hP z hz h

/-- `t₁ + t₂`, over pairs of pieces (Appendix A.3). -/
def addPieces (P Q : List (Form σ × Term σ × ℕ)) : List (Form σ × Term σ × ℕ) :=
  P.flatMap fun x => Q.map fun y => (.and x.1 y.1, .add x.2.1 y.2.1, x.2.2 + y.2.2)

theorem covers_addPieces {P Q : List (Form σ × Term σ × ℕ)} {w : List σ} {i n₁ n₂ : ℕ}
    (hP : Covers P w i n₁) (hQ : Covers Q w i n₂) : Covers (addPieces P Q) w i (n₁ + n₂) := by
  obtain ⟨⟨x, hx, hgx⟩, hP⟩ := hP
  obtain ⟨⟨y, hy, hgy⟩, hQ⟩ := hQ
  simp only [Covers, addPieces, List.mem_flatMap, List.mem_map]
  refine ⟨⟨_, ⟨x, hx, y, hy, rfl⟩, by simp [Form.sat, hgx, hgy]⟩, ?_⟩
  rintro _ ⟨x', hx', y', hy', rfl⟩ h
  simp only [Form.sat, Bool.and_eq_true] at h
  have := hP x' hx' h.1
  have := hQ y' hy' h.2
  simp only [Term.val]
  omega

/-- The hypotheses of `sat_ltPieces`, `covers_condPieces` and `covers_addPieces` are
satisfiable:
`1` is represented by `(⊤, 1, 0)`. -/
example : Covers [(Form.topAt 0, (.one : Term Bool), 0)] [] 0 1 :=
  ⟨⟨_, List.mem_singleton_self _, Form.sat_topAt ..⟩, fun x hx _ => by
    rw [List.mem_singleton] at hx; subst hx; rfl⟩

end CRASP
end Transformer

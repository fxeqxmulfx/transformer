/-
# The key the compiler emits is not the lift

`Transformer.ALM.HullLift` proves that no erase rule fires on a family of
lifted keys, and `Transformer.ALM.HullCover` reads the container's final size
off that.  Both are theorems about `liftKey k = (2k, -k²)`, and the compiler
never emits it.  `_to_2d_key` (`transformer_vm/graph/core.py`, lines 293-319,
ported verbatim as `vm-rs/alm-compile/src/graph.rs::embed_key`) builds the
intercept of a `latest` head as

  `ky = -k² - clear · BIG + LATEST_ALPHA · inv_log_pos`,

so every live key of the shipped model sits *above* the paraboloid by a
positive amount.  The amount is bounded: `LATEST_ALPHA = 0.3` and
`inv_log_pos p = 1/log 2 - 1/log (p+2)` (`transformer_vm/evaluator.py`, line
230) rises from `0` to `1/log 2`, so the offset stays under
`0.3/log 2 = 0.4329` at every position a trace can reach — never zero after
the first token, never as much as the unit gap between distinct integer keys.

That bound is the whole of it.  Three lifted keys put the middle one above the
chord of the other two by `(k₂-k₁)(k₃-k₂) ≥ 1`, and an offset under `1` cannot
spend that, so `lineEval_markKey_lt` recovers the strict domination
`lineEval_liftKey_lt_of_ne` gave for free and every consequence of it follows
unchanged: `not_eraseStep_of_marked`, `erasesTo_eq_of_marked`,
`build_eq_of_marked`, and one line per distinct key in
`build_card_eq_of_marked`.  What the perturbation costs is a hypothesis the
lift did not need — the keys must be a unit apart, which on the integer keys
the machine compiles to (`Transformer.ALM.ScalarInt`) they are.

`marked_sep_of_shipped` is the shipped constant discharging that hypothesis.
The perturbation's own purpose — separating two writes to one logical key, so
that `HullMeta::last_seq` is a fallback rather than the mechanism — is
`todo3.md` §2a, measured by
`vm-rs/alm-vm/tests/reference.rs::nothing_on_the_released_weights_is_decided_by_rounding`
at `6.0e-6` key steps in `hello`.  The other term, the clear marker, does not
fit under this bound and is not covered here.

Source: `transformer_vm/graph/core.py` lines 293-319; `todo3.md` §2a.
-/

import Transformer.ALM.HullLift
import Mathlib.Analysis.Complex.ExponentialBounds

namespace Transformer
namespace ALM

/-! ### The key `_to_2d_key` builds -/

/-- The 2D key of a live entry of a `latest` head: the lift of `k`, raised by
the recency term `δ = LATEST_ALPHA · inv_log_pos p`. -/
def markKey (δ k : ℝ) : ℝ × ℝ := (2 * k, -k ^ 2 + δ)

/-- With no recency term it is the lift, so everything below specializes to
`Transformer.ALM.HullLift`. -/
@[simp] theorem markKey_zero (k : ℝ) : markKey 0 k = liftKey k := by
  unfold markKey liftKey; norm_num

/-- The recency term moves the intercept and nothing else, so the slope still
names the key. -/
theorem markKey_fst (δ k : ℝ) : (markKey δ k).1 = 2 * k := rfl

/-- At `x = k` the perturbed key evaluates to `k² + δ`: the paraboloid's
tangent, lifted off it by the term. -/
theorem lineEval_markKey_self (δ k : ℝ) : lineEval (markKey δ k) k = k ^ 2 + δ := by
  unfold lineEval markKey; ring

/-- **And a key a unit away is still strictly below it there.**  The shortfall
of `k'` at `k` is `(k - k')² - (δ' - δ)`, and a perturbation spread under one
cannot spend a squared key gap of at least one.  This is
`lineEval_liftKey_lt_of_ne` with the offsets carried through, and it is the
only place the unit separation is used. -/
theorem lineEval_markKey_lt {δ δ' k k' : ℝ} (hsep : 1 ≤ (k - k') ^ 2)
    (hspread : δ' - δ < 1) :
    lineEval (markKey δ' k') k < lineEval (markKey δ k) k := by
  unfold lineEval markKey
  nlinarith

/-- The hypotheses are satisfiable, at the shipped spread: the key `0` raised
by nothing falls `1 - 0.4` below the key `1` raised by `0.4`, at `x = 1`. -/
example : lineEval (markKey 0 (0 : ℝ)) 1 < lineEval (markKey 0.4 (1 : ℝ)) 1 :=
  lineEval_markKey_lt (by norm_num) (by norm_num)

/-! ### The family a `latest` head holds -/

/-- A family of perturbed lifts, as a head holds it: one line per *integer*
key, each raised by an offset in `[0, A]`.  `onePerKey` is not an extra
assumption about the data but the container's own invariant — two lines of
equal slope are two writes to one key, and `eraseStep_of_slope_eq` removes one
of them before this predicate is asked about. -/
structure Marked (A : ℝ) (s : Finset (ℝ × ℝ)) : Prop where
  perturbed : ∀ l ∈ s, ∃ (z : ℤ) (δ : ℝ), 0 ≤ δ ∧ δ ≤ A ∧ l = markKey δ (z : ℝ)
  onePerKey : ∀ l ∈ s, ∀ l' ∈ s, l.1 = l'.1 → l = l'

/-- Both clauses survive shrinking, which is what the build induction needs. -/
theorem Marked.subset {A : ℝ} {s t : Finset (ℝ × ℝ)} (hs : Marked A s) (h : t ⊆ s) :
    Marked A t :=
  ⟨fun l hl => hs.perturbed l (h hl), fun l hl l' hl' => hs.onePerKey l (h hl) l' (h hl')⟩

/-- Distinct integer keys are a unit apart, so their squared gap clears the
perturbation spread. -/
private theorem one_le_sq_sub_int {z z' : ℤ} (h : z ≠ z') : (1 : ℝ) ≤ ((z : ℝ) - (z' : ℝ)) ^ 2 := by
  have hz : (1 : ℤ) ≤ (z - z') ^ 2 := by
    rcases lt_trichotomy z z' with hlt | heq | hgt
    · nlinarith
    · exact absurd heq h
    · nlinarith
  have := (Int.cast_le (R := ℝ)).mpr hz
  push_cast at this
  linarith

/-- **No erase step applies to the family the compiler emits.**  The line under
test strictly beats every survivor at its own key, so — exactly as in
`not_eraseStep_of_lift` — no rule of the form "this line is matched by a
surviving one at every query" can hold, whatever comparisons an implementation
performs. -/
theorem not_eraseStep_of_marked {A : ℝ} {s t : Finset (ℝ × ℝ)} (hA : A < 1) (hs : Marked A s) :
    ¬ EraseStep s t := by
  rintro ⟨l, hl, rfl, hdom⟩
  obtain ⟨z, δ, hδ0, hδA, rfl⟩ := hs.perturbed l hl
  obtain ⟨l', hl', hle⟩ := hdom (z : ℝ)
  have hmem : l' ∈ s := Finset.mem_of_mem_erase hl'
  obtain ⟨z', δ', hδ'0, hδ'A, rfl⟩ := hs.perturbed l' hmem
  have hne : markKey δ' (z' : ℝ) ≠ markKey δ (z : ℝ) := (Finset.mem_erase.mp hl').1
  have hzz : z ≠ z' := by
    rintro rfl
    exact hne (hs.onePerKey _ hmem _ hl (by rw [markKey_fst, markKey_fst]))
  exact absurd hle (not_le.mpr
    (lineEval_markKey_lt (one_le_sq_sub_int hzz) (by linarith)))

/-- The hypotheses are satisfiable, and on data an erase rule does apply to
elsewhere: the keys `0` and `1` raised by `0` and `0.4` are a `Marked 0.4`
family, while the two parallel lines of `eraseStep_of_slope_eq` are not. -/
example : Marked 0.4 ({markKey 0 (0 : ℝ), markKey 0.4 (1 : ℝ)} : Finset (ℝ × ℝ)) := by
  constructor
  · intro l hl
    simp only [Finset.mem_insert, Finset.mem_singleton] at hl
    rcases hl with rfl | rfl
    · exact ⟨0, 0, by norm_num⟩
    · exact ⟨1, 0.4, by norm_num⟩
  · intro l hl l' hl'
    simp only [Finset.mem_insert, Finset.mem_singleton] at hl hl'
    rcases hl with rfl | rfl <;> rcases hl' with rfl | rfl <;> simp [markKey]

/-- **So no chain of erase loops removes anything from it either.** -/
theorem erasesTo_eq_of_marked {A : ℝ} {s t : Finset (ℝ × ℝ)} (hA : A < 1)
    (h : ErasesTo s t) (hs : Marked A s) : t = s := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => exact absurd hstep (not_eraseStep_of_marked hA (ih ▸ hs))

/-- **And the build keeps every line it inserts.**  The container the shipped
compiler's keys leave behind is the set of lines it was given, so `hno : ∀ p ∈
ps, p = 0` is discharged for the family that actually arrives and not only for
its idealization. -/
theorem build_eq_of_marked {A : ℝ} {p : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)} (hA : A < 1)
    (h : Relation.ReflTransGen BuildStep (∅, ∅) p) (hs : Marked A p.2) : p.1 = p.2 := by
  induction h with
  | refl => rfl
  | tail _ hstep ih =>
      rename_i b _ _
      rcases hstep with ⟨l, rfl⟩ | ⟨hers, h2⟩
      · have hb : Marked A b.2 := hs.subset (Finset.subset_insert _ _)
        simp only []
        rw [ih hb]
      · have hb : Marked A b.2 := h2 ▸ hs
        exact absurd hers (not_eraseStep_of_marked hA ((ih hb) ▸ hb))

/-- **So it holds one line per distinct key.**  `HullCover`'s conclusion, over
the keys the compiler emits: the slope is `2k`, `onePerKey` makes it injective
on the container, and the container is the whole insertion history. -/
theorem build_card_eq_of_marked {A : ℝ} {c d : Finset (ℝ × ℝ)} (hA : A < 1)
    (h : Relation.ReflTransGen BuildStep (∅, ∅) (c, d)) (hs : Marked A d) :
    c.card = (d.image Prod.fst).card := by
  have hcd : c = d := build_eq_of_marked (p := (c, d)) hA h hs
  subst hcd
  exact (Finset.card_image_of_injOn fun l hl l' hl' hf => hs.onePerKey l hl l' hl' hf).symm

/-- The hypotheses are satisfiable: inserting the single perturbed key `0`
builds a container of one line, whose one slope is the one key. -/
example : (insert (markKey 0.4 (0 : ℝ)) (∅ : Finset (ℝ × ℝ))).card
    = ((insert (markKey 0.4 (0 : ℝ)) (∅ : Finset (ℝ × ℝ))).image Prod.fst).card := by
  refine build_card_eq_of_marked (A := 0.4) (by norm_num)
    (Relation.ReflTransGen.single (Or.inl ⟨markKey 0.4 (0 : ℝ), rfl⟩)) ?_
  constructor
  · intro l hl
    simp only [Finset.mem_insert, Finset.notMem_empty, or_false] at hl
    exact ⟨0, 0.4, by norm_num [hl]⟩
  · intro l hl l' hl' _
    simp only [Finset.mem_insert, Finset.notMem_empty, or_false] at hl hl'
    rw [hl, hl']

/-! ### The shipped constant clears the bound -/

/-- **`LATEST_ALPHA / log 2 < 1`.**  `inv_log_pos p = 1/log 2 - 1/log (p+2)` is
bounded above by `1/log 2`, so the offset `0.3 · inv_log_pos p` never reaches
`0.4329` — under the `1` every theorem above asks for, with room for a factor
of two. -/
theorem marked_sep_of_shipped : (0.3 : ℝ) / Real.log 2 < 1 := by
  have h : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  rw [div_lt_one (by linarith)]
  linarith

end ALM
end Transformer

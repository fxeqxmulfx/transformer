module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Mixture
public import Mathlib.Data.Fintype.BigOperators

/-!
# Full positional interactions for bounded causal prefixes

Source: `review/convex_causal.md`, CCa3. Padding with `none` preserves length;
each position remains separate. The feature map multiplies categorical input
indicators only. Its parameters are the joint output coefficients, not factors
inside these products. It is a universal, exponentially large fallback, not
an assertion that memorization learns unseen labels or extrapolates in length.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {A C : Type*} {B : ℕ}

/-- A visible prefix with a declared maximum context (CCa3). -/
abbrev Context (A : Type*) (B : ℕ) := Fin B → Option A

/-- Positional padding uses no answer, task identifier or arithmetic rule (CCa3). -/
def encode (B : ℕ) (xs : List A) : Context A B := fun i => xs[i.val]?

/-- No two bounded visible prefixes are identified by the encoding (CCa3). -/
theorem encode_injective {xs ys : List A} (hx : xs.length ≤ B) (hy : ys.length ≤ B)
    (h : encode B xs = encode B ys) : xs = ys := by
  apply List.ext_getElem?
  intro i
  by_cases hi : i < B
  · exact congrFun h ⟨i, hi⟩
  · have hxi : xs.length ≤ i := le_trans hx (Nat.le_of_not_gt hi)
    have hyi : ys.length ≤ i := le_trans hy (Nat.le_of_not_gt hi)
    simp [List.getElem?_eq_none hxi, List.getElem?_eq_none hyi]

/-- All positional categorical interactions, evaluated only on inputs (CCa3). -/
def feature [DecidableEq A] (template x : Context A B) : ℝ :=
  ∏ i, if x i = template i then 1 else 0

/-- The product selects exactly one complete prefix template (CCa3). -/
theorem feature_eq [DecidableEq A] (template x : Context A B) :
    feature template x = if template = x then 1 else 0 := by
  classical
  by_cases he : template = x
  · subst template
    simp [feature]
  · rw [ite_eq_right he]
    obtain ⟨i, hi⟩ := Function.ne_iff.mp he
    exact Finset.prod_eq_zero (Finset.mem_univ i) (ite_eq_right (Ne.symm hi))

/-- The complete joint coefficient head over the fixed interaction features (CCa3). -/
noncomputable def evaluate [Fintype A] [DecidableEq A]
    (W : Context A B → C → ℝ) (x : Context A B) (c : C) : ℝ :=
  ∑ t, W t c * feature t x

/-- Executing the feature expansion realizes every conditional coordinate table (CCa3). -/
theorem evaluate_eq [Fintype A] [DecidableEq A] (W : Context A B → C → ℝ)
    (x : Context A B) (c : C) : evaluate W x c = W x c := by
  classical
  unfold evaluate
  simp_rw [feature_eq]
  simp [mul_ite]

/-- Every finite-context conditional distribution has a feasible exact head (CCa3).
This includes stochastic targets; deterministic targets are its one-hot boundary. -/
theorem universal_rows [Fintype A] [DecidableEq A] [Fintype C]
    (Q : Context A B → C → ℝ) (hQ : Rows Q) :
    ∃ W : Context A B → C → ℝ, Rows W ∧ evaluate W = Q := by
  refine ⟨Q, hQ, ?_⟩
  funext x c
  exact evaluate_eq Q x c

/-- The exact allocated feature count makes the universality cost explicit (CCa3). -/
theorem context_card [Fintype A] :
    Fintype.card (Context A B) = (Fintype.card A + 1) ^ B := by
  classical
  rw [Fintype.card_fun]
  simp

/-- No token rule is inserted by the features: all logits are preserved by the head (CCa3). -/
theorem evaluate_loss [Fintype A] [DecidableEq A] [Fintype C]
    {S : Type*} [Fintype S] (ε : ℝ) (input : S → Context A B) (label : S → C)
    (W : Context A B → C → ℝ) :
    loss ε input label (evaluate W) = loss ε input label W := by
  have he : evaluate W = W := by
    funext x c
    exact evaluate_eq W x c
  rw [he]

example : encode 3 ([false, true] : List Bool) ≠ encode 3 [true, false] := by
  intro h
  have he := encode_injective (by decide : [false, true].length ≤ 3)
    (by decide : [true, false].length ≤ 3) h
  cases he

example := universal_rows (B := 2) (A := Bool)
  (point (fun x => xor (x 0 == some true) (x 1 == some true))) (rows_point _)

example : Fintype.card (Context Bool 4) = 81 := by rw [context_card]; norm_num

example := evaluate_loss (A := Bool) (B := 2) (C := Bool) (1 / 4)
  (fun b : Bool => encode 2 [b, !b]) (fun b => b)
  (point (fun x => xor (x 0 == some true) (x 1 == some true)))

end Transformer.Grokking.FlatSpectrum.ConvexCausal

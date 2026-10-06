import Transformer.Basis.Tasks

/-!
# Necessary answer properties of any Basis solver

Source: the actual depth, MQAR and parity oracles at cbafbe9. These
requirements concern the next answer, rather than the whole extended
list: retaining different inputs would trivially distinguish whole lists
even if their predictions were identical. Correct depth predictions must
retain order, MQAR predictions must retain binding and last-write order,
and parity predictions must distinguish odd and even bit counts.

Theorems apply to any given continuation function, including the actual
GPTMini adapter. They do not assert that arbitrary parameters already
satisfy these task-dependent conditions.
-/

namespace Transformer.Basis

/-- A bag-only answer function is invariant under every permutation of its raw input.
Source: the order/binding obstruction exhibited by the raw Basis examples below. -/
def PermutationInvariant (next : Tokens → ℤ) : Prop :=
  ∀ x y, x.Perm y → next x = next y

/-- Any solver must give different next answers to valid prefixes with different labels.
Source: the independently defined task semantics, using only the candidate's correctness property. -/
theorem answer_separation (next : Tokens → ℤ) (mode : Mode) (task : Task)
    (hsolve : SolvesTask (extend next) mode task) (x y : Tokens)
    (hx : TaskPrefix mode task x) (hy : TaskPrefix mode task y)
    (hdiff : taskNext mode task x ≠ taskNext mode task y) : next x ≠ next y := by
  have hnext := (solvesTask_extend_iff next mode task).mp hsolve
  rw [hnext x hx, hnext y hy]
  exact hdiff

example : SolvesTask (extend (taskNext .easy .depth)) .easy .depth ∧
    TaskPrefix .easy .depth [1, 9, 9, 10, 10] ∧
    TaskPrefix .easy .depth [1, 9, 10, 9, 10] ∧
    taskNext .easy .depth [1, 9, 9, 10, 10] ≠ taskNext .easy .depth [1, 9, 10, 9, 10] := by
  refine ⟨fun _ _ => rfl, ?_, ?_, by decide⟩
  · exact ⟨[some false, some false, some true, some true], by decide, by decide, rfl⟩
  · exact ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩

/-- Two same-bag depth words require opposite E_2 answers within the real context cap.
Source: Basis easy depth, with aabb accepted and abab rejected. -/
theorem depth_order_pair :
    TaskPrefix .easy .depth [1, 9, 9, 10, 10] ∧
    TaskPrefix .easy .depth [1, 9, 10, 9, 10] ∧
    ([1, 9, 9, 10, 10] : Tokens).Perm [1, 9, 10, 9, 10] ∧
    taskNext .easy .depth [1, 9, 9, 10, 10] = accept ∧
    taskNext .easy .depth [1, 9, 10, 9, 10] = reject := by
  refine ⟨?_, ?_, by decide, by decide, by decide⟩
  · exact ⟨[some false, some false, some true, some true], by decide, by decide, rfl⟩
  · exact ⟨[some false, some true, some false, some true], by decide, by decide, rfl⟩

/-- Every correct easy depth predictor must preserve an order distinction lost by bag encoding.
Source: the same-bag raw depth pair, without any transformer-weight existence claim. -/
theorem depth_requires_order (next : Tokens → ℤ)
    (hsolve : SolvesTask (extend next) .easy .depth) :
    next [1, 9, 9, 10, 10] ≠ next [1, 9, 10, 9, 10] := by
  rcases depth_order_pair with ⟨hx, hy, _, ha, hr⟩
  apply answer_separation next .easy .depth hsolve _ _ hx hy
  rw [ha, hr]
  decide

example : SolvesTask (extend (taskNext .easy .depth)) .easy .depth := fun _ _ => rfl

/-- An actual eight-write MQAR prefix, exposing the first two bound values as arguments.
Source: Basis easy recall's eight distinct keys and raw adjacent key/value format. -/
def bindingPrefix (first second : ℤ) : Tokens :=
  [1, 36, first, 37, second, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36]

/-- Same query and same token multiset can require different answers in actual Basis easy recall.
Source: swapping the first two values in the real eight-record table, not a reduced two-record task. -/
theorem recall_binding_pair :
    TaskPrefix .easy .recall (bindingPrefix 292 293) ∧
    TaskPrefix .easy .recall (bindingPrefix 293 292) ∧
    (bindingPrefix 292 293).Perm (bindingPrefix 293 292) ∧
    taskNext .easy .recall (bindingPrefix 292 293) = 292 ∧
    taskNext .easy .recall (bindingPrefix 293 292) = 293 := by
  exact ⟨⟨by decide, 292, by decide⟩, ⟨by decide, 293, by decide⟩,
    by decide, by decide, by decide⟩

/-- A correct MQAR predictor must preserve which adjacent value is bound to the queried key.
Source: the actual easy-mode raw-binding counterexample and the necessary answer separation theorem. -/
theorem recall_requires_binding (next : Tokens → ℤ)
    (hsolve : SolvesTask (extend next) .easy .recall) :
    next (bindingPrefix 292 293) ≠ next (bindingPrefix 293 292) := by
  rcases recall_binding_pair with ⟨hx, hy, _, ha, hb⟩
  apply answer_separation next .easy .recall hsolve _ _ hx hy
  rw [ha, hb]
  decide

example : SolvesTask (extend (taskNext .easy .recall)) .easy .recall := fun _ _ => rfl

/-- An actual sixteen-write hard prefix with two chronological values of query key 36.
Source: Basis hard recall's eight original writes followed by eight changed-value rewrites. -/
def rewritePrefix (first last : ℤ) : Tokens :=
  [1, 36, first, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299,
    36, last, 37, 301, 38, 302, 39, 303, 40, 304, 41, 305, 42, 306, 43, 307, 36]

/-- Last-write order changes the hard-mode answer although all raw tokens are preserved as a multiset.
Source: Basis hard MQAR's overwrite oracle on its actual sixteen-record configuration. -/
theorem recall_overwrite_pair :
    TaskPrefix .hard .recall (rewritePrefix 292 300) ∧
    TaskPrefix .hard .recall (rewritePrefix 300 292) ∧
    (rewritePrefix 292 300).Perm (rewritePrefix 300 292) ∧
    taskNext .hard .recall (rewritePrefix 292 300) = 300 ∧
    taskNext .hard .recall (rewritePrefix 300 292) = 292 := by
  exact ⟨⟨by decide, 300, by decide⟩, ⟨by decide, 292, by decide⟩,
    by decide, by decide, by decide⟩

/-- Correct hard recall predictions must distinguish the chronological overwrite order.
Source: the actual hard-mode pair and the general necessary separation condition. -/
theorem recall_requires_last_write (next : Tokens → ℤ)
    (hsolve : SolvesTask (extend next) .hard .recall) :
    next (rewritePrefix 292 300) ≠ next (rewritePrefix 300 292) := by
  rcases recall_overwrite_pair with ⟨hx, hy, _, ha, hb⟩
  apply answer_separation next .hard .recall hsolve _ _ hx hy
  rw [ha, hb]
  decide

example : SolvesTask (extend (taskNext .hard .recall)) .hard .recall := fun _ _ => rfl

/-- Equal-length legal bit prompts require different answers after changing one bit.
Source: Basis parity's no-scratchpad, one-bit valid input and its actual integer vocabulary. -/
theorem parity_bit_pair :
    TaskPrefix .easy .parity (parityPrompt [false]) ∧
    TaskPrefix .easy .parity (parityPrompt [true]) ∧
    taskNext .easy .parity (parityPrompt [false]) = evenToken ∧
    taskNext .easy .parity (parityPrompt [true]) = oddToken := by
  exact ⟨⟨[false], by decide, by decide, by decide, Or.inl rfl⟩,
    ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩, by decide, by decide⟩

/-- A correct parity predictor must respond to the bit content, even with the same BOS/SEP positions.
Source: the actual legal one-bit pair; causal position handling alone does not imply this condition. -/
theorem parity_requires_bits (next : Tokens → ℤ)
    (hsolve : SolvesTask (extend next) .easy .parity) :
    next (parityPrompt [false]) ≠ next (parityPrompt [true]) := by
  rcases parity_bit_pair with ⟨hx, hy, he, ho⟩
  apply answer_separation next .easy .parity hsolve _ _ hx hy
  rw [he, ho]
  decide

example : SolvesTask (extend (taskNext .easy .parity)) .easy .parity := fun _ _ => rfl

/-- A bag-invariant answer function cannot solve actual easy recall.
Source: the validated same-bag eight-write pair, isolating the encoder limitation from optimization. -/
theorem recall_not_bag_solver (next : Tokens → ℤ) (hbag : PermutationInvariant next) :
    ¬SolvesTask (extend next) .easy .recall := by
  intro hsolve
  exact recall_requires_binding next hsolve
    (hbag _ _ recall_binding_pair.2.2.1)

example : PermutationInvariant (fun tokens => (tokens.length : ℤ)) := by
  intro x y h
  exact congrArg (fun n : ℕ => (n : ℤ)) h.length_eq

end Transformer.Basis

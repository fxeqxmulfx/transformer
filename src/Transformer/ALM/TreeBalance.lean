/-
# Rebalancing moves nodes and not the order they are in

`Transformer.ALM.TreeQuery` prices and verifies the descent
`Tree::lower_bound_slope`, but `lowerBound_eq_find` asks for a hypothesis the
descent cannot check: that the predicate rises along `t.toList`, which is the
tree's ordering read through the test.  A freshly built tree has it.  What
`tree.rs` does between two queries is `fix_insert` and `fix_erase`, three
hundred lines of pointer surgery whose correctness is checked by `audit` in the
tests and by nothing else — and if a rotation could move a node past its
neighbour, the hypothesis would be false and the answer wrong, not merely late.

It cannot, and the reason is structural rather than arithmetic.  Read
`tree.rs` lines 520-680: every write those two functions perform is either
`lk_mut(_).red = _`, which is a colour, or a call to `rotate_left` /
`rotate_right`, which is the standard rotation.  So each step is `Step` below,
each acts on one subtree at one position (`RBNode.Path`, Batteries' zipper for
this same tree type), and `toList_of_stepAt` is that none of them changes the
in-order sequence.  `toList_of_rebalances` closes that under any number of
steps, and `lowerBound_eq_find_of_rebalances` is the conclusion the container
needs: after a rebalancing, the descent returns what a search of the *original*
order returns.

What this does not say is that the tree is still balanced — the colours are
carried but nothing is proved about them, and `the_rotation_moves_the_depth`
is a rotation that changes the depth while fixing the order, so the two
questions really are separate.  The depth bound `lbCount_le_two_log` still
holds of trees satisfying the red-black invariant and still rests on `audit`
for the claim that these are such trees.  The gap that leaves is a cost, not
an answer.

Source: `vm-rs/alm-hull/src/tree.rs`, `rotate_left`, `rotate_right`,
`fix_insert`, `fix_erase`.
-/

import Transformer.ALM.TreeQuery
import Mathlib.Logic.Relation

namespace Transformer
namespace ALM

open RBTree RBNode

variable {α : Type*}

/-! ### The two writes the rebalancing is made of -/

/-- **`rotate_left`.**  `y` is the right child, `b` is `y`'s left child; `b`
becomes the right child of `x` and `x` becomes the left child of `y`.  The
colours ride along untouched: `fix_insert` sets them in its own statements,
never inside a rotation.  A node with no right child is left alone, which is
the `NIL` guard the caller keeps. -/
def rotateLeft : RBNode α → RBNode α
  | .node c a x (.node c' b y r) => .node c' (.node c a x b) y r
  | t => t

/-- **`rotate_right`**, the mirror of it. -/
def rotateRight : RBNode α → RBNode α
  | .node c (.node c' l y b) x r => .node c' l y (.node c b x r)
  | t => t

/-- **A recolouring**, which is every other write the two loops perform. -/
def paint (c : RBColor) : RBNode α → RBNode α
  | .nil => .nil
  | .node _ l v r => .node c l v r

/-- One write of `fix_insert` or `fix_erase`, as the code is able to perform
them: a rotation either way, or a colour. -/
inductive Step where
  | rotateLeft
  | rotateRight
  | paint (c : RBColor)

/-- The step, run. -/
def Step.run : Step → RBNode α → RBNode α
  | .rotateLeft => ALM.rotateLeft
  | .rotateRight => ALM.rotateRight
  | .paint c => ALM.paint c

/-! ### None of them moves a node past its neighbour -/

/-- A rotation is the associativity of the in-order sequence, and that is all
it is. -/
theorem toList_rotateLeft (t : RBNode α) : (rotateLeft t).toList = t.toList := by
  match t with
  | .nil => rfl
  | .node _ _ _ .nil => rfl
  | .node _ _ _ (.node _ _ _ _) => simp [rotateLeft, toList_node]

/-- And so is the other one. -/
theorem toList_rotateRight (t : RBNode α) : (rotateRight t).toList = t.toList := by
  match t with
  | .nil => rfl
  | .node _ .nil _ _ => rfl
  | .node _ (.node _ _ _ _) _ _ => simp [rotateRight, toList_node]

/-- A colour is not a position. -/
theorem toList_paint (c : RBColor) (t : RBNode α) : (paint c t).toList = t.toList := by
  cases t <;> simp [paint, toList_node]

/-- **So no write of the rebalancing does.** -/
theorem toList_run (s : Step) (t : RBNode α) : (s.run t).toList = t.toList := by
  cases s with
  | rotateLeft => exact toList_rotateLeft t
  | rotateRight => exact toList_rotateRight t
  | paint c => exact toList_paint c t

/-! ### And a step is taken at a node, not at the root -/

/-- One step of the rebalancing, taken where the loop is standing: a subtree at
the end of some path is replaced by a rotation or a recolouring of itself.  The
path is Batteries' zipper for this tree, which is the `parent` chain `tree.rs`
walks. -/
inductive StepAt : RBNode α → RBNode α → Prop where
  | mk (p : Path α) (s : Step) (t : RBNode α) : StepAt (p.fill t) (p.fill (s.run t))

/-- A whole rebalancing: `fix_insert` and `fix_erase` walk up recolouring, and
stop after at most a rotation or two, so any number of steps. -/
abbrev Rebalances : RBNode α → RBNode α → Prop := Relation.ReflTransGen StepAt

/-- **A step leaves the order alone, wherever it is taken.**  The subtree's own
order is untouched, and the path contributes the same elements either side of
it. -/
theorem toList_of_stepAt {t t' : RBNode α} (h : StepAt t t') : t'.toList = t.toList := by
  obtain ⟨p, s, u⟩ := h
  rw [Path.fill_toList, Path.fill_toList, toList_run]

/-- **And so does a whole rebalancing.** -/
theorem toList_of_rebalances {t t' : RBNode α} (h : Rebalances t t') : t'.toList = t.toList := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => rw [toList_of_stepAt hstep, ih]

/-! ### Which is what the descent needs -/

/-- **The container answers the same after a rebalancing as before it.**  The
monotonicity `lowerBound_eq_find` asks for is a property of the in-order
sequence, the rebalancing does not touch that sequence, so the descent through
the rebalanced tree returns the first element the predicate accepts in the
order the envelope was built in.

This is the half of `tree.rs` the tests were carrying alone.  The other half,
that the tree is still *balanced*, is not here: see
`the_rotation_moves_the_depth`. -/
theorem lowerBound_eq_find_of_rebalances (p : α → Bool) {t t' : RBNode α}
    (h : Rebalances t t') (hmono : t.toList.Pairwise (fun x y => p x = true → p y = true))
    (acc : Option α) : lowerBound p t' acc = (t.toList.find? p).or acc := by
  have ht := toList_of_rebalances h
  rw [lowerBound_eq_find p t' acc (by rw [ht]; exact hmono), ht]

/-- **And the depth is a different question.**  One rotation of a three-node
chain: the order is the same and the depth is not, so nothing above can be
read as a statement about balance. -/
theorem the_rotation_moves_the_depth :
    let t : RBNode ℕ :=
      .node .black .nil 0 (.node .red .nil 1 (.node .red .nil 2 .nil))
    (rotateLeft t).toList = t.toList ∧ (rotateLeft t).depth ≠ t.depth := by
  refine ⟨toList_rotateLeft _, by decide⟩

/-! ### The hypotheses are satisfiable -/

/-- A two-node tree whose order is `[0, 1]`, rotated: `Rebalances` relates it to
its rotation, and the predicate `1 ≤ ·` rises along the order. -/
example :
    Rebalances (α := ℕ) (.node .black .nil 0 (.node .red .nil 1 .nil))
        (rotateLeft (.node .black .nil 0 (.node .red .nil 1 .nil))) ∧
      (RBNode.toList (α := ℕ) (.node .black .nil 0 (.node .red .nil 1 .nil))).Pairwise
        (fun x y => decide (1 ≤ x) = true → decide (1 ≤ y) = true) := by
  refine ⟨Relation.ReflTransGen.single ?_, by decide⟩
  exact .mk .root .rotateLeft _

end ALM
end Transformer

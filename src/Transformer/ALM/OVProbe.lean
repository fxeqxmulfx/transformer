/-
# Orthogonal Vectors for algorithms that can only test a pair

`Transformer.ALM.QueryModel` proves SETH for SAT algorithms whose only
operation is evaluating the formula on an assignment.  This file builds the
matching model on the Orthogonal Vectors side: an algorithm is an adaptive
decision tree whose only operation is testing one pair `(A i, B j)` for
orthogonality, and it costs the number of tests it makes — or the size of its
input, whichever is larger.

The `max` with `N · d` is not padding.  `Reduces.cost_ge_input` of
`Transformer.ALM.SETH` requires every algorithm to cost at least its input
size, *including* the incorrect ones, and a pure tree depth does not: the
one-node tree `done false` has depth `0`.  Charging `max (depth, N · d)` is
the least that satisfies it, and it changes nothing about the bound, since the
depths at stake are quadratic in `N`.

The reduction itself is `Transformer.ALM.Unconditional`.

Source of the model: the query/decision-tree model as used for fine-grained
lower bounds, e.g. V. Vassilevska Williams, *On some fine-grained questions in
algorithms and complexity*, ICM 2018, §8.
-/

import Transformer.ALM.QueryModel

namespace Transformer
namespace ALM

variable {d n N : ℕ}

/-! ### Orthogonality as a test the tree can perform -/

/-- Orthogonality, decided.  `orth_iff` makes it a finite conjunction of
Boolean conditions, so the tree has something it can actually branch on. -/
def orthB (a b : BVec d) : Bool := decide (∀ i, (a i && b i) = false)

lemma orthB_eq_true_iff (a b : BVec d) : orthB a b = true ↔ Orth a b := by
  rw [orthB, decide_eq_true_iff, orth_iff]

lemma orthB_eq_false_iff (a b : BVec d) : orthB a b = false ↔ ¬ Orth a b := by
  rw [← Bool.not_eq_true, orthB_eq_true_iff]

/-! ### The trees -/

/-- An adaptive decision tree over orthogonality tests on `N` vectors: name a
pair, branch on the answer, stop with a verdict. -/
inductive OProbe (N : ℕ) : Type
  /-- Stop and answer. -/
  | done : Bool → OProbe N
  /-- Test the pair `(i, j)` and continue on the answer. -/
  | ask : Fin N → Fin N → (Bool → OProbe N) → OProbe N

/-- Running a tree on an instance: each `ask` is answered by the instance. -/
def orun (A B : Fin N → BVec d) : OProbe N → Bool
  | .done b => b
  | .ask i j k => orun A B (k (orthB (A i) (B j)))

/-- The number of tests the tree makes in the worst case. -/
def odepth : OProbe N → ℕ
  | .done _ => 0
  | .ask _ _ k => 1 + max (odepth (k true)) (odepth (k false))

/-! ### Exhaustive search is such an algorithm -/

/-- Try the listed pairs in order, stopping at the first orthogonal one. -/
def oscan : List (Fin N × Fin N) → OProbe N
  | [] => .done false
  | p :: t => .ask p.1 p.2 fun b => bif b then .done true else oscan t

/-- And it is correct on the pairs it lists. -/
theorem orun_oscan (A B : Fin N → BVec d) :
    ∀ L : List (Fin N × Fin N),
      orun A B (oscan L) = true ↔ ∃ p ∈ L, Orth (A p.1) (B p.2) := by
  intro L
  induction L with
  | nil => simp [oscan, orun]
  | cons p t ih =>
      rw [oscan, orun]
      by_cases hp : Orth (A p.1) (B p.2)
      · rw [(orthB_eq_true_iff (A p.1) (B p.2)).mpr hp]
        exact iff_of_true rfl ⟨p, List.mem_cons_self, hp⟩
      · rw [(orthB_eq_false_iff (A p.1) (B p.2)).mpr hp]
        show orun A B (oscan t) = true ↔ _
        rw [ih]
        constructor
        · rintro ⟨q, hq, horth⟩
          exact ⟨q, List.mem_cons_of_mem _ hq, horth⟩
        · rintro ⟨q, hq, horth⟩
          rcases List.mem_cons.mp hq with rfl | hq'
          · exact absurd horth hp
          · exact ⟨q, hq', horth⟩

/-- Its depth is exactly the number of pairs it lists: one test each, and the
`true` branch stops immediately. -/
theorem odepth_oscan : ∀ L : List (Fin N × Fin N), odepth (oscan L) = L.length := by
  intro L
  induction L with
  | nil => rfl
  | cons p t ih =>
      show 1 + max (odepth (OProbe.done true)) (odepth (oscan t)) = t.length + 1
      rw [ih]
      show 1 + max 0 t.length = t.length + 1
      omega

/-- Exhaustive scan over all pairs. -/
noncomputable def oscanAll (N : ℕ) : OProbe N := oscan Finset.univ.toList

/-- Its depth is `N²`: every pair, once. -/
theorem odepth_oscanAll (N : ℕ) : odepth (oscanAll N) = N * N := by
  rw [oscanAll, odepth_oscan, Finset.length_toList, Finset.card_univ,
    Fintype.card_prod, Fintype.card_fin]

/-! ### The model -/

/-- The cost of a tree on `N` vectors of dimension `d`: the tests it makes, or
the input it must read, whichever is larger. -/
noncomputable def ovProbeCost (t : (N : ℕ) → OProbe N) (N d : ℕ) : ℝ :=
  max (odepth (t N) : ℝ) ((N : ℝ) * (d : ℝ))

/-- **The black-box Orthogonal Vectors model.**  An algorithm is a family of
decision trees, one per instance size; it answers what its tree answers, and
it costs what the tree costs. -/
noncomputable def ovProbeModel : CostModel where
  Alg := (N : ℕ) → OProbe N
  decides := fun t => fun {_d _N} A B => orun A B (t _N) = true
  cost := ovProbeCost

lemma ovProbeModel_decides {t : (N : ℕ) → OProbe N} {d N : ℕ}
    (A B : Fin N → BVec d) :
    ovProbeModel.decides t A B ↔ orun A B (t N) = true := Iff.rfl

/-- **The model contains a correct algorithm.**  Without this every lower
bound below would hold vacuously, for want of anything to bound. -/
theorem ovProbeModel_solves_oscanAll : ovProbeModel.Solves oscanAll := by
  intro d N _ A B
  rw [ovProbeModel_decides, oscanAll, orun_oscan]
  constructor
  · rintro ⟨p, -, horth⟩
    exact ⟨p.1, p.2, horth⟩
  · rintro ⟨i, j, horth⟩
    exact ⟨(i, j), Finset.mem_toList.mpr (Finset.mem_univ _), horth⟩

/-- The cost is at least the input size, for every tree — correct or not. -/
lemma le_ovProbeCost (t : (N : ℕ) → OProbe N) (N d : ℕ) :
    (N : ℝ) * (d : ℝ) ≤ ovProbeCost t N d := le_max_right _ _

/-- And at least the depth, which is what the reduction bills. -/
lemma odepth_le_ovProbeCost (t : (N : ℕ) → OProbe N) (N d : ℕ) :
    (odepth (t N) : ℝ) ≤ ovProbeCost t N d := le_max_left _ _

/-- The model is not empty and the scan really does pay `N²`: on two vectors
in dimension one the exhaustive tree makes four tests. -/
example : odepth (oscanAll 2) = 4 := by rw [odepth_oscanAll]

end ALM
end Transformer

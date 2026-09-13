/-
# The whole chain, with nothing assumed

Every earlier link carried a hypothesis.  `Transformer.ALM.Hardness` assumed
Orthogonal Vectors is hard; `Transformer.ALM.SETH` replaced that by SETH,
which is open.  In the black-box model both disappear.

`Transformer.ALM.QueryModel` proves SETH for SAT algorithms that can only
evaluate the formula.  `Transformer.ALM.OVProbe` builds the matching
Orthogonal Vectors model.  What is missing is the bridge, and Williams'
reduction supplies it verbatim: an orthogonality test on the constructed
instance *is* an evaluation of the formula (`orth_aVec_bVec_iff`), so
`toProbe` rewrites an Orthogonal Vectors tree as a SAT tree of the same depth.
`OVHard_of_SETH` then applies with its hypothesis discharged, and
`query_ge_of_build_small_black_box` is the barrier with no conjecture in it.

What this does and does not say.  It is a genuine, unconditional lower bound,
and it is exactly as strong as its model: it constrains indices that learn
about the keys only by asking whether a given key answers a given query.  The
planar hull of `transformer_vm/attention/hull2d_cht.h` reads coordinates, so
it is outside the model and untouched — as it must be, since
`Transformer.ALM.FixedDim` shows no bound at fixed dimension can hold.  Nor
does the result lift: a lower bound in a black-box model does not imply one
for algorithms that see the input (Baker–Gill–Solovay).  SETH stays open.
-/

import Transformer.ALM.OVProbe

namespace Transformer
namespace ALM

/-! ### An orthogonality test is an evaluation of the formula -/

/-- **The two tests coincide.**  On the instance Williams' reduction builds,
asking whether two vectors are orthogonal is asking whether the two halves
satisfy the formula.  This is `orth_aVec_bVec_iff` as an equality of the
Booleans the two trees branch on. -/
lemma satB_eq_orthB {n m : ℕ} (φ : CNF n m) (α β : Fin n → Bool) :
    satB φ α β = orthB (aVec φ α) (bVec φ β) := by
  rw [Bool.eq_iff_iff, satB_eq_true_iff, orthB_eq_true_iff, orth_aVec_bVec_iff]

/-- Rewriting an Orthogonal Vectors tree as a satisfiability tree: each test
of the pair `(i, j)` becomes an evaluation on the assignments those indices
enumerate. -/
noncomputable def toProbe {n : ℕ} : OProbe (2 ^ n) → Probe n
  | .done b => .done b
  | .ask i j k => .ask (assign n i) (assign n j) fun b => toProbe (k b)

/-- The rewritten tree answers what the original does. -/
theorem run_toProbe {n m : ℕ} (φ : CNF n m) :
    ∀ t : OProbe (2 ^ n),
      run φ (toProbe t) =
        orun (fun i => aVec φ (assign n i)) (fun j => bVec φ (assign n j)) t := by
  intro t
  induction t with
  | done b => rfl
  | ask i j k ih =>
      show run φ (toProbe (k (satB φ (assign n i) (assign n j)))) = _
      rw [satB_eq_orthB]
      exact ih _

/-- And makes exactly as many tests. -/
theorem depth_toProbe {n : ℕ} : ∀ t : OProbe (2 ^ n), depth (toProbe t) = odepth t := by
  intro t
  induction t with
  | done b => rfl
  | ask i j k ih =>
      show 1 + max (depth (toProbe (k true))) (depth (toProbe (k false)))
        = 1 + max (odepth (k true)) (odepth (k false))
      rw [ih true, ih false]

/-! ### The reduction, in the black-box model -/

/-- **Williams' reduction between the two black-box models.**  Every field is
a theorem here: correctness from `run_toProbe`, the cost bound from
`depth_toProbe`, and `cost_ge_input` from the `max` in `ovProbeCost` — which
is precisely why that `max` is there. -/
noncomputable def ovReduces : Reduces probeModel ovProbeModel where
  alg := fun t => fun n => toProbe (t (2 ^ n))
  decides_reduce := fun t {n _m} φ => by
    show run φ (toProbe (t (2 ^ n))) = true ↔
      orun (fun i => aVec φ (assign n i)) (fun j => bVec φ (assign n j))
        (t (2 ^ n)) = true
    rw [run_toProbe]
  cost_le := fun t n m => by
    show (depth (toProbe (t (2 ^ n))) : ℝ) ≤ (2 : ℝ) ^ n * m + ovProbeCost t (2 ^ n) m
    rw [depth_toProbe]
    have hmain : (odepth (t (2 ^ n)) : ℝ) ≤ ovProbeCost t (2 ^ n) m :=
      odepth_le_ovProbeCost t (2 ^ n) m
    have hpos : (0 : ℝ) ≤ (2 : ℝ) ^ n * m := by positivity
    linarith
  cost_ge_input := fun t N d => le_ovProbeCost t N d

/-- **Orthogonal Vectors is hard in the black-box model**, unconditionally.
`OVHard_of_SETH` with its hypothesis discharged by `probeModel_SETH`. -/
theorem ovProbeModel_OVHard : ovProbeModel.OVHard :=
  OVHard_of_SETH probeModel_SETH ovReduces

/-! ### The barrier -/

/-- **The lower bound on exact lookup, with no conjecture.**  A black-box
index whose preprocessing stays within half the budget must spend `n^{1-ε}/2`
on a single query at dimension `Θ(log n)`, for arbitrarily large `n`.

This is `query_ge_of_build_small_of_SETH` of `Transformer.ALM.SETH` with the
last hypothesis removed. -/
theorem query_ge_of_build_small_black_box {I : NNIndex}
    (hI : ovProbeModel.Implements I) {ε : ℝ} (hε : 0 < ε)
    (hbuild : ∀ n d : ℕ, I.build n d ≤ (n : ℝ) ^ (2 - ε) / 2) :
    ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
      (n : ℝ) ^ (1 - ε) / 2 ≤
        I.query n (c * Nat.log 2 n + c * Nat.log 2 n) :=
  query_ge_of_build_small ovProbeModel_OVHard hI hε hbuild

/-! ### The hypotheses are satisfiable -/

/-- The exhaustive scan as an index, charged for its comparisons as well as
its reads: `n` keys of `m` coordinates, plus one comparison per key.  This is
what `bruteForce` costs when the comparisons are counted too, and it is what
the black-box model bills for `oscanAll`. -/
noncomputable def scanIndex : NNIndex :=
  { bruteForce with query := fun n m => (n : ℝ) * (m : ℝ) + (n : ℝ) }

/-- The black-box model implements it, through the exhaustive tree. -/
noncomputable def ovProbeModel_implements : ovProbeModel.Implements scanIndex where
  alg := oscanAll
  decides_reduce := fun A B =>
    (ovProbeModel_solves_oscanAll A B).trans (scanIndex.reduce_iff A B).symm
  cost_le := fun n d => by
    show max (odepth (oscanAll n) : ℝ) ((n : ℝ) * (d : ℝ))
      ≤ 0 + (n : ℝ) * ((n : ℝ) * (((d + d : ℕ)) : ℝ) + (n : ℝ))
    rw [odepth_oscanAll]
    have hn : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
    have hd : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
    push_cast
    refine max_le (by nlinarith) ?_
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp
    · have h1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hpos
      have hnd : (d : ℝ) ≤ (n : ℝ) * (d : ℝ) := by nlinarith
      have h2 : (d : ℝ) ≤ (n : ℝ) * ((d : ℝ) + (d : ℝ)) + (n : ℝ) := by linarith
      have key := mul_le_mul_of_nonneg_left h2 hn
      linarith

/-- So the barrier is not a statement about nothing: an index and a black-box
implementation of it exist, with preprocessing well inside the budget. -/
example : ∃ c : ℕ, ∀ N : ℕ, ∃ n, N ≤ n ∧
    (n : ℝ) ^ (1 - (1 / 2 : ℝ)) / 2 ≤
      scanIndex.query n (c * Nat.log 2 n + c * Nat.log 2 n) :=
  query_ge_of_build_small_black_box ovProbeModel_implements (by norm_num)
    (fun n d => by
      show (0 : ℝ) ≤ (n : ℝ) ^ (2 - (1 / 2 : ℝ)) / 2
      positivity)

end ALM
end Transformer

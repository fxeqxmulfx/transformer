/-
# A model where the two hardness hypotheses hold together

`Transformer.ALM.SATModel.SETH_of_general` assumes both `SETHGeneral` and a
`Sparsification`.  Contradictory hypotheses would make it provable and
worthless, so this file exhibits a model satisfying both at once.

The model is degenerate in that every one of its algorithms is exponential —
it is a witness for satisfiability, not evidence for SETH.  It is not
degenerate in the way that matters here: its cost grows with the clause count,
so the identity does *not* satisfy `Sparsification.cost_le` and the wrapper
has to be a genuinely different algorithm.
-/

import Transformer.ALM.Sparsification

namespace Transformer
namespace ALM

/-- `b ≤ t` survives multiplication of the larger side by a factor `≥ 1`. -/
private lemma le_mul_of_one_le_factor {c b t : ℝ} (hc : 1 ≤ c) (hb : 0 ≤ b) (h : b ≤ t) :
    b ≤ c * t := by
  have := mul_nonneg (sub_nonneg.mpr hc) (hb.trans h)
  linarith

/-- A model in which the sparsifying wrapper is a genuinely different
algorithm with the cost the lemma prescribes.  An algorithm *is* the rate `ε`
its wrapper runs at, `0` being the plain scan; the cost grows with the clause
count, so sparsification is doing real work here and the identity would not
satisfy `Sparsification.cost_le`. -/
noncomputable def sparseModel : SATModel where
  Alg := ℝ
  decides := fun _ => fun φ => Satisfiable φ
  cost := fun ε n m =>
    (2 : ℝ) ^ (max ε 0 * (n : ℝ))
      * ((n : ℝ) * m + (2 : ℝ) ^ (2 * (n : ℝ)) * ((n : ℝ) + 1))

/-- The wrapper's factor never helps the algorithm. -/
private lemma one_le_sparse_factor (ε : ℝ) (n : ℕ) :
    (1 : ℝ) ≤ (2 : ℝ) ^ (max ε 0 * (n : ℝ)) := by
  have := Real.rpow_le_rpow_of_exponent_le (by norm_num : (1:ℝ) ≤ 2)
    (mul_nonneg (le_max_right ε 0) (Nat.cast_nonneg n))
  simpa using this

/-- Every algorithm of the model costs at least `2^{2n}`. -/
lemma sparseModel_cost_ge (ε : ℝ) (n m : ℕ) :
    (2 : ℝ) ^ (2 * (n : ℝ)) ≤ sparseModel.cost ε n m := by
  have hx : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hp : (0 : ℝ) < (2 : ℝ) ^ (2 * (n : ℝ)) := Real.rpow_pos_of_pos (by norm_num) _
  refine le_mul_of_one_le_factor (one_le_sparse_factor ε n) hp.le ?_
  have hnm : (0 : ℝ) ≤ (n : ℝ) * m := by positivity
  nlinarith

/-- `SETHGeneral` holds in it — trivially, its algorithms all being
exponential.  This is not evidence for SETH; it witnesses that the hypothesis
of `SETH_of_general` is satisfiable. -/
example : sparseModel.SETHGeneral := by
  intro δ hδ
  refine ⟨0, fun a _ N => ⟨max N 1, 0, le_max_left _ _, Nat.zero_le _, ?_⟩⟩
  have hx : (0 : ℝ) ≤ ((max N 1 : ℕ) : ℝ) := Nat.cast_nonneg _
  exact le_trans (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by nlinarith))
    (sparseModel_cost_ge a (max N 1) 0)

/-- And the sparsification is available in it, so both hypotheses of
`SETH_of_general` hold at once. -/
example : Sparsification sparseModel where
  dens := fun _ => 1
  alg := fun ε _ => ε
  solves := fun _ _ _ => fun _ => Iff.rfl
  cost_le := fun ε hε a n m => by
    have hx : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
    have hp : (0 : ℝ) < (2 : ℝ) ^ (2 * (n : ℝ)) := Real.rpow_pos_of_pos (by norm_num) _
    have hbase : (2 : ℝ) ^ (2 * (n : ℝ)) * ((n : ℝ) + 1)
        ≤ sparseModel.cost a n (1 * n) := by
      refine le_mul_of_one_le_factor (one_le_sparse_factor a n) (by positivity) ?_
      have hnn : (0 : ℝ) ≤ (n : ℝ) * (((1 * n : ℕ)) : ℝ) := by positivity
      linarith
    show (2 : ℝ) ^ (max ε 0 * (n : ℝ))
        * ((n : ℝ) * m + (2 : ℝ) ^ (2 * (n : ℝ)) * ((n : ℝ) + 1))
      ≤ (2 : ℝ) ^ (ε * (n : ℝ)) * ((n : ℝ) * m + sparseModel.cost a n (1 * n))
    rw [max_eq_left hε.le]
    exact mul_le_mul_of_nonneg_left (by linarith)
      (Real.rpow_pos_of_pos (by norm_num) _).le

end ALM
end Transformer

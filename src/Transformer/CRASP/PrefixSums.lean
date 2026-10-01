/-
# Reindexing a masked prefix by natural positions

arXiv:2506.16055v3, Appendix B.2 and F: a masked attention sum is one
BOS contribution plus the contributions at positions `1,...,i`.
-/

import Transformer.CRASP.TransformerPrefix

namespace Transformer.CRASP.RTfr

/-- Reindex a future-masked sum, separating its initial position (B.2/F). -/
theorem sum_masked_nat {n : ℕ} (i : Fin n) (f : ℕ → ℝ) :
    (∑ j ∈ masked i, f j.val) = f 0 + ∑ j ∈ Finset.Icc 1 i.val, f j := by
  have hsum : (∑ j ∈ masked i, f j.val) = ∑ j ∈ Finset.Icc 0 i.val, f j := by
    refine Finset.sum_bij (fun j _ => j.val) ?_ ?_ ?_ ?_
    · intro j hj
      simp only [masked, Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_def] at hj
      exact Finset.mem_Icc.mpr ⟨Nat.zero_le _, hj⟩
    · intro j hj t ht heq
      exact Fin.ext heq
    · intro j hj
      have hjle := (Finset.mem_Icc.mp hj).2
      refine ⟨⟨j, lt_of_le_of_lt hjle i.isLt⟩, ?_, rfl⟩
      simpa only [masked, Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_def] using hjle
    · intro j hj
      rfl
  have hsplit : Finset.Icc 0 i.val = insert 0 (Finset.Icc 1 i.val) := by
    ext j
    simp only [Finset.mem_Icc, Finset.mem_insert]
    omega
  rw [hsum, hsplit, Finset.sum_insert (by simp)]

end Transformer.CRASP.RTfr

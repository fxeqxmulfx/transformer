/-
# Complete causal recall on aligned key-value pairs

Arora et al., arXiv:2312.04927v1, §4, Proposition `prop: attention-ar`,
and Appendix Proposition `prop: app-attention`.  After the paper's first
local-shift layer has associated each value with its key, one unnormalized
causal attention layer computes every MQAR answer and zero on no-match rows.
The proof assumes distinct keys, as does the synthetic generator.
-/

import Transformer.Zoology.Appendix_Attention

namespace Transformer.Zoology

/-- One-hot encoding of the preceding-key answer relation, with the zero
vector at queries having no prior match.  Source: Appendix
`sec: intro-general-ar`, Setup. -/
noncomputable def expectedPairedAnswer {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : ℝ := by
  classical
  exact if PriorAnswer x i v then 1 else 0

/-- For every query and every value coordinate, paired causal attention
returns the indicator that the value is associated to an earlier matching
key.  This includes the no-match case and any number of query positions.
Source: Appendix Proposition `prop: app-attention`, corrected with the
causal mask and unique-key hypothesis. -/
theorem paired_attention_solves_mqar {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (i : Fin n) (v : Fin c) :
    pairedAttentionOutput x i v = expectedPairedAnswer x i v := by
  classical
  by_cases hmatch : ∃ j : Fin n, j < i ∧ x.key j = x.query i
  · obtain ⟨j, hj, hkey⟩ := hmatch
    rw [paired_attention_exact x hx i j hj hkey v]
    have hiff : PriorAnswer x i v ↔ x.value j = v := by
      constructor
      · rintro ⟨k, _, hk, hv⟩
        have hkj : k = j := hx (hk.trans hkey.symm)
        simpa [hkj] using hv
      · intro hv
        exact ⟨j, hj, hkey, hv⟩
    by_cases hv : x.value j = v
    · simp [oneHot, hv, expectedPairedAnswer, hiff.mpr hv]
    · have hnoanswer : ¬ PriorAnswer x i v := hiff.not.mpr hv
      have hrev : v ≠ x.value j := Ne.symm hv
      simp [oneHot, expectedPairedAnswer, hnoanswer, hrev]
  · have hnone : ∀ j : Fin n, j < i → x.key j ≠ x.query i := by
      intro j hj hkey
      exact hmatch ⟨j, hj, hkey⟩
    rw [paired_attention_no_match x i hnone v]
    have hnoanswer : ¬ PriorAnswer x i v := by
      rintro ⟨j, hj, hkey, _⟩
      exact hnone j hj hkey
    simp [expectedPairedAnswer, hnoanswer]

/-- Distinct-key inputs with a prior query are satisfiable: the second query
may refer to the first key. -/
example : ∃ x : MQARInstance 2 2, UniqueKeys x ∧
    PriorAnswer x 1 0 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_⟩
  · intro i j h
    exact h
  · exact ⟨0, by decide, rfl, rfl⟩

end Transformer.Zoology

/-
# One-hot autocorrelation and the distance-selection gap

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`,
Autocorrelations and The Input-Dependent Kernels. A one-hot cyclic
autocorrelation counts equal-token pairs at each raw-token lag. The concrete
example shows that a needed key-query lag need not maximize that count:
unrelated repeated values and queries can create a larger peak. Thus a
`Top` selector requires an additional coverage argument or assumption.
-/

import Transformer.Zoology.Appendix_RequiredDistances

open scoped BigOperators

namespace Transformer.Zoology

/-- Flatten an untagged key-value-query triple sequence into raw token
positions `3j`, `3j+1`, `3j+2`. Source: Appendix equation
`eq: kqv-indices` and `sec: data-dep-ar`, Setup. -/
def flattenTriples {n c : ℕ} (x : MQARInstance n c) :
    TokenSequence (3 * n) c := fun p =>
  let j : Fin n := ⟨p.val / 3, by omega⟩
  if p.val % 3 = 0 then x.key j
  else if p.val % 3 = 1 then x.value j
  else x.query j

/-- Equal-token pair count at a cyclic raw-token lag. For one-hot tokens,
this is the sum over feature coordinates of the cyclic autocorrelation.
Source: Appendix `sec: data-dep-ar`, Autocorrelations. -/
def cyclicMatchCount {n c : ℕ} (u : TokenSequence n c)
    (lag : Fin n) : ℕ :=
  ∑ i : Fin n, if u i = u (i + lag) then 1 else 0

/-- The corresponding real-valued one-hot autocorrelation.
Source: Appendix `sec: data-dep-ar`, displayed autocorrelation formula. -/
def cyclicOneHotCorrelation {n c : ℕ} (u : TokenSequence n c)
    (lag : Fin n) : ℝ :=
  ∑ i : Fin n, ∑ q : Fin c,
    oneHot (u i) q * oneHot (u (i + lag)) q

/-- One-hot cyclic autocorrelation is exactly an equal-token pair count.
Source: Appendix `sec: data-dep-ar`, Autocorrelations. -/
theorem cyclicOneHotCorrelation_eq_count {n c : ℕ}
    (u : TokenSequence n c) (lag : Fin n) :
    cyclicOneHotCorrelation u lag = cyclicMatchCount u lag := by
  classical
  simp [cyclicOneHotCorrelation, cyclicMatchCount, oneHot_inner]

/-- Three triples with unique keys. Only the last query recalls an earlier
key, at tuple distance two and raw-token lag eight. The repeated values and
nonmatching queries create a separate autocorrelation peak. -/
def autocorrelationExample : MQARInstance 3 5 := {
  key := fun i => ⟨i.val, by omega⟩
  value := fun _ => 3
  query := fun i => if i = 2 then 0 else 4
}

/-- The example has distinct keys, and its final query recalls the value
associated with the first key. Source: Appendix `sec: data-dep-ar`, Setup. -/
theorem autocorrelationExample_valid :
    UniqueKeys autocorrelationExample ∧
      PriorAnswer autocorrelationExample 2 3 := by
  constructor
  · intro i j h
    apply Fin.ext
    simpa [autocorrelationExample] using congrArg Fin.val h
  · exact ⟨0, by decide, rfl, rfl⟩

/-- The example has exactly one required tuple interaction distance.
Source: Appendix `sec: data-dep-ar`, Setup. -/
theorem autocorrelationExample_distances :
    requiredDistances autocorrelationExample = {2} := by
  decide

/-- Unrelated raw-token pairs create a larger one-hot autocorrelation peak
at lag three than the required key-query lag eight. Source: Appendix
`sec: data-dep-ar`, proposed autocorrelation `Top` selector. -/
theorem autocorrelationExample_scores :
    cyclicMatchCount (flattenTriples autocorrelationExample) 3 = 4 ∧
      cyclicMatchCount (flattenTriples autocorrelationExample) 8 = 1 := by
  decide

/-- Therefore the required raw-token lag eight is not even a maximizing
positive lag for this valid unique-key MQAR instance. This disproves the
unqualified inference that `Top` of raw one-hot autocorrelation necessarily
contains each needed interaction lag. It does not rule out a different
input-dependent selector. Source: Appendix `sec: data-dep-ar`, paragraph
“The Input-Dependent Kernels.” -/
theorem required_lag_not_autocorrelation_maximum :
    ¬ ∀ lag : Fin 9, 0 < lag.val →
      cyclicMatchCount (flattenTriples autocorrelationExample) lag ≤
        cyclicMatchCount (flattenTriples autocorrelationExample) 8 := by
  intro h
  have h3 := h 3 (by decide)
  obtain ⟨hscore3, hscore8⟩ := autocorrelationExample_scores
  rw [hscore3, hscore8] at h3
  omega

end Transformer.Zoology

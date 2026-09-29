/-
# Multi-query associative recall

Arora et al., arXiv:2312.04927v1, §3, Definition `def: general-AR`, and
Appendix `sec: intro-general-ar`.  The paper first describes recall on a raw
token sequence and then uses separate key, value, and query streams for the
theoretical analysis.  Both formulations are represented here.
-/

import Mathlib

namespace Transformer.Zoology

/-- A token sequence over a vocabulary of size `c`.  Source: §3, Definition
`def: general-AR`. -/
abbrev TokenSequence (n c : ℕ) := Fin n → Fin c

/-- The answers allowed by the raw-sequence wording of MQAR.  The printed
definition writes `u_i` where its input sequence is named `x_i`; we use `x`.
Its `j < i` condition does not specify which occurrence wins when a key has
several successors.  Source: §3, Definition `def: general-AR`. -/
def RawAnswer {n c : ℕ} (x : TokenSequence n c) (i : Fin n) (v : Fin c) : Prop :=
  ∃ j : Fin n, j < i ∧ ∃ h : j.val + 1 < n,
    x j = x i ∧ x ⟨j.val + 1, h⟩ = v

/-- A sequence of key, value, and query triples, as used in the appendix's
more general version of MQAR.  Source: Appendix `sec: intro-general-ar`,
Setup. -/
structure MQARInstance (n c : ℕ) where
  key : Fin n → Fin c
  value : Fin n → Fin c
  query : Fin n → Fin c

/-- A value associated to some earlier matching key.  This relation follows
the appendix's `j < i` quantifier literally; it permits several answers when
the input associates different values with the same key.  Source: Appendix
`sec: intro-general-ar`, Setup. -/
def PriorAnswer {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : Prop :=
  ∃ j : Fin n, j < i ∧ x.key j = x.query i ∧ x.value j = v

/-- Consistency of the key-value dictionary.  This is the additional
hypothesis needed for the paper's phrase "output the associated value" to
specify a unique answer when keys may recur.  The synthetic generator in
Appendix `app:synthetic` pairs each key with one value. -/
def Consistent {n c : ℕ} (x : MQARInstance n c) : Prop :=
  ∀ j k, x.key j = x.key k → x.value j = x.value k

/-- Under a consistent dictionary, every prior matching key yields the same
answer.  Source: Appendix `sec: intro-general-ar`, Setup, with the explicit
dictionary-consistency correction noted above. -/
theorem priorAnswer_unique {n c : ℕ} (x : MQARInstance n c)
    (hx : Consistent x) (i : Fin n) (v w : Fin c)
    (hv : PriorAnswer x i v) (hw : PriorAnswer x i w) : v = w := by
  obtain ⟨j, _, hj, rfl⟩ := hv
  obtain ⟨k, _, hk, rfl⟩ := hw
  exact hx j k (hj.trans hk.symm)

/-- The consistency hypothesis is satisfiable even with several keys. -/
example : Consistent ({
    key := fun _ : Fin 2 => (0 : Fin 1)
    value := fun _ : Fin 2 => (0 : Fin 1)
    query := fun _ : Fin 2 => (0 : Fin 1)
  } : MQARInstance 2 1) := by
  intro j k _
  rfl

/-- Without dictionary consistency, the appendix's MQAR relation can
require two different outputs at the same query.  This refutes a literal
single-valued interpretation of its unrestricted Setup.  Source: Appendix
`sec: intro-general-ar`, Setup. -/
theorem unrestricted_mqar_ambiguous :
    ∃ x : MQARInstance 3 2, ∃ i : Fin 3, ∃ v w : Fin 2,
      v ≠ w ∧ PriorAnswer x i v ∧ PriorAnswer x i w := by
  let x : MQARInstance 3 2 := {
    key := fun _ => 0
    value := fun j => if j = 1 then 1 else 0
    query := fun _ => 0
  }
  refine ⟨x, 2, 0, 1, by decide, ?_, ?_⟩
  · exact ⟨0, by decide, rfl, by decide⟩
  · exact ⟨1, by decide, rfl, by decide⟩

end Transformer.Zoology

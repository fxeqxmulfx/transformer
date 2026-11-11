module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Separation
public import Transformer.Zoology.Section4_PairedRecall

/-!+# Learned matching solves causal MQAR

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR5;
Arora et al., arXiv:2312.04927v1, §4 and Appendix, paired recall after
the local key/value shift. A directly learned matching matrix is applied
to each prior key and copies its associated value. The map is linear in
all parameters. Training on separating dictionaries with ordinary answer
CE identifies the matrix, so it solves every distinct-key paired MQAR
stream, of every length. The no-match output is zero, as in the existing
paired MQAR specification. Token parsing and the pairing shift are given.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators
open Zoology

/-- Causal value copying with the learned key matching scores (CR5).
Only input keys, input values, the query, and the causal mask occur. -/
def streamLogits {n c : ℕ} (A : Matrix (Fin c) (Fin c) ℝ)
    (x : MQARInstance n c) (i : Fin n) (v : Fin c) : ℝ :=
  ∑ j : Fin n, (if j < i then A (x.query i) (x.key j) else 0) * oneHot (x.value j) v

/-- The stream computation remains linear in the complete learned matrix
for every input and every causal query position (CR5). -/
theorem streamLogits_linear {n c : ℕ} (A B : Matrix (Fin c) (Fin c) ℝ)
    (a b : ℝ) (x : MQARInstance n c) (i : Fin n) (v : Fin c) :
    streamLogits (a • A + b • B) x i v =
      a * streamLogits A x i v + b * streamLogits B x i v := by
  simp only [streamLogits, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  by_cases h : j < i <;> simp [h, add_mul, mul_assoc]

/-- Identity matching is exactly the existing paired-attention operator,
including its causal mask; it is not assumed to solve recall (CR5). -/
theorem streamLogits_one {n c : ℕ} (x : MQARInstance n c) (i : Fin n) (v : Fin c) :
    streamLogits (1 : Matrix (Fin c) (Fin c) ℝ) x i v = pairedAttentionOutput x i v := by
  simp [streamLogits, pairedAttentionOutput, Matrix.one_apply, oneHot, eq_comm]

variable {S ν : Type*} [Fintype S] [Fintype ν] [DecidableEq ν] [Nontrivial ν]

/-- **Answer-CE training solves every aligned distinct-key MQAR stream.**
Training values may even use a different alphabet from stream values:
the learned matrix acts on keys; categorical copying has no parameters.
All training hypotheses concern the ordinary dictionary-answer objective
and explicit input separation (`review/flat_spectrum.md`, CR5). -/
theorem trained_solves_mqar {n c : ℕ} (query : S → Fin c) (data : S → Fin c → ν)
    (hsep : Separates query data) {A : Matrix (Fin c) (Fin c) ℝ}
    (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (x : MQARInstance n c) (hx : UniqueKeys x) (i : Fin n) (v : Fin c) :
    streamLogits A x i v = expectedPairedAnswer x i v := by
  rw [eq_one_of_answerLoss_le query data hsep hA (hmin stochastic_one), streamLogits_one]
  exact paired_attention_solves_mqar x hx i v

/-- Concrete, satisfiable training and stream hypotheses: a two-key
dictionary is queried at both keys; the second stream query recalls a
prior binding. -/
example :
    let x : MQARInstance 2 2 := { key := id, value := id, query := fun _ => 0 }
    streamLogits (1 : Matrix (Fin 2) (Fin 2) ℝ) x 1 0 = expectedPairedAnswer x 1 0 := by
  dsimp only
  exact trained_solves_mqar (id : Fin 2 → Fin 2) (fun _ => (id : Fin 2 → Fin 2))
    (fun q k hk => ⟨q, rfl, hk⟩) stochastic_one (isMinOn_answerLoss_one _ _)
    _ (fun _ _ h => h) 1 0

end Transformer.Grokking.FlatSpectrum.ConvexRecall

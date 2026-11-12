module

public import Transformer.Grokking.Protocol.SolverForward

/-!
# The solver answers every equation

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, the
decoder and the validation accuracy; openai/grok at 3d64b1d8, `grok/transformer.py`,
`Transformer.forward`, as `Model.logits`. The parameters are those of `SolverParams`.

After the two blocks the row at `=` is `solverScale` times the centred `markedRow`
(`solver_decode`): each of the four normalizations multiplies by a positive factor and takes
off a constant. The readout of the residue `r` is the entry `5 + r` less the entry `4`, so the
constant cancels, and the logits are `solverScale` times the logits that mark the answer
(`answerLogits_solver`): the answer has the logit `solverScale`, every other token `0`.
So the solver answers every equation (`predict_answerLogits_solver`), and the paper's decoder,
in exact reals with its causal attention, its normalizations and `max 0`, has parameters that
answer all 9312 equations (`exists_param_forall_predict`).

This says that the answers are in the range of the architecture. It says nothing about what
training reaches: `solver` is built, not trained.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- **The factor of the solver's logits**: that of the four normalizations between the
attention of the first block and the readout. -/
noncomputable def solverScale (x : Equation) : ℝ :=
  normScale (normScale (normScale (normScale 1 (gapRow x)) (markedRow x)) (markedRow x))
    (markedRow x)

lemma solverScale_pos (x : Equation) : 0 < solverScale x :=
  normScale_pos (normScale_pos (normScale_pos (normScale_pos one_pos _) _) _) _

/-- **The output of the second block at `=`**: a positive multiple of the centred row that
carries the mark of the quotient. -/
theorem solver_decode (x : Equation) (i : Fin 128) :
    decodeTo solver (relu paperSizes) (read x) 2 4 i =
      solverScale x * (markedRow x i - rowMean (markedRow x)) := by
  have e0 : block solver (relu paperSizes) (solverRows x) ⟨0, by decide⟩ 4 =
      fun i => normScale (normScale 1 (gapRow x)) (markedRow x) *
        (markedRow x i - rowMean (markedRow x)) := funext (solver_block_zero rfl x)
  have e2 : ∀ t : ℝ, layerNorm (fun _ => 1) (fun _ => 0)
      (fun i => t * (markedRow x i - rowMean (markedRow x))) =
      fun i => normScale t (markedRow x) * (markedRow x i - rowMean (markedRow x)) :=
    fun t => funext (layerNorm_mul_sub (by norm_num) t _ _)
  change block solver (relu paperSizes)
    (fun t' => block solver (relu paperSizes) (solverRows x) ⟨0, by decide⟩ t')
    ⟨1, by decide⟩ 4 i = _
  rw [solver_block_of_ne (by decide)]
  change layerNorm (fun _ => 1) (fun _ => 0) (layerNorm (fun _ => 1) (fun _ => 0)
    (block solver (relu paperSizes) (solverRows x) ⟨0, by decide⟩ 4)) i = _
  rw [e0, e2, e2]
  rfl

/-- The entry of the residue `r` in the marked row: whether `r` is the quotient. -/
lemma markedRow_answer (x : Equation) (r : Fin 97) (h : (r : ℕ) + 5 < 128) :
    markedRow x ⟨(r : ℕ) + 5, h⟩ = if r = x.1 then 1 else 0 := by
  have h2 : ((⟨(r : ℕ) + 5, h⟩ : Fin 128) : ℕ) ≠ 2 := fun e => by
    have e' : (r : ℕ) + 5 = 2 := e
    omega
  have h3 : ((⟨(r : ℕ) + 5, h⟩ : Fin 128) : ℕ) ≠ 3 := fun e => by
    have e' : (r : ℕ) + 5 = 3 := e
    omega
  rw [markedRow, gapRow_of_ne x h2 h3, zero_add, hiddenOut_logGap, ind]
  exact if_congr ⟨fun h' => Fin.ext (Nat.add_right_cancel h'),
    fun h' => congrArg (fun s : Fin 97 => (s : ℕ) + 5) h'⟩ rfl rfl

/-- The entry `4` of the marked row is zero: the readout subtracts it. -/
lemma markedRow_four (x : Equation) (h : 4 < 128) : markedRow x ⟨4, h⟩ = 0 := by
  rw [markedRow, gapRow_of_ne x (o := ⟨4, h⟩) (by norm_num) (by norm_num), zero_add,
    hiddenOut_logGap]
  exact ind_of_ne fun h' => by
    have e : 4 = ZMod.val (quotient x) + 5 := h'
    omega

/-- **The logits of the solver** are a positive multiple of the logits that mark the
answer. -/
theorem answerLogits_solver (x : Equation) (tok : Fin 239) :
    answerLogits solver x tok = solverScale x * markLogits x tok := by
  have hl : answerLogits solver x tok = ∑ i : Fin 128,
      solverOut tok i * decodeTo solver (relu paperSizes) (read x) 2 4 i := rfl
  rw [hl, Finset.sum_congr rfl fun i _ => by rw [solver_decode]]
  by_cases ht : ∃ r, residueToken r = tok
  · obtain ⟨r, rfl⟩ := ht
    have hr := r.isLt
    have e : ∀ i : Fin 128, solverOut (residueToken r) i *
        (solverScale x * (markedRow x i - rowMean (markedRow x))) =
        solverScale x * (ind ((r : ℕ) + 5) i * (markedRow x i - rowMean (markedRow x))) -
          solverScale x * (ind 4 i * (markedRow x i - rowMean (markedRow x))) := fun i => by
      rw [solverOut_residueToken]
      ring
    rw [Finset.sum_congr rfl fun i _ => e i, Finset.sum_sub_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, sum_ind_mul (by omega : (r : ℕ) + 5 < 128),
      sum_ind_mul (by norm_num : 4 < 128), markedRow_answer, markedRow_four, markLogits,
      if_congr (residueToken_injective.eq_iff (a := r) (b := x.1)) rfl rfl]
    ring
  · have h0 : ∀ i, solverOut tok i = 0 := solverOut_of_not_residue fun r hr => ht ⟨r, hr⟩
    rw [Finset.sum_congr rfl fun i _ => by rw [h0 i, zero_mul], Finset.sum_const_zero,
      markLogits, ite_eq_right fun h => ht ⟨x.1, h.symm⟩, mul_zero]

/-- The margin of the solver's answer is at least the factor of its logits. -/
lemma solverScale_le_margin (x : Equation) :
    solverScale x ≤ margin (answerLogits solver x) (residueToken x.1) := by
  have : Nonempty {c : Fin 239 // c ≠ residueToken x.1} :=
    ⟨⟨eosToken, (residueToken_ne_eosToken x.1).symm⟩⟩
  refine le_ciInf fun c => ?_
  rw [answerLogits_solver, answerLogits_solver, markLogits, markLogits, ite_eq_left rfl,
    ite_eq_right c.2]
  linarith

/-- **The solver answers every equation.** -/
theorem predict_answerLogits_solver (x : Equation) :
    predict (answerLogits solver x) = residueToken x.1 :=
  predict_eq_of_margin_pos ((solverScale_pos x).trans_le (solverScale_le_margin x))

/-- **The paper's decoder has parameters that answer all 9312 equations.** -/
theorem exists_param_forall_predict :
    ∃ θ : Param paperSizes → ℝ, ∀ x : Equation, predict (answerLogits θ x) = residueToken x.1 :=
  ⟨solver, predict_answerLogits_solver⟩

example := markedRow_answer (20, 4) 20 (by norm_num)

example := markedRow_four (20, 4) (by norm_num)

example := predict_answerLogits_solver (20, 4)

end Transformer.Grokking.Protocol

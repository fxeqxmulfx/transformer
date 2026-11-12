module

public import Transformer.Grokking.Protocol.SolverLogits

/-!
# The solver is certified by two of its answers

Source: `experiments/grok_division/README.md`, *The decoders that answer every equation are
not quotient-equivariant*, for the certificate; arXiv:2201.02177v1 (Power et al.), appendix
*Additional experimental details*, for the decoder. The parameters are those of
`SolverParams`.

The factor of the solver's logits depends on the equation, through the variance of the rows
that are normalized (`rowVar_gapRow`, `rowVar_markedRow`): its inverse square is explicit
(`inv_solverScale_sq`). The ballast `10000` makes the gap, at most `300`, a small part of
that variance, and the factor is between `1/900` and `1/870` for every equation
(`solverScale_mem`).

So the logits of two equations of an orbit differ, after the relabelling and a constant, by
at most `1/52200` (`abs_answerLogits_solver_scaleToken`), and the margins are at least
`1/900 > 2/52200`: the hypotheses of `predict_eq_residueToken_of_two_anchors` hold for the
decoder's own logits, with a positive error (`solver_certified`). The certificate is
therefore not empty on the architecture; whether training produces parameters that satisfy
it is not addressed here.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

lemma markedRow_eq (x : Equation) (o : Fin 128) :
    markedRow x o =
      ballast * ind 3 o + (logGap x : ℝ) * ind 2 o + 1 * ind ((quotient x).val + 5) o := by
  rw [markedRow, gapRow, hiddenOut_logGap, one_mul]

/-- The variance of the row that carries the mark. -/
lemma rowVar_markedRow (x : Equation) :
    rowVar (markedRow x) =
      ((ballast ^ 2 + (logGap x : ℝ) ^ 2 + 1 ^ 2) - (ballast + logGap x + 1) ^ 2 / 128) / 128 := by
  have hq : (quotient x).val + 5 < 128 := by
    have := ZMod.val_lt (quotient x)
    omega
  rw [rowVar_eq (by norm_num), Finset.sum_congr rfl fun j _ => by rw [markedRow_eq],
    Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => markedRow_eq x j,
    sum_sq_three (by norm_num) (by norm_num) hq (by norm_num) (by omega) (by omega),
    sum_three (by norm_num) (by norm_num) hq]
  norm_num

/-- The variance of the row after the attention. -/
lemma rowVar_gapRow (x : Equation) :
    rowVar (gapRow x) =
      ((ballast ^ 2 + (logGap x : ℝ) ^ 2 + 0 ^ 2) - (ballast + logGap x + 0) ^ 2 / 128) / 128 := by
  have e : ∀ o : Fin 128, gapRow x o =
      ballast * ind 3 o + (logGap x : ℝ) * ind 2 o + 0 * ind 4 o := fun o => by
    rw [gapRow]
    ring
  rw [rowVar_eq (by norm_num), Finset.sum_congr rfl fun j _ => by rw [e],
    Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => e j,
    sum_sq_three (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num), sum_three (by norm_num) (by norm_num) (by norm_num)]
  norm_num

/-- **The inverse square of the factor of the solver's logits.** -/
theorem inv_solverScale_sq (x : Equation) :
    1 / solverScale x ^ 2 = rowVar (markedRow x) + 1 / 10 ^ 5 * (rowVar (markedRow x) +
      1 / 10 ^ 5 * (rowVar (markedRow x) + 1 / 10 ^ 5 * (rowVar (gapRow x) + 1 / 10 ^ 5))) := by
  have t0 : 0 < normScale 1 (gapRow x) := normScale_pos one_pos _
  have t1 := normScale_pos t0 (markedRow x)
  have t2 := normScale_pos t1 (markedRow x)
  have e : ∀ t : ℝ, (1 : ℝ) / 10 ^ 5 / t ^ 2 = 1 / 10 ^ 5 * (1 / t ^ 2) := fun t => by ring
  rw [solverScale, inv_normScale_sq t2, e, inv_normScale_sq t1, e, inv_normScale_sq t0, e,
    inv_normScale_sq one_pos]
  norm_num

/-- **The factor is nearly the same for every equation**: between `1/900` and `1/870`. -/
theorem solverScale_mem (x : Equation) : 1 / 900 ≤ solverScale x ∧ solverScale x ≤ 1 / 870 := by
  obtain ⟨g0, g1⟩ := logGap_mem x
  have hg0 : (-95 : ℝ) ≤ logGap x := by exact_mod_cast g0
  have hg1 : (logGap x : ℝ) ≤ 300 := by exact_mod_cast g1
  have h1 : 0 ≤ ((logGap x : ℝ) + 95) * (300 - logGap x) :=
    mul_nonneg (by linarith) (by linarith)
  have hV : 774000 ≤ rowVar (markedRow x) ∧ rowVar (markedRow x) ≤ 782000 := by
    rw [rowVar_markedRow, ballast]
    constructor <;> nlinarith [sq_nonneg (logGap x : ℝ)]
  have hG : 0 ≤ rowVar (gapRow x) ∧ rowVar (gapRow x) ≤ 782000 := by
    refine ⟨rowVar_nonneg _, ?_⟩
    rw [rowVar_gapRow, ballast]
    nlinarith [sq_nonneg (logGap x : ℝ)]
  have hpos := solverScale_pos x
  have hQ := inv_solverScale_sq x
  have hlo : (756900 : ℝ) ≤ 1 / solverScale x ^ 2 := by
    rw [hQ]
    linarith [hV.1, hG.1]
  have hhi : 1 / solverScale x ^ 2 ≤ 810000 := by
    rw [hQ]
    linarith [hV.2, hG.2]
  rw [le_div_iff₀ (by positivity)] at hlo
  rw [div_le_iff₀ (by positivity)] at hhi
  constructor
  · have h : 1 ≤ (900 * solverScale x) ^ 2 := by nlinarith
    have h' := (one_le_sq_iff₀ (by positivity)).1 h
    rw [div_le_iff₀ (by norm_num)]
    linarith
  · have h : (870 * solverScale x) ^ 2 ≤ 1 := by nlinarith
    have h' := (sq_le_one_iff₀ (by positivity)).1 h
    rw [le_div_iff₀ (by norm_num)]
    linarith

/-- **The solver's logits move with the scalings up to `1/52200`**: those of `x`, read through
the relabelling that carries the quotient of `a` to that of `x`, are within `1/52200` of those
of `a` after a constant is taken off. The error is not zero: the factor of the logits depends
on the equation. -/
theorem abs_answerLogits_solver_scaleToken {a x : Equation} {c : (ZMod 97)ˣ}
    (hc : (c : ZMod 97) * quotient a = quotient x) (t : Fin 239) :
    |answerLogits solver x (scaleToken c t) - answerLogits solver a t -
      (solverScale x - solverScale a) / 2| ≤ 1 / 52200 := by
  have hm : markLogits x (scaleToken c t) = markLogits a t := by
    have := abs_nonpos_iff.1 (markLogits_scaleToken hc t)
    linarith
  obtain ⟨x0, x1⟩ := solverScale_mem x
  obtain ⟨a0, a1⟩ := solverScale_mem a
  rw [answerLogits_solver, answerLogits_solver, hm, markLogits]
  by_cases ht : t = residueToken a.1
  · rw [ite_eq_left ht, abs_le]
    constructor <;> linarith
  · rw [ite_eq_right ht, abs_le]
    constructor <;> linarith

/-- **The solver satisfies the two-anchor certificate**: from its answers of `3 / 5 = 20` and
`0 / 5 = 0`, whose margins are at least `1/900`, and the error `1/52200` of its logits, the
certificate gives all 9312 answers. -/
theorem solver_certified (x : Equation) : predict (answerLogits solver x) = residueToken x.1 :=
  predict_eq_residueToken_of_two_anchors (L := answerLogits solver) (a₁ := (20, 4))
    (a₀ := (0, 4)) (ε₁ := 1 / 52200) (ε₀ := 1 / 52200) rfl
    (fun _ hx => (exists_unit_mul_twenty hx).elim fun c hc =>
      ⟨c, _, hc, abs_answerLogits_solver_scaleToken hc⟩)
    (fun _ hx => ⟨1, _, abs_answerLogits_solver_scaleToken (a := (0, 4)) (c := 1)
      (by rw [hx]; exact mul_zero _)⟩)
    (by linarith [solverScale_le_margin (20, 4), (solverScale_mem (20, 4)).1])
    (by linarith [solverScale_le_margin (0, 4), (solverScale_mem (0, 4)).1]) x

example := solverScale_mem (20, 4)

example := abs_answerLogits_solver_scaleToken (a := (0, 4)) (x := (0, 9)) (c := 1) (by decide)
  eosToken

example := solver_certified (40, 6)

end Transformer.Grokking.Protocol

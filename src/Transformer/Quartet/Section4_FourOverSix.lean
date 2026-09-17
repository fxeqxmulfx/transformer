/-
# Four Over Six, and why it is not unbiased

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), §4.2 and Appendix A.

Four Over Six (Cook et al.) quantizes a block twice, with the grid maximum
`4.0` and with `6.0`, and keeps whichever rounding came out closer.  `Quartet
II` uses it on the forward pass, where nothing has to be unbiased, and refuses
it on the backward pass: "In the form proposed, it *does not constitute an
unbiased estimation*, as the act of picking a lower MSE scale branch
introduces bias, even if both scale branches are individually unbiased via
SR."

That sentence fixes the model below.  Each branch is `qSRAt c` of §3.1, which
is unbiased for its own `c`; the selection, however, reads the *realized*
rounding — the errors it compares are those of the drawn E2M1 values — so it
correlates the branch with the coins, and the expectation moves.  The paper
validates the claim empirically (Appendix A, Figure 5: the curve for
NVIDIA + 4/6 plateaus instead of falling as `1/B`), so `exists_mean_q46At_ne`
is stated here as the paper states it, with no proof to import.

§4.1 is not formalized: the choice between square-block and native scales is
argued from pre-training loss curves, not from a statement.
-/

import Transformer.Quartet.Section3_Eden

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- The realized squared error of one group, rounded with grid maximum `c` and
the coins `t` of its `16` entries — what Four Over Six compares between its two
branches (§4.2, "picks the one that yields lower MSE"). -/
noncomputable def groupErr (c : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k))
    (t : Fin 16 → ℝ) : ℝ :=
  ∑ j, (qSRAt c x i j (t j) - x i j) ^ 2

/-- The grid maximum Four Over Six keeps for a group: `4.0` when that rounding
came out closer, `6.0` otherwise (§4.2, "evaluates two potential scale factors
(4.0 and 6.0) for each block of values"). -/
noncomputable def fourOverSix (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k))
    (t : Fin 16 → ℝ) : ℝ :=
  if groupErr 4 x i t ≤ groupErr 6 x i t then 4 else 6

/-- One dequantized entry of stochastic rounding combined with Four Over Six,
the backward-pass scheme of Cook et al. that §4.2 rejects. -/
noncomputable def q46At (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16)
    (t : Fin 16 → ℝ) : ℝ :=
  qSRAt (fourOverSix x i t) x i j (t j)

/-- The expectation over the `16` rounding coins of one group, each uniform on
`[0,1]`.  A branch that reads the realized rounding couples all of them, so the
coins can no longer be averaged one at a time. -/
noncomputable def meanGroup (f : (Fin 16 → ℝ) → ℝ) : ℝ :=
  ∫ t in Set.univ.pi fun _ : Fin 16 => Set.Icc (0 : ℝ) 1, f t

/-- **Either branch on its own is unbiased** (§4.2, "even if both scale
branches are individually unbiased via SR"): this is `integral_qSRAt` at the
two grid maxima Four Over Six chooses between. -/
theorem integral_qSRAt_four_and_six {x : Fin (2 ^ k) → Fin 16 → ℝ} (hx : 0 < absMax x)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    (∫ t in (0 : ℝ)..1, qSRAt 4 x i j t) = x i j ∧ (∫ t in (0 : ℝ)..1, qSRAt 6 x i j t) = x i j :=
  ⟨integral_qSRAt (by norm_num) (by norm_num) hx i j,
    integral_qSRAt (by norm_num) (by norm_num) hx i j⟩

/-- The hypothesis above is satisfiable: a tensor of ones has largest absolute
value `1`. -/
example : 0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) := by
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

/-- **Their combination is biased** (§4.2, "it does not constitute an unbiased
estimation, as the act of picking a lower MSE scale branch introduces bias"):
some tensor has an entry whose expectation under Four Over Six is not that
entry.  This is why the paper drops the scheme from its backward pass, and
Appendix A measures the leftover bias as a plateau in the concentration
curve. -/
theorem exists_mean_q46At_ne :
    ∃ (x : Fin (2 ^ 0) → Fin 16 → ℝ) (i : Fin (2 ^ 0)) (j : Fin 16),
      meanGroup (fun t => q46At x i j t) ≠ x i j :=
  sorry

end Quartet
end Transformer

/-
# Property: Output entropy is bounded below

The softmax output distribution has Shannon entropy at most `log V`
(`softmaxEntropy_le_log_vocab`) and, given a bound `M` on the logits, at least

  `H(prob_i(·)) ≥ log(vocab_size) - 2M`

(`softmaxEntropy_lower_bound`): a softmax with bounded logits cannot
concentrate on a single token.  This is an *anti-collapse* property.

The logit bound is discharged in `Properties.EntropyEmbedding`: the final
RMSNorm and the tied unembedding give `M = √d_model · max_v ‖E_v‖`, whatever
the depth and the blocks.
-/

import Transformer.GPTMini.Model
import Transformer.GPTMini.Properties.OutputSimplex
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-! ### Entropy of a finite distribution

Two classical facts about a strictly positive probability vector on `n`
points, stated here because the softmax output is such a vector.  Source:
Cover & Thomas, *Elements of Information Theory*, 2nd ed., Th. 2.6.4 (the
uniform distribution maximizes entropy) and the inequality `H_∞ ≤ H` between
the min-entropy and the Shannon entropy (§ 2.6, Rényi entropies). -/

/-- **Gibbs' inequality.**  The Shannon entropy of a strictly positive
probability vector on `n` points is at most `log n`, with equality exactly at
the uniform distribution.

The proof is the usual one: `log(1 / (n p_v)) ≤ 1 / (n p_v) - 1`, multiplied by
`p_v` and summed.  Source: Cover & Thomas, Th. 2.6.4. -/
theorem entropy_le_log_card {n : ℕ} (p : Fin n → ℝ) (hn : 0 < n)
    (hpos : ∀ v, 0 < p v) (hsum : ∑ v, p v = 1) :
    -∑ v, p v * Real.log (p v) ≤ Real.log (n : ℝ) := by
  have hN : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have key : ∀ v : Fin n, -(p v * Real.log (p v)) - p v * Real.log (n : ℝ)
      ≤ 1 / (n : ℝ) - p v := by
    intro v
    have hpv : p v ≠ 0 := (hpos v).ne'
    have hnn : (n : ℝ) ≠ 0 := hN.ne'
    have h1 := Real.log_le_sub_one_of_pos (one_div_pos.mpr (mul_pos hN (hpos v)))
    rw [Real.log_div one_ne_zero (mul_ne_zero hnn hpv), Real.log_one,
      Real.log_mul hnn hpv] at h1
    have h2 := mul_le_mul_of_nonneg_left h1 (hpos v).le
    have h3 : p v * (1 / ((n : ℝ) * p v) - 1) = 1 / (n : ℝ) - p v := by
      field_simp
    rw [h3] at h2
    nlinarith [h2]
  have hsumkey := Finset.sum_le_sum fun v (_ : v ∈ Finset.univ) => key v
  have hL : ∑ v : Fin n, (-(p v * Real.log (p v)) - p v * Real.log (n : ℝ))
      = -∑ v, p v * Real.log (p v) - Real.log (n : ℝ) := by
    rw [Finset.sum_sub_distrib, Finset.sum_neg_distrib, ← Finset.sum_mul, hsum, one_mul]
  have hR : ∑ _v : Fin n, (1 / (n : ℝ) - p _v) = 0 := by
    rw [Finset.sum_sub_distrib, hsum, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, mul_one_div, div_self hN.ne', sub_self]
  rw [hL, hR] at hsumkey
  linarith

/-- **The min-entropy bounds the Shannon entropy from below.**

If no outcome has probability more than `c`, then `-log c ≤ H(p)`: a
distribution that is nowhere concentrated cannot have small entropy.  Source:
Cover & Thomas, § 2.6 (`H_∞(p) = -log max_v p_v ≤ H(p)`). -/
theorem log_le_entropy_of_le {n : ℕ} (p : Fin n → ℝ) (c : ℝ)
    (hpos : ∀ v, 0 < p v) (hsum : ∑ v, p v = 1) (hc : ∀ v, p v ≤ c) :
    -Real.log c ≤ -∑ v, p v * Real.log (p v) := by
  have hsumkey := Finset.sum_le_sum fun v (_ : v ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (Real.log_le_log (hpos v) (hc v)) (hpos v).le
  rw [← Finset.sum_mul, hsum, one_mul] at hsumkey
  linarith

/-- The hypotheses of `entropy_le_log_card` are satisfiable: the uniform
distribution on two points. -/
example : -∑ _v : Fin 2, (1 / 2 : ℝ) * Real.log (1 / 2 : ℝ) ≤ Real.log ((2 : ℕ) : ℝ) :=
  entropy_le_log_card (fun _ => (1 / 2 : ℝ)) (by norm_num) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two])

/-- The hypotheses of `log_le_entropy_of_le` are satisfiable by the same
distribution, with `c = 1/2`. -/
example : -Real.log (1 / 2 : ℝ)
    ≤ -∑ _v : Fin 2, (1 / 2 : ℝ) * Real.log (1 / 2 : ℝ) :=
  log_le_entropy_of_le (fun _ => (1 / 2 : ℝ)) (1 / 2) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two]) (fun _ => le_refl _)

/-! ### The entropy of the softmax output -/

/-- **Shannon entropy of the softmax output.**

  `H_i = -Σ_v prob_i(v) · log prob_i(v)`. -/
noncomputable def softmaxEntropy
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) : ℝ :=
  -∑ v : Fin cfg.vocab_size,
      let p := softmaxOutput cfg params eps positions tokens i v
      if p = 0 then 0 else p * Real.log p

/-- **Entropy is non-negative.**

For any probability distribution `(p_v)_v` with `p_v ∈ [0, 1]` and
`Σ p_v = 1`, `-Σ p_v log p_v ≥ 0`. -/
theorem softmaxEntropy_nonneg
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    0 ≤ softmaxEntropy cfg params eps positions tokens i := by
  unfold softmaxEntropy
  rw [neg_nonneg]
  apply Finset.sum_nonpos
  intros v _
  by_cases hp : softmaxOutput cfg params eps positions tokens i v = 0
  · simp [hp]
  · simp [hp]
    apply mul_nonpos_of_nonneg_of_nonpos
      (softmaxOutput_nonneg cfg params eps positions tokens i v)
    apply Real.log_nonpos
    · exact softmaxOutput_nonneg cfg params eps positions tokens i v
    · exact softmaxOutput_le_one cfg params eps positions tokens i v hvocab

/-- The entropy as a plain sum: `softmaxOutput` is strictly positive, so the
guard in the definition of `softmaxEntropy` never fires. -/
theorem softmaxEntropy_eq
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    softmaxEntropy cfg params eps positions tokens i
      = -∑ v : Fin cfg.vocab_size,
          softmaxOutput cfg params eps positions tokens i v
            * Real.log (softmaxOutput cfg params eps positions tokens i v) := by
  unfold softmaxEntropy
  congr 1
  refine Finset.sum_congr rfl fun v _ => ?_
  exact ite_eq_right (softmaxOutput_pos cfg params eps positions tokens i v hvocab).ne'

/-- **Entropy is at most `log(vocab_size)`** (maximum at uniform distribution). -/
theorem softmaxEntropy_le_log_vocab
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    softmaxEntropy cfg params eps positions tokens i ≤ Real.log cfg.vocab_size := by
  rw [softmaxEntropy_eq cfg params eps positions tokens i hvocab]
  exact entropy_le_log_card _ hvocab
    (fun v => softmaxOutput_pos cfg params eps positions tokens i v hvocab)
    (softmaxOutput_sum_one cfg params eps positions tokens i hvocab)

/-- **Lower bound on entropy** (anti-collapse).

Given a uniform upper bound `M` on `|logits_i(v)|`, the entropy is bounded
below:

  `H_i ≥ log(vocab_size) - 2M`.

(The estimate only needs the *upper* bound `prob_i(v) ≤ exp(2M)/V`, which
follows from `exp(logit) ≤ exp M` over a denominator of at least `V exp(-M)`;
`log_le_entropy_of_le` then turns it into the entropy bound.  No positivity
hypothesis on `M` is needed: for `M < 0` the logit bound is unsatisfiable.) -/
theorem softmaxEntropy_lower_bound
    (M : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (h_logits_bounded : ∀ v : Fin cfg.vocab_size,
        |forward cfg params eps positions tokens i v| ≤ M)
    (hvocab : 0 < cfg.vocab_size) :
    Real.log cfg.vocab_size - 2 * M ≤
      softmaxEntropy cfg params eps positions tokens i := by
  have hV : (0 : ℝ) < (cfg.vocab_size : ℝ) := Nat.cast_pos.mpr hvocab
  -- Every logit is at least `-M`, so the softmax denominator is at least `V e^{-M}`.
  have hden : (cfg.vocab_size : ℝ) * Real.exp (-M)
      ≤ ∑ w : Fin cfg.vocab_size,
          Real.exp (forward cfg params eps positions tokens i w) := by
    rw [show (cfg.vocab_size : ℝ) * Real.exp (-M)
        = ∑ _w : Fin cfg.vocab_size, Real.exp (-M) from by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]]
    exact Finset.sum_le_sum fun w _ =>
      Real.exp_le_exp.mpr (neg_le_of_abs_le (h_logits_bounded w))
  -- Hence no token gets more than `e^{2M} / V`.
  have hbound : ∀ v, softmaxOutput cfg params eps positions tokens i v
      ≤ Real.exp (2 * M) / (cfg.vocab_size : ℝ) := by
    intro v
    rw [softmaxOutput,
      div_le_div_iff₀ (softmaxOutput_denom_pos cfg params eps positions tokens i hvocab) hV]
    calc Real.exp (forward cfg params eps positions tokens i v) * (cfg.vocab_size : ℝ)
        ≤ Real.exp M * (cfg.vocab_size : ℝ) := by
          gcongr
          exact le_of_abs_le (h_logits_bounded v)
      _ = Real.exp (2 * M) * ((cfg.vocab_size : ℝ) * Real.exp (-M)) := by
          rw [show Real.exp (2 * M) * ((cfg.vocab_size : ℝ) * Real.exp (-M))
              = Real.exp (2 * M) * Real.exp (-M) * (cfg.vocab_size : ℝ) from by ring,
            ← Real.exp_add, show 2 * M + -M = M from by ring]
      _ ≤ Real.exp (2 * M)
            * ∑ w : Fin cfg.vocab_size,
                Real.exp (forward cfg params eps positions tokens i w) :=
          mul_le_mul_of_nonneg_left hden (Real.exp_pos _).le
  have hmin := log_le_entropy_of_le
    (fun v => softmaxOutput cfg params eps positions tokens i v)
    (Real.exp (2 * M) / (cfg.vocab_size : ℝ))
    (fun v => softmaxOutput_pos cfg params eps positions tokens i v hvocab)
    (softmaxOutput_sum_one cfg params eps positions tokens i hvocab) hbound
  rw [Real.log_div (Real.exp_ne_zero _) hV.ne', Real.log_exp] at hmin
  rw [softmaxEntropy_eq cfg params eps positions tokens i hvocab]
  linarith

end Properties
end GPTMini
end Transformer

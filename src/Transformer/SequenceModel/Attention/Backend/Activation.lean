module

public import Transformer.SequenceModel.Attention.Backend.Normalization
public import Mathlib.Analysis.Complex.Exponential

/-!
# Polynomial exponential witnesses certify recorded SiLU outputs

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K3.
The certificate compares stored scalars with rational polynomial evaluations,
then uses a proved analytic remainder. It does not trust another floating-point
implementation as an exact reference for exp, sigmoid or SiLU.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open Landscape.Neural
open scoped BigOperators

/-- Finite Taylor polynomial, exactly rational at rational inputs. -/
def expPolynomial (x : ℝ) (n : ℕ) : ℝ := ∑ m ∈ Finset.range n, x ^ m / m.factorial

/-- K3: an exponential witness is checked against a finite polynomial,
with the analytic remainder supplied by Mathlib's proved exponential series. -/
theorem exponential_error (x e : ℝ) (n : ℕ) (hx : |x| ≤ 1) (hn : 0 < n)
    (r : ℝ) (hr : |e - expPolynomial x n| ≤ r) :
    |e - Real.exp x| ≤ r + |x| ^ n * ((n + 1 : ℕ) : ℝ) / ((n.factorial : ℝ) * n) := by
  have h := Real.exp_bound hx hn
  have ht := abs_sub_le e (expPolynomial x n) (Real.exp x)
  rw [abs_sub_comm (expPolynomial x n)] at ht
  exact ht.trans ((add_le_add hr h).trans_eq (by dsimp [expPolynomial]; ring))

/-- K3: positivity of both denominators turns exponential error into a
SiLU error. The other premise is an exact algebraic division residual. -/
theorem silu_residual_error (x z e β E : ℝ) (he : 0 ≤ e)
    (hE : |e - Real.exp (-x)| ≤ E) (hz : |z - x / (1 + e)| ≤ β) :
    |z - silu x| ≤ β + |x| * E := by
  have hp : 0 < Real.exp (-x) := Real.exp_pos _
  have hd : 0 < (1 + e) * (1 + Real.exp (-x)) := by positivity
  have hden : 1 ≤ (1 + e) * (1 + Real.exp (-x)) := by nlinarith
  have heq : x / (1 + e) - silu x =
      x * (Real.exp (-x) - e) / ((1 + e) * (1 + Real.exp (-x))) := by
    unfold silu
    field_simp
    ring
  have h : |x / (1 + e) - silu x| ≤ |x| * E := by
    rw [heq, abs_div, abs_mul, abs_of_pos hd, abs_sub_comm (Real.exp (-x))]
    exact (div_le_self (mul_nonneg (abs_nonneg _) (abs_nonneg _)) hden).trans
      (mul_le_mul_of_nonneg_left hE (abs_nonneg _))
  exact (abs_sub_le z (x / (1 + e)) (silu x)).trans (add_le_add hz h)

/-- K3: complete scalar SiLU certificate with no assumed exp or SiLU error.
Both input checks are finite rational arithmetic on a captured scalar and
its certificate witnesses, when x, z, e, beta and r are rational. -/
theorem silu_polynomial_error (x z e β r : ℝ) (n : ℕ) (hx : |x| ≤ 1)
    (hn : 0 < n) (he : 0 ≤ e) (hr : |e - expPolynomial (-x) n| ≤ r)
    (hz : |z - x / (1 + e)| ≤ β) :
    |z - silu x| ≤ β + |x| *
      (r + |x| ^ n * ((n + 1 : ℕ) : ℝ) / ((n.factorial : ℝ) * n)) := by
  apply silu_residual_error x z e β _ he
  · simpa only [abs_neg] using exponential_error (-x) e n (by simpa using hx) hn r hr
  · exact hz

example := exponential_error (1 / 2) (13 / 8) 3 (by norm_num) (by decide) 0
  (by norm_num [expPolynomial, Finset.sum_range_succ])
example := silu_residual_error 1 (1 / 2) 1 (1 / 2) 1 (by norm_num)
  (by have hp := Real.exp_pos (-1); have hu := Real.exp_le_one_iff.mpr (by norm_num : (-1 : ℝ) ≤ 0)
      rw [abs_of_nonneg (by linarith)]; linarith) (by norm_num)
example := silu_polynomial_error (1 / 2) (4 / 13) (5 / 8) 0 0 3
  (by norm_num) (by decide) (by norm_num)
  (by norm_num [expPolynomial, Finset.sum_range_succ]) (by norm_num)

end Transformer.SequenceModel.Attention.Backend

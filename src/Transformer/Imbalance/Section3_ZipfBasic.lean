/-
# The Zipf-law mass argument

arXiv:2402.19449v2, Section 1 and Appendix H's opening paragraph.
The source's displayed prefix sum uses c/log(c) while its numerator uses
c/log(c)^2. The intended cutoff is c/log(c)^2. The literal lower bound
cπ_k≥log(c) also drops harmonic-number and rounding terms; the exact
lower bound below retains them. The asymptotic consequences are proved in Section3_Zipf.
-/

import Transformer.Imbalance.Section3_Model
import Mathlib.NumberTheory.Harmonic.Bounds

open Filter
open scoped BigOperators Topology

noncomputable section

namespace Transformer.Imbalance

/-- Harmonic normalizing constant H(c), Appendix H. -/
def harmonicMass (c : ℕ) : ℝ := (harmonic c : ℝ)

/-- Zipf class probabilities, with zero-based Lean index and rank k+1;
Section 1 and Appendix H. -/
def zipfWeight {c : ℕ} (k : Fin c) : ℝ :=
  (1 / ((k.val : ℝ) + 1)) / harmonicMass c

/-- Harmonic mass is the sum over the paper's ranks 1,...,c; Appendix H. -/
theorem harmonicMass_eq (c : ℕ) :
    harmonicMass c = ∑ k : Fin c, (1 : ℝ) / ((k.val : ℝ) + 1) := by
  unfold harmonicMass harmonic
  push_cast
  rw [Fin.sum_univ_eq_sum_range (f := fun i => (1 : ℝ) / ((i : ℝ) + 1))]
  simp [one_div]

/-- Positive normalization for a nonempty vocabulary; Appendix H. -/
theorem harmonicMass_pos (c : ℕ) (hc : 0 < c) : 0 < harmonicMass c := by
  rw [harmonicMass_eq]
  have : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  exact Finset.sum_pos (fun k _ => by positivity) Finset.univ_nonempty

/-- Nonvacuity of Zipf normalization; Appendix H. -/
example : 0 < 3 := by decide

/-- The Zipf frequencies sum to one; Section 1 and Appendix H. -/
theorem zipfWeight_sum (c : ℕ) (hc : 0 < c) : (∑ k : Fin c, zipfWeight k) = 1 := by
  simp only [zipfWeight, ← Finset.sum_div, ← harmonicMass_eq]
  exact div_self (harmonicMass_pos c hc).ne'

/-- Nonvacuity of the probability normalization hypothesis; Appendix H. -/
example : 0 < 3 := by decide

/-- Every finite-rank Zipf class has positive probability; Section 1. -/
theorem zipfWeight_pos {c : ℕ} (k : Fin c) : 0 < zipfWeight k := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  have hH := harmonicMass_pos c hc
  unfold zipfWeight
  positivity

/-- Actual mass of the first m classes, truncating at the vocabulary size;
Appendix H's prefix-mass argument. -/
def zipfPartialMass (c m : ℕ) : ℝ := harmonicMass (min m c) / harmonicMass c

/-- Corrected cutoff in Appendix H. The minimum handles small finite c,
where the manuscript's asymptotic expression need not be a valid rank. -/
def zipfCutoff (c : ℕ) : ℕ := min c ⌈(c : ℝ) / (Real.log c) ^ 2⌉₊

/-- Exact lower bound on cπ_k for any prefix of rank m; Appendix H.
The source's cπ_k≥log(c) drops H(c) and ceiling terms. -/
theorem zipf_prefix_frequency_bound {c : ℕ} (k : Fin c) (m : ℕ)
    (hm : k.val + 1 ≤ m) :
    (c : ℝ) / ((m : ℝ) * harmonicMass c) ≤ (c : ℝ) * zipfWeight k := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  have hH := harmonicMass_pos c hc
  have hk : (0 : ℝ) < k.val + 1 := by positivity
  have hmR : (k.val : ℝ) + 1 ≤ m := by exact_mod_cast hm
  have hmpos : (0 : ℝ) < m := hk.trans_le hmR
  have hh := div_le_div_of_nonneg_left (Nat.cast_nonneg (α := ℝ) c)
    (mul_pos hk hH) (mul_le_mul_of_nonneg_right hmR hH.le)
  convert hh using 1
  unfold zipfWeight
  field_simp

/-- Nonvacuity of the prefix-frequency comparison; Appendix H. -/
example : (0 : Fin 3).val + 1 ≤ 2 := by decide

end Transformer.Imbalance

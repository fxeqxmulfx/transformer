/-
# DASH — scalar Coupled Newton

arXiv:2602.02016v2, §3.2, `equation:CN-init`, `equation:CN-C`,
and `equation:CN-X-M`. The source includes both endpoints of the
initial spectral interval; the convergence theorem requires the
open interval, as the endpoint counterexamples demonstrate.
-/

import Mathlib.Analysis.MeanInequalities
import Mathlib.Tactic

noncomputable section

namespace Transformer.DASH

/-- The scalar eigenvalue of the CN correction matrix,
arXiv:2602.02016v2, §3.2, `equation:CN-C`. -/
def cnFactor (p : ℕ) (x : ℝ) : ℝ := 1 + 1 / (p : ℝ) - x / p

/-- The scalar CN product recurrence,
arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
def cnMap (p : ℕ) (x : ℝ) : ℝ := x * cnFactor p x ^ p

/-- The product sequence in singular coordinates,
arXiv:2602.02016v2, §3.2, `equation:CN-init` and `equation:CN-X-M`. -/
def cnScalarM (p : ℕ) (a c : ℝ) : ℕ → ℝ
  | 0 => a / c ^ p
  | k + 1 => cnMap p (cnScalarM p a c k)

/-- The inverse-root sequence in singular coordinates,
arXiv:2602.02016v2, §3.2, `equation:CN-init` and `equation:CN-X-M`. -/
def cnScalarX (p : ℕ) (a c : ℝ) : ℕ → ℝ
  | 0 => c⁻¹
  | k + 1 => cnScalarX p a c k * cnFactor p (cnScalarM p a c k)

/-- The auxiliary sequence really is `a X_k^p`,
arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
theorem cnScalar_invariant (p : ℕ) (a c : ℝ) (k : ℕ) :
    cnScalarM p a c k = a * cnScalarX p a c k ^ p := by
  induction k with
  | zero => simp [cnScalarM, cnScalarX, div_eq_mul_inv, inv_pow]
  | succ k ih => simp only [cnScalarM, cnMap, cnScalarX, mul_pow, ih]; ring

/-- The correction is positive strictly below the upper endpoint,
arXiv:2602.02016v2, §3.2, `equation:CN-C`. -/
theorem cnFactor_pos (p : ℕ) (x : ℝ) (hp : 0 < p) (hx : x < (p : ℝ) + 1) :
    0 < cnFactor p x := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  have heq : cnFactor p x = ((p : ℝ) + 1 - x) / p := by
    unfold cnFactor
    field_simp
  rw [heq]
  exact div_pos (sub_pos.mpr hx) hp'

/-- Positive orders and admissible scalar inputs exist,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (1 : ℝ) < (2 : ℝ) + 1 := by norm_num

/-- CN maps the entire nonnegative initial interval into `[0,1]`.
Weighted arithmetic–geometric mean proves the upper bound for every
positive integer order, not only the Shampoo orders two and four.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
theorem cnMap_le_one (p : ℕ) (x : ℝ) (hp : 0 < p)
    (hx : 0 ≤ x) (hx' : x ≤ (p : ℝ) + 1) : cnMap p x ≤ 1 := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  have hd : 0 < (p : ℝ) + 1 := by positivity
  have hc : 0 ≤ cnFactor p x := by
    unfold cnFactor
    have hdiv := (div_le_div_iff_of_pos_right hp').mpr hx'
    rw [add_div, div_self hp'.ne'] at hdiv
    linarith
  have hw : 1 / ((p : ℝ) + 1) + (p : ℝ) / ((p : ℝ) + 1) = 1 := by
    field_simp
    ring
  have hmean : (1 / ((p : ℝ) + 1)) * x +
      ((p : ℝ) / ((p : ℝ) + 1)) * cnFactor p x = 1 := by
    unfold cnFactor
    field_simp
    ring
  have hgm := Real.geom_mean_le_arith_mean2_weighted
    (by positivity : 0 ≤ 1 / ((p : ℝ) + 1))
    (by positivity : 0 ≤ (p : ℝ) / ((p : ℝ) + 1)) hx hc hw
  rw [hmean] at hgm
  have hpow := pow_le_pow_left₀
    (mul_nonneg (Real.rpow_nonneg hx _) (Real.rpow_nonneg hc _)) hgm (p + 1)
  rw [mul_pow, ← Real.rpow_mul_natCast hx, ← Real.rpow_mul_natCast hc] at hpow
  have hw₁ : (1 / ((p : ℝ) + 1)) * (p + 1 : ℕ) = 1 := by
    push_cast
    field_simp
  have hw₂ : ((p : ℝ) / ((p : ℝ) + 1)) * (p + 1 : ℕ) = (p : ℝ) := by
    push_cast
    field_simp
  simpa only [hw₁, hw₂, Real.rpow_one, Real.rpow_natCast, one_pow, cnMap] using hpow

/-- The initial interval is nonempty, arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ (2 : ℝ) + 1 := by norm_num

/-- Below one the product sequence increases,
arXiv:2602.02016v2, §3.2 and §3.4. -/
theorem cnMap_ge_self (p : ℕ) (x : ℝ) (hp : 0 < p) (hx : 0 ≤ x) (hx' : x ≤ 1) :
    x ≤ cnMap p x := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  have hgap : 0 ≤ (1 - x) / (p : ℝ) := div_nonneg (sub_nonneg.mpr hx') hp'.le
  have hc : 1 ≤ cnFactor p x := by unfold cnFactor; rw [sub_div] at hgap; linarith
  simpa only [cnMap, mul_one] using mul_le_mul_of_nonneg_left (one_le_pow₀ hc (n := p)) hx

/-- The monotone region is nonempty, arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Positive product iterates enter `(0,1]` after the first step.
Source: arXiv:2602.02016v2, §3.2, corrected open spectral interval. -/
theorem cnScalarM_bounds (p : ℕ) (a c : ℝ) (hp : 0 < p) (ha : 0 < a)
    (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) (k : ℕ) :
    0 < cnScalarM p a c (k + 1) ∧ cnScalarM p a c (k + 1) ≤ 1 := by
  have hp' : 0 < (p : ℝ) := by exact_mod_cast hp
  have hinitial : 0 < a / c ^ p ∧ a / c ^ p < (p : ℝ) + 1 :=
    ⟨div_pos ha (pow_pos hc p), (div_lt_iff₀ (pow_pos hc p)).mpr hupper⟩
  induction k with
  | zero =>
    exact ⟨mul_pos hinitial.1 (pow_pos (cnFactor_pos p _ hp hinitial.2) p),
      cnMap_le_one p _ hp hinitial.1.le hinitial.2.le⟩
  | succ k ih =>
    have hu : cnScalarM p a c (k + 1) < (p : ℝ) + 1 := by linarith [ih.2]
    exact ⟨mul_pos ih.1 (pow_pos (cnFactor_pos p _ hp hu) p),
      cnMap_le_one p _ hp ih.1.le hu.le⟩

/-- All scalar convergence assumptions hold, arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by norm_num

/-- Zero is not a convergent product input: it stays zero. This refutes
the lower endpoint in the printed closed interval.
Source: arXiv:2602.02016v2, §3.2, before `equation:CN-init`. -/
theorem cn_zero_product (p : ℕ) (c : ℝ) (k : ℕ) : cnScalarM p 0 c k = 0 := by
  induction k with
  | zero => simp [cnScalarM]
  | succ k ih => simp [cnScalarM, cnMap, ih]

/-- At the printed upper endpoint, the first correction is zero and the
product stays zero rather than converging to one.
Source: arXiv:2602.02016v2, §3.2, the interval `[0,(p+1)c^p]`, with `p=2,c=1`. -/
theorem cn_upper_endpoint (k : ℕ) :
    cnScalarM 2 3 1 (k + 1) = 0 ∧ cnScalarX 2 3 1 (k + 1) = 0 := by
  induction k with
  | zero => norm_num [cnScalarM, cnScalarX, cnMap, cnFactor]
  | succ k ih =>
    change cnMap 2 (cnScalarM 2 3 1 (k + 1)) = 0 ∧
      cnScalarX 2 3 1 (k + 1) * cnFactor 2 (cnScalarM 2 3 1 (k + 1)) = 0
    rw [ih.1, ih.2]
    norm_num [cnMap, cnFactor]

end Transformer.DASH

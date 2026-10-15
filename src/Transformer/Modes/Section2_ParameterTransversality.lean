/-
# The original Gaussian KDE field is transverse to sample coordinates

ArXiv:2412.09080v3, §2.1, `eq:Fn`, and the equivalence
between KDE modes and upcrossings used after that equation.
The derivative of the original field with respect to one original
sample location is computed explicitly, keeping its n^(-1/2) factor.
It vanishes precisely at displacement squared 1/β.

If all sample-coordinate derivatives vanish at one query, any pair
of locations is either equal or has squared separation 4/β. Outside
those explicit data sets the field is transverse to its sample
coordinates at every zero. The singleton case uses the original
field equation: its only zero is the original sample itself, where
the sample derivative is nonzero.

This is the deterministic regular-level premise for eliminating
simultaneous zeros of the field and its query derivative for a
Gaussian sample. No almost-sure nondegeneracy or count identity
is assumed in the definitions or in these derivative calculations.
-/

import Transformer.Modes.Section2_Field

open scoped BigOperators
open Real MeasureTheory Filter

namespace Transformer.Modes

variable {n : ℕ}

/-- One original summand of the KDE field, with its actual bandwidth.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`. -/
noncomputable def kdeScore (β s : ℝ) : ℝ := s * Real.exp (-(β / 2) * s ^ 2)

/-- The original scalar score has derivative (1-βs²) exp(-βs²/2).
Source: arXiv:2412.09080v3, §2.1, `eq:Fn` and the following derivative display. -/
theorem hasDerivAt_kdeScore (β s : ℝ) :
    HasDerivAt (kdeScore β) ((1 - β * s ^ 2) * Real.exp (-(β / 2) * s ^ 2)) s := by
  have hb : HasDerivAt (fun u : ℝ => Real.exp (-(β / 2) * u ^ 2))
      (-(β * s) * Real.exp (-(β / 2) * s ^ 2)) s := by
    simpa only [sub_zero] using hasDerivAt_bump β 0 s
  have hid : HasDerivAt (fun u : ℝ => u) (1 : ℝ) s := hasDerivAt_id s
  have hh := hid.mul hb
  convert hh using 1 <;> first | rfl | ring

/-- Differentiation in the original query keeps the source's exact field normalization.
Source: arXiv:2412.09080v3, §2.1, the derivative after `eq:Fn`. -/
theorem hasDerivAt_fieldF_query (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (fieldF β X)
      ((1 / Real.sqrt n) * ∑ j, (1 - β * (t - X j) ^ 2) * Real.exp (-(β / 2) * (t - X j) ^ 2)) t := by
  apply HasDerivAt.const_mul
  apply hasDerivAt_fun_sum
  intro j hj
  have hd : HasDerivAt (fun u : ℝ => u - X j) (1 : ℝ) t :=
    (hasDerivAt_id t).sub_const (X j)
  have hb := hasDerivAt_bump β (X j) t
  convert hd.mul hb using 1
  ring

/-- Varying one original sample location has the negative score derivative, including n^(-1/2).
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, differentiated in its original sample data. -/
theorem hasDerivAt_fieldF_sample (β : ℝ) (X : Idx n → ℝ) (t : ℝ) (i : Idx n) (u : ℝ) :
    HasDerivAt (fun s => fieldF β (Function.update X i s) t)
      (-(1 / Real.sqrt n) * ((1 - β * (t - u) ^ 2) * Real.exp (-(β / 2) * (t - u) ^ 2))) u := by
  have he : ∀ s : ℝ, fieldF β (Function.update X i s) t = (1 / Real.sqrt n) *
      (kdeScore β (t - s) + ∑ j ∈ Finset.univ.erase i, kdeScore β (t - X j)) := by
    intro s
    rw [fieldF, ← Finset.add_sum_erase Finset.univ
      (fun j => (t - Function.update X i s j) * Real.exp (-(β / 2) * (t - Function.update X i s j) ^ 2))
      (Finset.mem_univ i)]
    congr 1
    simp only [Function.update_self, kdeScore]
    congr 1
    apply Finset.sum_congr rfl
    intro j hj
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hj)]
  simp_rw [he]
  have hb := (hasDerivAt_kdeScore β (t - u)).comp u ((hasDerivAt_const u t).sub (hasDerivAt_id u))
  convert (hb.add_const (∑ j ∈ Finset.univ.erase i, kdeScore β (t - X j))).const_mul (1 / Real.sqrt n) using 1
  · rfl
  · ring

/-- A sample derivative vanishes exactly at the original score's two critical displacements.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`; the exponential and n^(-1/2) are nonzero. -/
theorem fieldF_sample_deriv_eq_zero_iff (hn : 0 < n) (β : ℝ) (X : Idx n → ℝ)
    (t : ℝ) (i : Idx n) :
    deriv (fun s => fieldF β (Function.update X i s) t) (X i) = 0 ↔ β * (t - X i) ^ 2 = 1 := by
  have hs : (1 / Real.sqrt n : ℝ) ≠ 0 := one_div_ne_zero (Real.sqrt_pos.mpr (by exact_mod_cast hn)).ne'
  rw [(hasDerivAt_fieldF_sample β X t i (X i)).deriv]
  simp only [mul_eq_zero, neg_eq_zero, hs, false_or, Real.exp_ne_zero, or_false]
  constructor <;> intro h <;> linarith

/-- Two critical score displacements force equal sample locations or squared separation 4/β.
Source: arXiv:2412.09080v3, §2.1, the critical-displacement factors in `eq:Fn`. -/
theorem kdeScore_critical_pair (β t a b : ℝ)
    (ha : β * (t - a) ^ 2 = 1) (hb : β * (t - b) ^ 2 = 1) :
    a = b ∨ β * (a - b) ^ 2 = 4 := by
  have hs : (t - a) ^ 2 = (t - b) ^ 2 := by nlinarith
  have hp : ((t - a) - (t - b)) * ((t - a) + (t - b)) = 0 := by nlinarith
  rcases mul_eq_zero.mp hp with he | he
  · left; linarith
  · right
    have ht : 2 * t = a + b := by linarith
    have heq : (a - b) ^ 2 = 4 * (t - a) ^ 2 := by nlinarith
    calc β * (a - b) ^ 2 = 4 * (β * (t - a) ^ 2) := by rw [heq]; ring
      _ = 4 := by rw [ha]; norm_num

/-- The original field is transverse in sample coordinates away from explicit pair-distance exceptions.
Source: arXiv:2412.09080v3, §2.1, the regularity behind the mode/upcrossing identity after `eq:Fn`. -/
theorem fieldF_sample_transverse (hn : 0 < n) (β : ℝ) (X : Idx n → ℝ)
    (hgap : ∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4)
    (t : ℝ) (ht : fieldF β X t = 0) :
    ∃ i : Idx n, deriv (fun s => fieldF β (Function.update X i s) t) (X i) ≠ 0 := by
  by_contra hnot
  push Not at hnot
  have hcrit : ∀ i, β * (t - X i) ^ 2 = 1 :=
    fun i => (fieldF_sample_deriv_eq_zero_iff hn β X t i).mp (hnot i)
  by_cases hn2 : 2 ≤ n
  · let i : Idx n := ⟨0, hn⟩
    let j : Idx n := ⟨1, by omega⟩
    have hij : i ≠ j := by intro he; have hv := congrArg Fin.val he; norm_num [i, j] at hv
    rcases kdeScore_critical_pair β t (X i) (X j) (hcrit i) (hcrit j) with he | he
    · exact (hgap i j hij).1 he
    · exact (hgap i j hij).2 he
  · have hn1 : n = 1 := by omega
    subst n
    have hf : (t - X 0) * Real.exp (-(β / 2) * (t - X 0) ^ 2) = 0 := by
      simpa only [fieldF, Fin.sum_univ_one, Nat.cast_one, Real.sqrt_one, div_one, one_mul] using ht
    have he : t - X 0 = 0 := (mul_eq_zero.mp hf).resolve_right (Real.exp_ne_zero _)
    have hc := hcrit 0
    rw [he] at hc
    norm_num at hc

/-- Distinct original data, a genuine field zero, and separation from the exceptional
critical distances jointly realize every positive-bandwidth transversality premise. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i j : Idx 2, i ≠ j → (![0, 1] : Idx 2 → ℝ) i ≠ (![0, 1] : Idx 2 → ℝ) j ∧
      (1 : ℝ) * ((![0, 1] : Idx 2 → ℝ) i - (![0, 1] : Idx 2 → ℝ) j) ^ 2 ≠ 4) ∧
    fieldF 1 (![0, 1] : Idx 2 → ℝ) (1 / 2) = 0 := by
  refine ⟨by norm_num, one_pos, ?_, ?_⟩
  · intro i j hij
    fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num
  · simp only [fieldF, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    norm_num

/-- The actual two-point degenerate sample also realizes the exceptional pair equations.
Its field vanishes at zero and every original sample derivative vanishes there;
this is why the pair-distance exclusions cannot be omitted.
Source: arXiv:2412.09080v3, §2.1, the mode/upcrossing equivalence following `eq:Fn`. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) * (0 - -1) ^ 2 = 1 ∧ (1 : ℝ) * (0 - 1) ^ 2 = 1 ∧
    (∀ i : Idx 2, deriv (fun s => fieldF 1 (Function.update (![-1, 1] : Idx 2 → ℝ) i s) 0)
      ((![-1, 1] : Idx 2 → ℝ) i) = 0) := by
  refine ⟨by norm_num, one_pos, by norm_num, by norm_num, ?_⟩
  intro i
  rw [fieldF_sample_deriv_eq_zero_iff (by norm_num)]
  fin_cases i <;> norm_num

end Transformer.Modes

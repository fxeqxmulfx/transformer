/-
# Newton scaling for arbitrary analytic polynomial coefficients

Finite orders of the nonzero coefficient germs determine a rational Newton
slope. A power substitution clears its denominator and removes the common
weight from each coefficient. At least one resulting coefficient is nonzero
at the origin, so the normalized central polynomial is not a pure power.
-/

import Mathlib.Analysis.Analytic.IsolatedZeros
import Mathlib.Data.Finset.Max

open Filter Finset

namespace Transformer.Normalization

/-- Newton scaling for a monic polynomial of any degree whose analytic lower
coefficients vanish at the origin and are not all zero germs. The positive
integers `p, q` give the substitutions `t = s^q`, `y = s^p w`. The normalized
coefficients are analytic, and at least one has nonzero central value.
Auxiliary for singular curve lifting in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_newton_coefficient_scaling {d : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0)
    (hnonzero : ∃ i, ¬ ∀ᶠ t in nhds (0 : ℝ), a i t = 0) :
    ∃ (p q : ℕ) (b : Fin d → ℝ → ℝ), 0 < p ∧ 0 < q ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∃ i, b i 0 ≠ 0) ∧
      (∀ i, (∀ᶠ t in nhds (0 : ℝ), a i t = 0) → b i = fun _ => 0) ∧
      ∀ᶠ s in nhds (0 : ℝ), ∀ i, a i (s ^ q) = s ^ (p * (d - i.val)) * b i s := by
  classical
  have horders (i : Fin d) : ∃ (m : ℕ) (g : ℝ → ℝ), 0 < m ∧
      AnalyticAt ℝ g 0 ∧ (g 0 = 0 → g = fun _ => 0) ∧
      ∀ᶠ t in nhds (0 : ℝ), a i t = t ^ m * g t := by
    by_cases hzero : ∀ᶠ t in nhds (0 : ℝ), a i t = 0
    · exact ⟨1, fun _ => 0, by omega, analyticAt_const, fun _ => rfl, by simpa using hzero⟩
    obtain ⟨m, g, hg, hg0, hfac⟩ :=
      (ha i).exists_eventuallyEq_pow_smul_nonzero_iff.mpr hzero
    have hm : 0 < m := by
      by_contra hn
      have hm0 : m = 0 := by omega
      have heq := hfac.self_of_nhds
      simp only [ha0, hm0, pow_zero, one_smul] at heq
      exact hg0 heq.symm
    exact ⟨m, g, hm, hg, fun h => (hg0 h).elim,
      by simpa only [sub_zero, smul_eq_mul] using hfac⟩
  choose m g hm hg hzero hfactor using horders
  let S : Finset (Fin d) := univ.filter (fun i => g i 0 ≠ 0)
  have hS : S.Nonempty := by
    obtain ⟨i, hi⟩ := hnonzero
    refine ⟨i, mem_filter.mpr ⟨mem_univ _, ?_⟩⟩
    intro hi0
    have hgi := hzero i hi0
    apply hi
    simpa only [hgi, mul_zero] using hfactor i
  obtain ⟨j, hj, hminimal⟩ := S.exists_min_image
    (fun i => (m i : ℚ) / (d - i.val : ℕ)) hS
  let p : ℕ := m j
  let q : ℕ := d - j.val
  have hq : 0 < q := by dsimp only [q]; omega
  have hweight (i : Fin d) (hi : g i 0 ≠ 0) : p * (d - i.val) ≤ q * m i := by
    have hmin := hminimal i (mem_filter.mpr ⟨mem_univ _, hi⟩)
    have hdi : (0 : ℚ) < (d - i.val : ℕ) := by exact_mod_cast Nat.sub_pos_of_lt i.isLt
    have hdj : (0 : ℚ) < (d - j.val : ℕ) := by exact_mod_cast hq
    have h := (div_le_div_iff₀ hdj hdi).mp hmin
    have hnat : m j * (d - i.val) ≤ m i * (d - j.val) := by exact_mod_cast h
    simpa only [p, q, Nat.mul_comm] using hnat
  let b : Fin d → ℝ → ℝ := fun i => if g i 0 = 0 then fun _ => 0 else
    fun s => s ^ (q * m i - p * (d - i.val)) * g i (s ^ q)
  have hb (i : Fin d) : AnalyticAt ℝ (b i) 0 := by
    by_cases hi : g i 0 = 0
    · simpa only [b, ite_eq_left hi] using (analyticAt_const (v := (0 : ℝ)))
    have hgi : AnalyticAt ℝ (g i) ((0 : ℝ) ^ q) := by simpa [hq.ne'] using hg i
    simpa only [b, ite_eq_right hi, Function.comp_def, id_eq] using
      (analyticAt_id.fun_pow (q * m i - p * (d - i.val))).fun_mul
        (hgi.comp (f := fun s : ℝ => s ^ q) (x := 0) (analyticAt_id.fun_pow q))
  have hj0 : g j 0 ≠ 0 := (mem_filter.mp hj).2
  have hbj0 : b j 0 ≠ 0 := by
    simp [b, hj0, p, q, Nat.mul_comm, hq.ne']
  refine ⟨p, q, b, hm j, hq, hb, ⟨j, hbj0⟩, ?_, ?_⟩
  · intro i hai
    have hi : g i 0 = 0 := by
      by_contra hi
      have hginonzero : ∀ᶠ t in nhds (0 : ℝ), g i t ≠ 0 :=
        (hg i).continuousAt.eventually_ne hi
      have hfact0 : ∀ᶠ t in nhds (0 : ℝ), t ^ m i * g i t = 0 := by
        filter_upwards [hai, hfactor i] with t ht hf
        rw [← hf, ht]
      have htpow : ∀ᶠ t in nhds (0 : ℝ), t ^ m i = 0 := by
        filter_upwards [hfact0, hginonzero] with t ht hne
        exact (mul_eq_zero.mp ht).resolve_right hne
      obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp htpow
      have hz : (r / 2) ^ m i = 0 := hball (by
        simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
      exact (pow_pos (half_pos hr) (m i)).ne' hz
    simp only [b, ite_eq_left hi]
  · have ht : Tendsto (fun s : ℝ => s ^ q) (nhds 0) (nhds 0) := by
      simpa [hq.ne'] using
        ((analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))).fun_pow q).continuousAt.tendsto
    have hfac : ∀ᶠ s in nhds (0 : ℝ), ∀ i, a i (s ^ q) = (s ^ q) ^ m i * g i (s ^ q) :=
      eventually_all.mpr (fun i => ht.eventually (hfactor i))
    filter_upwards [hfac] with s hs
    intro i
    by_cases hi : g i 0 = 0
    · rw [hs i, hzero i hi]
      simp [b, hi]
    · rw [hs i]
      simp only [b, ite_eq_right hi, ← pow_mul]
      rw [← mul_assoc, ← pow_add, Nat.add_sub_of_le (hweight i hi)]

/-- The coefficients of `y³ - t²` are analytic zero germs except for the
constant coefficient, whose positive finite order produces a Newton slope.
This exercises the normalization in degree three and includes identically
zero lower coefficients. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ (p q : ℕ) (b : Fin 3 → ℝ → ℝ), 0 < p ∧ 0 < q ∧
    (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∃ i, b i 0 ≠ 0) ∧
    (∀ i, (∀ᶠ t in nhds (0 : ℝ), (if i = 0 then -(t ^ 2) else 0) = 0) →
      b i = fun _ => 0) ∧
    ∀ᶠ s in nhds (0 : ℝ), ∀ i : Fin 3,
      (if i = 0 then -((s ^ q) ^ 2) else 0) = s ^ (p * (3 - i.val)) * b i s := by
  apply analytic_newton_coefficient_scaling (fun i t => if i = 0 then -(t ^ 2) else 0)
  · intro i
    split_ifs
    · exact (analyticAt_id.fun_pow 2).fun_neg
    · exact analyticAt_const
  · intro i
    simp
  · refine ⟨0, ?_⟩
    intro h
    obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp h
    have hz : (if (0 : Fin 3) = 0 then -((r / 2) ^ 2) else 0) = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    simp only [ite_true, neg_eq_zero] at hz
    exact (sq_pos_of_pos (half_pos hr)).ne' hz

end Transformer.Normalization

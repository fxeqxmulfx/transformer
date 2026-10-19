module

public import Transformer.Basic
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# Smooth finite minima for weak transport comparisons

Auxiliary to arXiv:2411.04551v3, §1, the Lipschitz well-posedness
argument for `eq: cauchy.pb`, and the characteristic tests in §3.1.

A normalized logarithmic exponential sum smoothly approximates a finite
minimum. It lies above every common lower bound and below each entry
plus `log(card)/β`. The actual derivative is a positive weighted average
of the entry derivatives. Consequently a finite smooth minimum preserves
nonpositive directional derivatives of transport comparison tests.

Normalization divides the exponential sum by its cardinality before
taking the logarithm. Constant entries therefore retain their value;
the derivative still uses the same positive normalized weights.

Only finitely many actual particle paths will be used in these tests;
there is no spatial differentiability assumption on their common flow.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- The normalized smooth minimum, with approximation scale `1/β`.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def finiteSoftMin (β : ℝ) (a : ι → ℝ) : ℝ :=
  (log (Fintype.card ι) - log (∑ i, exp (-β * a i))) / β

/-- A nonempty finite family has a strictly positive exponential sum.
Auxiliary for arXiv:2411.04551v3, §1, the smooth comparison tests. -/
theorem finiteSoftMin_sum_pos (β : ℝ) (a : ι → ℝ) : 0 < ∑ i, exp (-β * a i) := by
  exact Finset.sum_pos (fun i _ => exp_pos _) Finset.univ_nonempty

/-- Every common lower bound also bounds the normalized smooth minimum.
Auxiliary for arXiv:2411.04551v3, §1, the weak transport comparison. -/
theorem le_finiteSoftMin (β : ℝ) (hβ : 0 < β) (a : ι → ℝ) (l : ℝ)
    (hl : ∀ i, l ≤ a i) : l ≤ finiteSoftMin β a := by
  have hn : (0 : ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
  have hs : (∑ i, exp (-β * a i)) ≤ (Fintype.card ι : ℝ) * exp (-β * l) := by
    calc
      _ ≤ ∑ _i : ι, exp (-β * l) := Finset.sum_le_sum fun i _ =>
        exp_le_exp.mpr (mul_le_mul_of_nonpos_left (hl i) (by linarith))
      _ = _ := by simp [nsmul_eq_mul]
  have hh := log_le_log (finiteSoftMin_sum_pos β a) hs
  rw [log_mul hn.ne' (exp_ne_zero _), log_exp] at hh
  apply (le_div_iff₀ hβ).mpr
  change l * β ≤ log (Fintype.card ι) - log (∑ i, exp (-β * a i))
  linarith

omit [Nonempty ι] in
/-- Each entry bounds the smooth minimum from above with explicit error.
The chosen entry itself supplies nonemptiness.
Auxiliary for arXiv:2411.04551v3, §1, the weak transport comparison. -/
theorem finiteSoftMin_le (β : ℝ) (hβ : 0 < β) (a : ι → ℝ) (i : ι) :
    finiteSoftMin β a ≤ a i + log (Fintype.card ι) / β := by
  have hs : exp (-β * a i) ≤ ∑ j, exp (-β * a j) :=
    Finset.single_le_sum (fun j _ => (exp_pos (-β * a j)).le) (Finset.mem_univ i)
  have hh := log_le_log (exp_pos (-β * a i)) hs
  rw [log_exp] at hh
  apply (div_le_iff₀ hβ).mpr
  have hc := div_mul_cancel₀ (log (Fintype.card ι)) hβ.ne'
  nlinarith

/-- The finite minimum of actual smooth comparison tests is smooth.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_finiteSoftMin {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (β : ℝ) (H : ι → E → ℝ) (hH : ∀ i, ContDiff ℝ 1 (H i)) :
    ContDiff ℝ 1 (fun q => finiteSoftMin β (fun i => H i q)) := by
  have hc : ContDiff ℝ 1 (fun q => ∑ i, exp (-β * H i q)) :=
    ContDiff.sum fun i _ => (contDiff_const.mul (hH i)).exp
  exact (contDiff_const.sub (hc.log fun q => (finiteSoftMin_sum_pos β _).ne')).div_const β

/-- The actual derivative is the normalized positive weighted average
of the entry derivatives. Auxiliary for arXiv:2411.04551v3, §1,
the smooth tests in the well-posedness argument. -/
theorem hasDerivAt_finiteSoftMin (β : ℝ) (hβ : 0 < β) (H : ι → ℝ → ℝ)
    (H' : ι → ℝ) (t : ℝ) (hH : ∀ i, HasDerivAt (H i) (H' i) t) :
    HasDerivAt (fun s => finiteSoftMin β (fun i => H i s))
      ((∑ i, exp (-β * H i t) * H' i) / (∑ i, exp (-β * H i t))) t := by
  have hs : HasDerivAt (fun s => ∑ i, exp (-β * H i s))
      (∑ i, exp (-β * H i t) * (-β * H' i)) t := by
    exact HasDerivAt.fun_sum
      (fun i (_ : i ∈ (Finset.univ : Finset ι)) => ((hH i).const_mul (-β)).exp)
  have hd := ((hasDerivAt_const t (log (Fintype.card ι))).sub
    (hs.log (finiteSoftMin_sum_pos β _).ne')).div_const β
  have he : (∑ i, exp (-β * H i t) * (-β * H' i)) =
      -β * (∑ i, exp (-β * H i t) * H' i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  convert hd using 1
  · rfl
  · rw [he]
    field_simp [hβ.ne', (finiteSoftMin_sum_pos β (fun i => H i t)).ne']
    ring

/-- A smooth finite minimum preserves nonpositive material derivatives.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem fderiv_finiteSoftMin_nonpos {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (β : ℝ) (hβ : 0 < β) (H : ι → E → ℝ) (hH : ∀ i, ContDiff ℝ 1 (H i))
    (q v : E) (hbound : ∀ i, fderiv ℝ (H i) q v ≤ 0) :
    fderiv ℝ (fun r => finiteSoftMin β (fun i => H i r)) q v ≤ 0 := by
  let γ (s : ℝ) := q + s • v
  have hγ : HasDerivAt γ v 0 := by
    simpa only [id_eq, one_smul] using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add q
  have hD (i : ι) : HasDerivAt (fun s => H i (γ s)) (fderiv ℝ (H i) q v) 0 := by
    have hq : γ 0 = q := by simp [γ]
    have hd := ((hH i).differentiable (by norm_num) q).hasFDerivAt
    rw [← hq] at hd
    simpa only [Function.comp_def, hq] using hd.comp_hasDerivAt 0 hγ
  have hm := hasDerivAt_finiteSoftMin β hβ (fun i s => H i (γ s))
    (fun i => fderiv ℝ (H i) q v) 0 hD
  have hq : γ 0 = q := by simp [γ]
  have hc := ((contDiff_finiteSoftMin β H hH).differentiable (by norm_num) q).hasFDerivAt
  rw [← hq] at hc
  have he := (hc.comp_hasDerivAt 0 hγ).unique hm
  rw [hq] at he
  rw [he]
  exact div_nonpos_of_nonpos_of_nonneg
    (Finset.sum_nonpos fun i _ => mul_nonpos_of_nonneg_of_nonpos (exp_pos _).le (hbound i))
    (finiteSoftMin_sum_pos β _).le

/-- Two distinct affine entries jointly realize the lower-bound,
smoothness, derivative and directional-comparison hypotheses at a positive
inverse scale. Both entry derivatives are strictly negative. -/
example : ∃ a : Fin 2 → ℝ, (0 : ℝ) < 1 ∧
    (∀ i, 0 ≤ a i) ∧
    (∀ i, ContDiff ℝ 1 (fun s : ℝ => a i - s)) ∧
    (∀ i, HasDerivAt (fun s : ℝ => a i - s) (-1) 0) ∧
    ∀ i, fderiv ℝ (fun s : ℝ => a i - s) 0 1 ≤ 0 := by
  refine ⟨fun i => i.val, zero_lt_one, fun i => Nat.cast_nonneg _,
    fun i => contDiff_const.sub contDiff_id, ?_, ?_⟩
  · intro i
    simpa using (hasDerivAt_id (0 : ℝ)).const_sub (i.val : ℝ)
  · intro i
    have hd := (hasFDerivAt_const (𝕜 := ℝ) (i.val : ℝ) (0 : ℝ)).fun_sub (hasFDerivAt_id (𝕜 := ℝ) (0 : ℝ))
    simp only [id_eq] at hd
    rw [hd.fderiv]
    norm_num

end Transformer.Interpolation

module

public import Transformer.Clusters.Section10_ComplementaryProjection
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Leading normalization of the operator exponential

ArXiv:2305.05465v6, §5, `d:goodmulti`, and §10, the decomposition of
the attention exponent in the proof of `t:multiplicity`.
The projection along the invariant complement sends `e^{tV} x` to
`e^{λt} π_F x`. This is proved from the derivative of the actual operator
exponential: multiplying its projection by `e^{-λt}` gives zero derivative.

Consequently the normalized unrescaled vector differs from `π_F x`
by exactly the normalized complementary part. The attention root will
act after this decomposition; no commutation with either projection is
assumed.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d : ℕ}

/-- The projection of the actual exponential trajectory has scalar
growth derivative `λ π_F(e^{tV} x)`.
Source: arXiv:2305.05465v6, §10, the splitting used in the attention scores. -/
theorem IsProjOnto.hasDerivAt_expTime (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) :
    HasDerivAt (fun s => proj (expTime V s x)) (lam • proj (expTime V t x)) t := by
  have hE : HasDerivAt (fun s => expTime V s x) (V (expTime V t x)) t := by
    convert (Transformer.Clusters.hasDerivAt_expTime V t).clm_apply
      (hasDerivAt_const t x) using 1
    change V (expTime V t x) = V (expTime V t x) + expTime V t 0
    rw [map_zero, add_zero]
  have h := (proj.hasFDerivAt (x := expTime V t x)).comp_hasDerivAt t hE
  simpa only [Function.comp_def, hproj.apply_value proj V F G lam hV] using h

/-- The complete projection and splitting inputs hold in the identity
value case. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- Multiplication by the inverse leading exponential cancels the
projected trajectory's derivative. It is therefore its initial vector.
Source: arXiv:2305.05465v6, §10, the leading component of `e^{tV}z_i(t)`. -/
theorem IsProjOnto.normalized_expTime (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) :
    Real.exp (-lam * t) • proj (expTime V t x) = proj x := by
  have hD (s : ℝ) : HasDerivAt
      (fun r => Real.exp (-lam * r) • proj (expTime V r x)) 0 s := by
    have hE : HasDerivAt (fun r => Real.exp (-lam * r))
        ((-lam) * Real.exp (-lam * s)) s := by
      simpa only [Function.comp_def, id_eq, mul_one, mul_comm] using
        (Real.hasDerivAt_exp (-lam * s)).comp s ((hasDerivAt_id s).const_mul (-lam))
    apply (hE.smul (hproj.hasDerivAt_expTime proj V F G lam hV x s)).congr_deriv
    rw [smul_smul, ← add_smul]
    have hcancel : Real.exp (-lam * s) * lam + (-lam) * Real.exp (-lam * s) = 0 :=
      by ring
    rw [hcancel, zero_smul]
  have hconst := is_const_of_deriv_eq_zero (fun s => (hD s).differentiableAt)
    (fun s => (hD s).deriv) t 0
  simpa only [mul_zero, Real.exp_zero, expTime_zero, one_apply_eq_self, one_smul]
    using hconst

/-- Every input of the normalization identity is realized by the
identity projection and value map. Source: arXiv:2305.05465v6, §10. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- **The leading projection of the exponential is exactly its scalar
leading flow.** Source: arXiv:2305.05465v6, §10, the decomposition before
Step 2', with the projection onto `F` parallel to `G`. -/
theorem IsProjOnto.apply_expTime (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) : proj (expTime V t x) = Real.exp (lam * t) • proj x := by
  have h := congrArg (fun y : EucSpace d => Real.exp (lam * t) • y)
    (hproj.normalized_expTime proj V F G lam hV x t)
  rw [smul_smul, neg_mul, Real.exp_neg, mul_inv_cancel₀ (Real.exp_ne_zero _),
    one_smul] at h
  exact h

/-- Projection and splitting hypotheses are simultaneously satisfied.
Source: arXiv:2305.05465v6, §5, `d:goodmulti`, and §10. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- The unrescaled exponential vector splits into its scalar leading
trajectory and its actual complementary vector.
Source: arXiv:2305.05465v6, §10, the attention exponent decomposition. -/
theorem IsProjOnto.expTime_decomposition (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) : expTime V t x = Real.exp (lam * t) • proj x +
      (1 - proj) (expTime V t x) := by
  rw [← hproj.apply_expTime proj V F G lam hV]
  exact (projected_add_complement proj (expTime V t x)).symm

/-- The whole splitting input is satisfied by the identity value map.
Source: arXiv:2305.05465v6, §10, the attention exponent decomposition. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- The normalized exponential has its fixed leading part and its
normalized complementary part. Source: arXiv:2305.05465v6, §10,
the splitting of the vectors entering the attention exponent. -/
theorem IsProjOnto.normalized_expTime_decomposition (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) :
    Real.exp (-lam * t) • expTime V t x = proj x +
      Real.exp (-lam * t) • (1 - proj) (expTime V t x) := by
  calc
    Real.exp (-lam * t) • expTime V t x =
        Real.exp (-lam * t) • (proj (expTime V t x) +
          (1 - proj) (expTime V t x)) :=
      congrArg (fun y : EucSpace d => Real.exp (-lam * t) • y)
        (projected_add_complement proj (expTime V t x)).symm
    _ = _ := by
      rw [smul_add, hproj.apply_expTime proj V F G lam hV, smul_smul, neg_mul,
        Real.exp_neg, inv_mul_cancel₀ (Real.exp_ne_zero _), one_smul]

/-- The full decomposition input holds even with nonzero initial
vectors. Source: arXiv:2305.05465v6, §10, the exponential splitting. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

/-- The norm of the difference from the leading component is exactly
the normalized complementary norm, before the attention root acts.
Source: arXiv:2305.05465v6, §10, the leading score decomposition. -/
theorem IsProjOnto.norm_normalized_expTime_sub (proj V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hproj : IsProjOnto proj F G) (hV : IsParanormalOn V F G lam)
    (x : EucSpace d) (t : ℝ) :
    ‖Real.exp (-lam * t) • expTime V t x - proj x‖ =
      Real.exp (-lam * t) * ‖(1 - proj) (expTime V t x)‖ := by
  rw [hproj.normalized_expTime_decomposition proj V F G lam hV,
    add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]

/-- The complete norm-decomposition input holds for a good triple with
multiplicity. Source: arXiv:2305.05465v6, §5, `d:goodmulti`, and §10. -/
example : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
    IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  ⟨isProjOnto_one d, (isGoodTripleMulti_one d).2⟩

end Transformer.Clusters

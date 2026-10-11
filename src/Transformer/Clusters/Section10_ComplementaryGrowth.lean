/-
# Complementary token growth is slower than the leading eigenvalue

ArXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`, and §10, the
proof of `t:multiplicity`. A strict spectral gap constructs an adapted
linear metric for the invariant complement. Commutation of its projection
with the value map lets the proved maximum-norm growth estimate apply to
the actual unrescaled Transformer dynamics.

The resulting exponential rate is strictly below the positive leading
eigenvalue. Normalization by the leading exponential therefore makes the
complementary component decay. No diagonalizability or boundedness of the
full rescaled configuration is assumed.
-/

import Transformer.Clusters.Section10_PowerContraction
import Transformer.Clusters.Section10_ComplementaryProjection
import Transformer.Clusters.Section9_Growth

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The spectral-gap metric composed with the complementary projection
satisfies the exact operator inequality needed by the growth theorem.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`, and §10. -/
theorem exists_complementary_contraction (V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G) :
    ∃ (R : ℝ) (B : ParamMatrix d), 0 < R ∧ R < lam ∧
      (∀ x : EucSpace d, ‖x‖ ≤ ‖B x‖) ∧
      IsProjContraction V (B.comp (1 - proj)) R := by
  obtain ⟨R, B, hR, hRlam, hdom, hcontract⟩ :=
    hV.2.2.2.2.2.exists_contracting_powerMetric G V lam
  refine ⟨R, B, hR, hRlam, hdom, fun x => ?_⟩
  simp only [ContinuousLinearMap.comp_apply]
  rw [hproj.complement_apply_value proj V F G lam hV]
  apply hcontract
  simpa only [sub_apply, one_apply_eq_self] using
    hproj.mem_complement proj F G hV.2.2.1 x

/-- Paranormality and the specified projection coexist for the identity
value map. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example : IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ :=
  ⟨(isGoodTripleMulti_one d).2, isProjOnto_one d⟩

/-- **The actual unrescaled complementary tokens grow at a rate below
the leading eigenvalue.** Source: arXiv:2305.05465v6, §9, `e:notdi`, and
§10, the spectral-gap estimate used in `t:multiplicity`.
The contraction metric is constructed from the spectral hypothesis. -/
theorem TransformerDynamics.exists_bound_complementary (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (X : ℝ → Idx n → EucSpace d) (hX : TransformerDynamics Q K V X) :
    ∃ R C : ℝ, 0 < R ∧ R < lam ∧ 0 < C ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
        ‖(1 - proj) (X t i)‖ ≤ C * Real.exp (R * t) := by
  obtain ⟨R, B, hR, hRlam, hdom, hcontract⟩ :=
    exists_complementary_contraction V proj F G lam hV hproj
  obtain ⟨C, hC, hbound⟩ := norm_proj_le Q K V (B.comp (1 - proj)) R hR.le
    hcontract X hX
  refine ⟨R, C, hR, hRlam, hC, fun t ht i => ?_⟩
  exact (hdom ((1 - proj) (X t i))).trans (hbound t ht i)

/-- A genuine evolving solution with nonzero tokens satisfies every
growth input: the identity value map sends a stationary rescaled token
to its exponentially growing unrescaled trajectory.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §9, `e:notdi`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    TransformerDynamics (1 : ParamMatrix 1) 1 1
      (fun t (_ : Idx 2) => expTime 1 t (basePoint 0 : EucSpace 1)) := by
  refine ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, ?_⟩
  exact (transformerDynamics_iff_rescaled 1 1 1 _
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) (fun _ _ => rfl)).mpr
    (rescaledDynamics_one_const 1 1 _)

/-- The same growth estimate applies to `e^{tV} z_i(t)` for every actual
solution of the rescaled ODE. Source: arXiv:2305.05465v6, §3, `e:Rres`,
§9, `e:notdi`, and §10, the complementary component in the scores. -/
theorem RescaledDynamics.exists_bound_unrescaled_complementary
    (Q K V proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ R C : ℝ, 0 < R ∧ R < lam ∧ 0 < C ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
        ‖(1 - proj) (expTime V t (Z t i))‖ ≤ C * Real.exp (R * t) := by
  have hX : TransformerDynamics Q K V (fun t i => expTime V t (Z t i)) :=
    (transformerDynamics_iff_rescaled Q K V _ Z (fun _ _ => rfl)).mpr hZ
  exact hX.exists_bound_complementary Q K V proj F G lam hV hproj _

/-- The splitting, projection, and actual rescaled ODE inputs are all
satisfied by coincident nonzero tokens.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §5, `d:goodmulti`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- **After leading normalization the complementary component decays
exponentially along the actual dynamics.**
Source: arXiv:2305.05465v6, §9, `e:majorizeweights`, `r:notdiagonalizable`,
and §10, the separation of leading and complementary attention terms.
This supplies a decaying error without assuming the rescaled complement
is bounded. The decay rate comes from the strict spectral gap. -/
theorem RescaledDynamics.exists_bound_normalized_complementary
    (Q K V proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ C : ℝ, 0 < δ ∧ 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      Real.exp (-lam * t) * ‖(1 - proj) (expTime V t (Z t i))‖ ≤
        C * Real.exp (-δ * t) := by
  obtain ⟨R, C, _, hRlam, hC, hbound⟩ :=
    hZ.exists_bound_unrescaled_complementary Q K V proj F G lam hV hproj Z
  refine ⟨lam - R, C, sub_pos.mpr hRlam, hC, fun t ht i => ?_⟩
  calc
    Real.exp (-lam * t) * ‖(1 - proj) (expTime V t (Z t i))‖ ≤
        Real.exp (-lam * t) * (C * Real.exp (R * t)) :=
      mul_le_mul_of_nonneg_left (hbound t ht i) (Real.exp_pos _).le
    _ = C * Real.exp (-(lam - R) * t) := by
      rw [← mul_assoc, mul_comm (Real.exp (-lam * t)) C, mul_assoc, ← Real.exp_add]
      congr 1
      congr 1
      ring

/-- Every input for the normalized decay estimate is satisfied by an
actual solution; the complementary part is zero in this identity case.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, `π_G`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- The normalized complementary norm actually tends to zero for each
token. Source: arXiv:2305.05465v6, §9, `e:majorizeweights`, and §10,
the strict separation between the leading and complementary scores. -/
theorem RescaledDynamics.tendsto_normalized_complementary
    (Q K V proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    Tendsto (fun t => Real.exp (-lam * t) *
      ‖(1 - proj) (expTime V t (Z t i))‖) atTop (nhds 0) := by
  obtain ⟨δ, C, hδ, _, hbound⟩ :=
    hZ.exists_bound_normalized_complementary Q K V proj F G lam hV hproj Z
  have hlim : Tendsto (fun t : ℝ => C * Real.exp (-δ * t)) atTop (nhds 0) := by
    simpa only [Function.comp_def, id_eq, neg_mul, mul_zero] using
      (Real.tendsto_exp_neg_atTop_nhds_zero.comp
        (Filter.Tendsto.const_mul_atTop hδ tendsto_id)).const_mul C
  refine squeeze_zero' (Eventually.of_forall fun t =>
    mul_nonneg (Real.exp_pos _).le (norm_nonneg _)) ?_ hlim
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
  exact hbound t ht i

/-- The entire dynamical input holds for two coincident nonzero tokens.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, `π_G`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters

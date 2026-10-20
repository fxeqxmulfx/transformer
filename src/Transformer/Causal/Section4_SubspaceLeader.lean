module

public import Transformer.Causal.Section3_TransverseDecay
public import Transformer.Causal.Section4_ParticleDrift

/-!
# The actual leader converges to its leading projection

ArXiv:2411.04990v2, §3, `lemma1`, and §4, `thm2`.
The strict transverse quadratic condition gives exponential decay of
the original trajectory's positive linear lift. Its leading projection
is fixed. Normalizing the lift therefore gives the claimed limit of
the original leader, for every initial point with nonzero leading
projection and arbitrary original query and key matrices.

This proves the first-token part of the conjecture and its full one-token
case. Convergence of all followers remains a separate open assertion;
it is not assumed anywhere in these proofs.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- The original projected value solution converges to its normalized
initial leading component under the exact spectral assumptions of `thm2`.
The leading eigenvalue need not be positive and its multiplicity need
not exceed one for this single-token conclusion.
Source: arXiv:2411.04990v2, §3, `lemma1`, specialized as in §4, `thm2`. -/
theorem singleToken_tendsto_normalized_starProjection (V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x)
    (hunit : ‖x 0‖ = 1) (hp : L.starProjection (x 0) ≠ 0) :
    Tendsto x atTop (nhds (‖L.starProjection (x 0)‖⁻¹ • L.starProjection (x 0))) := by
  let y := singleTokenLift V lam x
  let p := L.starProjection (x 0)
  obtain ⟨c, hc, hdec⟩ := singleTokenLift_transverse_decay V L lam heig hinv hdecay x hx
  have he : Tendsto (fun t : ℝ => Real.exp (-(2 * c * t))) atTop (nhds 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.const_mul_atTop (by positivity : 0 < 2 * c))
  have hsq : Tendsto (fun t => ‖y t - p‖ ^ 2) atTop (nhds 0) := by
    apply squeeze_zero' (Eventually.of_forall fun t => sq_nonneg _)
      ((eventually_ge_atTop (0 : ℝ)).mono fun t ht => hdec t ht)
    simpa only [mul_zero] using he.const_mul (‖x 0 - p‖ ^ 2)
  have hn : Tendsto (fun t => ‖y t - p‖) atTop (nhds 0) := by
    simpa only [Function.comp_def, Real.sqrt_sq_eq_abs, abs_norm, Real.sqrt_zero] using
      Real.continuous_sqrt.continuousAt.tendsto.comp hsq
  have hy : Tendsto y atTop (nhds p) := by
    have hz := (tendsto_zero_iff_norm_tendsto_zero.mpr hn).add_const p
    simpa only [sub_add_cancel, zero_add] using hz
  have hnormalize := (hy.norm.inv₀ (norm_ne_zero_iff.mpr hp)).smul hy
  apply hnormalize.congr
  intro t
  have hyt : ‖y t‖ = singleTokenScale V lam x t := by
    change ‖singleTokenScale V lam x t • x t‖ = _
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (singleTokenScale_pos V lam x t),
      norm_eq_one_of_singleTokenODE V hx hunit t, mul_one]
  rw [hyt]
  change (singleTokenScale V lam x t)⁻¹ • (singleTokenScale V lam x t • x t) = x t
  rw [smul_smul, inv_mul_cancel₀ (singleTokenScale_pos V lam x t).ne', one_smul]

/-- The first token of every actual causal attention solution has the
normalized leading-projection limit. No exceptional initialization is
needed beyond the explicit nonzero projection, and no score assumption
is used. Source: arXiv:2411.04990v2, §4, `thm2`, the leader furnished by `lemma1`. -/
theorem csaLeader_tendsto_normalized_starProjection (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X)
    (hp : L.starProjection (X 0 0 : EucSpace d) ≠ 0) :
    Tendsto (fun t => (X t 0 : EucSpace d)) atTop
      (nhds (‖L.starProjection (X 0 0 : EucSpace d)‖⁻¹ •
        L.starProjection (X 0 0 : EucSpace d))) :=
  singleToken_tendsto_normalized_starProjection V L lam heig hinv hdecay _
    (csaLeader_singleTokenODE β Q K V X hX) (mem_sphere_zero_iff_norm.mp (X 0 0).2) hp

/-- The complete leading-subspace conclusion holds pointwise for one
token, at arbitrary scores, under the same spectral conditions.
Source: arXiv:2411.04990v2, §4, `thm2`, restricted to one token. -/
theorem subspace_cluster_singleton (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (X : ℝ → SphereTuple d 1) (hX : CSA d 1 β Q K V X)
    (hp : L.starProjection (X 0 0 : EucSpace d) ≠ 0) :
    ∀ i : Idx 1, Tendsto (fun t => (X t i : EucSpace d)) atTop
      (nhds (‖L.starProjection (X 0 0 : EucSpace d)‖⁻¹ •
        L.starProjection (X 0 0 : EucSpace d))) := by
  intro i
  rw [Subsingleton.elim i (0 : Idx 1)]
  exact csaLeader_tendsto_normalized_starProjection β Q K V L lam heig hinv hdecay X hX hp

/-- A genuine original causal flow with any nonempty tuple satisfies all
leader hypotheses, at arbitrary original scores. The identity value map
gives the full leading plane and a nonzero initial projection.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) (m : ℕ) :
    ∃ (X : ℝ → SphereTuple 2 (m + 1)),
      (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
        inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
      CSA 2 (m + 1) β Q K (ContinuousLinearMap.id ℝ _) X ∧
      (⊤ : Submodule ℝ (EucSpace 2)).starProjection (X 0 0 : EucSpace 2) ≠ 0 := by
  let X := csaFlow β Q K (ContinuousLinearMap.id ℝ _) (fun _ => basePoint 1 : SphereTuple 2 (m + 1))
  refine ⟨X, fun v _ => by simp, fun z hz => hz, ?_, (csaFlow_spec β Q K _ _).2, ?_⟩
  · intro z hz hz0
    have hz' : z = 0 := by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz
    exact (hz0 hz').elim
  · simp only [X, csaFlow_zero, Submodule.starProjection_top, ContinuousLinearMap.id_apply]
    exact norm_ne_zero_iff.mp (by rw [mem_sphere_zero_iff_norm.mp (basePoint 1).2]; norm_num)

end Transformer.Causal

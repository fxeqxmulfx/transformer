/-
# The unrestricted nGPT clustering claim has a zero-step counterexample

ArXiv:2510.22026v2, §3, `thm:convergence`, includes nGPT for the
layer-dependent real parameter `α` of Table 1 without a step-size
hypothesis. With `α=0`, its actual angular update is zero. Two distinct
unit tokens therefore remain distinct for all time, even in dimension
two. The radial variables of nGPT remain constant as in Table 2.

The counterexample uses the actual uniform product law. Its two tokens
are linearly independent almost surely, so they are distinct and their
actual attention vectors are nonzero. Thus the original normalization
of the attention vector is defined on this event; the failure is not
caused by a zero attention vector or a one-dimensional tangent space.

Appendix C, the nGPT paragraph after its overlap ODE, explicitly assumes
that `∫₀ᵗ α(s) ds` diverges before concluding convergence. That assumption
is absent from `thm:convergence`. Appendix D's nGPT time change also
needs an unbounded clock (and prints the inverse step size there).
The unrestricted first half is refuted below without replacing it by
a weaker theorem under the source's name. The Gaussian alternative
and its conditional corollary remain in `Clustering`.
-/

import Transformer.Normalization.ClusteringLine
import Transformer.Perspective.UniformHemisphere
import Transformer.Metastability.Section4_UniformLaw

open Real MeasureTheory Filter
open scoped BigOperators

namespace Transformer.Normalization

/-- Zero step size freezes the actual nGPT angular and radial equations.
Source: arXiv:2510.22026v2, §2, Tables 1 and 2, and §3,
`thm:convergence`. In Table 1 the angular velocity is
`α(t) • proj θ (‖A‖⁻¹ • A)`, which is zero at `α=0` as well. -/
theorem schemeDynamics_ngpt_zero_step {d n : ℕ} (β τ : ℝ)
    (Q K V : ℝ → ParamMatrix d) (Θ : Idx n → EucSpace d) (r₀ : Idx n → ℝ) :
    SchemeDynamics d n β Q K V 0 τ .nGPT (fun _ => Θ) (fun _ => r₀) := by
  intro t _ j
  constructor
  · simpa only [speedFactor, Pi.zero_apply, inv_zero, zero_mul, zero_smul] using
      (hasDerivWithinAt_const t (Set.Ici 0) (Θ j))
  · exact hasDerivWithinAt_const t (Set.Ici 0) (r₀ j)

/-- An independent nonempty tuple has a nonzero actual attention vector
with identity value map. Source: arXiv:2510.22026v2, §2,
the positive exponential weights in `A_j`, used in the nGPT row of
Table 1. All actual query and key parameters are retained. -/
theorem attentionVec_id_ne_zero_of_linearIndependent {d n : ℕ}
    (β : ℝ) (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d)
    (hΘ : LinearIndependent ℝ Θ) (j : Idx n) :
    attentionVec d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) Θ j ≠ 0 := by
  let w : Idx n → ℝ := fun k => Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k)))
  have hZ : 0 < ∑ k, w k :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩
  have hsum : (∑ k, w k • Θ k) ≠ 0 := by
    intro hz
    have hj := (linearIndependent_iff'.mp hΘ) Finset.univ w hz j (Finset.mem_univ j)
    exact (Real.exp_ne_zero _) hj
  change (∑ k, w k)⁻¹ • (∑ k, w k • Θ k) ≠ 0
  intro hz
  exact hsum ((smul_eq_zero.mp hz).resolve_left (inv_ne_zero hZ.ne'))

/-- The independence input is realized by the two actual unit coordinate
vectors in dimension two. Source: arXiv:2510.22026v2, §3,
the nondegenerate uniform initialization in `thm:convergence`. -/
example : LinearIndependent ℝ (fun j : Idx 2 => (EuclideanSpace.single j 1 : EucSpace 2)) ∧
    (∀ j : Idx 2, ‖(EuclideanSpace.single j 1 : EucSpace 2)‖ = 1) := by
  refine ⟨?_, fun j => by simp [PiLp.norm_single]⟩
  have h := (EuclideanSpace.basisFun (Fin 2) ℝ).toBasis.linearIndependent
  rw [OrthonormalBasis.coe_toBasis] at h
  convert h using 1
  funext j
  exact (EuclideanSpace.basisFun_apply (Fin 2) ℝ j).symm

/-- Two distinct actual unit vectors, positive temperature and nonzero
attention vectors realize the zero-step path explicitly. The source's
Table 1 angular update is defined at every time for this configuration.
Source: arXiv:2510.22026v2, §3, `thm:convergence`. -/
example : SchemeDynamics 2 2 1 (idParams 2) (idParams 2) (idParams 2) 0 0 .nGPT
      (fun _ j => (EuclideanSpace.single j 1 : EucSpace 2)) (fun _ _ => 1) ∧
    ¬ Synchronizes 2 2 (fun _ j => (EuclideanSpace.single j 1 : EucSpace 2)) ∧
    ∀ j : Idx 2, attentionVec 2 2 1 (idParams 2 0) (idParams 2 0) (idParams 2 0)
      (fun k => (EuclideanSpace.single k 1 : EucSpace 2)) j ≠ 0 := by
  have hlin : LinearIndependent ℝ (fun j : Idx 2 => (EuclideanSpace.single j 1 : EucSpace 2)) := by
    have h := (EuclideanSpace.basisFun (Fin 2) ℝ).toBasis.linearIndependent
    rw [OrthonormalBasis.coe_toBasis] at h
    convert h using 1
    funext j
    exact (EuclideanSpace.basisFun_apply (Fin 2) ℝ j).symm
  refine ⟨schemeDynamics_ngpt_zero_step 1 0 _ _ _ _ _, ?_, fun j =>
    attentionVec_id_ne_zero_of_linearIndependent 1 _ _ _ hlin j⟩
  apply not_synchronizes_const _ 0 1
  intro heq
  have h01 : (0 : Idx 2) = 1 := hlin.injective heq
  norm_num at h01

/-- The actual two-token zero-step nGPT path fails to synchronize almost
surely under every original uniform law on the circle. Its normalization
is defined: every actual attention vector is nonzero at every time.
Source: arXiv:2510.22026v2, §3, `thm:convergence`, nGPT clause. -/
theorem ngpt_zero_step_counterexample_ae (P : Measure (SphereTuple 2 2))
    (hP : Perspective.UniformTuple 2 2 P) :
    ∀ᵐ Θ₀ ∂P,
      SchemeDynamics 2 2 1 (idParams 2) (idParams 2) (idParams 2) 0 0 .nGPT
        (fun _ => tupleCoe Θ₀) (fun _ _ => 1) ∧
      (∀ t : ℝ, ∀ j : Idx 2,
        attentionVec 2 2 1 (idParams 2 t) (idParams 2 t) (idParams 2 t)
          (tupleCoe Θ₀) j ≠ 0) ∧
      ¬ Synchronizes 2 2 (fun _ => tupleCoe Θ₀) := by
  filter_upwards [Perspective.ae_linearIndependent_of_uniformTuple 2 le_rfl P hP]
    with Θ₀ hΘ
  refine ⟨schemeDynamics_ngpt_zero_step 1 0 _ _ _ _ _, ?_, ?_⟩
  · intro t j
    exact attentionVec_id_ne_zero_of_linearIndependent 1 (idParams 2 t) (idParams 2 t)
      (tupleCoe Θ₀) hΘ j
  · apply not_synchronizes_const (tupleCoe Θ₀) 0 1
    intro heq
    have h01 : (0 : Idx 2) = 1 := hΘ.injective heq
    norm_num at h01

/-- The full uniform-law hypothesis above is realized by the constructed
spherical Lebesgue law, with two independent actual circle tokens.
Source: arXiv:2510.22026v2, §3, `thm:convergence`. -/
example : Perspective.UniformTuple 2 2
    (Measure.pi fun _ : Idx 2 => MeanField.uniformLaw 2) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨MeanField.uniformLaw 2, hν.1, hν.2, rfl⟩

/-- **The unrestricted first half of `thm:convergence` is false.**
Even after excluding dimension one, it asserts synchronization for every
real step-size function in nGPT. At `d=n=2`, `β=1`, `α=0`, the actual
uniform initialization is independent almost surely and the actual paths
remain fixed. Appendix C requires a diverging accumulated step size;
the printed theorem has no such hypothesis. The negated statement retains
all schemes, parameters, initial data and actual equations of the claim.
Source: arXiv:2510.22026v2, §3, `thm:convergence`, first half,
§2, Table 1, and Appendix C, the nGPT overlap equation. -/
theorem not_clusters_from_uniform :
    ¬ ∀ (d n : ℕ), 2 ≤ d → ∀ (β : ℝ) (α : ℝ → ℝ) (τ : ℝ) (scheme : Scheme),
      (scheme = .post ∨ scheme = .nGPT ∨ scheme = .CoD) →
      ∀ P : Measure (SphereTuple d n), Perspective.UniformTuple d n P →
      ∀ᵐ Θ₀ ∂P, ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
        θ 0 = tupleCoe Θ₀ →
        SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
          Synchronizes d n θ := by
  intro h
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  let P : Measure (SphereTuple 2 2) := Measure.pi fun _ => MeanField.uniformLaw 2
  have hP : Perspective.UniformTuple 2 2 P := ⟨MeanField.uniformLaw 2, hν.1, hν.2, rfl⟩
  have hcluster := h 2 2 le_rfl 1 0 0 .nGPT (Or.inr (Or.inl rfl)) P hP
  have hfalse : ∀ᵐ Θ₀ ∂P, False := by
    filter_upwards [hcluster, ngpt_zero_step_counterexample_ae P hP] with Θ₀ hgood hbad
    exact hbad.2.2 (hgood (fun _ => tupleCoe Θ₀) (fun _ _ => 1) rfl hbad.1)
  exact Filter.eventually_const.mp hfalse

end Transformer.Normalization

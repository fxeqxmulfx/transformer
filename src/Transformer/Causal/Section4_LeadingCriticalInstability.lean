module

public import Transformer.Causal.Section4_TransverseLinearization
public import Transformer.Causal.Section4_SubspaceTransverse

/-!
# Positive tangent eigenvalues at negative leading critical points

ArXiv:2411.04990v2, §4, `thm2`, and its preceding discussion of critical
manifolds. At an actual critical last token of a leading consensus prefix,
negative longitudinal overlap gives a positive common leading coefficient.
Every transverse leading component vanishes there. Differentiating its
exact equation therefore makes it a left eigenvector of the actual field
derivative. Leading dimension at least two supplies a nonzero such vector.

The resulting positive right eigenvalue is tangent to the original sphere
and belongs to the actual derivative, including the original attention
weight derivatives. Neither a symmetric value matrix nor a frozen-weight
approximation is used. This local algebraic instability does not establish
measure zero of the basin of a general moving-prefix trajectory.
-/

@[expose] public section

open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- A transverse leading component vanishes at a negative critical
follower of the original leading consensus prefix.
Source: arXiv:2411.04990v2, §4, the critical-manifold analysis of `thm2`. -/
theorem inner_csaFrozenCritical_transverse_eq_zero (n : ℕ) (hn : 1 ≤ n) (β : ℝ)
    (Q K V : ParamMatrix d) (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (x : EucSpace d)
    (hcrit : csaFrozenFollowerField n β Q K V u x = 0)
    (hq : inner ℝ (u : EucSpace d) x < 0) (v : EucSpace d) (hv : v ∈ L)
    (hvu : inner ℝ v u = 0) : inner ℝ v x = 0 := by
  have hc := csaFrozenLeadingCoefficient_pos_of_critical n hn β Q K V L lam hlam
    heig hinv u hu x hcrit hq
  have h := inner_csaFrozenFollowerField n β Q K V L lam heig hinv v hv u x
  rw [hcrit, inner_zero_right, hvu, mul_zero, zero_add] at h
  exact (mul_eq_zero.mp h.symm).resolve_left hc.ne'

/-- The exact actual last-row derivative has the actual leading
coefficient as a left eigenvalue on every transverse leading covector.
Source: arXiv:2411.04990v2, §4, the critical configurations in `thm2`.
The derivative of the original scalar coefficient is included and
vanishes from this component because the actual component is zero. -/
theorem inner_fderiv_csaFrozenFollowerField_transverse (n : ℕ) (hn : 1 ≤ n) (β : ℝ)
    (Q K V : ParamMatrix d) (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (x : EucSpace d)
    (hcrit : csaFrozenFollowerField n β Q K V u x = 0)
    (hq : inner ℝ (u : EucSpace d) x < 0) (v : EucSpace d) (hv : v ∈ L)
    (hvu : inner ℝ v u = 0) (w : EucSpace d) :
    inner ℝ v (fderiv ℝ (csaFrozenFollowerField n β Q K V u) x w) =
      csaFrozenLeadingCoefficient n β Q K V lam u x * inner ℝ v w := by
  apply inner_fderiv_of_transverse_factor (csaFrozenFollowerField n β Q K V u)
    (csaFrozenLeadingCoefficient n β Q K V lam u) v x
    ((contDiff_csaFrozenFollowerField n β Q K V u).differentiable (by simp) x)
    ((contDiff_csaFrozenLeadingCoefficient n β Q K V lam u).differentiable (by simp) x)
    ?_ (inner_csaFrozenCritical_transverse_eq_zero n hn β Q K V L lam hlam
      heig hinv u hu x hcrit hq v hv hvu) w
  intro y
  simpa only [hvu, mul_zero, zero_add] using
    inner_csaFrozenFollowerField n β Q K V L lam heig hinv v hv u y

/-- The actual derivative maps each original tangent direction back to
the original tangent space at an actual critical follower.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
theorem csaFrozenFollowerField_fderiv_tangent (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (u : EucSpace d) (x : SSphere d)
    (hcrit : csaFrozenFollowerField n β Q K V u x = 0) (w : EucSpace d)
    (hxw : inner ℝ (x : EucSpace d) w = 0) :
    inner ℝ (x : EucSpace d) (fderiv ℝ (csaFrozenFollowerField n β Q K V u) x w) = 0 := by
  let a : EucSpace d → EucSpace d := fun y =>
    csaAverage β Q K V (Fin.snoc (fun _ : Idx n => u) y) (Fin.last n)
  have ha : DifferentiableAt ℝ a x :=
    ((contDiff_csaAverage β Q K V (Fin.last n)).comp
      (contDiff_csaConsensusInsertion n u)).differentiable (by simp) x
  have he : (fun y => proj d y (a y)) = csaFrozenFollowerField n β Q K V u := by
    funext y
    simp only [csaFrozenFollowerField, csaAmbientField, Fin.snoc_last, a]
  have h := inner_fderiv_projectedField_of_critical a x ha (by
    change (fun y => proj d y (a y)) x = 0
    rw [he]
    exact hcrit) w hxw
  rwa [he] at h

/-- Every negative-overlap critical follower of a leading consensus
prefix has a positive real eigenvalue with a nonzero original tangent
eigenvector. All actual query, key, value, weight and projection
derivatives are retained. Source: arXiv:2411.04990v2, §4, the
critical-manifold obstruction preceding `thm2`.
The strict transverse decay hypothesis is not needed for this local result. -/
theorem csaFrozenNegativeCritical_exists_positive_tangent_eigenvalue
    (n : ℕ) (hn : 1 ≤ n) (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdim : 2 ≤ Module.finrank ℝ L) (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (x : SSphere d) (hcrit : csaFrozenFollowerField n β Q K V u x = 0)
    (hq : inner ℝ (u : EucSpace d) (x : EucSpace d) < 0) :
    ∃ c : ℝ, 0 < c ∧ ∃ w : EucSpace d, w ≠ 0 ∧ inner ℝ (x : EucSpace d) w = 0 ∧
      fderiv ℝ (csaFrozenFollowerField n β Q K V u) x w = c • w := by
  let c := csaFrozenLeadingCoefficient n β Q K V lam u x
  have hc : 0 < c := csaFrozenLeadingCoefficient_pos_of_critical n hn β Q K V L lam
    hlam heig hinv u hu x hcrit hq
  have hp : L.starProjection (u : EucSpace d) ≠ 0 := by
    rw [L.starProjection_eq_self_iff.mpr hu]
    exact norm_ne_zero_iff.mp (by rw [mem_sphere_zero_iff_norm.mp u.2]; norm_num)
  obtain ⟨v, hv, hv0, hvu⟩ := exists_leading_transverse_normal L hdim u hp
  have hvx := inner_csaFrozenCritical_transverse_eq_zero n hn β Q K V L lam hlam
    heig hinv u hu x hcrit hq v hv hvu
  obtain ⟨w, hw0, hxw, hw⟩ := exists_tangent_eigenvector_of_left_inner
    (fderiv ℝ (csaFrozenFollowerField n β Q K V u) x) x v hv0 hvx c hc
    (inner_fderiv_csaFrozenFollowerField_transverse n hn β Q K V L lam hlam
      heig hinv u hu x hcrit hq v hv hvu)
  refine ⟨c, hc, w, hw0, hxw, ?_⟩
  rwa [csaTangentProjection_apply, proj,
    csaFrozenFollowerField_fderiv_tangent n β Q K V u x hcrit w hxw,
    zero_smul, sub_zero] at hw

/-- The source hypotheses and every actual negative critical-point
premise hold jointly for the antipodal pair with two leading dimensions.
The nonzero second coordinate supplies a transverse leading covector.
Source: arXiv:2411.04990v2, §4, `thm2`, its exceptional critical state. -/
example : ∃ (u x : SSphere 2) (v : EucSpace 2), (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (∀ y ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) y = (1 : ℝ) • y) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    csaFrozenFollowerField (d := 2) 1 1 0 0 (ContinuousLinearMap.id ℝ _) u x = 0 ∧
    inner ℝ (u : EucSpace 2) (x : EucSpace 2) < 0 ∧
    v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧ v ≠ 0 ∧ inner ℝ v u = 0 ∧
    inner ℝ (x : EucSpace 2) v = 0 := by
  let u : SSphere 2 := basePoint 1
  let x : SSphere 2 := ⟨-(u : EucSpace 2), by simp [mem_sphere_zero_iff_norm.mp u.2]⟩
  let v : EucSpace 2 := EuclideanSpace.single (1 : Fin 2) 1
  refine ⟨u, x, v, le_rfl, one_pos, fun y _ => by simp, fun z hz => hz,
    by simp, Submodule.mem_top, ?_, ?_, Submodule.mem_top, ?_, ?_, ?_⟩
  · simp [x, csaFrozenFollowerField, csaAmbientField, csaAverage, csaTokenWeight,
      csaPartition, Fin.sum_univ_two, Fin.snoc, proj]
  · simp [x, mem_sphere_zero_iff_norm.mp u.2]
  · intro hv
    have he := congrArg (fun z : EucSpace 2 => z (1 : Fin 2)) hv
    simp [v] at he
  · simp [u, v, basePoint, EuclideanSpace.inner_single_left]
  · rw [real_inner_comm v (x : EucSpace 2)]
    simp [x, u, v, basePoint, EuclideanSpace.inner_single_left]

end Transformer.Causal

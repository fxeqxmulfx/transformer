module

public import Transformer.Causal.Section4_LeadingCriticalInstability
public import Transformer.Causal.Section4_NegativeLeadingBranch

/-!
# Actual non-consensus follower limits have positive tangent eigenvalues

ArXiv:2411.04990v2, §4, `thm2`, and its critical-manifold discussion.
An actual converging original prefix and actual converging original last
token give a critical point of the original limiting follower field.
Indeed, a nonzero limiting velocity would give an eventually positive
scalar drift on a bounded unit-sphere trajectory.

If this actual last limit differs from the leading prefix limit, the
proved negative-branch dichotomy puts its overlap strictly below zero.
The preceding exact derivative calculation supplies a positive real
tangent eigenvalue there. This does not assume that every follower has
a limit and does not assert nullity of the exceptional moving-prefix basin.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Actual original token limits give the limit of the actual original
last velocity field. This uses the original continuous weights, not an
independent limiting-weight assumption. Source: arXiv:2411.04990v2,
§4, the critical-manifold analysis for `thm2`. -/
theorem tendsto_csaLastField_of_limits (β : ℝ) (Q K V : ParamMatrix d)
    (u x : EucSpace d) (X : ℝ → SphereTuple d (n + 1))
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop (nhds u))
    (hx : Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop (nhds x)) :
    Tendsto (fun t => csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) (Fin.last n))
      atTop (nhds (csaFrozenFollowerField n β Q K V u x)) := by
  let Y : Idx (n + 1) → EucSpace d := Fin.snoc (fun _ : Idx n => u) x
  have hY : Tendsto (fun t => fun j => (X t j : EucSpace d)) atTop (nhds Y) := by
    apply tendsto_pi_nhds.mpr
    intro j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · simpa only [Y, Fin.snoc_last] using hx
    · simpa only [Y, Fin.snoc_castSucc] using hp k
  have hf := ((contDiff_csaAmbientField β Q K V).continuous.tendsto Y).comp hY
  exact (tendsto_pi_nhds.mp hf) (Fin.last n)

/-- Actual original prefix and last-token limits give an actual critical
point of the original limiting follower field. No spectral or query/key
assumptions are required for this bounded-velocity fact.
Source: arXiv:2411.04990v2, §4, the limiting critical configurations. -/
theorem csaFrozenFollowerField_eq_zero_of_limits (β : ℝ) (Q K V : ParamMatrix d)
    (u x : EucSpace d) (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop (nhds u))
    (hx : Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop (nhds x)) :
    csaFrozenFollowerField n β Q K V u x = 0 := by
  have hl := tendsto_csaLastField_of_limits β Q K V u x X hp hx
  by_contra hz
  let y := csaFrozenFollowerField n β Q K V u x
  let r : ℝ → ℝ := fun t => inner ℝ y (X t (Fin.last n) : EucSpace d)
  let r' : ℝ → ℝ := fun t => inner ℝ y (csaAmbientField β Q K V
    (fun j => (X t j : EucSpace d)) (Fin.last n))
  have hy : 0 < inner ℝ y y := real_inner_self_pos.mpr hz
  have hr (t : ℝ) : HasDerivAt r (r' t) t :=
    hasDerivAt_csaLast_fieldOverlap β Q K V X hX y t
  have hlim : Tendsto r' atTop (nhds (inner ℝ y y)) := tendsto_const_nhds.inner hl
  let c := inner ℝ y y / 2
  have hc : 0 < c := div_pos hy (by norm_num)
  have hhalf : c < inner ℝ y y := by dsimp [c]; linarith
  obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually (lt_mem_nhds hhalf))
  have hbound (t : ℝ) : r t ≤ ‖y‖ := by
    calc
      r t ≤ |inner ℝ y (X t (Fin.last n) : EucSpace d)| := le_abs_self _
      _ ≤ ‖y‖ * ‖(X t (Fin.last n) : EucSpace d)‖ := abs_real_inner_le_norm _ _
      _ = ‖y‖ := by rw [mem_sphere_zero_iff_norm.mp (X t (Fin.last n)).2, mul_one]
  have hbad := csaScalar_lt_of_tail_growth r r' ‖y‖ (r T) c T
    (fun t _ => hr t) (fun t _ => hbound t) hc (fun t ht _ => (hT t ht).le) T le_rfl
  exact (lt_irrefl (r T)) hbad

/-- Any actual non-consensus limit of the original last token has a
positive tangent eigenvalue of its actual limiting original field.
Source: arXiv:2411.04990v2, §4, a necessary instability property for
exceptional convergent trajectories in `thm2`. Actual prefix convergence
and existence of this last limit are explicit; general follower
convergence and exceptional-basin nullity remain separate assertions. -/
theorem csaLast_nonconsensus_limit_has_positive_tangent_eigenvalue
    (β : ℝ) (Q K V : ParamMatrix d) (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hlam : 0 < lam) (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (hdim : 2 ≤ Module.finrank ℝ L) (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d)))
    (x : SSphere d) (hx : Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop
      (nhds (x : EucSpace d))) (hxu : x ≠ u) :
    ∃ c : ℝ, 0 < c ∧ ∃ w : EucSpace d, w ≠ 0 ∧ inner ℝ (x : EucSpace d) w = 0 ∧
      fderiv ℝ (csaFrozenFollowerField n β Q K V u) x w = c • w := by
  have hbad : ¬Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop
      (nhds (u : EucSpace d)) := by
    intro hg
    exact hxu (Subtype.ext (tendsto_nhds_unique hx hg))
  obtain ⟨δ, hδ, hneg, -⟩ := csaLast_negative_leading_of_not_tendsto β Q K V L lam hlam
    heig hinv hdecay u hu X hX hn hp hbad
  have hq : inner ℝ (u : EucSpace d) (x : EucSpace d) ≤ -δ :=
    le_of_tendsto (tendsto_const_nhds.inner hx) hneg
  exact csaFrozenNegativeCritical_exists_positive_tangent_eigenvalue n hn β Q K V L lam
    hlam heig hinv hdim u hu x (csaFrozenFollowerField_eq_zero_of_limits β Q K V u x X hX hp hx)
    (by linarith)

/-- The actual stationary antipodal original pair realizes every
spectral, prefix, limit and non-consensus premise jointly.
Source: arXiv:2411.04990v2, §4, an exceptional initialization of `thm2`. -/
example : ∃ (X : ℝ → SphereTuple 2 2) (u x : SSphere 2),
    (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
      inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
    2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    CSA 2 2 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X ∧
    (∀ j : Idx 1, Tendsto (fun t => (X t j.castSucc : EucSpace 2)) atTop
      (nhds (u : EucSpace 2))) ∧
    Tendsto (fun t => (X t (Fin.last 1) : EucSpace 2)) atTop (nhds (x : EucSpace 2)) ∧ x ≠ u := by
  let u : SSphere 2 := basePoint 1
  let x : SSphere 2 := ⟨-(u : EucSpace 2), by simp [mem_sphere_zero_iff_norm.mp u.2]⟩
  refine ⟨fun _ => antipodalCircleTuple u, u, x, one_pos, le_rfl,
    fun v _ => by simp, fun z hz => hz, ?_, by simp, Submodule.mem_top,
    csa_const_antipodal 1 u, ?_, ?_, ?_⟩
  · intro z hz hz0
    exact (hz0 (by
      simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim
  · intro j
    have hj : j = 0 := by ext; omega
    simpa only [hj, Fin.castSucc_zero, antipodalCircleTuple, Matrix.cons_val_zero] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (u : EucSpace 2)) atTop (nhds (u : EucSpace 2)))
  · simpa only [antipodalCircleTuple, show Fin.last 1 = (1 : Fin 2) from rfl,
      Matrix.cons_val_one, Matrix.cons_val_zero, x] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => -(u : EucSpace 2)) atTop (nhds (-(u : EucSpace 2))))
  · intro he
    have hi := congrArg (fun z : SSphere 2 => inner ℝ (u : EucSpace 2) (z : EucSpace 2)) he
    norm_num [x, u, basePoint] at hi

end Transformer.Causal

/-
# Uniqueness of an original follower approaching a repelling limit

ArXiv:2411.04990v2, §5 and Appendix A, unstable-limit
avoidance for the actual smooth frozen system.
Two actual solutions with the same original prefix and the
same strictly repelling last-angle limit eventually coincide
in their last angle by the proved scalar repulsion argument.
Their full states then coincide at a finite time. Uniqueness
of the original global ODE propagates equality to all times.

For the actual canonical flow, a fixed prefix initialization
therefore permits at most one last initialization approaching
any given repelling last limit. If the allowed limits form a
countable set, their exceptional last-initialization fiber is
countable. A genuine original antipodal equilibrium witnesses
that such an exceptional fiber can be nonempty.
No general center-stable manifold theorem is assumed here.
-/

import Transformer.Causal.Section5_RowDerivative
import Transformer.Causal.Section5_RowConvergence
import Transformer.Causal.Section5_FrozenFlow
import Transformer.Causal.SectionA_ScalarRepulsion

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {n m : ℕ}

/-- Actual original solutions with the same converging prefix and the same strictly repelling last limit coincide on all real times.
Source: arXiv:2411.04990v2, §5 and Appendix A, the conditional original repeller-avoidance argument for `thm:fixed_centers`. -/
theorem frozenODE_eq_of_shared_prefix_repelling_limits (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ ψ : ℝ → Idx (n + 1) → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t)
    (hψ : ∀ t, HasDerivAt ψ (frozenVelocity (n + 1) m β θ a (ψ t)) t)
    (hshared : ∀ t, ∀ j : Idx n, φ t j.castSucc = ψ t j.castSucc)
    (pstar : Idx n → ℝ) (hp : Tendsto (fun t j => φ t j.castSucc) atTop (nhds pstar))
    (l : ℝ) (hpos : 0 < frozenLastVelocityDeriv β θ a pstar l)
    (hx : Tendsto (fun t => φ t (Fin.last n)) atTop (nhds l))
    (hy : Tendsto (fun t => ψ t (Fin.last n)) atTop (nhds l)) : φ = ψ := by
  have hψs := frozenODE_last_scalar β θ a ψ hψ
  have hpre (t : ℝ) : (fun j : Idx n => ψ t j.castSucc) = (fun j => φ t j.castSucc) :=
    funext (fun j => (hshared t j).symm)
  obtain ⟨T, hT⟩ := frozenScalar_eventually_eq_of_repelling_limit
    (fun (p : Idx n → ℝ) u => frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n))
    (frozenLastVelocityDeriv β θ a) (fun p u => hasDerivAt_frozenLastVelocity β θ a ha p u)
    pstar l (continuous_frozenLastVelocityDeriv_joint β θ a ha).continuousAt hpos
    (fun t j => φ t j.castSucc) hp (fun t => φ t (Fin.last n)) (fun t => ψ t (Fin.last n))
    (frozenODE_last_scalar β θ a φ hφ) (fun t => by simpa only [hpre t] using hψs t) hx hy
  have hstate : φ T = ψ T := funext (fun j => Fin.lastCases (hT T le_rfl) (fun k => hshared T k) j)
  have hs := frozenODE_unique β θ a ha (fun t => φ (t + T)) (fun t => ψ (t + T))
    (frozenODE_time_shift β θ a φ hφ T) (frozenODE_time_shift β θ a ψ hψ T)
    (by simpa only [zero_add] using hstate)
  funext t
  simpa only [sub_add_cancel] using congrFun hs (t - T)

/-- With an actual fixed original prefix initialization, a specified repelling last limit has at most one original last initialization.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original conditional exceptional-fiber argument for `thm:fixed_centers`. -/
theorem frozenFlow_last_initial_eq_of_repelling_limits (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p₀ pstar : Idx n → ℝ) (hp : Tendsto (frozenFlow β θ a ha p₀) atTop (nhds pstar))
    (x₀ y₀ l : ℝ) (hpos : 0 < frozenLastVelocityDeriv β θ a pstar l)
    (hx : Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l))
    (hy : Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ y₀) t (Fin.last n)) atTop (nhds l)) : x₀ = y₀ := by
  have hshared : ∀ t, ∀ j : Idx n, frozenFlow β θ a ha (Fin.snoc p₀ x₀) t j.castSucc =
      frozenFlow β θ a ha (Fin.snoc p₀ y₀) t j.castSucc := by
    intro t j
    simp only [frozenFlow_castSucc, Fin.snoc_castSucc]
  have hprefix : Tendsto (fun t j => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t j.castSucc) atTop (nhds pstar) := by
    simpa! only [frozenFlow_castSucc, Fin.snoc_castSucc] using hp
  have hs := frozenODE_eq_of_shared_prefix_repelling_limits β θ a ha _ _
    (frozenFlow_spec β θ a ha (Fin.snoc p₀ x₀)).2 (frozenFlow_spec β θ a ha (Fin.snoc p₀ y₀)).2
    hshared pstar hprefix l hpos hx hy
  have h := congrArg (fun q : ℝ → Idx (n + 1) → ℝ => q 0 (Fin.last n)) hs
  simpa only [frozenFlow_zero, Fin.snoc_last] using h

/-- For a fixed actual prefix and a specified strictly repelling limit, the actual follower-initialization fiber contains at most one point.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original scalar exceptional fiber in `thm:fixed_centers`. -/
theorem frozenFlow_repelling_limit_fiber_subsingleton (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p₀ pstar : Idx n → ℝ) (hp : Tendsto (frozenFlow β θ a ha p₀) atTop (nhds pstar))
    (l : ℝ) (hpos : 0 < frozenLastVelocityDeriv β θ a pstar l) :
    {x₀ : ℝ | Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l)}.Subsingleton := by
  intro x₀ hx y₀ hy
  exact frozenFlow_last_initial_eq_of_repelling_limits β θ a ha p₀ pstar hp x₀ y₀ l hpos hx hy

/-- A countable set of possible actual repelling last limits yields a countable exceptional fiber over each fixed converging original prefix.
Source: arXiv:2411.04990v2, §5 and Appendix A, the conditional null-fiber step of `thm:fixed_centers`, before the separate measurable Fubini argument. -/
theorem frozenFlow_repelling_fiber_countable (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p₀ pstar : Idx n → ℝ) (hp : Tendsto (frozenFlow β θ a ha p₀) atTop (nhds pstar))
    (C : Set ℝ) (hC : C.Countable) :
    {x₀ : ℝ | ∃ l ∈ C, 0 < frozenLastVelocityDeriv β θ a pstar l ∧
      Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l)}.Countable := by
  have hfiber (l : ℝ) : {x₀ : ℝ | 0 < frozenLastVelocityDeriv β θ a pstar l ∧
      Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l)}.Subsingleton := by
    intro x₀ hx y₀ hy
    exact frozenFlow_last_initial_eq_of_repelling_limits β θ a ha p₀ pstar hp x₀ y₀ l hx.1 hx.2 hy.2
  have hUnion : (⋃ l ∈ C, {x₀ : ℝ | 0 < frozenLastVelocityDeriv β θ a pstar l ∧
      Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l)}).Countable :=
    hC.biUnion (fun l hl => (hfiber l).countable)
  apply hUnion.mono
  intro x₀ hx
  obtain ⟨l, hl, hd, hlim⟩ := hx
  exact mem_iUnion.mpr ⟨l, mem_iUnion.mpr ⟨hl, hd, hlim⟩⟩

/-- The actual exceptional repelling fiber has zero original Lebesgue measure when its possible limiting critical angles form a countable set.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original scalar null-fiber conclusion for `thm:fixed_centers`, prior to measurable Fubini. -/
theorem frozenFlow_repelling_fiber_volume_zero (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p₀ pstar : Idx n → ℝ) (hp : Tendsto (frozenFlow β θ a ha p₀) atTop (nhds pstar))
    (C : Set ℝ) (hC : C.Countable) :
    volume {x₀ : ℝ | ∃ l ∈ C, 0 < frozenLastVelocityDeriv β θ a pstar l ∧
      Tendsto (fun t => frozenFlow β θ a ha (Fin.snoc p₀ x₀) t (Fin.last n)) atTop (nhds l)} = 0 :=
  (frozenFlow_repelling_fiber_countable β θ a ha p₀ pstar hp C hC).measure_zero volume

/-- A genuine actual antipodal equilibrium lies in the countable repelling fiber of an actual fixed original prefix.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original one-center repelling configuration in `thm:fixed_centers`. -/
example : let θ : Idx 1 → ℝ := fun _ => 0
    let a : Idx 1 → ℝ := fun _ => 1
    let ha : ∀ j : Idx 1, (0 : ℝ) ≤ a j := fun _ => zero_le_one
    let p₀ : Idx 1 → ℝ := fun _ => 0
    {x₀ : ℝ | ∃ l ∈ ({Real.pi} : Set ℝ), 0 < frozenLastVelocityDeriv 1 θ a p₀ l ∧
      Tendsto (fun t => frozenFlow 1 θ a ha (Fin.snoc p₀ x₀) t (Fin.last 1)) atTop (nhds l)}.Countable ∧
    volume {x₀ : ℝ | ∃ l ∈ ({Real.pi} : Set ℝ), 0 < frozenLastVelocityDeriv 1 θ a p₀ l ∧
      Tendsto (fun t => frozenFlow 1 θ a ha (Fin.snoc p₀ x₀) t (Fin.last 1)) atTop (nhds l)} = 0 ∧
    ∃ l ∈ ({Real.pi} : Set ℝ), 0 < frozenLastVelocityDeriv 1 θ a p₀ l ∧
      Tendsto (fun t => frozenFlow 1 θ a ha (Fin.snoc p₀ Real.pi) t (Fin.last 1)) atTop (nhds l) := by
  dsimp only
  let θ : Idx 1 → ℝ := fun _ => 0
  let a : Idx 1 → ℝ := fun _ => 1
  let ha : ∀ j : Idx 1, (0 : ℝ) ≤ a j := fun _ => zero_le_one
  let p₀ : Idx 1 → ℝ := fun _ => 0
  let X : Idx 2 → ℝ := Fin.snoc p₀ Real.pi
  have hpv : frozenVelocity 1 1 1 θ a p₀ = 0 := by funext k; simp [frozenVelocity, θ, a, p₀]
  have hXv : frozenVelocity 2 1 1 θ a X = 0 := by
    funext k
    refine Fin.lastCases ?_ (fun j => ?_) k
    · change frozenVelocity 2 1 1 θ a (Fin.snoc p₀ Real.pi) (Fin.last 1) = 0
      rw [frozenVelocity_last_eq_force]
      simp [frozenLastForce, h_pot, θ, a, p₀]
    · rw [frozenVelocity_castSucc]
      simpa only [X, Fin.snoc_castSucc, Pi.zero_apply] using congrFun hpv j
  have hpf : frozenFlow 1 θ a ha p₀ = fun _ => p₀ := by
    exact frozenFlow_eq_of_ODE 1 θ a ha (fun _ => p₀) (fun t => by simpa only [hpv] using hasDerivAt_const t p₀)
  have hXf : frozenFlow 1 θ a ha X = fun _ => X := by
    exact frozenFlow_eq_of_ODE 1 θ a ha (fun _ => X) (fun t => by simpa only [hXv] using hasDerivAt_const t X)
  have hp : Tendsto (frozenFlow 1 θ a ha p₀) atTop (nhds p₀) := by rw [hpf]; exact tendsto_const_nhds
  have hcrit : frozenVelocity 2 1 1 θ a (Fin.snoc p₀ Real.pi) (Fin.last 1) = 0 := congrFun hXv (Fin.last 1)
  have hpos : 0 < frozenLastVelocityDeriv 1 θ a p₀ Real.pi := by
    apply (frozenLastVelocityDeriv_pos_iff 1 θ a ha p₀ Real.pi hcrit).mpr
    norm_num [frozenLastCurvature, Fin.sum_univ_one, g_pot, θ, a, p₀]
    positivity
  refine ⟨frozenFlow_repelling_fiber_countable 1 θ a ha p₀ p₀ hp {Real.pi} (countable_singleton _),
    frozenFlow_repelling_fiber_volume_zero 1 θ a ha p₀ p₀ hp {Real.pi} (countable_singleton _),
    Real.pi, mem_singleton _, hpos, ?_⟩
  change Tendsto (fun t => frozenFlow 1 θ a ha X t (Fin.last 1)) atTop (nhds Real.pi)
  simpa only [hXf, X, Fin.snoc_last] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => Real.pi) atTop (nhds Real.pi))

end Transformer.Causal

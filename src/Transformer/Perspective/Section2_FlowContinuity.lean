/-
# Continuity of the actual spherical attention flow

ArXiv:2312.10794v5, §2, `SA`, and Appendix D, Step 1,
`eq: stability.4ortho`; arXiv:2411.04551v3, §5,
`prop:generic-discrete`.

The flow below is chosen from the proved existence theorem for the original
attention equation. Its uniqueness makes the choice deterministic.
Grönwall's inequality gives continuous dependence on the initial tuple.
The tuple norm is the maximum of the token norms, as in the source's bound.

The temperature and forward-time sign conditions are explicit. No limit,
measurability or continuity is imposed in the definition of the flow.
-/

import Transformer.Perspective.Section2_SphereFlow
import Transformer.Perspective.SALipschitz
import Mathlib.Analysis.ODE.Gronwall

open Real Set

namespace Transformer.Perspective

variable {d n : ℕ}

/-- The uniquely determined trajectory of `SA` from a spherical tuple.
Source: arXiv:2312.10794v5, §2, `SA`; arXiv:2411.04551v3, §5,
the deterministic flow in `prop:generic-discrete`. -/
noncomputable def saFlow (β : ℝ) (X₀ : SphereTuple d n) : ℝ → SphereTuple d n :=
  Classical.choose (exists_SA β X₀)

/-- The chosen trajectory has the prescribed initial value and solves the
actual attention equation. Source: arXiv:2312.10794v5, §2, `SA`. -/
theorem saFlow_spec (β : ℝ) (X₀ : SphereTuple d n) :
    saFlow β X₀ 0 = X₀ ∧ SA d n β (saFlow β X₀) :=
  Classical.choose_spec (exists_SA β X₀)

/-- The initial-value identity of the source's deterministic `SA` flow.
Source: arXiv:2312.10794v5, §2, `SA`. -/
theorem saFlow_zero (β : ℝ) (X₀ : SphereTuple d n) : saFlow β X₀ 0 = X₀ :=
  (saFlow_spec β X₀).1

/-- Any solution is the chosen flow from its initial tuple.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem saFlow_eq_of_SA (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : SA d n β X) : saFlow β (X 0) = X :=
  SA_unique β _ X (saFlow_spec β (X 0)).2 hX (saFlow_zero β (X 0))

example (β : ℝ) (x : SSphere 1) : SA 1 1 β (fun _ _ => x) :=
  SA_const_consensus 1 1 one_pos β x

/-- A consensus initial tuple gives the actual stationary trajectory.
Source: arXiv:2312.10794v5, §2, `SA`; all attention weights average
the same token, and its tangent projection is zero. -/
theorem saFlow_consensus (β : ℝ) (hn : 0 < n) (x : SSphere d) :
    saFlow β (fun _ : Idx n => x) = (fun _ : ℝ => fun _ : Idx n => x) := by
  exact saFlow_eq_of_SA β (fun _ : ℝ => fun _ : Idx n => x)
    (SA_const_consensus d n hn β x)

example : (0 : ℕ) < 1 := one_pos

/-- Each actual trajectory is continuous, including as a sphere-valued
tuple. Source: arXiv:2312.10794v5, §2, `SA`. -/
theorem continuous_saFlow_path (β : ℝ) (X₀ : SphereTuple d n) :
    Continuous (saFlow β X₀) := by
  apply continuous_pi_iff.mpr
  intro i
  apply continuous_induced_rng.mpr
  exact continuous_iff_continuousAt.mpr fun t =>
    ((SA_iff d n β (saFlow β X₀)).mp (saFlow_spec β X₀).2 t i).continuousAt

/-- The maximum token distance is unchanged by the spherical embedding.
Source: the tuple norm used in arXiv:2312.10794v5, Appendix D,
`eq: stability.4ortho`. -/
theorem dist_sphereTuple_coe (X Y : SphereTuple d n) :
    dist (fun i => (X i : EucSpace d)) (fun i => (Y i : EucSpace d)) = dist X Y := by
  simp only [dist_pi_def, Subtype.nndist_eq]

/-- **Continuous dependence on initial tokens, with the source's
Grönwall exponent.** The maximum distance at time `t ≥ 0` is bounded by
its initial value times `exp (10 n max{1,β} t)`.
Source: arXiv:2312.10794v5, Appendix D, `eq: stability.4ortho`.
The inverse temperature is explicitly nonnegative, as in the source. -/
theorem dist_SA_le_exp (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (X Y : ℝ → SphereTuple d n) (hX : SA d n β X) (hY : SA d n β Y)
    (t : ℝ) (ht : 0 ≤ t) :
    dist (X t) (Y t) ≤ dist (X 0) (Y 0) * Real.exp (10 * n * max 1 β * t) := by
  let x (s : ℝ) (i : Idx n) := (X s i : EucSpace d)
  let y (s : ℝ) (i : Idx n) := (Y s i : EucSpace d)
  have hx (s : ℝ) : HasDerivAt x (saField d n β (x s)) s :=
    hasDerivAt_pi.mpr ((SA_iff d n β X).mp hX s)
  have hy (s : ℝ) : HasDerivAt y (saField d n β (y s)) s :=
    hasDerivAt_pi.mpr ((SA_iff d n β Y).mp hY s)
  have hdiff (s : ℝ) : HasDerivAt (fun r => x r - y r)
      (saField d n β (x s) - saField d n β (y s)) s := (hx s).sub (hy s)
  have hc : Continuous (fun s => x s - y s) :=
    continuous_iff_continuousAt.mpr fun s => (hdiff s).continuousAt
  have hK : 0 ≤ 10 * (n : ℝ) * max 1 β := by positivity
  have hbound (s : ℝ) :
      ‖saField d n β (x s) - saField d n β (y s)‖
        ≤ 10 * (n : ℝ) * max 1 β * ‖x s - y s‖ := by
    have hxu : x s ∈ unitTuples d n :=
      fun i => mem_sphere_zero_iff_norm.mp (X s i).2
    have hyu : y s ∈ unitTuples d n :=
      fun i => mem_sphere_zero_iff_norm.mp (Y s i).2
    simpa only [Real.coe_toNNReal _ hK] using
      lipschitzOnWith_iff_norm_sub_le.mp (lipschitzOnWith_saField d n β hβ hn) hxu hyu
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖x 0 - y 0‖) (K := 10 * (n : ℝ) * max 1 β) (ε := 0)
    hc.continuousOn (fun s _ => (hdiff s).hasDerivWithinAt) le_rfl
    (fun s _ => by simpa only [add_zero] using hbound s) t ⟨ht, le_rfl⟩
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm, x, y,
    dist_sphereTuple_coe] using hG

/-- Both ODE hypotheses are realized by nonzero stationary tokens, with
nonnegative temperature, a nonempty tuple, and forward time. -/
example (x y : SSphere 1) : (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 ∧
    SA 1 1 1 (fun _ _ => x) ∧ SA 1 1 1 (fun _ _ => y) ∧ (0 : ℝ) ≤ 1 :=
  ⟨zero_le_one, one_pos, SA_const_consensus 1 1 one_pos 1 x,
    SA_const_consensus 1 1 one_pos 1 y, zero_le_one⟩

/-- The actual flow map is Lipschitz at each forward time.
Source: arXiv:2312.10794v5, Appendix D, `eq: stability.4ortho`;
arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem lipschitzWith_saFlow (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (t : ℝ) (ht : 0 ≤ t) :
    LipschitzWith (Real.toNNReal (Real.exp (10 * n * max 1 β * t)))
      (fun X₀ : SphereTuple d n => saFlow β X₀ t) := by
  apply LipschitzWith.of_dist_le_mul
  intro X₀ Y₀
  rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
  simpa only [saFlow_zero] using
    dist_SA_le_exp β hβ hn _ _ (saFlow_spec β X₀).2 (saFlow_spec β Y₀).2 t ht

example : (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) ≤ 1 :=
  ⟨zero_le_one, one_pos, zero_le_one⟩

/-- The flow map is continuous in its initial tuple at forward times.
Source: arXiv:2411.04551v3, §5, the deterministic measurable flow
used to define the cluster map in `prop:generic-discrete`. -/
theorem continuous_saFlow (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (t : ℝ) (ht : 0 ≤ t) :
    Continuous (fun X₀ : SphereTuple d n => saFlow β X₀ t) :=
  (lipschitzWith_saFlow β hβ hn t ht).continuous

example : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) ≤ 0 :=
  ⟨le_rfl, one_pos, le_rfl⟩

end Transformer.Perspective

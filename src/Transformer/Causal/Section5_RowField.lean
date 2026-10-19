module

public import Transformer.Causal.Section5_FrozenLastRow
public import Mathlib.Analysis.Calculus.LocalExtr.Rolle

/-!
# Periodicity and limits of the actual frozen scalar row

ArXiv:2411.04990v2, §5, the actual scalar equations in the
proof of `thm:fixed_centers`.
The original force, curvature, and partition are jointly
continuous in the actual predecessor tuple and last angle.
Their original periodicity passes to the exact normalized
last velocity. Consequently convergence of the predecessor
tuple gives the actual field limit at every fixed angle,
and convergence of both arguments gives the velocity limit.

The partition derivative is exactly `-beta` times the original
force. Equivalently the source's energy derivative is `beta`
times that force; its displayed gradient formula omits this
factor. We retain it here. Periodicity and Rolle's theorem
then supply an actual force zero within an angular period.
Every original interaction and center mass is retained.
-/

@[expose] public section

open scoped BigOperators
open Real Set Filter

namespace Transformer.Causal

variable {n m : ℕ}

/-- The original last force is jointly continuous in the actual predecessor tuple and actual last angle.
Source: arXiv:2411.04990v2, §5, the original `sum h` in `thm:fixed_centers`. -/
theorem continuous_frozenLastForce_joint (β : ℝ) (θ a : Idx m → ℝ) :
    Continuous (fun q : (Idx n → ℝ) × ℝ => frozenLastForce β θ a q.1 q.2) := by
  unfold frozenLastForce h_pot
  fun_prop

/-- The original last curvature is jointly continuous in the actual predecessor tuple and actual last angle.
Source: arXiv:2411.04990v2, §5, the original `sum g` in the diagonal calculation of `thm:fixed_centers`. -/
theorem continuous_frozenLastCurvature_joint (β : ℝ) (θ a : Idx m → ℝ) :
    Continuous (fun q : (Idx n → ℝ) × ℝ => frozenLastCurvature β θ a q.1 q.2) := by
  unfold frozenLastCurvature g_pot
  fun_prop

/-- The original last partition is jointly continuous in the actual predecessor tuple and actual last angle.
Source: arXiv:2411.04990v2, §5, the original normalization in `thm:fixed_centers`. -/
theorem continuous_frozenLastPartition_joint (β : ℝ) (θ a : Idx m → ℝ) :
    Continuous (fun q : (Idx n → ℝ) × ℝ => frozenLastPartition β θ a q.1 q.2) := by
  unfold frozenLastPartition
  fun_prop

/-- The exact original normalized last velocity is jointly continuous for the original nonnegative center masses.
Source: arXiv:2411.04990v2, §5, the smooth original scalar field in `thm:fixed_centers`. -/
theorem continuous_frozenLastVelocity_joint (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    Continuous (fun q : (Idx n → ℝ) × ℝ => frozenVelocity (n + 1) m β θ a (Fin.snoc q.1 q.2) (Fin.last n)) := by
  have hfun : (fun q : (Idx n → ℝ) × ℝ => frozenVelocity (n + 1) m β θ a (Fin.snoc q.1 q.2) (Fin.last n)) =
      (fun q => -frozenLastForce β θ a q.1 q.2 / frozenLastPartition β θ a q.1 q.2) :=
    funext (fun q => frozenVelocity_last_eq_force β θ a q.1 q.2)
  rw [hfun]
  exact (continuous_frozenLastForce_joint β θ a).neg.div (continuous_frozenLastPartition_joint β θ a)
    (fun q => (frozenLastPartition_pos β θ a ha q.1 q.2).ne')

/-- The actual original last force is periodic in the original last angle, for every fixed actual predecessor tuple.
Source: arXiv:2411.04990v2, Appendix B and §5, the original circular interaction in `thm:fixed_centers`. -/
theorem frozenLastForce_periodic (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    Function.Periodic (frozenLastForce β θ a p) (2 * Real.pi) := by
  intro u
  have hshift (v : ℝ) : u + 2 * Real.pi - v = (u - v) + 2 * Real.pi := by ring
  simp only [frozenLastForce, hshift, h_pot_periodic]

/-- The actual original last curvature is periodic in the original last angle.
Source: arXiv:2411.04990v2, Appendix B and §5, the original curvature in the diagonal calculation of `thm:fixed_centers`. -/
theorem frozenLastCurvature_periodic (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    Function.Periodic (frozenLastCurvature β θ a p) (2 * Real.pi) := by
  intro u
  have hshift (v : ℝ) : u + 2 * Real.pi - v = (u - v) + 2 * Real.pi := by ring
  simp only [frozenLastCurvature, hshift, g_pot_periodic]

/-- The exact original last normalization is periodic in its original last angle.
Source: arXiv:2411.04990v2, §5, the original partition of `thm:fixed_centers`. -/
theorem frozenLastPartition_periodic (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    Function.Periodic (frozenLastPartition β θ a p) (2 * Real.pi) := by
  intro u
  have hshift (v : ℝ) : u + 2 * Real.pi - v = (u - v) + 2 * Real.pi := by ring
  simp only [frozenLastPartition, hshift, Real.cos_add_two_pi]

/-- The actual normalized original last velocity is periodic in its last angle with the original angular period.
Source: arXiv:2411.04990v2, §5, the actual circular scalar field of `thm:fixed_centers`. -/
theorem frozenLastVelocity_periodic (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    Function.Periodic (fun u => frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n)) (2 * Real.pi) := by
  intro u
  dsimp only
  rw [frozenVelocity_last_eq_force, frozenVelocity_last_eq_force,
    frozenLastForce_periodic β θ a p u, frozenLastPartition_periodic β θ a p u]

/-- The actual partition derivative is minus the original temperature times the actual force.
Source: arXiv:2411.04990v2, §5, the energy and scalar field in `thm:fixed_centers`; the source omits the temperature factor in its displayed energy-gradient relation, which is retained here. -/
theorem hasDerivAt_frozenLastPartition (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) :
    HasDerivAt (frozenLastPartition β θ a p) (-β * frozenLastForce β θ a p u) u := by
  have he (v : ℝ) : HasDerivAt (fun x => Real.exp (β * (Real.cos (x - v) - 1))) (-β * h_pot β (u - v)) u := by
    have h := (((((hasDerivAt_id u).sub_const v).cos).sub_const 1).const_mul β).exp
    apply h.congr_deriv
    simp only [id_eq, mul_one, h_pot]
    ring
  have hp := ((hasDerivAt_const u 1).fun_add (HasDerivAt.fun_sum (u := Finset.univ) (fun j _ => he (p j)))).fun_add
    (HasDerivAt.fun_sum (u := Finset.univ) (fun j _ => (he (θ j)).const_mul (a j)))
  apply hp.congr_deriv
  unfold frozenLastForce
  simp only [zero_add, mul_add, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- A nonzero original temperature guarantees an actual original force zero strictly within an angular period, for every actual predecessor tuple.
Source: arXiv:2411.04990v2, §5 and Appendix A, existence of a critical angle for the original periodic energy in `thm:fixed_centers`. -/
theorem frozenLastForce_exists_zero (β : ℝ) (hβ : β ≠ 0) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    ∃ u ∈ Ioo (0 : ℝ) (2 * Real.pi), frozenLastForce β θ a p u = 0 := by
  have hperiod : frozenLastPartition β θ a p 0 = frozenLastPartition β θ a p (2 * Real.pi) := by
    simpa only [zero_add] using (frozenLastPartition_periodic β θ a p 0).symm
  obtain ⟨u, hu, hd⟩ := exists_deriv_eq_zero (show (0 : ℝ) < 2 * Real.pi by positivity)
    (contDiff_frozenLastPartition β θ a p).continuous.continuousOn hperiod
  rw [(hasDerivAt_frozenLastPartition β θ a p u).deriv, mul_eq_zero] at hd
  exact ⟨u, hu, hd.resolve_left (neg_ne_zero.mpr hβ)⟩

/-- Convergence of the actual predecessor tuple gives the actual original last-field limit at every fixed angle.
Source: arXiv:2411.04990v2, §5 and Appendix A, the limiting scalar field used for the original frozen convergence proof. -/
theorem frozenLastVelocity_tendsto_fixed_angle (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : ℝ → Idx n → ℝ) (pstar : Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar)) (u : ℝ) :
    Tendsto (fun t => frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) u) (Fin.last n)) atTop
      (nhds (frozenVelocity (n + 1) m β θ a (Fin.snoc pstar u) (Fin.last n))) := by
  have hpair : Tendsto (fun t => (p t, u)) atTop (nhds (pstar, u)) := hp.prodMk_nhds tendsto_const_nhds
  have h := (continuous_frozenLastVelocity_joint β θ a ha).tendsto (pstar, u)
  simpa! only [Function.comp_def] using h.comp hpair

/-- Convergence of the actual predecessor tuple and last angle gives the limit of the exact original last velocity.
Source: arXiv:2411.04990v2, §5 and Appendix A, the criticality step for the actual smooth frozen system. -/
theorem frozenLastVelocity_tendsto (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : ℝ → Idx n → ℝ) (pstar : Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar))
    (u : ℝ → ℝ) (ustar : ℝ) (hu : Tendsto u atTop (nhds ustar)) :
    Tendsto (fun t => frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) (u t)) (Fin.last n)) atTop
      (nhds (frozenVelocity (n + 1) m β θ a (Fin.snoc pstar ustar) (Fin.last n))) := by
  have hpair : Tendsto (fun t => (p t, u t)) atTop (nhds (pstar, ustar)) := hp.prodMk_nhds hu
  have h := (continuous_frozenLastVelocity_joint β θ a ha).tendsto (pstar, ustar)
  simpa! only [Function.comp_def] using h.comp hpair

/-- Genuine positive original center masses and constant actual predecessor and last trajectories jointly satisfy the field-limit and critical-point existence hypotheses.
Source: arXiv:2411.04990v2, §5, the original consensus equilibrium in `thm:fixed_centers`. -/
example : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧ (1 : ℝ) ≠ 0 ∧
    Tendsto (fun _ : ℝ => (fun _ : Idx 1 => (0 : ℝ))) atTop (nhds (fun _ : Idx 1 => (0 : ℝ))) ∧
    Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) :=
  ⟨fun _ => zero_le_one, one_ne_zero, tendsto_const_nhds, tendsto_const_nhds⟩

/-- A genuine original predecessor and frozen center have an actual force zero at the antipodal angle inside one full angular period.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original periodic critical-angle geometry in `thm:fixed_centers`. -/
example : Real.pi ∈ Ioo (0 : ℝ) (2 * Real.pi) ∧
    frozenLastForce 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) Real.pi = 0 := by
  refine ⟨⟨Real.pi_pos, by linarith [Real.pi_pos]⟩, ?_⟩
  simp [frozenLastForce, h_pot]

end Transformer.Causal

module

public import Transformer.Causal.Section5_RowField

/-!
# The actual derivative of the original normalized frozen row

ArXiv:2411.04990v2, §5, the diagonal scalar derivative used
in the proof of `thm:fixed_centers`.
Away from criticality, differentiating the exact original
velocity `-F/Z` retains the partition term. Since `F'=G` and
`Z'=-beta F`, its derivative is
`(-G Z-beta F^2)/Z^2`.

This expression is jointly continuous in the actual predecessor
tuple and last angle for nonnegative original center masses.
At an actual critical angle it reduces to `-G/Z`, with the
strict curvature signs corresponding to attraction or repulsion.
Both the original normalization and all original masses remain
in the formula. Joint continuity also gives the derivative
limit along converging actual predecessor and last trajectories.
-/

@[expose] public section

open Real Set Filter

namespace Transformer.Causal

variable {n m : ℕ}

/-- The exact original last-angle velocity derivative, including its original partition derivative away from criticality.
Source: arXiv:2411.04990v2, §5, differentiating the original scalar field in `thm:fixed_centers`; the full quotient term is retained. -/
noncomputable def frozenLastVelocityDeriv (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) : ℝ :=
  (-frozenLastCurvature β θ a p u * frozenLastPartition β θ a p u - β * frozenLastForce β θ a p u ^ 2) /
    frozenLastPartition β θ a p u ^ 2

/-- The exact original scalar row has the displayed derivative at every actual angle.
Source: arXiv:2411.04990v2, §5, the original quotient field of `thm:fixed_centers`, with `Z'=-beta F`. -/
theorem hasDerivAt_frozenLastVelocity (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ) :
    HasDerivAt (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n))
      (frozenLastVelocityDeriv β θ a p u) u := by
  have h := (hasDerivAt_frozenLastForce β θ a p u).neg.fun_div (hasDerivAt_frozenLastPartition β θ a p u)
    (frozenLastPartition_pos β θ a ha p u).ne'
  have hfun : (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) =
      (fun v => -frozenLastForce β θ a p v / frozenLastPartition β θ a p v) :=
    funext (frozenVelocity_last_eq_force β θ a p)
  rw [hfun]
  apply h.congr_deriv
  unfold frozenLastVelocityDeriv
  simp only [Pi.neg_apply]
  ring

/-- The actual original last-angle derivative agrees with the full original quotient formula everywhere.
Source: arXiv:2411.04990v2, §5, the original last-row derivative for `thm:fixed_centers`. -/
theorem deriv_frozenLastVelocity (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ) :
    deriv (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) u =
      frozenLastVelocityDeriv β θ a p u := (hasDerivAt_frozenLastVelocity β θ a ha p u).deriv

/-- The full original last-angle derivative is jointly continuous in the actual predecessor tuple and actual last angle.
Source: arXiv:2411.04990v2, §5, the smooth original scalar field used for local repulsion and stability in `thm:fixed_centers`. -/
theorem continuous_frozenLastVelocityDeriv_joint (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    Continuous (fun q : (Idx n → ℝ) × ℝ => frozenLastVelocityDeriv β θ a q.1 q.2) := by
  have hF := continuous_frozenLastForce_joint (n := n) β θ a
  have hG := continuous_frozenLastCurvature_joint (n := n) β θ a
  have hZ := continuous_frozenLastPartition_joint (n := n) β θ a
  exact ((hG.neg.mul hZ).sub (continuous_const.mul (hF.pow 2))).div (hZ.pow 2)
    (fun q => pow_ne_zero 2 (frozenLastPartition_pos β θ a ha q.1 q.2).ne')

/-- At an actual original critical angle, the full derivative reduces exactly to negative original curvature over original partition.
Source: arXiv:2411.04990v2, §5, the critical diagonal derivative display in `thm:fixed_centers`. -/
theorem frozenLastVelocityDeriv_of_critical (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    frozenLastVelocityDeriv β θ a p u = -frozenLastCurvature β θ a p u / frozenLastPartition β θ a p u := by
  rw [← deriv_frozenLastVelocity β θ a ha p u]
  exact deriv_frozenLastVelocity_of_critical β θ a ha p u hcrit

/-- An actual critical original last angle is strictly repelling exactly when its original curvature is strictly negative.
Source: arXiv:2411.04990v2, §5, the repelling diagonal sign in `thm:fixed_centers`. -/
theorem frozenLastVelocityDeriv_pos_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    0 < frozenLastVelocityDeriv β θ a p u ↔ frozenLastCurvature β θ a p u < 0 := by
  rw [frozenLastVelocityDeriv_of_critical β θ a ha p u hcrit,
    lt_div_iff₀ (frozenLastPartition_pos β θ a ha p u), zero_mul, neg_pos]

/-- An actual critical original last angle has strictly negative velocity derivative exactly when its original curvature is positive.
Source: arXiv:2411.04990v2, §5, the attracting diagonal sign in `thm:fixed_centers`. -/
theorem frozenLastVelocityDeriv_neg_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    frozenLastVelocityDeriv β θ a p u < 0 ↔ 0 < frozenLastCurvature β θ a p u := by
  rw [frozenLastVelocityDeriv_of_critical β θ a ha p u hcrit,
    div_lt_iff₀ (frozenLastPartition_pos β θ a ha p u), zero_mul, neg_lt_zero]

/-- At an actual original critical angle, nonzero last-velocity derivative is equivalent to nonzero original curvature.
Source: arXiv:2411.04990v2, §5, the diagonal nondegeneracy criterion in `thm:fixed_centers`. -/
theorem frozenLastVelocityDeriv_ne_zero_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    frozenLastVelocityDeriv β θ a p u ≠ 0 ↔ frozenLastCurvature β θ a p u ≠ 0 := by
  rw [frozenLastVelocityDeriv_of_critical β θ a ha p u hcrit, ne_eq, div_eq_zero_iff]
  simp only [neg_eq_zero, (frozenLastPartition_pos β θ a ha p u).ne', or_false]

/-- The actual last-angle derivative is continuous as a function of the full original angular state.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original smooth derivative sampled along the actual flow in `thm:fixed_centers`. -/
theorem continuous_frozenLastVelocityDeriv_full (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    Continuous (fun φ : Idx (n + 1) → ℝ =>
      frozenLastVelocityDeriv β θ a (fun j => φ j.castSucc) (φ (Fin.last n))) := by
  have hpair : Continuous (fun φ : Idx (n + 1) → ℝ => ((fun j : Idx n => φ j.castSucc), φ (Fin.last n))) := by fun_prop
  have h := (continuous_frozenLastVelocityDeriv_joint β θ a ha).comp hpair
  simpa! only [Function.comp_def] using h

/-- Converging actual predecessor and last trajectories have the actual limiting original last-angle derivative.
Source: arXiv:2411.04990v2, §5 and Appendix A, the continuity step for repelling-limit avoidance in `thm:fixed_centers`. -/
theorem frozenLastVelocityDeriv_tendsto (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : ℝ → Idx n → ℝ) (pstar : Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar))
    (u : ℝ → ℝ) (ustar : ℝ) (hu : Tendsto u atTop (nhds ustar)) :
    Tendsto (fun t => frozenLastVelocityDeriv β θ a (p t) (u t)) atTop
      (nhds (frozenLastVelocityDeriv β θ a pstar ustar)) := by
  have hpair : Tendsto (fun t => (p t, u t)) atTop (nhds (pstar, ustar)) := hp.prodMk_nhds hu
  have h := (continuous_frozenLastVelocityDeriv_joint β θ a ha).tendsto (pstar, ustar)
  simpa! only [Function.comp_def] using h.comp hpair

/-- A genuine original frozen center and actual preceding token jointly give a repelling critical last angle at the antipode.
Source: arXiv:2411.04990v2, §5, the actual one-center critical configuration in `thm:fixed_centers`. -/
example : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧
    frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (Fin.snoc (fun _ : Idx 1 => 0) Real.pi) (Fin.last 1) = 0 ∧
    frozenLastCurvature 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) Real.pi < 0 := by
  refine ⟨fun _ => zero_le_one, ?_, ?_⟩
  · rw [frozenVelocity_last_eq_force]
    simp [frozenLastForce, h_pot]
  · norm_num [frozenLastCurvature, Fin.sum_univ_one, g_pot]
    positivity

/-- The genuine actual antipodal critical angle has strictly positive original normalized velocity derivative.
Source: arXiv:2411.04990v2, §5, the actual one-center repelling configuration in `thm:fixed_centers`. -/
example : 0 < frozenLastVelocityDeriv 1 (fun _ : Idx 1 => 0) (fun _ => 1)
    (fun _ : Idx 1 => 0) Real.pi := by
  have hv : frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1)
      (Fin.snoc (fun _ : Idx 1 => 0) Real.pi) (Fin.last 1) = 0 := by
    rw [frozenVelocity_last_eq_force]
    simp [frozenLastForce, h_pot]
  apply (frozenLastVelocityDeriv_pos_iff 1 (fun _ => 0) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) Real.pi hv).mpr
  norm_num [frozenLastCurvature, Fin.sum_univ_one, g_pot]
  positivity

/-- The same actual predecessor and frozen center jointly give an attracting critical last angle at the center, with genuine predecessor and last limits.
Source: arXiv:2411.04990v2, §5, the actual one-center consensus configuration in `thm:fixed_centers`. -/
example : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧
    frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (Fin.snoc (fun _ : Idx 1 => 0) 0) (Fin.last 1) = 0 ∧
    0 < frozenLastCurvature 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) 0 ∧
    Tendsto (fun _ : ℝ => (fun _ : Idx 1 => (0 : ℝ))) atTop (nhds (fun _ : Idx 1 => (0 : ℝ))) ∧
    Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) := by
  refine ⟨fun _ => zero_le_one, ?_, ?_, tendsto_const_nhds, tendsto_const_nhds⟩
  · rw [frozenVelocity_last_eq_force]
    simp [frozenLastForce, h_pot]
  · norm_num [frozenLastCurvature, Fin.sum_univ_one, g_pot]

end Transformer.Causal

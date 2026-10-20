module

public import Transformer.Normalization.Section2_SphereInvariant
public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# §3 — The logarithmic clock and the rescaled radii

Karagodin, Polyanskiy, Rigollet, arXiv:2510.22026v2, supplement,
Appendix D, proof of Theorem 3.1, Step 1 and the scheme-specific remarks.

The change `t = T exp u`, `q_j(u) = r_j(t)/t`, is derived from the original
forward equations, including their radial component. The source uses `T = 1`;
any `T > 0` works and lets the Mix-LN proof start after its switching time.
The transformed equations hold at every real `u`, since `T exp u > 0` lies
in the interior of the original forward half-line. They require neither a
backward-global original solution nor positive radii. Bounds and positivity
of the rescaled radii are separate questions, not consequences of this
algebraic change of variables.
-/

@[expose] public section

open Real Set

namespace Transformer.Normalization

/-- The inverse of the shifted logarithmic clock: `t = T exp u`.
Source: arXiv:2510.22026v2, Appendix D, Step 1 (`T = 1`). -/
noncomputable def logTime (T u : ℝ) : ℝ := T * Real.exp u

/-- Positive starting time keeps the entire logarithmic clock strictly
inside the original forward domain. Source: Appendix D, Step 1. -/
theorem logTime_pos {T : ℝ} (hT : 0 < T) (u : ℝ) : 0 < logTime T u :=
  mul_pos hT (Real.exp_pos u)

example : (0 : ℝ) < 1 := one_pos

/-- The logarithmic clock satisfies `dt/du = t`.
Source: arXiv:2510.22026v2, Appendix D, Step 1. -/
theorem hasDerivAt_logTime (T u : ℝ) : HasDerivAt (logTime T) (logTime T u) u :=
  (Real.hasDerivAt_exp u).const_mul T

/-- The positive clock visits every positive original time.
Source: arXiv:2510.22026v2, Appendix D, inverse logarithmic time change. -/
theorem logTime_log_div {T t : ℝ} (hT : 0 < T) (ht : 0 < t) :
    logTime T (Real.log (t / T)) = t := by
  rw [logTime, Real.exp_log (div_pos ht hT)]
  field_simp

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 := ⟨one_pos, two_pos⟩

/-- The logarithmic clock preserves the limit at positive infinity.
Source: arXiv:2510.22026v2, Appendix D, `sprop:time_change` and Step 1. -/
theorem tendsto_comp_logTime_iff {E : Type*} {f : ℝ → E} {l : Filter E}
    {T : ℝ} (hT : 0 < T) :
    Filter.Tendsto (fun u => f (logTime T u)) Filter.atTop l ↔
      Filter.Tendsto f Filter.atTop l := by
  constructor
  · intro h
    have hlog := Real.tendsto_log_atTop.comp (Filter.tendsto_id.atTop_div_const hT)
    apply Filter.Tendsto.congr' (f₁ := fun t => f (logTime T (Real.log (t / T)))) ?_ (h.comp hlog)
    filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with t ht
    rw [logTime_log_div hT ht]
  · intro h
    exact h.comp (Real.tendsto_exp_atTop.const_mul_atTop hT)

example : (0 : ℝ) < 2 := two_pos

/-- The source's rescaled radius `q_j(u) = r_j(t(u))/t(u)`.
Source: arXiv:2510.22026v2, Appendix D, Step 1. -/
noncomputable def rescaledRadius {n : ℕ} (T : ℝ) (r : ℝ → Idx n → ℝ)
    (u : ℝ) (j : Idx n) : ℝ := r (logTime T u) j / logTime T u

variable {d n : ℕ} {β τ T : ℝ} {Q K V : ℝ → ParamMatrix d} {α : ℝ → ℝ}
  {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}

/-- Both polar equations transform under the logarithmic clock:
the speed becomes `s_j/t`, and `q_j' = r_j' - q_j`.
The original one-sided derivatives become ordinary derivatives at `t > 0`.
Source: arXiv:2510.22026v2, Appendix D, Step 1 and `sprop:time_change`. -/
theorem SchemeDynamics.logTime_equations {scheme : Scheme}
    (hD : SchemeDynamics d n β Q K V α τ scheme θ r) (hT : 0 < T)
    (u : ℝ) (j : Idx n) :
    let t := logTime T u
    let A := attentionVec d n β (Q t) (K t) (V t) (θ t) j
    HasDerivAt (fun v => θ (logTime T v) j)
        ((speedFactor d n β Q K V α θ r τ scheme t j / t)⁻¹ • proj d (θ t j) A) u ∧
      HasDerivAt (fun v => rescaledRadius T r v j)
        (radialDerivative d n β Q K V θ τ scheme t j - rescaledRadius T r u j) u := by
  have ht := logTime_pos hT u
  have hθ := ((hD _ ht.le j).1.hasDerivAt (Ici_mem_nhds ht)).scomp u (hasDerivAt_logTime T u)
  have hr : HasDerivAt (fun v => r (logTime T v) j)
      (logTime T u * radialDerivative d n β Q K V θ τ scheme (logTime T u) j) u := by
    simpa only [Function.comp_def, smul_eq_mul] using
      ((hD _ ht.le j).2.hasDerivAt (Ici_mem_nhds ht)).scomp u (hasDerivAt_logTime T u)
  constructor
  · have heq : (speedFactor d n β Q K V α θ r τ scheme (logTime T u) j /
        logTime T u)⁻¹ = logTime T u *
          (speedFactor d n β Q K V α θ r τ scheme (logTime T u) j)⁻¹ := by
      rw [inv_div, div_eq_mul_inv]
    rw [heq, mul_smul]
    exact hθ
  · change HasDerivAt (fun v => r (logTime T v) j / logTime T v) _ u
    apply (hr.fun_div (hasDerivAt_logTime T u) ht.ne').congr_deriv
    dsimp [rescaledRadius]
    field_simp [ht.ne']

/-- The original identity singleton with radius `1+t` realizes every
hypothesis, without assuming the transformed equations. -/
example := (schemeDynamics_singleton (d := 2) 1 1 0 0 (basePoint 1)
  (by simp [basePoint, PiLp.norm_single]) .pre (Or.inl rfl)).logTime_equations
    (T := 1) one_pos 0 0

/-- Pre-LN becomes `Θ_j' = q_j⁻¹ P_j A_j`,
`q_j' = ⟨Θ_j,A_j⟩ - q_j`, exactly as printed in Appendix D, Step 1.
Time-dependent original parameters are evaluated at `T exp u`. -/
theorem SchemeDynamics.pre_logTime_equations
    (hD : SchemeDynamics d n β Q K V α τ .pre θ r) (hT : 0 < T)
    (u : ℝ) (j : Idx n) :
    let t := logTime T u
    let A := attentionVec d n β (Q t) (K t) (V t) (θ t) j
    HasDerivAt (fun v => θ (logTime T v) j)
        ((rescaledRadius T r u j)⁻¹ • proj d (θ t j) A) u ∧
      HasDerivAt (fun v => rescaledRadius T r v j)
        (inner (𝕜 := ℝ) (θ t j) A - rescaledRadius T r u j) u := by
  simpa only [speedFactor, radialDerivative, rescaledRadius] using hD.logTime_equations hT u j

example := (schemeDynamics_singleton (d := 2) 1 1 0 0 (basePoint 1)
  (by simp [basePoint, PiLp.norm_single]) .pre (Or.inl rfl)).pre_logTime_equations
    (T := 1) one_pos 0 0

/-- Peri-LN becomes `Θ_j' = (q_j ‖A_j‖)⁻¹ P_j A_j`,
`q_j' = ⟨Θ_j,A_j⟩/‖A_j‖ - q_j`.
The identity includes the formal zero-attention branch, but does not assert
smoothness there. Source: Appendix D, remark adapting the proof to Peri-LN. -/
theorem SchemeDynamics.peri_logTime_equations
    (hD : SchemeDynamics d n β Q K V α τ .peri θ r) (hT : 0 < T)
    (u : ℝ) (j : Idx n) :
    let t := logTime T u
    let A := attentionVec d n β (Q t) (K t) (V t) (θ t) j
    HasDerivAt (fun v => θ (logTime T v) j)
        ((rescaledRadius T r u j * ‖A‖)⁻¹ • proj d (θ t j) A) u ∧
      HasDerivAt (fun v => rescaledRadius T r v j)
        (inner (𝕜 := ℝ) (θ t j) A / ‖A‖ - rescaledRadius T r u j) u := by
  simpa only [speedFactor, radialDerivative, rescaledRadius, mul_div_right_comm] using
    hD.logTime_equations hT u j

example := (schemeDynamics_singleton (d := 2) 1 1 0 0 (basePoint 1)
  (by simp [basePoint, PiLp.norm_single]) .peri (Or.inr rfl)).peri_logTime_equations
    (T := 1) one_pos 0 0

/-- At a time strictly after its switch, Mix-LN has the same logarithmic
equations as Pre-LN. No differentiability across the switch is inferred.
Source: arXiv:2510.22026v2, Appendix D, remark adapting the proof to Mix-LN. -/
theorem SchemeDynamics.mix_logTime_equations
    (hD : SchemeDynamics d n β Q K V α τ .mix θ r) (hT : 0 < T)
    (u : ℝ) (hswitch : τ < logTime T u) (j : Idx n) :
    let t := logTime T u
    let A := attentionVec d n β (Q t) (K t) (V t) (θ t) j
    HasDerivAt (fun v => θ (logTime T v) j)
        ((rescaledRadius T r u j)⁻¹ • proj d (θ t j) A) u ∧
      HasDerivAt (fun v => rescaledRadius T r v j)
        (inner (𝕜 := ℝ) (θ t j) A - rescaledRadius T r u j) u := by
  simpa only [speedFactor, radialDerivative, ite_eq_left hswitch, ite_eq_right (not_le.mpr hswitch),
    rescaledRadius] using hD.logTime_equations hT u j

/-- A genuine Mix-LN singleton realizes all hypotheses with the switch
before initialization, so that its original forward dynamics is Pre-LN. -/
example : SchemeDynamics 2 1 1 (idParams 2) (idParams 2) (idParams 2) 0 (-1) .mix
    (fun _ _ => (basePoint 1 : EucSpace 2)) (fun t _ => 1 + t) ∧
    (0 : ℝ) < 1 ∧ (-1 : ℝ) < logTime 1 0 := by
  refine ⟨?_, one_pos, by norm_num [logTime]⟩
  have hpre := schemeDynamics_singleton (d := 2) 1 1 (-1) 0 (basePoint 1)
    (by simp [basePoint, PiLp.norm_single]) .pre (Or.inl rfl)
  intro t ht j
  have hs : (-1 : ℝ) < t := by linarith
  simpa only [speedFactor, radialDerivative, ite_eq_left hs, ite_eq_right (not_le.mpr hs)] using
    hpre t ht j

end Transformer.Normalization

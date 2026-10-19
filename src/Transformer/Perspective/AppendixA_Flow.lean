module

public import Transformer.Perspective.AppendixA_CriticalShape
public import Transformer.Perspective.Section2_FlowContinuity
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Appendix A — the flow of `e:gradfl` and its Lipschitz inverse

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

`e:gradfl` is `SA` at `β = 0` run at double speed (`beta0Dynamics_smul`), so
its solutions are the time-changed `SA` flow `e0Flow X₀ t = saFlow 0 X₀ (2t)`
(`eq_e0Flow`). The proof of `l:nosaddleconv` pulls the local center-stable set
back along this flow: it needs the backward flow map `X₀ ↦ e0Flow X₀ (-T)`,
which recovers the start of a trajectory from its state at time `T`
(`e0Flow_neg_eq`) and is Lipschitz by the Grönwall bound of
`eq: stability.4ortho` run backward in time (`lipschitzWith_e0Flow_neg`).
-/

@[expose] public section

open Real Set

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- At `β = 0` the attention weights of `SA` are uniform: the drift is
`Proj_{x_i}((1/n) Σ_j x_j)`. -/
theorem SA_zero_drift (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    (partitionSA d n 0 X t i)⁻¹ •
        ∑ j : Idx n, Real.exp (0 * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          • ((X t j : EucSpace d))
      = ((n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d) := by
  simp [partitionSA]

/-- **`e:gradfl` is `SA` at `β = 0` run at double speed:** if `X` solves
`e:gradfl`, then `s ↦ X (s / 2)` solves `SA` at `β = 0`. -/
theorem SA_zero_comp_half {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X) :
    SA d n 0 (fun s => X (s / 2)) := by
  intro t i
  have h := (hX (t / 2) i).scomp t ((hasDerivAt_id' t).div_const 2)
  refine h.congr_deriv ?_
  rw [SA_zero_drift (fun s => X (s / 2)) t i, ← proj_smul, smul_smul]
  congr 2
  ring

/-- **The flow of `e:gradfl`:** the `SA` flow at `β = 0`, run at double speed. -/
noncomputable def e0Flow (X₀ : SphereTuple d n) (t : ℝ) : SphereTuple d n :=
  saFlow 0 X₀ (2 * t)

/-- **Every solution of `e:gradfl` is the flow from its initial tuple.** -/
theorem eq_e0Flow {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X) (t : ℝ) :
    X t = e0Flow (X 0) t := by
  have h := saFlow_eq_of_SA 0 _ (SA_zero_comp_half hX)
  simp only [zero_div] at h
  rw [e0Flow, h]
  show X t = X (2 * t / 2)
  rw [mul_div_cancel_left₀ t two_ne_zero]

/-- **The backward flow recovers the start:** a solution of `e:gradfl` starts
at the image of its state at time `T` under the flow map at time `-T`. -/
theorem e0Flow_neg_eq {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X) (T : ℝ) :
    e0Flow (X T) (-T) = X 0 := by
  have h := eq_e0Flow (fun t i => (hX (T + t) i).comp_const_add T t) (-T)
  simp only [add_neg_cancel, add_zero] at h
  exact h.symm

/-- The hypothesis of `eq_e0Flow` and `e0Flow_neg_eq` holds for a consensus. -/
example (x : SSphere 2) : E0GradientAscent 2 3 (fun _ => signedTuple x ∅) :=
  E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅)

/-- **Grönwall backward in time:** two solutions of `SA` at time `-t ≤ 0` are at
most `e^{10 n max{1,β} t}` times as far apart as at time `0`. The bound of
`dist_SA_le_exp`, for the reversed trajectories `s ↦ X (-s)`, which follow the
reversed field. -/
theorem dist_SA_neg_le_exp (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (X Y : ℝ → SphereTuple d n) (hX : SA d n β X) (hY : SA d n β Y)
    (t : ℝ) (ht : 0 ≤ t) :
    dist (X (-t)) (Y (-t)) ≤ dist (X 0) (Y 0) * Real.exp (10 * n * max 1 β * t) := by
  let x (s : ℝ) (i : Idx n) := (X (-s) i : EucSpace d)
  let y (s : ℝ) (i : Idx n) := (Y (-s) i : EucSpace d)
  have hx (s : ℝ) : HasDerivAt x (-saField d n β (x s)) s := by
    have h := (hasDerivAt_pi.mpr ((SA_iff d n β X).mp hX (-s))).scomp s (hasDerivAt_neg s)
    rw [neg_one_smul] at h
    exact h
  have hy (s : ℝ) : HasDerivAt y (-saField d n β (y s)) s := by
    have h := (hasDerivAt_pi.mpr ((SA_iff d n β Y).mp hY (-s))).scomp s (hasDerivAt_neg s)
    rw [neg_one_smul] at h
    exact h
  have hdiff (s : ℝ) : HasDerivAt (fun r => x r - y r)
      (-saField d n β (x s) - -saField d n β (y s)) s := (hx s).sub (hy s)
  have hc : Continuous (fun s => x s - y s) :=
    continuous_iff_continuousAt.mpr fun s => (hdiff s).continuousAt
  have hK : 0 ≤ 10 * (n : ℝ) * max 1 β := by positivity
  have hbound (s : ℝ) :
      ‖-saField d n β (x s) - -saField d n β (y s)‖
        ≤ 10 * (n : ℝ) * max 1 β * ‖x s - y s‖ := by
    have hxu : x s ∈ unitTuples d n :=
      fun i => mem_sphere_zero_iff_norm.mp (X (-s) i).2
    have hyu : y s ∈ unitTuples d n :=
      fun i => mem_sphere_zero_iff_norm.mp (Y (-s) i).2
    rw [neg_sub_neg, norm_sub_rev]
    simpa only [Real.coe_toNNReal _ hK] using
      lipschitzOnWith_iff_norm_sub_le.mp (lipschitzOnWith_saField d n β hβ hn) hxu hyu
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖x 0 - y 0‖) (K := 10 * (n : ℝ) * max 1 β) (ε := 0)
    hc.continuousOn (fun s _ => (hdiff s).hasDerivWithinAt) le_rfl
    (fun s _ => by simpa only [add_zero] using hbound s) t ⟨ht, le_rfl⟩
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm, x, y, neg_zero,
    dist_sphereTuple_coe] using hG

/-- The hypotheses of `dist_SA_neg_le_exp` hold for two consensus solutions. -/
example (x y : SSphere 1) : (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 ∧
    SA 1 1 1 (fun _ _ => x) ∧ SA 1 1 1 (fun _ _ => y) ∧ (0 : ℝ) ≤ 1 :=
  ⟨zero_le_one, one_pos, SA_const_consensus 1 1 one_pos 1 x,
    SA_const_consensus 1 1 one_pos 1 y, zero_le_one⟩

/-- **The backward flow map of `e:gradfl` is Lipschitz**, with constant
`e^{20 n T}` at time `-T ≤ 0`. -/
theorem lipschitzWith_e0Flow_neg (hn : 0 < n) (T : ℝ) (hT : 0 ≤ T) :
    LipschitzWith (Real.toNNReal (Real.exp (20 * n * T)))
      (fun X₀ : SphereTuple d n => e0Flow X₀ (-T)) := by
  apply LipschitzWith.of_dist_le_mul
  intro X₀ Y₀
  rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
  have h := dist_SA_neg_le_exp 0 le_rfl hn _ _ (saFlow_spec 0 X₀).2 (saFlow_spec 0 Y₀).2
    (2 * T) (by positivity)
  rw [saFlow_zero, saFlow_zero, max_eq_left zero_le_one, mul_one,
    show 10 * (n : ℝ) * (2 * T) = 20 * n * T by ring] at h
  simpa only [e0Flow, mul_neg] using h

/-- The hypotheses of `lipschitzWith_e0Flow_neg` hold for `n = 1`, `T = 0`. -/
example : (0 : ℕ) < 1 ∧ (0 : ℝ) ≤ 0 := ⟨one_pos, le_rfl⟩

end Perspective
end Transformer

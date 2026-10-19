module

public import Transformer.Interpolation.Section1_PhaseDerivatives
public import Transformer.Interpolation.Section1_ControlPartitions
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Atomicity across all original complete control phases

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: vf` and `eq: cauchy.pb`.
For any actual original weak law, the spherical variance is continuous
through switches and obeys a linear differential inequality inside
each constant phase. An integrating factor propagates zero variance
across the closed phase, without differentiating at either endpoint.

Induction over the source's actual finite switching grid gives a unit
barycenter everywhere on the prescribed horizon. The original law is
therefore a Dirac at that barycenter whenever its initial law is a
Dirac. The complete nonlinear field and all original weak solutions
are covered; no atomicity or weak uniqueness is assumed.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- An integrating factor fences variance on a closed phase, using
only continuity at the endpoints. Source auxiliary:
arXiv:2411.04551v3, §1, piecewise constant `eq: cauchy.pb`. -/
theorem nonpos_on_Icc_of_deriv_le_mul (e e' : ℝ → ℝ) (C a b : ℝ)
    (hc : ContinuousOn e (Icc a b))
    (hd : ∀ s ∈ Ioo a b, HasDerivAt e (e' s) s) (h0 : e a ≤ 0)
    (hle : ∀ s ∈ Ioo a b, e' s ≤ C * e s) (t : ℝ) (ht : t ∈ Icc a b) :
    e t ≤ 0 := by
  have hab : a ≤ b := ht.1.trans ht.2
  let g (s : ℝ) := exp (-C * s) * e s
  have he (s : ℝ) : HasDerivAt (fun r => exp (-C * r)) (exp (-C * s) * (-C)) s := by
    simpa only [mul_one, id_eq] using ((hasDerivAt_id s).const_mul (-C)).exp
  have hg (s : ℝ) (hs : s ∈ Ioo a b) :
      HasDerivAt g (exp (-C * s) * (e' s - C * e s)) s := by
    convert (he s).mul (hd s hs) using 1
    ring
  have ha : AntitoneOn g (Icc a b) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc a b)
    · exact ((continuous_const.mul continuous_id).rexp).continuousOn.mul hc
    · intro s hs
      rw [interior_Icc] at hs
      exact (hg s hs).differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Icc] at hs
      rw [(hg s hs).deriv]
      exact mul_nonpos_of_nonneg_of_nonpos (exp_pos _).le (sub_nonpos.mpr (hle s hs))
  have hgt : g t ≤ 0 := (ha ⟨le_rfl, hab⟩ ht ht.1).trans
    (mul_nonpos_of_nonneg_of_nonpos (exp_pos _).le h0)
  exact nonpos_of_mul_nonpos_right hgt (exp_pos _)

/-- Zero original variance propagates across a complete constant control
phase, including its endpoints. Source auxiliary:
arXiv:2411.04551v3, §1, the atomic case of `eq: cauchy.pb`. -/
theorem variance_nonpos_on_constant_phase (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (a b : ℝ)
    (hθ : ∀ s ∈ Ico a b, ∀ r ∈ Ico a b, θ s = θ r)
    (h0 : 1 - ‖barycenter d (μ a)‖ ^ 2 ≤ 0) (t : ℝ) (ht : t ∈ Icc a b) :
    1 - ‖barycenter d (μ t)‖ ^ 2 ≤ 0 := by
  let e (s : ℝ) := 1 - ‖barycenter d (μ s)‖ ^ 2
  let e' (s : ℝ) := -2 * inner ℝ (barycenter d (μ s))
    (∫ x : SSphere d, fullVF d θ (μ s) s x ∂(μ s : Measure (SSphere d)))
  let C := 2 * (‖(θ a).V‖ * (2 * exp (2 * ‖(θ a).B‖) * ‖(θ a).B‖) +
    ‖(θ a).W‖ * ‖(θ a).U‖ +
    2 * (‖(θ a).V‖ + ‖(θ a).W‖ * (‖(θ a).U‖ + ‖(θ a).b‖)))
  refine nonpos_on_Icc_of_deriv_le_mul e e' C a b
    (continuous_cauchyPB_variance θ μ hμ).continuousOn ?_ h0 ?_ t ht
  · intro s hs
    exact hasDerivAt_cauchyPB_locally_constant_variance θ μ hμ (θ s) s
      (eventuallyEq_const_of_constantOn_Ico θ a b s hθ hs)
  · intro s hs
    have hsA := hθ s ⟨hs.1.le, hs.2⟩ a ⟨le_rfl, hs.1.trans hs.2⟩
    dsimp only [e', e, C]
    simpa only [hsA] using cauchyPB_variance_flux_le θ (μ s) s

/-- Every original weak law from a unit atom has a unit barycenter
throughout an arbitrary finite complete control sequence. Source:
arXiv:2411.04551v3, §1, `eq: cauchy.pb` and the point-mass setting. -/
theorem norm_barycenter_eq_one_of_piecewise_dirac (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (p : SSphere d) (h0 : μ 0 = Perspective.diracProb d p)
    (T : ℝ) (K : ℕ) (hθ : PiecewiseConstant d θ T K) (t : ℝ) (ht : t ∈ Icc 0 T) :
    ‖barycenter d (μ t)‖ = 1 := by
  obtain ⟨τ, hmono, hzero, hlast, hconst⟩ := hθ
  let e (s : ℝ) := 1 - ‖barycenter d (μ s)‖ ^ 2
  have he0 : e 0 ≤ 0 := by
    dsimp only [e]
    rw [h0]
    simp [barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp p.2]
  have hstep (i : Fin K) (hi : e (τ i.castSucc) ≤ 0) :
      ∀ s ∈ Icc (τ i.castSucc) (τ i.succ), e s ≤ 0 :=
    fun s hs => variance_nonpos_on_constant_phase θ μ hμ _ _
      (fun r hr r' hr' => hconst i r r' hr hr') hi s hs
  have hgrid (i : Fin (K + 1)) : ∀ s ∈ Icc 0 (τ i), e s ≤ 0 := by
    induction i using Fin.induction with
    | zero =>
      intro s hs
      rw [hzero] at hs
      have hs0 : s = 0 := le_antisymm hs.2 hs.1
      rw [hs0]
      exact he0
    | succ i hi =>
      have ha : 0 ≤ τ i.castSucc := by
        rw [← hzero]
        exact hmono (Fin.zero_le _)
      have heA : e (τ i.castSucc) ≤ 0 := hi _ ⟨ha, le_rfl⟩
      intro s hs
      by_cases hsa : s ≤ τ i.castSucc
      · exact hi s ⟨hs.1, hsa⟩
      · exact hstep i heA s ⟨(lt_of_not_ge hsa).le, hs.2⟩
  have he : e t ≤ 0 := hgrid (Fin.last K) t (by simpa only [hlast] using ht)
  have hn : ‖barycenter d (μ t)‖ ≤ 1 := norm_spherical_mean_le_one (μ t)
  have hp := norm_nonneg (barycenter d (μ t))
  dsimp only [e] at he
  nlinarith

/-- Every actual original weak solution from a unit atom stays an
actual Dirac throughout the full piecewise constant network. Source:
arXiv:2411.04551v3, §1, `eq: ips` as a case of `eq: cauchy.pb`. -/
theorem exists_dirac_of_cauchyPB_piecewise (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (p : SSphere d) (h0 : μ 0 = Perspective.diracProb d p)
    (T : ℝ) (K : ℕ) (hθ : PiecewiseConstant d θ T K) (t : ℝ) (ht : t ∈ Icc 0 T) :
    ∃ q : SSphere d, (q : EucSpace d) = barycenter d (μ t) ∧ μ t = Perspective.diracProb d q := by
  have hn := norm_barycenter_eq_one_of_piecewise_dirac θ μ hμ p h0 T K hθ t ht
  let q : SSphere d := ⟨barycenter d (μ t), mem_sphere_zero_iff_norm.mpr hn⟩
  refine ⟨q, rfl, eq_diracProb_of_integral_squared_distance_eq_zero (μ t) q ?_⟩
  rw [integral_spherical_squared_distance, integral_inner (integrable_spherical_coe (μ t))]
  change 2 - 2 * inner ℝ (barycenter d (μ t)) (barycenter d (μ t)) = 0
  rw [real_inner_self_eq_norm_sq, hn]
  norm_num

/-- A genuine stationary unit atom under nonzero original attention
satisfies all weak-law, initial-law and finite-control-grid hypotheses.
The nonzero constant scalar function below also witnesses every
integrating-factor premise, with equality in its growth bound. -/
example : ∃ (θ : TimeParams 2) (μ : ℝ → Perspective.ProbSphere 2) (p : SSphere 2),
    cauchyPB 2 θ μ ∧ μ 0 = Perspective.diracProb 2 p ∧ PiecewiseConstant 2 θ 1 1 ∧
    (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (∀ s ∈ Ico (0 : ℝ) 1, ∀ r ∈ Ico (0 : ℝ) 1, θ s = θ r) ∧
    ContinuousOn (fun _ : ℝ => (-1 : ℝ)) (Icc (0 : ℝ) 1) ∧
    (∀ s ∈ Ioo (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (-1 : ℝ)) 0 s) ∧
    (-1 : ℝ) ≤ 0 ∧ (∀ s ∈ Ioo (0 : ℝ) 1, (0 : ℝ) ≤ 0 * (-1)) := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let A : Params 2 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 2 := basePoint 1
  have hθ : pureAttentionParams 2 1 = (fun _ => A) := by
    funext s
    simp only [pureAttentionParams, one_smul]
    rfl
  have hμ : cauchyPB 2 (fun _ => A) (fun _ => Perspective.diracProb 2 p) := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  exact ⟨(fun _ => A), (fun _ => Perspective.diracProb 2 p), p, hμ, rfl,
    piecewiseConstant_constant A 1 zero_le_one, by norm_num, (fun _ _ _ _ => rfl),
    continuous_const.continuousOn, (fun s _ => hasDerivAt_const s (-1)), by norm_num,
    (fun _ _ => by norm_num)⟩

end Transformer.Interpolation

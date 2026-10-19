module

public import Transformer.Interpolation.Section1_PiecewiseAtomicity
public import Transformer.Interpolation.Section1_AtomicField
public import Transformer.Interpolation.Wasserstein

/-!
# Quantitative separation limit for original atomic inputs

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: vf`, `eq: cauchy.pb`,
and the parameter estimate in `thm: targets.atoms`.
Every weak solution from a unit atom stays a Dirac under the actual
complete piecewise constant parameters. Its original barycenter
velocity is therefore the complete one-atom velocity.

For two different original weak laws driven by the same bounded
parameters, their squared barycenter gap grows at most exponentially
with constant twice 3K+5K². The proof uses the original weak equation
inside each phase and continuity across the actual finite grid.
There is no assumed common flow map or weak uniqueness.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Bounded original complete controls cannot separate arbitrarily
close unit atoms by an arbitrary amount in fixed time. Source auxiliary:
arXiv:2411.04551v3, §1.1, `thm: targets.atoms`, applied to `eq: ips`
through the actual original weak equation. -/
theorem squared_barycenter_gap_le_exp_of_dirac (θ : TimeParams d)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (p q : SSphere d) (hμ0 : μ 0 = Perspective.diracProb d p)
    (hν0 : ν 0 = Perspective.diracProb d q) (T M : ℝ) (K : ℕ)
    (hθ : PiecewiseConstant d θ T K)
    (hM : ∀ s ∈ Icc (0 : ℝ) T,
      ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖ ≤ M) :
    ‖barycenter d (μ T) - barycenter d (ν T)‖ ^ 2 ≤
      ‖(p : EucSpace d) - (q : EucSpace d)‖ ^ 2 * exp (2 * (3 * M + 5 * M ^ 2) * T) := by
  let v (ρ : ℝ → Perspective.ProbSphere d) (s : ℝ) :=
    ∫ x : SSphere d, fullVF d θ (ρ s) s x ∂(ρ s : Measure (SSphere d))
  let L := 3 * M + 5 * M ^ 2
  let e (s : ℝ) := ‖barycenter d (μ s) - barycenter d (ν s)‖ ^ 2
  let e' (s : ℝ) := 2 * inner ℝ (barycenter d (μ s) - barycenter d (ν s)) (v μ s - v ν s)
  have hv (s : ℝ) (hs : s ∈ Icc (0 : ℝ) T) :
      ‖v μ s - v ν s‖ ≤ L * ‖barycenter d (μ s) - barycenter d (ν s)‖ := by
    obtain ⟨a, ha, hμs⟩ := exists_dirac_of_cauchyPB_piecewise θ μ hμ p hμ0 T K hθ s hs
    obtain ⟨b, hb, hνs⟩ := exists_dirac_of_cauchyPB_piecewise θ ν hν q hν0 T K hθ s hs
    have hvm : v μ s = fullVF d θ (Perspective.diracProb d a) s a := by
      dsimp only [v]
      rw [hμs]
      change (∫ x : SSphere d, fullVF d θ (Perspective.diracProb d a) s x ∂Measure.dirac a) = _
      rw [integral_dirac]
    have hvn : v ν s = fullVF d θ (Perspective.diracProb d b) s b := by
      dsimp only [v]
      rw [hνs]
      change (∫ x : SSphere d, fullVF d θ (Perspective.diracProb d b) s x ∂Measure.dirac b) = _
      rw [integral_dirac]
    rw [hvm, hvn, ← ha, ← hb]
    exact norm_fullVF_dirac_pair_sub_le_of_sum_bound θ s M a b (hM s hs)
  have he (s : ℝ) (hs : s ∈ Icc (0 : ℝ) T) : e' s ≤ (2 * L) * e s := by
    have hi := real_inner_le_norm (barycenter d (μ s) - barycenter d (ν s)) (v μ s - v ν s)
    have hh := mul_le_mul_of_nonneg_left (hv s hs)
      (norm_nonneg (barycenter d (μ s) - barycenter d (ν s)))
    dsimp only [e', e]
    nlinarith
  let a0 := ‖(p : EucSpace d) - (q : EucSpace d)‖ ^ 2
  let z (s : ℝ) := e s - a0 * exp ((2 * L) * s)
  let z' (s : ℝ) := e' s - a0 * (exp ((2 * L) * s) * (2 * L))
  have hzc : Continuous z := by
    exact (((continuous_cauchyPB_mean θ μ hμ).sub
      (continuous_cauchyPB_mean θ ν hν)).norm.pow 2).sub
        (continuous_const.mul (continuous_const.mul continuous_id).rexp)
  have hz0 : z 0 ≤ 0 := by
    dsimp only [z, e, a0]
    rw [hμ0, hν0]
    simp [barycenter, Perspective.diracProb]
  obtain ⟨τ, hmono, hzero, hlast, hconst⟩ := hθ
  have hstep (i : Fin K) (hzi : z (τ i.castSucc) ≤ 0) :
      ∀ s ∈ Icc (τ i.castSucc) (τ i.succ), z s ≤ 0 := by
    have hi0 : 0 ≤ τ i.castSucc := by rw [← hzero]; exact hmono (Fin.zero_le _)
    have hiT : τ i.succ ≤ T := by rw [← hlast]; exact hmono (Fin.le_last _)
    have hglob (s : ℝ) (hs : s ∈ Ioo (τ i.castSucc) (τ i.succ)) : s ∈ Icc (0 : ℝ) T :=
      ⟨hi0.trans hs.1.le, hs.2.le.trans hiT⟩
    refine nonpos_on_Icc_of_deriv_le_mul z z' (2 * L) _ _ hzc.continuousOn ?_ hzi ?_
    · intro s hs
      have hloc := eventuallyEq_const_of_constantOn_Ico θ _ _ s
        (fun r hr r' hr' => hconst i r r' hr hr') hs
      have hde : HasDerivAt e (e' s) s :=
        ((hasDerivAt_cauchyPB_locally_constant_barycenter θ μ hμ (θ s) s hloc).sub
          (hasDerivAt_cauchyPB_locally_constant_barycenter θ ν hν (θ s) s hloc)).norm_sq
      have hdx : HasDerivAt (fun r => exp ((2 * L) * r))
          (exp ((2 * L) * s) * (2 * L)) s := by
        simpa only [mul_one, id_eq] using ((hasDerivAt_id s).const_mul (2 * L)).exp
      exact hde.sub (hdx.const_mul a0)
    · intro s hs
      have hh := he s (hglob s hs)
      dsimp only [z', z]
      nlinarith
  have hgrid (i : Fin (K + 1)) : ∀ s ∈ Icc 0 (τ i), z s ≤ 0 := by
    induction i using Fin.induction with
    | zero =>
      intro s hs
      rw [hzero] at hs
      have hs0 : s = 0 := le_antisymm hs.2 hs.1
      rw [hs0]
      exact hz0
    | succ i hi =>
      have ha : 0 ≤ τ i.castSucc := by rw [← hzero]; exact hmono (Fin.zero_le _)
      have hza : z (τ i.castSucc) ≤ 0 := hi _ ⟨ha, le_rfl⟩
      intro s hs
      by_cases hsa : s ≤ τ i.castSucc
      · exact hi s ⟨hs.1, hsa⟩
      · exact hstep i hza s ⟨(lt_of_not_ge hsa).le, hs.2⟩
  have hT : 0 ≤ T := by rw [← hlast, ← hzero]; exact hmono (Fin.zero_le _)
  have hzT : z T ≤ 0 := hgrid (Fin.last K) T (by rw [hlast]; exact ⟨hT, le_rfl⟩)
  exact sub_nonpos.mp hzT

/-- The actual W₂ distance controls original spherical barycenters.
Source auxiliary: arXiv:2411.04551v3, §1, `thm: targets.atoms`. -/
theorem norm_barycenter_sub_le_W2 (μ ν : Perspective.ProbSphere d) :
    ‖barycenter d μ - barycenter d ν‖ ≤
      W2 d (μ : Measure (SSphere d)) (ν : Measure (SSphere d)) := by
  simpa only [barycenter, W2, one_mul] using Wasserstein.norm_integral_sub_le_W2
    (μ : Measure (SSphere d)) (ν : Measure (SSphere d))
    MeasurableSet.univ Metric.isBounded_of_compactSpace (by simp) (by simp)
    (fun x : SSphere d => (x : EucSpace d)) (integrable_spherical_coe μ) (integrable_spherical_coe ν)
    zero_le_one (fun x _ y _ => by
      change ‖(x : EucSpace d) - (y : EucSpace d)‖ ≤
        1 * ‖(x : EucSpace d) - (y : EucSpace d)‖
      rw [one_mul])

/-- An actual W₂ approximation to a unit atom bounds barycenter error.
Source auxiliary: arXiv:2411.04551v3, §1, `thm: targets.atoms`. -/
theorem norm_barycenter_sub_unit_le_W2 (μ : Perspective.ProbSphere d) (p : SSphere d) :
    ‖barycenter d μ - (p : EucSpace d)‖ ≤
      W2 d (μ : Measure (SSphere d)) (Measure.dirac p) := by
  simpa [barycenter, Perspective.diracProb] using
    norm_barycenter_sub_le_W2 μ (Perspective.diracProb d p)

/-- Two accurate original atomic targets require a corresponding
barycenter gap. Source auxiliary: arXiv:2411.04551v3, §1,
`thm: targets.atoms`, with its original W₂ accuracy. -/
theorem target_distance_le_barycenter_gap_add_errors (μ ν : Perspective.ProbSphere d)
    (p q : SSphere d) (ε : ℝ)
    (hp : W2 d (μ : Measure (SSphere d)) (Measure.dirac p) ≤ ε)
    (hq : W2 d (ν : Measure (SSphere d)) (Measure.dirac q) ≤ ε) :
    ‖(p : EucSpace d) - (q : EucSpace d)‖ ≤ 2 * ε + ‖barycenter d μ - barycenter d ν‖ := by
  have hpm : ‖(p : EucSpace d) - barycenter d μ‖ ≤ ε := by
    rw [norm_sub_rev]
    exact (norm_barycenter_sub_unit_le_W2 μ p).trans hp
  have hqm := (norm_barycenter_sub_unit_le_W2 ν q).trans hq
  have ht₁ := norm_sub_le_norm_sub_add_norm_sub (p : EucSpace d) (barycenter d μ) (q : EucSpace d)
  have ht₂ := norm_sub_le_norm_sub_add_norm_sub (barycenter d μ) (barycenter d ν) (q : EucSpace d)
  linarith


/-- Distinct unit atomic inputs, actual original nonzero constant
attention, genuine weak laws and a positive finite horizon realize all
hypotheses jointly, with the actual source sum norm bounded by two. -/
example : ∃ (θ : TimeParams 2) (μ ν : ℝ → Perspective.ProbSphere 2) (p q : SSphere 2),
    p ≠ q ∧ cauchyPB 2 θ μ ∧ cauchyPB 2 θ ν ∧
    μ 0 = Perspective.diracProb 2 p ∧ ν 0 = Perspective.diracProb 2 q ∧
    PiecewiseConstant 2 θ 1 1 ∧ (∀ s ∈ Icc (0 : ℝ) 1,
      ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖ ≤ 2) := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let A : Params 2 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 2 := basePoint 1
  let q := antipode 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => A) := by
    funext s
    simp only [pureAttentionParams, one_smul]
    rfl
  have hμ (r : SSphere 2) : cauchyPB 2 (fun _ => A) (fun _ => Perspective.diracProb 2 r) := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 r
  refine ⟨(fun _ => A), (fun _ => Perspective.diracProb 2 p),
    (fun _ => Perspective.diracProb 2 q), p, q, (antipode_ne 2 p).symm, hμ p, hμ q,
    rfl, rfl, piecewiseConstant_constant A 1 zero_le_one, ?_⟩
  intro s hs
  norm_num [A, I]

/-- Genuine atomic laws realize both original target-accuracy premises. -/
example (p q : SSphere 2) :
    W2 2 (Measure.dirac p) (Measure.dirac p) ≤ (1 / 4 : ℝ) ∧
    W2 2 (Measure.dirac q) (Measure.dirac q) ≤ (1 / 4 : ℝ) := by
  constructor <;> rw [W2_self] <;> norm_num

end Transformer.Interpolation

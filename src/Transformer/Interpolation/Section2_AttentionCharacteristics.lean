module

public import Transformer.Interpolation.Section2_AttentionExtension
public import Transformer.Interpolation.Section2_AttentionWeakTests

/-!
# Complete original attention characteristics

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`, and §2,
`prop: targets.atoms`. Given any original weak solution, its frozen
law-dependent field has a unique complete trajectory through each unit
initial point. The bounded extension is continuous in time and globally
Lipschitz in space. Its squared-norm defect solves a homogeneous scalar
ODE, so trajectories starting on the sphere remain there and solve the
original equation exactly. No characteristic representation of the law
is assumed in this construction.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Time continuity of the actual extended field along the weak law.
Auxiliary for arXiv:2411.04551v3, §1, characteristics of `eq: vf`. -/
theorem continuous_attentionGlobalField_time (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : EucSpace d) : Continuous (fun t => attentionGlobalField B (μ t) x) := by
  have hm := continuous_attentionAverage_time B μ hμ (Landscape.GlobalFlow.ballRetract 2 x)
  change Continuous (fun t => attentionCutoff x •
    (attentionAverage B (μ t) (Landscape.GlobalFlow.ballRetract 2 x) -
      inner ℝ (Landscape.GlobalFlow.ballRetract 2 x) (attentionAverage B (μ t) (Landscape.GlobalFlow.ballRetract 2 x)) •
        Landscape.GlobalFlow.ballRetract 2 x))
  have hx : Continuous (fun _ : ℝ => Landscape.GlobalFlow.ballRetract 2 x) := continuous_const
  have hcut : Continuous (fun _ : ℝ => attentionCutoff x) := continuous_const
  exact hcut.smul (hm.sub ((hx.inner (𝕜 := ℝ) hm).smul hx))

/-- Joint continuity of the extended field along the actual law.
Auxiliary for arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem continuous_attentionGlobalField_joint (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ) :
    Continuous (fun q : ℝ × EucSpace d => attentionGlobalField B (μ q.1) q.2) := by
  exact continuous_prod_of_continuous_lipschitzWith' _ _
    (fun t => lipschitzWith_attentionGlobalField B (μ t))
    (fun x => continuous_attentionGlobalField_time B μ hμ x)

/-- The retracted original average is globally Lipschitz, uniformly in
the law. Auxiliary for arXiv:2411.04551v3, §1, characteristics. -/
theorem lipschitzWith_retracted_attentionAverage (B : ParamMatrix d) (ν : Perspective.ProbSphere d) :
    LipschitzWith ⟨4 * exp (2 * (‖B‖ * 2)) * ‖B‖, by positivity⟩
      (fun x : EucSpace d => attentionAverage B ν (Landscape.GlobalFlow.ballRetract 2 x)) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  change ‖attentionAverage B ν (Landscape.GlobalFlow.ballRetract 2 x) -
    attentionAverage B ν (Landscape.GlobalFlow.ballRetract 2 y)‖ ≤
      (4 * exp (2 * (‖B‖ * 2)) * ‖B‖) * ‖x - y‖
  have ha := norm_attentionAverage_sub_le B ν 2 (Landscape.GlobalFlow.ballRetract 2 x) (Landscape.GlobalFlow.ballRetract 2 y)
    (Landscape.GlobalFlow.norm_ballRetract_le (by norm_num) x) (Landscape.GlobalFlow.norm_ballRetract_le (by norm_num) y)
  have hr := (Landscape.GlobalFlow.lipschitzWith_ballRetract (by norm_num : (0 : ℝ) < 2)).dist_le_mul x y
  rw [dist_eq_norm, dist_eq_norm] at hr
  calc
    _ ≤ (2 * exp (2 * (‖B‖ * 2)) * ‖B‖) * (2 * ‖x - y‖) :=
      ha.trans (mul_le_mul_of_nonneg_left hr (by positivity))
    _ = _ := by ring

/-- Joint continuity of the retracted average supplies the norm-defect
coefficient. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_retracted_attentionAverage_joint (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ) :
    Continuous (fun q : ℝ × EucSpace d => attentionAverage B (μ q.1) (Landscape.GlobalFlow.ballRetract 2 q.2)) := by
  exact continuous_prod_of_continuous_lipschitzWith' _ _
    (fun t => lipschitzWith_retracted_attentionAverage B (μ t))
    (fun x => continuous_attentionAverage_time B μ hμ (Landscape.GlobalFlow.ballRetract 2 x))

/-- The radial velocity is proportional to the squared-norm defect,
also outside the cutoff. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem inner_attentionGlobalField_norm_defect (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x : EucSpace d) : 2 * inner ℝ x (attentionGlobalField B ν x) =
      (-2 * attentionCutoff x * inner ℝ x (attentionAverage B ν (Landscape.GlobalFlow.ballRetract 2 x))) *
        (‖x‖ ^ 2 - 1) := by
  by_cases hx : ‖x‖ ≤ 2
  · rw [attentionGlobalField, Landscape.GlobalFlow.ballRetract_of_norm_le (by norm_num : (0 : ℝ) < 2) hx]
    simp only [real_inner_smul_right, proj, inner_sub_right, real_inner_self_eq_norm_sq]
    ring
  · have hc := attentionCutoff_eq_zero x (le_of_not_ge hx)
    simp only [attentionGlobalField, hc, zero_smul, inner_zero_right, mul_zero, zero_mul]

/-- Every extended trajectory with a unit initial point stays on the
original sphere. Source: arXiv:2411.04551v3, §1, tangent projection. -/
theorem norm_eq_one_of_attentionGlobalField (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (Y : ℝ → EucSpace d) (hY : ∀ t, HasDerivAt Y (attentionGlobalField B (μ t) (Y t)) t)
    (h0 : ‖Y 0‖ = 1) (t : ℝ) : ‖Y t‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let a (s : ℝ) := -2 * attentionCutoff (Y s) *
    inner ℝ (Y s) (attentionAverage B (μ s) (Landscape.GlobalFlow.ballRetract 2 (Y s)))
  have hm : Continuous (fun s => attentionAverage B (μ s) (Landscape.GlobalFlow.ballRetract 2 (Y s))) := by
    have hp : Continuous (fun s : ℝ => (s, Y s)) := continuous_id.prodMk hc
    simpa only [Function.comp_def] using (continuous_retracted_attentionAverage_joint B μ hμ).comp hp
  have ha : Continuous a := (continuous_const.mul
    (lipschitzWith_attentionCutoff.continuous.comp hc)).mul (hc.inner hm)
  have hd (s : ℝ) : HasDerivAt (fun r => ‖Y r‖ ^ 2 - 1) (a s * (‖Y s‖ ^ 2 - 1)) s := by
    convert (hY s).norm_sq.sub_const 1 using 1
    exact (inner_attentionGlobalField_norm_defect B (μ s) (Y s)).symm
  have hz := Perspective.scalar_eq_zero_of_linearODE a _ ha hd (by rw [h0]; norm_num) t
  nlinarith [norm_nonneg (Y t)]

/-- Every unit initial point has a complete actual attention particle
along the given weak law. Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem exists_attention_particle (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) : ∃ Y : ℝ → SSphere d, Y 0 = x ∧ ∀ t,
      HasDerivAt (fun s => (Y s : EucSpace d))
        (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t (Y t)) t := by
  obtain ⟨Z, h0, hZ⟩ := Landscape.GlobalFlow.exists_timeDependent_global
    (fun t x => attentionGlobalField B (μ t) x) _ 5 (by norm_num)
    (fun t => lipschitzWith_attentionGlobalField B (μ t))
    (fun x => continuous_attentionGlobalField_time B μ hμ x)
    (fun t x => norm_attentionGlobalField_le B (μ t) x) (x : EucSpace d)
  have hnorm (t : ℝ) : ‖Z t‖ = 1 := norm_eq_one_of_attentionGlobalField B μ hμ Z hZ
    (by rw [h0]; exact mem_sphere_zero_iff_norm.mp x.2) t
  let Y (t : ℝ) : SSphere d := ⟨Z t, mem_sphere_zero_iff_norm.mpr (hnorm t)⟩
  refine ⟨Y, Subtype.ext h0, fun t => ?_⟩
  rw [← attentionGlobalField_eq_fullVF B (μ t) t (Y t)]
  exact hZ t

/-- Original attention particles along the same evolving law are fixed
by their initial point. Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem unique_attention_particle (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (X Y : ℝ → SSphere d)
    (hX : ∀ t, HasDerivAt (fun s => (X s : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t (X t)) t)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t (Y t)) t)
    (h0 : X 0 = Y 0) : X = Y := by
  have he := Landscape.GlobalFlow.timeDependent_unique (fun t x => attentionGlobalField B (μ t) x) _
    (fun t => lipschitzWith_attentionGlobalField B (μ t))
    (fun s => (X s : EucSpace d)) (fun s => (Y s : EucSpace d))
    (fun t => by rw [attentionGlobalField_eq_fullVF]; exact hX t)
    (fun t => by rw [attentionGlobalField_eq_fullVF]; exact hY t) (congrArg Subtype.val h0)
  exact funext fun t => Subtype.ext (congrFun he t)

/-- A nonzero attention matrix, its actual stationary Dirac weak law,
and a genuine unit trajectory jointly satisfy the weak, extended and
original particle equations, the unit initial condition, and uniqueness
initial-value hypotheses. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2) (Y : ℝ → SSphere 2),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2)) (attentionGlobalField B (μ t) (Y t)) t) ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2))
      (fullVF 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) (μ t) t (Y t)) t) ∧
    ‖(Y 0 : EucSpace 2)‖ = 1 ∧ Y 0 = Y 0 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hY (t : ℝ) : HasDerivAt (fun _ : ℝ => (p : EucSpace 2))
      (fullVF 2 (fun _ => ⟨B, B, 0, 0, 0⟩) (μ t) t p) t := by
    rw [← hθ, fullVF_pureAttentionParams, Perspective.vectorField_diracProb_self]
    exact hasDerivAt_const t (p : EucSpace 2)
  refine ⟨B, μ, (fun _ => p), ?_, ?_, hY, mem_sphere_zero_iff_norm.mp p.2, rfl⟩
  · rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  · intro t
    rw [attentionGlobalField_eq_fullVF B (μ t) t p]
    exact hY t

end Transformer.Interpolation

module

public import Transformer.Interpolation.Section1_TestModulus
public import Transformer.Interpolation.Section1_CompactTests
public import Transformer.Interpolation.Section3_RidgeLaw

/-!
# Continuous tests and moving linear tests for autonomous weak solutions

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`.

The uniform time modulus extends the original weak equation's continuity
from `C¹` tests to every continuous ambient scalar test whenever the
original velocity is uniformly bounded. A continuous autonomous field
is bounded on the compact sphere, so its actual weak solutions have
this continuity without any added regularity premise on the measure curve.

Consequently the original flux of each `C¹` test is continuous and the
integrated weak equation differentiates pointwise. The compact moving-test
rule then differentiates the pairing with an actual moving particle.
This supplies the original squared-distance test used to transport atoms.
It requires continuity of the field, including across the ReLU kink,
and does not require differentiating the field or the flow in space.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A uniform velocity bound gives continuity of every continuous scalar
test integral along every original weak solution.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_bounded_test (θ : TimeParams d) (B : ℝ) (hB : 0 ≤ B)
    (hVF : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), ‖fullVF d θ ρ s x‖ ≤ B)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (f : EucSpace d → ℝ) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) := by
  apply continuous_iff_continuousAt.mpr
  intro t
  rw [Metric.continuousAt_iff]
  intro ε hε
  obtain ⟨δ, hδ, hm⟩ := exists_uniform_cauchyPB_test_radius θ B hB hVF f hf ε hε
  refine ⟨δ, hδ, fun s hs => ?_⟩
  simpa only [dist_eq_norm] using hm μ hμ t s (by simpa only [Real.dist_eq] using hs)

/-- For a continuous autonomous original field, compactness supplies the
bound needed to obtain continuity of every continuous scalar test integral.
Source: arXiv:2411.04551v3, §1, the weak solution in `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_autonomous_test (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (f : EucSpace d → ℝ) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) := by
  obtain ⟨B, hB⟩ := (isCompact_univ : IsCompact (univ : Set (SSphere d))).exists_bound_of_continuousOn
    (hF.comp continuous_subtype_val).continuousOn
  exact continuous_cauchyPB_bounded_test θ |B| (abs_nonneg _) (fun ρ s x => by
    rw [hfield]
    exact (hB x (mem_univ x)).trans (le_abs_self _)) μ hμ f hf

/-- The original weak equation has its pointwise test derivative for an
autonomous continuous field. Its actual flux is a continuous test integral;
no time regularity of the weak solution is assumed separately.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_autonomous_test (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d)) (F x)
        ∂(μ t : Measure (SSphere d))) t := by
  let f (x : EucSpace d) := fderiv ℝ φ x (F x)
  have hf : Continuous f :=
    (hφ.continuous_fderiv_apply (by norm_num)).comp (continuous_id.prodMk hF)
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d θ (μ s) s x) ∂(μ s : Measure (SSphere d))) =
      (fun s => ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))) := by
    funext s
    apply integral_congr_ae
    exact ae_of_all _ fun x => by dsimp only; rw [hfield, inner_gradient_left]
  have hc := continuous_cauchyPB_autonomous_test θ F hF hfield μ hμ f hf
  have hd := hasDerivAt_cauchyPB_test θ μ hμ φ hφ (heq ▸ hc) t
  convert hd using 1
  apply integral_congr_ae
  exact ae_of_all _ fun x => by dsimp only; rw [hfield]

/-- Pairing the actual law with an actual moving particle has both the
original weak-equation flux and the particle's own derivative.
The particle need only solve the original continuous field.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_moving_inner (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, inner ℝ (Y s : EucSpace d) (x : EucSpace d)
      ∂(μ s : Measure (SSphere d)))
      ((∫ x : SSphere d, inner ℝ (Y t : EucSpace d) (F x) ∂(μ t : Measure (SSphere d))) +
        ∫ x : SSphere d, inner ℝ (F (Y t)) (x : EucSpace d) ∂(μ t : Measure (SSphere d))) t := by
  let A (s : ℝ) (x : SSphere d) := inner ℝ (Y s : EucSpace d) (x : EucSpace d)
  let G (s : ℝ) (x : SSphere d) := inner ℝ (F (Y s)) (x : EucSpace d)
  have hYc : Continuous (fun s => (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  have hA : Continuous (Function.uncurry A) :=
    (hYc.comp continuous_fst).inner (continuous_subtype_val.comp continuous_snd)
  have hG : Continuous (Function.uncurry G) :=
    ((hF.comp hYc).comp continuous_fst).inner (continuous_subtype_val.comp continuous_snd)
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => A r x) (G s x) s := by
    simpa only [inner_zero_right, zero_add] using (hY s).inner ℝ (hasDerivAt_const s (x : EucSpace d))
  have hfixed := hasDerivAt_cauchyPB_autonomous_test θ F hF hfield μ hμ
    (fun x : EucSpace d => inner ℝ (Y t : EucSpace d) x) (contDiff_const.inner ℝ contDiff_id) t
  simp only [gradient_inner_linear] at hfixed
  exact hasDerivAt_spherical_compact_test μ A G hA hG hD t _ hfixed
    (continuous_cauchyPB_inner θ μ hμ (F (Y t))).continuousAt

/-- A genuine projection neuron, its transported law from a transverse
atom, its stationary unit output direction, and a nonconstant linear test
jointly realize all field, solution, path, and test hypotheses. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2)
    (μ : ℝ → Perspective.ProbSphere 2) (Y : ℝ → SSphere 2) (f : EucSpace 2 → ℝ),
    (0 : ℝ) ≤ 1 ∧ Continuous F ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), ‖fullVF 2 θ ρ s x‖ ≤ 1) ∧
    cauchyPB 2 θ μ ∧ Continuous f ∧ ContDiff ℝ 1 f ∧
    ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 ⟨EuclideanSpace.single 1 1, by simp⟩)
  let Y : ℝ → SSphere 2 := fun _ => basePoint 1
  let f (x : EucSpace 2) := inner ℝ a x
  have hF : Continuous F := by unfold F proj; fun_prop
  have hfield (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2) : fullVF 2 θ ρ s x = F x := by
    rw [fullVF_reluRidgeParams]
    simp [F]
  have ha : ‖a‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨θ, F, μ, Y, f, zero_le_one, hF, hfield, ?_, cauchyPB_ridgeLaw _ _ _ _ _,
    continuous_const.inner continuous_id, contDiff_const.inner ℝ contDiff_id, ?_⟩
  · intro ρ s x
    rw [hfield]
    exact (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) a).trans ha.le
  · intro t
    have he : F (Y t) = 0 := by simp [F, Y, a, proj, ha]
    rw [he]
    exact hasDerivAt_const t _

end Transformer.Interpolation

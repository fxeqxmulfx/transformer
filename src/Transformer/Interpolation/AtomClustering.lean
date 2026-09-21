/-
# Measure-to-measure interpolation — Clustering to a single point mass

`Proposition prop: targets.atoms` of §2 of arXiv:2411.04551v3: with
`(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)` and `supp μ_0` inside an open hemisphere, the
solution of `eq: cauchy.pb` collapses — the diameter of its support tends to
`0`, and for every `ε > 0` some `μ(T)` is within `W_∞`-distance `ε` of a point
mass.

**What the source says and what is changed here.**  `W_∞` is not available in
Mathlib, and the form this development gave the proposition carried it as a
plain parameter `Winf : Measure → Measure → ℝ`, constrained by nothing.  That
statement is false, and `not_forall_clustering_to_atom` refutes it: `Winf ≡ 1`
is a legal parameter, a Dirac mass sits in an open hemisphere, and the constant
curve at it solves `eq: cauchy.pb` — so the conclusion would give `1 ≤ 1/2`.

The repair is to pin `Winf` down by the property the proof uses,
`IsWinfToDirac`: a ball of radius `r ≥ 0` around `z` containing the support
bounds `W_∞(μ, δ_z)` by `r`.  That is one half of the identity
`W_∞(μ, δ_z) = sup_{x ∈ supp μ} d(x, z)`, and it is the half the proposition
needs.  Granted it, the second conclusion follows from the first, which is
`exists_dirac_close_of_diam_tendsto`; what stays unproved is the collapse of
the diameter itself.

Two further deviations are inherited from the source's own notation: the
geodesic convex hull `conv_g` is replaced by the support (the diameters agree
inside an open hemisphere, which is not proved here), and the membership
`z ∈ conv_g supp μ_0` is dropped.  The paper's companion rate
`inf{ t : W_2(μ(t), δ_z) ≤ ε } = O(log(1/ε))` is not formalized.

Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Mathlib.MeasureTheory.Measure.Support

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d : ℕ)

/-- **A Dirac mass is stationary for `eq: vf` when `𝐕 ≡ I_d` and `𝐖 ≡ 0`.**

The only point `δ_x` sees is `x` itself, so the softmax average is `x`
whatever `𝐁` is, the perceptron term is switched off, and `Proj_x` kills the
radial direction that remains.

Source: arXiv:2411.04551v3, §2, `eq: vf`. -/
theorem fullVF_diracProb_self (θ : TimeParams d) (t : ℝ) (x : SSphere d)
    (hV : (θ t).V = ContinuousLinearMap.id ℝ (EucSpace d)) (hW : (θ t).W = 0) :
    fullVF d θ (diracProb d x) t (x : EucSpace d) = 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d)) = Measure.dirac x := rfl
  rw [fullVF]
  simp only [hμ, integral_dirac, hV, hW, zero_apply,
    ContinuousLinearMap.id_apply, add_zero, smul_smul,
    inv_mul_cancel₀ (Real.exp_ne_zero _)]
  exact proj_smul_self hx 1

/-- **The constant curve at a Dirac mass solves `eq: cauchy.pb`.**

The integrand of the integrated form vanishes by `fullVF_diracProb_self`,
and the curve is constant.

Source: arXiv:2411.04551v3, §2, `eq: cauchy.pb`. -/
theorem cauchyPB_const_diracProb (θ : TimeParams d) (x : SSphere d)
    (hV : ∀ t : ℝ, (θ t).V = ContinuousLinearMap.id ℝ (EucSpace d))
    (hW : ∀ t : ℝ, (θ t).W = 0) :
    cauchyPB d θ (fun _ => diracProb d x) := by
  intro φ _ t
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d)) = Measure.dirac x := rfl
  have hrhs : (fun s => ∫ y, inner (𝕜 := ℝ) (gradient φ (y : EucSpace d))
        (fullVF d θ (diracProb d x) s (y : EucSpace d))
      ∂((diracProb d x : ProbSphere d) : Measure (SSphere d))) = fun _ => 0 := by
    funext s
    rw [hμ, integral_dirac, fullVF_diracProb_self d θ s x (hV s) (hW s), inner_zero_right]
  simp only [hrhs, intervalIntegral.integral_zero, add_zero]
  exact ⟨intervalIntegrable_const, trivial⟩

/-- The hypotheses of `cauchyPB_const_diracProb` are satisfiable: the
parameters of `prop: targets.atoms`, `(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)`. -/
example (B : ParamMatrix 1) :
    cauchyPB 1 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 1), B, 0, 0, 0⟩)
      (fun _ => diracProb 1 (basePoint 0)) :=
  cauchyPB_const_diracProb 1 _ (basePoint 0) (fun _ => rfl) (fun _ => rfl)

/-- **`prop: targets.atoms` with an unconstrained `W_∞` is false.**

A parameter `Winf` that no hypothesis constrains may be the constant `1`.  For
`d = 1` take `μ_0 = δ_{e_0}`: its support is `{e_0}`, which lies in the open
hemisphere around `e_0`, and the constant curve at `δ_{e_0}` solves
`eq: cauchy.pb` for the parameters of the proposition.  At `ε = 1/2` the
second conclusion would then read `1 ≤ 1/2`.

Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem not_forall_clustering_to_atom :
    ¬ ∀ (d : ℕ) (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ)
        (B : ParamMatrix d) (μ₀ : ProbSphere d),
      (∃ w : SSphere d, ∀ x ∈ (μ₀ : Measure (SSphere d)).support,
        0 < inner (𝕜 := ℝ) (x : EucSpace d) ((w : EucSpace d))) →
      ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ →
        cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ →
          Filter.Tendsto
              (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support))
              Filter.atTop (nhds 0) ∧
            ∀ ε : ℝ, 0 < ε → ∃ (z : SSphere d) (T : ℝ), 0 < T ∧
              Winf (μ T : Measure (SSphere d)) (Measure.dirac z) ≤ ε := by
  intro h
  have hhemi : ∃ w : SSphere 1,
      ∀ x ∈ ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support,
        0 < inner (𝕜 := ℝ) (x : EucSpace 1) ((w : EucSpace 1)) := by
    refine ⟨basePoint 0, fun x hx => ?_⟩
    have hxx : x = basePoint 0 := eq_of_mem_support_dirac hx
    have hx1 : ‖(x : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
    rw [← hxx, real_inner_self_eq_norm_mul_norm, hx1, mul_one]
    exact one_pos
  obtain ⟨-, hatom⟩ := h 1 (fun _ _ => 1) 0 (diracProb 1 (basePoint 0)) hhemi
    (fun _ => diracProb 1 (basePoint 0)) rfl
    (cauchyPB_const_diracProb 1 _ (basePoint 0) (fun _ => rfl) (fun _ => rfl))
  obtain ⟨z, T, -, hle⟩ := hatom (1 / 2) (by norm_num)
  norm_num at hle

/-- **`Winf` is a bound on the radius of a ball carrying the mass.**

`W_∞(μ, δ_z)` is `sup_{x ∈ supp μ} d(x, z)`; what the proof of
`prop: targets.atoms` uses is one inequality, that a ball of radius `r ≥ 0`
around `z` containing `supp μ` bounds it.  This is the property that pins the
parameter `Winf` down; `not_forall_clustering_to_atom` is what happens without
it.

Source: arXiv:2411.04551v3, §2, `W_∞`. -/
def IsWinfToDirac (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ) : Prop :=
  ∀ (μ : Measure (SSphere d)) (z : SSphere d) (r : ℝ), 0 ≤ r →
    (∀ x ∈ μ.support, dist x z ≤ r) → Winf μ (Measure.dirac z) ≤ r

/-- `IsWinfToDirac` is satisfiable, and it is a genuine constraint: the zero
function satisfies it, the constant `1` does not — a Dirac mass sits in the
ball of radius `0` around its own point. -/
example : IsWinfToDirac 1 (fun _ _ => 0) ∧ ¬ IsWinfToDirac 1 (fun _ _ => 1) := by
  refine ⟨fun _ _ _ hr _ => hr, fun h => ?_⟩
  have := h (Measure.dirac (basePoint 0)) (basePoint 0) 0 le_rfl
    (fun x hx => by rw [eq_of_mem_support_dirac hx, dist_self])
  norm_num at this

/-- **The second conclusion of `prop: targets.atoms` follows from the first.**

Once `diam(supp μ(t)) → 0`, pick a time `T > 0` where the diameter is below
`ε`.  The support of a probability measure is nonempty, so it holds some `z`,
and every point of the support is then within `ε` of `z` — which is exactly
what `IsWinfToDirac` turns into `W_∞(μ(T), δ_z) ≤ ε`.

Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem exists_dirac_close_of_diam_tendsto
    (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (hWinf : IsWinfToDirac d Winf) (μ : ℝ → ProbSphere d)
    (hdiam : Filter.Tendsto
        (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support))
        Filter.atTop (nhds 0)) :
    ∀ ε : ℝ, 0 < ε → ∃ (z : SSphere d) (T : ℝ), 0 < T ∧
      Winf (μ T : Measure (SSphere d)) (Measure.dirac z) ≤ ε := by
  intro ε hε
  have hev : ∀ᶠ t : ℝ in Filter.atTop,
      Metric.diam ((μ t : Measure (SSphere d)).support) < ε :=
    hdiam.eventually (gt_mem_nhds hε)
  obtain ⟨T, hTdiam, hTpos⟩ := (hev.and (Filter.eventually_gt_atTop (0 : ℝ))).exists
  have hne : ((μ T : Measure (SSphere d)).support).Nonempty := by
    rw [Set.nonempty_iff_ne_empty]
    intro hempty
    have h0 := MeasureTheory.Measure.measure_compl_support (μ := (μ T : Measure (SSphere d)))
    rw [hempty, Set.compl_empty, measure_univ] at h0
    exact one_ne_zero h0
  obtain ⟨z, hz⟩ := hne
  have : CompactSpace (SSphere d) := Metric.sphere.compactSpace (0 : EucSpace d) 1
  refine ⟨z, T, hTpos, hWinf _ z ε hε.le fun x hx => ?_⟩
  exact le_trans (Metric.dist_le_diam_of_mem Metric.isBounded_of_compactSpace hx hz) hTdiam.le

/-- **Proposition (prop: targets.atoms).**  *Clustering to a single point.*

If `𝐁 ∈ M_{d×d}(ℝ)` and `supp μ_0` is contained in an open hemisphere, then
the solution to `eq: cauchy.pb`–`eq: vf` with `(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)`
satisfies `diam(supp μ(t)) → 0` as `t → ∞`, and for any `ε > 0` there are
`z ∈ 𝕊^{d-1}` and `T > 0` with `W_∞(μ(T), δ_z) ≤ ε`.

Not proved here.  The second conclusion is not independent of the first:
`exists_dirac_close_of_diam_tendsto` derives it, so what is open is the
collapse of the diameter.

Source: arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem clustering_to_atom
    (Winf : Measure (SSphere d) → Measure (SSphere d) → ℝ)
    (hWinf : IsWinfToDirac d Winf)
    (B : ParamMatrix d) (μ₀ : ProbSphere d)
    (hhemi : ∃ w : SSphere d, ∀ x ∈ (μ₀ : Measure (SSphere d)).support,
      0 < inner (𝕜 := ℝ) (x : EucSpace d) ((w : EucSpace d))) :
    ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ →
      cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ →
        Filter.Tendsto
            (fun t : ℝ => Metric.diam ((μ t : Measure (SSphere d)).support))
            Filter.atTop (nhds 0) ∧
          ∀ ε : ℝ, 0 < ε → ∃ (z : SSphere d) (T : ℝ), 0 < T ∧
            Winf (μ T : Measure (SSphere d)) (Measure.dirac z) ≤ ε := by
  sorry

/-- The hypotheses of `clustering_to_atom` are satisfiable, and by more than a
contradiction: `Winf ≡ 0` is pinned down as `IsWinfToDirac` asks, and a Dirac
mass is supported at its own point, which lies in the open hemisphere around
itself. -/
example :
    IsWinfToDirac 1 (fun _ _ => 0) ∧
      ∃ w : SSphere 1,
        ∀ x ∈ ((diracProb 1 (basePoint 0) : ProbSphere 1) :
            Measure (SSphere 1)).support,
          0 < inner (𝕜 := ℝ) (x : EucSpace 1) ((w : EucSpace 1)) := by
  refine ⟨fun _ _ _ hr _ => hr, basePoint 0, fun x hx => ?_⟩
  have hxx : x = basePoint 0 := eq_of_mem_support_dirac hx
  have hx1 : ‖(x : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [← hxx, real_inner_self_eq_norm_mul_norm, hx1, mul_one]
  exact one_pos

end Interpolation
end Transformer

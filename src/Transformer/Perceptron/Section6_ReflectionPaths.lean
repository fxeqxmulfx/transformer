module

public import Transformer.Perceptron.Analytic
public import Transformer.Perspective.SphereInvariant
public import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
public import Mathlib.Analysis.Analytic.Linear
public import Mathlib.Analysis.Analytic.Constructions

/-!
# Actual orthogonal paths with analytic point orbits

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (a),
uses analytic uniqueness on the unit sphere. To apply the preceding
measure argument, we need actual orthogonal point maps depending
analytically on a real parameter, starting at the identity.

Compose reflection in the hyperplane perpendicular to `u` with
reflection perpendicular to `(1-t)u+tv`. The actual reflection formula
has the rational denominator `⟪(1-t)u+tv,(1-t)u+tv⟫`. If the affine normal
never vanishes, every original point orbit is analytic on the full real
line, and the original family is jointly continuous in time and point.

The maps are bundled linear isometries for every parameter. Their sphere
restrictions therefore preserve every law satisfying the source's actual
orthogonal-invariance predicate. No measure-preserving property is put
inside the definition, and no point or parameter is discarded.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The actual affine normal joining the two original normals.
Source: arXiv:2601.21366v2, §6, the spherical uniqueness argument. -/
noncomputable def reflectionNormal {d : ℕ} (u v : EucSpace d) (t : ℝ) : EucSpace d :=
  (1 - t) • u + t • v

/-- The actual orthogonal map obtained by the two hyperplane reflections.
Source: arXiv:2601.21366v2, §6, using orthogonal invariance for
analytic uniqueness on the sphere. Both original normals are retained. -/
noncomputable def reflectionPath {d : ℕ} (u v : EucSpace d) (t : ℝ) :
    EucSpace d ≃ₗᵢ[ℝ] EucSpace d :=
  ((ℝ ∙ u)ᗮ.reflection).trans ((ℝ ∙ reflectionNormal u v t)ᗮ.reflection)

/-- The actual family starts at the identity, since its two initial
hyperplane reflections agree. Source: arXiv:2601.21366v2, §6,
the identity-at-zero input of the spherical uniqueness argument. -/
theorem reflectionPath_zero_apply {d : ℕ} (u v z : EucSpace d) :
    reflectionPath u v 0 z = z := by
  simp only [reflectionPath, reflectionNormal, sub_zero, one_smul, zero_smul,
    add_zero, LinearIsometryEquiv.trans_apply]
  exact Submodule.reflection_involutive ((ℝ ∙ u)ᗮ) z

/-- The explicit actual point formula uses the original affine normal.
Source: arXiv:2601.21366v2, §6, the orthogonal point maps in
the spherical uniqueness argument. This is the standard Householder
formula, including its totalized value if the normal is zero. -/
theorem reflectionPath_apply {d : ℕ} (u v z : EucSpace d) (t : ℝ) :
    reflectionPath u v t z = (ℝ ∙ u)ᗮ.reflection z -
      (2 * (inner (𝕜 := ℝ) (reflectionNormal u v t) ((ℝ ∙ u)ᗮ.reflection z) /
        inner (𝕜 := ℝ) (reflectionNormal u v t) (reflectionNormal u v t))) •
        reflectionNormal u v t := by
  rw [reflectionPath, LinearIsometryEquiv.trans_apply,
    Submodule.reflection_orthogonal_apply, Submodule.reflection_singleton_apply,
    real_inner_self_eq_norm_sq, neg_sub, mul_smul]
  simp only [RCLike.ofReal_real_eq_id, id_eq, two_smul]

/-- The affine normal is an actual analytic curve.
Source: arXiv:2601.21366v2, §6, the analytic spherical paths. -/
theorem analyticOnNhd_reflectionNormal {d : ℕ} (u v : EucSpace d) :
    AnalyticOnNhd ℝ (reflectionNormal u v) univ := by
  intro t _
  exact ((analyticAt_const.sub analyticAt_id).smul analyticAt_const).add
    (analyticAt_id.smul analyticAt_const)

/-- Every actual point orbit of the two reflections is analytic when
the actual affine normal never vanishes. Source: arXiv:2601.21366v2,
§6, analytic uniqueness on the sphere in `prop: unified.log`.
The denominator's nonvanishing follows from its actual squared norm. -/
theorem analyticOnNhd_reflectionPath_orbit {d : ℕ} (u v : EucSpace d)
    (hne : ∀ t, reflectionNormal u v t ≠ 0) (z : EucSpace d) :
    AnalyticOnNhd ℝ (fun t => reflectionPath u v t z) univ := by
  intro t _
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have hp := analyticOnNhd_reflectionNormal u v t (mem_univ t)
  have hn : AnalyticAt ℝ (fun s => inner (𝕜 := ℝ) (reflectionNormal u v s)
      (reflectionNormal u v s)) t :=
    (L.analyticAt_bilinear (reflectionNormal u v t, reflectionNormal u v t)).comp
      (f := fun s => (reflectionNormal u v s, reflectionNormal u v s)) (hp.prod hp)
  have hc : AnalyticAt ℝ (fun s => inner (𝕜 := ℝ) (reflectionNormal u v s)
      ((ℝ ∙ u)ᗮ.reflection z)) t :=
    (L.analyticAt_bilinear (reflectionNormal u v t, (ℝ ∙ u)ᗮ.reflection z)).comp
      (f := fun s => (reflectionNormal u v s, (ℝ ∙ u)ᗮ.reflection z))
      (hp.prod analyticAt_const)
  have hden : inner (𝕜 := ℝ) (reflectionNormal u v t) (reflectionNormal u v t) ≠ 0 := by
    rw [real_inner_self_eq_norm_sq]
    exact pow_ne_zero 2 (norm_ne_zero_iff.mpr (hne t))
  have hquot := hc.fun_div hn hden
  simpa only [reflectionPath_apply] using
    (analyticAt_const.fun_sub ((analyticAt_const.fun_mul hquot).fun_smul hp))

/-- The actual reflection family is jointly continuous in time and
original point. Source: arXiv:2601.21366v2, §6, the spherical
uniqueness argument. Nonvanishing is required only for the actual normal;
no bound on time or original point is assumed. -/
theorem continuous_reflectionPath_apply {d : ℕ} (u v : EucSpace d)
    (hne : ∀ t, reflectionNormal u v t ≠ 0) :
    Continuous (fun p : ℝ × EucSpace d => reflectionPath u v p.1 p.2) := by
  have hp : Continuous (reflectionNormal u v) := (analyticOnNhd_reflectionNormal u v).continuous
  have hpf : Continuous (fun p : ℝ × EucSpace d => reflectionNormal u v p.1) :=
    hp.comp continuous_fst
  have hz : Continuous (fun p : ℝ × EucSpace d => (ℝ ∙ u)ᗮ.reflection p.2) :=
    ((ℝ ∙ u)ᗮ.reflection).continuous.comp continuous_snd
  have hden (p : ℝ × EucSpace d) :
      inner (𝕜 := ℝ) (reflectionNormal u v p.1) (reflectionNormal u v p.1) ≠ 0 := by
    rw [real_inner_self_eq_norm_sq]
    exact pow_ne_zero 2 (norm_ne_zero_iff.mpr (hne p.1))
  have hc := (hpf.inner hz).div (hpf.inner hpf) hden
  simpa only [reflectionPath_apply, Pi.div_apply] using
    hz.fun_sub ((continuous_const.fun_mul hc).fun_smul hpf)

/-- The same original family is jointly continuous on the actual sphere.
Source: arXiv:2601.21366v2, §6, passing the orthogonal maps to the
uniform spherical law in the positive-measure uniqueness argument. -/
theorem continuous_reflectionPath_sphere {d : ℕ} (u v : EucSpace d)
    (hne : ∀ t, reflectionNormal u v t ≠ 0) :
    Continuous (fun p : ℝ × SSphere d => Perspective.sphereMap d (reflectionPath u v p.1) p.2) := by
  have hc : Continuous (fun p : ℝ × SSphere d =>
      reflectionPath u v p.1 (p.2 : EucSpace d)) := by
    have hs : Continuous (fun p : ℝ × SSphere d => (p.1, (p.2 : EucSpace d))) :=
      continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd)
    simpa only [Function.comp_def] using
      (continuous_reflectionPath_apply u v hne).comp hs
  have hmem (p : ℝ × SSphere d) :
      reflectionPath u v p.1 (p.2 : EucSpace d) ∈ Metric.sphere (0 : EucSpace d) 1 := by
    rw [mem_sphere_zero_iff_norm, LinearIsometryEquiv.norm_map]
    exact norm_eq_of_mem_sphere p.2
  change Continuous (fun p : ℝ × SSphere d =>
    (⟨reflectionPath u v p.1 (p.2 : EucSpace d), hmem p⟩ : SSphere d))
  exact hc.subtype_mk hmem

/-- The nonvanishing-normal input is met by the two actual coordinate
normals in dimension two: the affine normal has coordinates `(1-t,t)`.
Both the ambient and sphere families consequently exist for all real times.
Source: arXiv:2601.21366v2, §6, the spherical uniqueness argument. -/
example : ∀ t : ℝ, reflectionNormal (EuclideanSpace.single (0 : Fin 2) 1)
    (EuclideanSpace.single (1 : Fin 2) 1) t ≠ 0 := by
  intro t ht
  have h0 := congrArg (fun z : EucSpace 2 => z 0) ht
  have h1 := congrArg (fun z : EucSpace 2 => z 1) ht
  simp only [reflectionNormal, PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply,
    PiLp.zero_apply, smul_eq_mul] at h0 h1
  norm_num at h0 h1
  linarith

end Transformer.Perceptron

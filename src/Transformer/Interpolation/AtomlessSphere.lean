/-
# Measure-to-measure interpolation — diffuse measures with separated hulls

The hypotheses of the clustering propositions of arXiv:2411.04551v3, §2, ask for initial measures
without atoms whose supports have disjoint geodesic convex hulls (`prop: compression`).  This module
meets them: the uniform law on the sphere, restricted to two opposite closed caps and normalized.
It is what makes the satisfiability examples of those propositions non-degenerate — an empty family
of initial measures satisfies every hypothesis of the kind vacuously.
-/

import Transformer.Causal.UniformAtoms
import Transformer.MeanField.UniformLaw
import Transformer.Interpolation.Basic

open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Perspective MeanField

/-- **A diffuse probability measure supported in a given closed set.**  The uniform law on the
sphere, restricted to a closed set `S` of positive mass and normalized, has no atoms and its
support lies in `S`. -/
theorem exists_diffuse_probSphere (d : ℕ) (hd : 2 ≤ d) {S : Set (SSphere d)} (hS : IsClosed S)
    (hpos : 0 < uniformLaw d S) :
    ∃ μ : ProbSphere d, (∀ y : SSphere d, (μ : Measure (SSphere d)) {y} = 0) ∧
      (μ : Measure (SSphere d)).support ⊆ S := by
  have hne : uniformLaw d S ≠ 0 := hpos.ne'
  have htop : uniformLaw d S ≠ ⊤ := by
    have := isProbabilityMeasure_uniformLaw d (by omega)
    exact measure_ne_top _ _
  have hprob : IsProbabilityMeasure ((uniformLaw d S)⁻¹ • (uniformLaw d).restrict S) :=
    ⟨by
      rw [Measure.smul_apply, Measure.restrict_apply_univ, smul_eq_mul,
        ENNReal.inv_mul_cancel hne htop]⟩
  refine ⟨⟨_, hprob⟩, fun y => ?_, ?_⟩
  · have h0 : uniformLaw d {y} = 0 := by
      rw [uniformLaw_apply]
      exact Causal.uniformSphere_singleton d hd y
    have hle : ((uniformLaw d).restrict S) {y} ≤ uniformLaw d {y} := Measure.restrict_le_self _
    have hle0 : ((uniformLaw d).restrict S) {y} = 0 := le_zero_iff.mp (h0 ▸ hle)
    show ((uniformLaw d S)⁻¹ • (uniformLaw d).restrict S) {y} = 0
    rw [Measure.smul_apply, smul_eq_mul, hle0, mul_zero]
  · refine Measure.support_subset_of_isClosed hS ?_
    have h : ∀ᵐ x ∂(uniformLaw d).restrict S, x ∈ S := ae_restrict_mem hS.measurableSet
    exact Measure.ae_smul_measure h _

/-- The hypotheses of `exists_diffuse_probSphere` are satisfiable: the whole circle is closed and
has positive uniform mass. -/
example : IsClosed (Set.univ : Set (SSphere 2)) ∧ 0 < uniformLaw 2 (Set.univ : Set (SSphere 2)) := by
  refine ⟨isClosed_univ, ?_⟩
  have := isProbabilityMeasure_uniformLaw 2 (by norm_num)
  rw [measure_univ]
  exact one_pos

/-- **The hull of a set inside a cap stays in the open half-space.**  If every point `y` of `A`
has `⟨y, w⟩ ≥ 3/4`, then every point of the geodesic convex hull `conv_g A` has `⟨x, w⟩ > 0`: the
Euclidean convex hull stays in the closed half-space `⟨·, w⟩ ≥ 3/4`, which excludes `0`, and the
radial projection multiplies by a positive number. -/
theorem convG_subset_pos (d : ℕ) (w : EucSpace d) (A : Set (SSphere d))
    (hA : ∀ y ∈ A, (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) (y : EucSpace d) w) :
    ∀ x ∈ convG d A, 0 < inner (𝕜 := ℝ) (x : EucSpace d) w := by
  have hconv : Convex ℝ {v : EucSpace d | (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) v w} :=
    convex_halfSpace_ge ⟨fun a b => inner_add_left a b w, fun c a => real_inner_smul_left a w c⟩ _
  have hsub : ((↑) '' A : Set (EucSpace d)) ⊆ {v : EucSpace d | (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) v w} := by
    rintro _ ⟨y, hy, rfl⟩
    exact hA y hy
  have hhull := convexHull_min hsub hconv
  rintro x (h0 | ⟨v, hv, c, hc, hx⟩)
  · have := hhull h0
    simp only [Set.mem_ofPred_eq, inner_zero_left] at this
    linarith
  · have := hhull hv
    simp only [Set.mem_ofPred_eq] at this
    rw [hx, real_inner_smul_left]
    exact mul_pos hc (by linarith)

/-- The hypotheses of `convG_subset_pos` are satisfiable: `A = ∅` in the circle. -/
example : ∀ y ∈ (∅ : Set (SSphere 2)),
    (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) (y : EucSpace 2) (EuclideanSpace.single 0 1) :=
  fun _ h => h.elim

/-- **Two diffuse measures with separated hulls.**  In every dimension `d ≥ 2` there are two
probability measures on `𝕊^{d-1}` without atoms whose supports have disjoint geodesic convex
hulls: the uniform law restricted to the caps `⟨·, u⟩ ≥ 3/4` and `⟨·, -u⟩ ≥ 3/4`, normalized. -/
theorem exists_two_diffuse_separated (d : ℕ) (hd : 2 ≤ d) :
    ∃ μ : Fin 2 → ProbSphere d,
      (∀ i : Fin 2, ∀ y : SSphere d, (μ i : Measure (SSphere d)) {y} = 0) ∧
      ∀ i j : Fin 2, i ≠ j →
        Disjoint (convG d (μ i : Measure (SSphere d)).support)
          (convG d (μ j : Measure (SSphere d)).support) := by
  set u : EucSpace d := EuclideanSpace.single ⟨0, by omega⟩ 1 with hu
  have hu1 : ‖u‖ = 1 := by simp [hu]
  set w : Fin 2 → EucSpace d := ![u, -u] with hw
  have hw1 : ∀ i, ‖w i‖ = 1 := by
    intro i
    fin_cases i <;> simp [hw, hu1]
  have hclosed : ∀ i, IsClosed {y : SSphere d | (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) (y : EucSpace d) (w i)} :=
    fun i => isClosed_le continuous_const (continuous_subtype_val.inner continuous_const)
  have hposmass : ∀ i, 0 < uniformLaw d {y : SSphere d | (3 / 4 : ℝ) ≤
      inner (𝕜 := ℝ) (y : EucSpace d) (w i)} := by
    intro i
    have hopen : IsOpen {y : SSphere d | (3 / 4 : ℝ) < inner (𝕜 := ℝ) (y : EucSpace d) (w i)} :=
      isOpen_lt continuous_const (continuous_subtype_val.inner continuous_const)
    have hmem : (⟨w i, mem_sphere_zero_iff_norm.mpr (hw1 i)⟩ : SSphere d) ∈
        {y : SSphere d | (3 / 4 : ℝ) < inner (𝕜 := ℝ) (y : EucSpace d) (w i)} := by
      show (3 / 4 : ℝ) < inner (𝕜 := ℝ) (w i) (w i)
      rw [real_inner_self_eq_norm_mul_norm, hw1 i]
      norm_num
    have hpos := Causal.uniformSphere_pos_of_isOpen d hopen ⟨_, hmem⟩
    rw [uniformLaw_apply]
    refine lt_of_lt_of_le hpos (measure_mono fun y (hy : (3 / 4 : ℝ) < _) => hy.le)
  choose μ hatom hsupp using fun i =>
    exists_diffuse_probSphere d hd (hclosed i) (hposmass i)
  refine ⟨μ, hatom, fun i j hij => ?_⟩
  have hpos : ∀ i, ∀ x ∈ convG d (μ i : Measure (SSphere d)).support,
      0 < inner (𝕜 := ℝ) (x : EucSpace d) (w i) := fun i =>
    convG_subset_pos d (w i) _ fun y hy => hsupp i hy
  rw [Set.disjoint_left]
  intro x hxi hxj
  have h1 := hpos i x hxi
  have h2 := hpos j x hxj
  fin_cases i <;> fin_cases j
  · exact hij rfl
  · simp only [hw, Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one,
      inner_neg_right] at h1 h2
    linarith
  · simp only [hw, Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one,
      inner_neg_right] at h1 h2
    linarith
  · exact hij rfl

/-- The hypothesis of `exists_two_diffuse_separated` is satisfiable: the circle, `d = 2`. -/
example : 2 ≤ 2 := le_rfl

end Interpolation
end Transformer

/-
# §6.1 — No smooth invariant measure

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The remark following `thm: boumal`: where almost every initial sequence
clusters, no measure with a density with respect to the uniform law can be
invariant under the flow.  The two inputs the survey uses without proof —
the clustering statement itself and the existence of solutions — are carried
as explicit hypotheses, so what is proved here is the implication, with its
dependence legible in the signature.

The measure-theoretic input is `Perspective.UniformAtomless`: the uniform law
charges neither a single configuration nor the diagonal.
-/

import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.UniformAtomless
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- *Invariant measures remark.* When `thm: beta.tiny` applies (e.g. always
for `d ≥ 3`), neither `SA` nor `USA` admits a smooth invariant measure.

"Smooth" is read as *having a density*: `μ ≪ P`, with `P` the uniform law
`UniformTuple` of §4.  That already rules out the Dirac masses on consensus
configurations, which *are* invariant but sit on a `P`-null set, and it is what
makes the remark a consequence of `boumal_clustering`: everything is swept into
that null set, so no measure with a density can be preserved.  The flow is
presented as any map `Φ` that transports initial data along solutions of `SA`.

**What the source says and what is changed here.**  Two inputs the survey uses
without proof are carried as explicit hypotheses rather than invoked, so that
the remark is proved outright and its dependence is legible in its signature:

* `h_clust` — the conclusion of `boumal_clustering`, which is a `sorry`;
* `h_exist` — existence of an `SA` solution through every initial sequence,
  which this development does not build: `Section2_FlowMap` states the flow
  map, it does not construct it.  Without `h_exist` the hypothesis on `Φ` is
  empty of content, since it constrains `Φ` only along initial data a solution
  already passes through, and the remark is not provable at all.

The source's `0 ≤ β` is dropped: with `h_clust` carried, no step of the proof
below uses the sign of `β`.

The proof.  Almost every orbit clusters, so `‖x_0(t) - x_1(t)‖ → 0` along it;
that function is bounded by `2`, so invariance of `μ` and dominated
convergence force `∫ ‖x_0 - x_1‖ dμ = 0`, i.e. `μ` lives on the diagonal
`{X | X_0 = X_1}`.  That set is `P`-null by `measure_coords_eq_eq_zero`, hence
`μ`-null, so `μ` is the zero measure — and it was a probability measure.

Source: arXiv:2312.10794v5, §6.1 (remark after `thm: boumal`). -/
theorem no_smooth_invariant_measure (β : ℝ) (hd : 3 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
    (∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β) →
    (∀ X₀ : SphereTuple d n, ∃ X : ℝ → SphereTuple d n,
        X 0 = X₀ ∧ Perspective.SA d n β X) →
    ∀ μ : Measure (SphereTuple d n), IsProbabilityMeasure μ → μ ≪ P →
      ¬ ∃ Φ : ℝ → SphereTuple d n → SphereTuple d n,
          (∀ (X : ℝ → SphereTuple d n), Perspective.SA d n β X →
              ∀ t : ℝ, Φ t (X 0) = X t) ∧
          ∀ t : ℝ, μ.map (Φ t) = μ := by
  intro P hP h_clust h_exist μ hμ hμP
  rintro ⟨Φ, hΦ1, hΦ2⟩
  have hd2 : 2 ≤ d := by omega
  set i₀ : Idx n := ⟨0, by omega⟩ with hi₀
  set j₀ : Idx n := ⟨1, by omega⟩ with hj₀
  have hij : i₀ ≠ j₀ := by simp [hi₀, hj₀, Fin.ext_iff]
  set F : SphereTuple d n → ℝ :=
    fun X => ‖(X i₀ : EucSpace d) - (X j₀ : EucSpace d)‖ with hFdef
  have hFcont : Continuous F :=
    ((continuous_apply i₀).subtype_val.sub (continuous_apply j₀).subtype_val).norm
  have hFmeas : Measurable F := hFcont.measurable
  have hFnonneg : ∀ X : SphereTuple d n, 0 ≤ F X := fun _ => norm_nonneg _
  have hFle : ∀ X : SphereTuple d n, ‖F X‖ ≤ 2 := by
    intro X
    rw [Real.norm_eq_abs, abs_of_nonneg (hFnonneg X)]
    refine (norm_sub_le _ _).trans ?_
    rw [mem_sphere_zero_iff_norm.mp (X i₀).2, mem_sphere_zero_iff_norm.mp (X j₀).2]
    norm_num
  have hμ0 : μ ≠ 0 := by
    intro h
    have h1 : (1 : ENNReal) = 0 := by rw [← measure_univ (μ := μ), h]; simp
    exact one_ne_zero h1
  -- `μ` is invariant under `Φ t`, hence `Φ t` is almost everywhere measurable:
  -- a non-measurable map would push `μ` forward to a Dirac mass, which is not
  -- absolutely continuous with respect to the atomless `P`.
  have haem : ∀ t : ℝ, AEMeasurable (Φ t) μ := by
    intro t
    by_contra hcon
    have hdir := Measure.map_of_not_aemeasurable_of_ne_zero hcon hμ0
    rw [hΦ2 t] at hdir
    obtain ⟨c, hc⟩ : ∃ c : SphereTuple d n, μ = Measure.dirac c := ⟨_, hdir⟩
    have hPc : μ {c} = 0 :=
      hμP (measure_singleton_tuple_eq_zero d n hd2 (by omega) P hP c)
    rw [hc, Measure.dirac_apply_of_mem (Set.mem_singleton c)] at hPc
    exact one_ne_zero hPc
  -- Almost every orbit clusters, so its first two particles come together.
  have hae : ∀ᵐ X₀ ∂μ, Filter.Tendsto (fun k : ℕ => F (Φ (k : ℝ) X₀))
      Filter.atTop (nhds 0) := by
    refine (h_clust.filter_mono hμP.ae_le).mono ?_
    rintro X₀ ⟨x_star, hstar⟩
    obtain ⟨X, hX0, hXsa⟩ := h_exist X₀
    have horb : ∀ t : ℝ, Φ t X₀ = X t := fun t => hX0 ▸ hΦ1 X hXsa t
    have hsub : Filter.Tendsto (fun t : ℝ =>
        ((X t i₀ : EucSpace d) - (x_star : EucSpace d))
          - ((X t j₀ : EucSpace d) - (x_star : EucSpace d)))
        Filter.atTop (nhds 0) := by
      simpa using (hstar X hX0 hXsa i₀).sub (hstar X hX0 hXsa j₀)
    have hdiff : Filter.Tendsto (fun t : ℝ => F (X t)) Filter.atTop (nhds 0) := by
      simpa [hFdef, sub_sub_sub_cancel_right] using hsub.norm
    simpa [horb, Function.comp_def] using hdiff.comp tendsto_natCast_atTop_atTop
  -- Invariance makes the integral of `F` constant along the orbit.
  have hint : ∀ k : ℕ, ∫ X, F (Φ (k : ℝ) X) ∂μ = ∫ X, F X ∂μ := fun k => by
    rw [← integral_map (haem (k : ℝ)) hFmeas.aestronglyMeasurable, hΦ2]
  have hFint : Integrable F μ :=
    (integrable_const (2 : ℝ)).mono' hFmeas.aestronglyMeasurable
      (Filter.Eventually.of_forall hFle)
  have hlim : Filter.Tendsto (fun k : ℕ => ∫ X, F (Φ (k : ℝ) X) ∂μ)
      Filter.atTop (nhds (∫ _X : SphereTuple d n, (0 : ℝ) ∂μ)) :=
    tendsto_integral_of_dominated_convergence (fun _ => (2 : ℝ))
      (fun k => (hFmeas.comp_aemeasurable (haem (k : ℝ))).aestronglyMeasurable)
      (integrable_const _)
      (fun k => Filter.Eventually.of_forall (fun X => hFle (Φ (k : ℝ) X))) hae
  simp only [hint, integral_zero] at hlim
  have hzero : ∫ X, F X ∂μ = 0 := tendsto_const_nhds_iff.mp hlim
  -- So `μ` lives on the diagonal, which is `P`-null.
  have hFae : F =ᵐ[μ] 0 := (integral_eq_zero_iff_of_nonneg hFnonneg hFint).mp hzero
  have hcompl : μ {X : SphereTuple d n | ¬ X i₀ = X j₀} = 0 := by
    rw [← ae_iff]
    filter_upwards [hFae] with X hX
    exact Subtype.ext (sub_eq_zero.mp (norm_eq_zero.mp hX))
  have hnull : μ {X : SphereTuple d n | X i₀ = X j₀} = 0 :=
    hμP (measure_coords_eq_eq_zero d n hd2 P hP hij)
  have hcov : (Set.univ : Set (SphereTuple d n))
      ⊆ {X : SphereTuple d n | X i₀ = X j₀} ∪ {X : SphereTuple d n | ¬ X i₀ = X j₀} :=
    fun X _ => em _
  have huniv : μ (Set.univ : Set (SphereTuple d n)) = 0 :=
    nonpos_iff_eq_zero.mp ((measure_mono hcov).trans_eq (measure_union_null hnull hcompl))
  rw [measure_univ] at huniv
  exact one_ne_zero huniv

/-- The hypotheses of `no_smooth_invariant_measure` are satisfiable: `d = 3`,
`n = 2`; the two carried hypotheses sit inside the quantifier over the uniform
law, which nothing has to exhibit. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

end Perspective
end Transformer

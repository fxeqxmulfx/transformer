/-
# Generic parameter sets for a fixed stationary measure

In `thm: any.d` (ii), §3.1 of arXiv:2601.21366v2, the probability measure
`μ` is fixed before the open dense parameter set `U_μ` is chosen. For these
quantifiers a direct argument suffices: pick one support point and require
its stationarity field to be nonzero. The resulting set is open and dense.

Density follows from the identity principle along lines in the neuron
parameters. The field cannot vanish identically: an input `x+v`, with `v`
orthogonal to `x`, produces a nonzero tangential drift. This proves the
stated conclusion for each fixed `μ`; it does not assert that a common
parameter set works for all measures, or that every field has finitely many zeros.
-/

import Transformer.Perceptron.Hyperplane
import Mathlib.Analysis.Analytic.Uniqueness
import Mathlib.MeasureTheory.Integral.Bochner.Set

open Real MeasureTheory Filter Topology
open scoped BigOperators

namespace Transformer.Perceptron

variable {d : ℕ}

/-- The perceptron parameters `(ω_j,a_j)_j` of arXiv:2601.21366v2, §3.1,
`thm: any.d` (ii). -/
abbrev NeuronParams (d : ℕ) := (Idx d → ℝ) × (Idx d → EucSpace d)

/-- The full parameter space of arXiv:2601.21366v2, §3.1,
`thm: any.d` (ii): `β` and the `d` neurons. Positivity is imposed on sets. -/
abbrev Params (d : ℕ) := ℝ × NeuronParams d

/-- The stationarity field is continuous in all parameters;
arXiv:2601.21366v2, §3.1, proof of `thm: any.d` (ii).
The compact sphere permits integration of the continuous attention kernel. -/
theorem continuous_energyGrad_params (σ : ℝ → ℝ) (hσ : Continuous σ)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    Continuous (fun p : Params d => energyGrad p.1 σ p.2.1 p.2.2 μ x) := by
  have hi : Continuous (fun β : ℝ => ∫ y, Real.exp (β * inner (𝕜 := ℝ) x
      (y : EucSpace d)) • proj d x (y : EucSpace d) ∂(μ : Measure (SSphere d))) := by
    have hc : Continuous (fun p : ℝ × SSphere d => Real.exp
        (p.1 * inner (𝕜 := ℝ) x (p.2 : EucSpace d)) • proj d x (p.2 : EucSpace d)) := by
      unfold proj
      fun_prop
    simpa only [MeasureTheory.setIntegral_univ] using
      (continuous_parametric_integral_of_continuous (μ := (μ : Measure (SSphere d)))
        (f := fun β (y : SSphere d) =>
        Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d))
        hc isCompact_univ)
  have hd : Continuous (fun p : Params d => drift σ p.2.1 p.2.2 x) := by
    unfold drift proj
    fun_prop
  exact (hi.comp continuous_fst).add hd

/-- The continuity hypothesis is satisfied by the identity activation;
arXiv:2601.21366v2, §3.1, `thm: any.d` (ii). -/
example : Continuous (id : ℝ → ℝ) := continuous_id

/-- With `β` fixed, the field is analytic along every line in neuron parameters;
arXiv:2601.21366v2, §3.1, proof of `thm: any.d` (ii).
The attention term is constant along this line. -/
theorem analytic_energyGrad_params_line (β : ℝ) (σ : ℝ → ℝ)
    (hσ : AnalyticOnNhd ℝ σ Set.univ) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) (p q : NeuronParams d) :
    AnalyticOnNhd ℝ (fun t : ℝ => energyGrad β σ
      (fun j => p.1 j + t * (q.1 j - p.1 j))
      (fun j => p.2 j + t • (q.2 j - p.2 j)) μ x) Set.univ := by
  intro t₀ _
  have hterm (j : Idx d) : AnalyticAt ℝ (fun t : ℝ =>
      ((p.1 j + t * (q.1 j - p.1 j)) *
        σ (inner (𝕜 := ℝ) (p.2 j + t • (q.2 j - p.2 j)) x)) •
        (p.2 j + t • (q.2 j - p.2 j))) t₀ := by
    have ha : AnalyticAt ℝ (fun t : ℝ => p.2 j + t • (q.2 j - p.2 j)) t₀ :=
      analyticAt_const.add (analyticAt_id.smul analyticAt_const)
    have hi : AnalyticAt ℝ (fun t : ℝ => inner (𝕜 := ℝ)
        (p.2 j + t • (q.2 j - p.2 j)) x) t₀ := by
      simpa only [Function.comp_def, innerSL_apply_apply, real_inner_comm] using
        ((innerSL ℝ x).analyticAt _).comp ha
    have hs := (hσ _ (Set.mem_univ _)).comp hi
    have hw : AnalyticAt ℝ (fun t : ℝ => p.1 j + t * (q.1 j - p.1 j)) t₀ :=
      analyticAt_const.add (analyticAt_id.mul analyticAt_const)
    exact (hw.mul hs).smul ha
  have hsum := Finset.univ.analyticAt_fun_sum (fun j _ => hterm j)
  have hi := ((innerSL ℝ x).analyticAt _).comp hsum
  unfold energyGrad drift proj
  exact analyticAt_const.add (hsum.sub (hi.smul analyticAt_const))

/-- The analytic hypothesis is satisfied by the identity activation;
arXiv:2601.21366v2, §3.1, `thm: any.d` (ii). -/
example : AnalyticOnNhd ℝ (id : ℝ → ℝ) Set.univ := analyticOnNhd_id

/-- A neuron can produce nonzero total drift at any sphere point in `d ≥ 2`;
arXiv:2601.21366v2, §3.1, the parameter perturbations in `thm: any.d` (ii).
Its input `x+v`, with `v` unit and orthogonal to `x`, has tangential part `v`. -/
theorem exists_energyGrad_params_ne_zero (hd : 2 ≤ d) (β : ℝ)
    (σ : ℝ → ℝ) (hσ1 : σ 1 ≠ 0) (μ : Perspective.ProbSphere d)
    (x : SSphere d) : ∃ q : NeuronParams d, energyGrad β σ q.1 q.2 μ x ≠ 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hx0 : (x : EucSpace d) ≠ 0 := norm_ne_zero_iff.mp (by rw [hx]; norm_num)
  obtain ⟨v, hv, hxv⟩ := exists_unit_inner_eq_zero hd hx0
  have hv0 : v ≠ 0 := norm_ne_zero_iff.mp (by rw [hv]; norm_num)
  let j : Idx d := ⟨0, by omega⟩
  by_cases h0 : energyGrad β σ 0 0 μ x = 0
  · refine ⟨⟨Pi.single j 1, fun _ => (x : EucSpace d) + v⟩, ?_⟩
    have hvx : inner (𝕜 := ℝ) v (x : EucSpace d) = 0 := by
      rw [real_inner_comm]
      exact hxv
    have hinner : inner (𝕜 := ℝ) ((x : EucSpace d) + v) (x : EucSpace d) = 1 := by
      rw [inner_add_left, real_inner_self_eq_norm_sq, hx, hvx]
      norm_num
    have hdrift : drift σ (Pi.single j 1) (fun _ => (x : EucSpace d) + v) x = σ 1 • v := by
      have hs : (∑ k : Idx d, ((Pi.single j 1 : Idx d → ℝ) k *
          σ (inner (𝕜 := ℝ) ((x : EucSpace d) + v) (x : EucSpace d))) •
          ((x : EucSpace d) + v)) = σ 1 • ((x : EucSpace d) + v) := by
        simp only [Pi.single_apply, hinner, ite_mul, one_mul, zero_mul,
          ite_smul, zero_smul]
        rw [Fintype.sum_ite_eq']
      rw [drift, hs, proj, inner_smul_right, inner_add_right,
        real_inner_self_eq_norm_sq, hx, hxv]
      simp [smul_add]
    have hi : (∫ y, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) •
        proj d (x : EucSpace d) (y : EucSpace d) ∂(μ : Measure (SSphere d))) = 0 := by
      simpa only [energyGrad, drift_zero, add_zero] using h0
    rw [energyGrad, hi, zero_add, hdrift]
    exact smul_ne_zero hσ1 hv0
  · exact ⟨⟨0, 0⟩, h0⟩

/-- Parameters with nonzero field at a fixed sphere point are dense at `β > 0`;
arXiv:2601.21366v2, §3.1, `thm: any.d` (ii).
Vanishing in an open neighborhood would extend along an analytic line to
a parameter with explicitly nonzero drift. -/
theorem mem_closure_nonzero_energyGrad_params (hd : 2 ≤ d)
    (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ Set.univ) (hσ1 : σ 1 ≠ 0)
    (μ : Perspective.ProbSphere d) (x : SSphere d) (p : Params d)
    (hβ : 0 < p.1) :
    p ∈ closure {q : Params d | 0 < q.1 ∧ energyGrad q.1 σ q.2.1 q.2.2 μ x ≠ 0} := by
  obtain ⟨q, hq⟩ := exists_energyGrad_params_ne_zero hd p.1 σ hσ1 μ x
  let path : ℝ → Params d := fun t => (p.1,
    (fun j => p.2.1 j + t * (q.1 j - p.2.1 j)),
    (fun j => p.2.2 j + t • (q.2 j - p.2.2 j)))
  have hc : Continuous path := by dsimp [path]; fun_prop
  have hzero : path 0 = p := by simp [path]
  have hone : path 1 = (p.1, q) := by simp [path]
  apply _root_.mem_closure_iff.mpr
  intro O hO hpO
  by_contra hn
  have hnear : ∀ᶠ t in 𝓝 (0 : ℝ), path t ∈ O := by
    have h := (hc.tendsto 0).eventually (hO.mem_nhds (hzero.symm ▸ hpO))
    exact h
  have heq : (fun t => energyGrad (path t).1 σ (path t).2.1 (path t).2.2 μ x) =ᶠ[𝓝 0]
      (fun _ => (0 : EucSpace d)) := by
    filter_upwards [hnear] with t ht
    by_contra hne
    exact hn ⟨path t, ht, hβ, hne⟩
  have ha := analytic_energyGrad_params_line p.1 σ hσ μ x p.2 q
  have hident := ha.eq_of_eventuallyEq analyticOnNhd_const heq
  have hlast : energyGrad p.1 σ q.1 q.2 μ x = 0 := by
    have h := congrFun hident 1
    change energyGrad (path 1).1 σ (path 1).2.1 (path 1).2.2 μ x = 0 at h
    simpa only [hone] using h
  exact hq hlast

/-- For a fixed probability measure, nonstationary parameters contain an open
set dense in `{β > 0}`; arXiv:2601.21366v2, §3.1, `thm: any.d` (ii).
A support point exists since the measure has total mass one. -/
theorem exists_open_dense_nonstationary_params (hd : 2 ≤ d)
    (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ Set.univ) (hσ1 : σ 1 ≠ 0)
    (μ : Perspective.ProbSphere d) :
    ∃ U : Set (Params d), IsOpen U ∧ U ⊆ {p | 0 < p.1} ∧
      {p | 0 < p.1} ⊆ closure U ∧
      ∀ p ∈ U, ¬ IsStationary p.1 σ p.2.1 p.2.2 μ := by
  obtain ⟨x, hx⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero
    (μ : Measure (SSphere d)))
  let U : Set (Params d) :=
    {p | 0 < p.1 ∧ energyGrad p.1 σ p.2.1 p.2.2 μ x ≠ 0}
  have hfield := continuous_energyGrad_params σ hσ.continuous μ x
  refine ⟨U, (isOpen_lt continuous_const continuous_fst).inter
    (isOpen_ne.preimage hfield), fun _ hp => hp.1, ?_, ?_⟩
  · intro p hp
    exact mem_closure_nonzero_energyGrad_params hd σ hσ hσ1 μ x p hp
  · intro p hp hs
    exact hp.2 (hs x hx)

/-- The dimension, activation and positive-temperature hypotheses of the
three parameter-witness and density results hold at `d = 2`, `σ = id`, `β = 1`;
arXiv:2601.21366v2, §3.1, `thm: any.d` (ii). -/
example : (2 : ℕ) ≤ 2 ∧ AnalyticOnNhd ℝ (id : ℝ → ℝ) Set.univ ∧
    id (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, analyticOnNhd_id, one_ne_zero, one_pos⟩

end Transformer.Perceptron

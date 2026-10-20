module

public import Transformer.Perspective.Section3_ClusteredSets
public import Transformer.Perspective.AppendixA_Saddle
public import Mathlib.MeasureTheory.Integral.Indicator

/-!
# §4 — Probability of clustering at small temperature

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §4, `th:beta_small`,
`eq: harry.potter` and `e:Ps0n`.

The events `Sset0 m` are open: uniqueness identifies the existentially
specified zero-temperature trajectory with `saFlow 0`, whose time maps are
continuous. Almost-sure zero-temperature consensus implies that almost every
start belongs to all sufficiently late events. Their probabilities therefore
tend to one. The finite-time perturbation estimate and hemisphere collapse
give `Sset0 m ⊆ clusteringSet β` for sufficiently small positive `β`.
These inclusions prove the original right-sided probability limit.

The source discusses both SA and USA and arbitrary query/key matrices. The
statements retained here concern SA with identity query/key matrices, exactly
as in their original `clusteringSet` definition. No measurability assumption
on `clusteringSet` is needed: its outer measure is squeezed between measurable
events of probability tending to one and the probability of the whole space.
-/

@[expose] public section

open Filter Set MeasureTheory
open scoped Topology ENNReal

namespace Transformer.Perspective

variable {d n : ℕ}

/-- A uniform initialization law is a probability measure, since each of its
independent sphere marginals is one. Source: arXiv:2312.10794v5, §3.2 and §4. -/
theorem isProbabilityMeasure_uniformTuple {P : Measure (SphereTuple d n)}
    (hP : UniformTuple d n P) : IsProbabilityMeasure P := by
  obtain ⟨σ, hσ, -, rfl⟩ := hP
  let := hσ
  infer_instance

example : UniformTuple 2 2 (Measure.pi fun _ : Idx 2 => shellLaw (d := 2) (by decide)) :=
  uniformTuple_shellLaw (by decide)

/-- The event in `eq: harry.potter` uses the actual, unique zero-temperature
solution, rather than a separately assumed trajectory.
Source: arXiv:2312.10794v5, §4, definition of `𝒮_0^m`. -/
theorem mem_Sset0_iff (m : ℕ) (X₀ : SphereTuple d n) :
    X₀ ∈ Sset0 d n m ↔ alphaClustered d n (3 / 4 : ℝ) (saFlow 0 X₀ m) := by
  constructor
  · rintro ⟨X, h0, hX, hcluster⟩
    have hflow := saFlow_eq_of_SA 0 X (SA_zero_of_beta0Dynamics hX)
    rw [h0] at hflow
    rwa [hflow]
  · intro hcluster
    exact ⟨saFlow 0 X₀, saFlow_zero 0 X₀, beta0Dynamics_saFlow X₀, hcluster⟩

/-- The sets `𝒮_0^m` are open: they are finite intersections of strict
inner-product inequalities along a continuous time map of the actual flow.
Source: arXiv:2312.10794v5, §4, immediately before `e:Ps0n`.
The proof also allows `m = 0`, while the source starts at `m = 1`. -/
theorem isOpen_Sset0 (hn : 0 < n) (m : ℕ) : IsOpen (Sset0 d n m) := by
  have hflow := continuous_saFlow (d := d) 0 le_rfl hn (m : ℝ) (Nat.cast_nonneg m)
  have htoken (i : Idx n) :
      Continuous (fun X₀ : SphereTuple d n => (saFlow 0 X₀ m i : EucSpace d)) :=
    continuous_subtype_val.comp ((continuous_apply i).comp hflow)
  have hset : Sset0 d n m = {X₀ | ∀ i j : Idx n, (3 / 4 : ℝ) <
      inner (𝕜 := ℝ) (saFlow 0 X₀ m i : EucSpace d) (saFlow 0 X₀ m j : EucSpace d)} := by
    ext X₀
    exact mem_Sset0_iff m X₀
  rw [hset]
  simp_rw [ofPred_forall]
  exact isOpen_iInter_of_finite fun i => isOpen_iInter_of_finite fun j =>
    isOpen_lt continuous_const ((htoken i).inner (htoken j))

example : (0 : ℕ) < 2 := by decide

/-- The finite-time events of `e:Ps0n` are measurable for the Borel structure
on sphere tuples. Source: arXiv:2312.10794v5, §4, openness of `𝒮_0^m`. -/
theorem measurableSet_Sset0 (hn : 0 < n) (m : ℕ) : MeasurableSet (Sset0 d n m) :=
  (isOpen_Sset0 hn m).measurableSet

example : (0 : ℕ) < 3 := by decide

variable (d n)

/-- **Equation (e:Ps0n).** `ℙ(𝒮_0^m) → 1` as `m → ∞`.

Almost-sure consensus makes the indicators of these measurable finite-time
events converge almost everywhere to one. Convergence of their probabilities
does not require proving the additional monotonicity asserted in the source.

Source: arXiv:2312.10794v5, §4, `e:Ps0n`. -/
theorem sset0_probability_tendsto_one (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      Filter.Tendsto (fun m : ℕ => (P (Sset0 d n m)).toReal)
        Filter.atTop (nhds 1) := by
  intro P hP
  let := isProbabilityMeasure_uniformTuple hP
  have hevent : ∀ᵐ X₀ ∂P, ∀ᶠ m : ℕ in atTop, X₀ ∈ Sset0 d n m ↔ X₀ ∈ univ := by
    filter_upwards [beta0_consensus d n hd P hP] with X₀ hX₀
    filter_upwards [eventually_mem_Sset0 hX₀] with m hm
    exact iff_of_true hm (mem_univ _)
  have hlim := tendsto_measure_of_ae_tendsto_indicator_of_isFiniteMeasure
    atTop MeasurableSet.univ (fun m => measurableSet_Sset0 (by omega : 0 < n) m) hevent
  have hreal := (ENNReal.tendsto_toReal (measure_ne_top P univ)).comp hlim
  simpa [Function.comp_def] using hreal

example : 2 ≤ 2 ∧ 2 ≤ 2 ∧
    UniformTuple 2 2 (Measure.pi fun _ : Idx 2 => shellLaw (d := 2) (by decide)) :=
  ⟨le_rfl, le_rfl, uniformTuple_shellLaw (by decide)⟩

/-- **Theorem (th:beta_small).** For fixed `d, n ≥ 2` and uniform
initialization, `ℙ(𝒮_β) → 1` as `β → 0⁺`.

For each `a < 1`, `e:Ps0n` supplies a measurable event `𝒮_0^m` of probability
greater than `a`. The source's finite-time perturbation bound puts that event
inside `𝒮_β` for all sufficiently small positive `β`. Monotonicity of outer
measure supplies the lower bound; total probability supplies the upper one.

Source: arXiv:2312.10794v5, §4, `th:beta_small` and its proof. -/
theorem clustering_probability_small_beta (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      Filter.Tendsto (fun β : ℝ => (P (clusteringSet d n β)).toReal)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  intro P hP
  let := isProbabilityMeasure_uniformTuple hP
  have hupper (β : ℝ) : (P (clusteringSet d n β)).toReal ≤ 1 := by
    have hle : P (clusteringSet d n β) ≤ 1 := by
      simpa using measure_mono (subset_univ (clusteringSet d n β)) (μ := P)
    simpa using ENNReal.toReal_mono (by simp : (1 : ℝ≥0∞) ≠ ∞) hle
  refine tendsto_order.mpr ⟨?_, ?_⟩
  · intro a ha
    obtain ⟨m, hm⟩ := ((sset0_probability_tendsto_one d n hd hn P hP).eventually_const_lt ha).exists
    obtain ⟨βm, hβm, hsubset⟩ := exists_Sset0_subset_clusteringSet (d := d) (by omega : 0 < n) m
    have hnear : ∀ᶠ β : ℝ in nhdsWithin 0 (Ioi 0), β < βm :=
      (eventually_lt_nhds hβm).filter_mono nhdsWithin_le_nhds
    filter_upwards [self_mem_nhdsWithin, hnear] with β hβ hβm'
    exact hm.trans_le (ENNReal.toReal_mono (measure_ne_top P _)
      (measure_mono (hsubset β hβ.le hβm'.le)))
  · intro b hb
    exact Eventually.of_forall fun β => (hupper β).trans_lt hb

example : 2 ≤ 2 ∧ 2 ≤ 2 ∧
    UniformTuple 2 2 (Measure.pi fun _ : Idx 2 => shellLaw (d := 2) (by decide)) :=
  ⟨le_rfl, le_rfl, uniformTuple_shellLaw (by decide)⟩

end Transformer.Perspective

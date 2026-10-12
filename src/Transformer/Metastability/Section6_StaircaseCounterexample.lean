/-
# An actual attention counterexample to the full uniform staircase

ArXiv:2410.06833v1, §6, `conj: saddle-to-saddle`.
The printed question uses an energy tending uniformly to zero and omits
its final plateau. The meaningful full-trajectory version in this tree
instead uses `2β Eβ` and includes the last unbounded plateau.

That version is false with uniform convergence on entire open intervals.
The actual two-token `SA` flow from orthogonal basis tokens supplies the
counterexample: its initial-temperature energy limit is `1/2`, while its
long-time energy limit at every positive temperature is `1`. A continuous
unbounded clock preserves the latter limit, whereas a full uniform
staircase forces those two limits to agree.

No dynamics, energy, or convergence conclusion is supplied by a new
definition. The source's original initial tuple, sphere constraints,
temperature range, and time-change requirements are all retained below.

The argument fixes one temperature when taking the long-time limit.
Unboundedness of its clock ensures that the same limiting physical energy
is seen after reparametrization. Its zero initial value also preserves the
initial energy for the separate limit in temperature.

Uniform convergence on each entire plateau controls both sides of every
finite jump time by continuity. Thus every proposed limiting height agrees
with the first one, including the height of the final unbounded plateau.
The two explicit endpoint limits then contradict the required agreement.
-/

import Transformer.Metastability.Section6_Profile
import Transformer.Metastability.Section6_UniformEndpoints
import Transformer.Metastability.Section6_TwoTokenEnergy

open Real Filter Topology

namespace Transformer.Metastability

/-- **No continuous unbounded clock produces the full uniform profile.**
The actual `SA` flow starts at the orthogonal basis pair. The conclusion
holds even without requiring the clock to be strictly increasing.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`, with the two
documented changes in `HasStaircaseProfile`: energy normalized by `2β`,
and the final plateau included as in `thm: staircase`. The entire-interval
uniform convergence quantifier is retained, not weakened. -/
theorem not_hasStaircaseProfile_basis_pair (τ : ℝ → ℝ → ℝ)
    (hτ : ∀ β : ℝ, 1 < β → Continuous (τ β))
    (hzero : ∀ β : ℝ, 1 < β → τ β 0 = 0)
    (hUnbounded : ∀ β : ℝ, 1 < β → Tendsto (τ β) atTop atTop) :
    ¬ HasStaircaseProfile 2 2
      (fun β => Perspective.saFlow β (Perspective.sphereBasisConfig 2)) τ := by
  let f (β t : ℝ) := if 1 < β then
    2 * β * Eβ 2 2 β (Perspective.saFlow β (Perspective.sphereBasisConfig 2) (τ β t))
    else 1 / 2
  have hf : ∀ β, Continuous (f β) := by
    intro β
    by_cases hβ : 1 < β
    · simpa only [f, ite_eq_left hβ, Function.comp_def] using
        (continuous_basis_pair_energy β).comp (hτ β hβ)
    · simpa only [f, ite_eq_right hβ] using
        (continuous_const : Continuous (fun _ : ℝ => (1 / 2 : ℝ)))
  have hinit : Tendsto (fun β => f β 0) atTop (nhds (1 / 2)) := by
    refine tendsto_basis_pair_initial_energy.congr' ?_
    filter_upwards [eventually_gt_atTop (1 : ℝ)] with β hβ
    simp only [f, ite_eq_left hβ, hzero β hβ]
  have hlong : ∀ β : ℝ, 1 < β → Tendsto (f β) atTop (nhds 1) := by
    intro β hβ
    simpa only [f, ite_eq_left hβ, Function.comp_def] using
      (tendsto_basis_pair_energy β (by linarith)).comp (hUnbounded β hβ)
  have hno := not_uniform_plateaux_of_distinct_endpoint_limits f hf
    (1 / 2) 1 hinit hlong (by norm_num)
  rintro ⟨k, T, φ, hk, -, hT0, hT, -, hconst, hlimit⟩
  apply hno
  refine ⟨k, T, fun i => φ (T i), by omega, hT0, hT, ?_⟩
  intro ε hε
  obtain ⟨B, hB⟩ := hlimit ε hε
  refine ⟨max B 1, ?_⟩
  intro β hβB i hi t ht hupper
  have hβ : 1 < β := (le_max_right B 1).trans_lt hβB
  have hvalue : φ (T i) = φ t :=
    hconst i hi (T i) t le_rfl ht.le (fun hil => hT i hil) hupper
  dsimp [f]
  rw [ite_eq_left hβ, hvalue]
  exact hB β ((le_max_left B 1).trans_lt hβB) i hi t ht hupper

/-- All clock hypotheses are satisfiable, and the clock traverses the
whole nonnegative trajectory: the identity clock is continuous, starts
at zero, is strictly increasing, and tends to infinity. -/
example : ∃ τ : ℝ → ℝ → ℝ,
    (∀ β : ℝ, 1 < β → Continuous (τ β)) ∧
    (∀ β : ℝ, 1 < β → τ β 0 = 0) ∧
    (∀ β : ℝ, 1 < β → Tendsto (τ β) atTop atTop) ∧
    (∀ β : ℝ, 1 < β → StrictMonoOn (τ β) (Set.Ici 0)) := by
  refine ⟨fun _ t => t, fun _ _ => continuous_id, fun _ _ => rfl,
    fun _ _ => tendsto_id, ?_⟩
  intro β _ a _ b _ hab
  exact hab

/-- **Counterexample to the full-trajectory normalized staircase problem.**
The source's `d,n ≥ 2` are realized by `d=n=2`, and every trajectory is the
actual global solution of `SA` from the same orthogonal initial tuple.
No continuous strictly increasing unbounded time change starting at zero
can give `HasStaircaseProfile` for this family.

This replaces the unproved, false universal statement of that version
with a proved counterexample. It does not refute a formulation with
convergence only on compact subintervals away from jump times. It also
does not call the printed, differently normalized problem false:
the printed energy tends uniformly to zero (`EnergyScale`).

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`; the two changes
are exactly those documented in `HasStaircaseProfile`. -/
theorem saddle_to_saddle_counterexample :
    ∃ (X₀ : SphereTuple 2 2) (X : ℝ → ℝ → SphereTuple 2 2),
      (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA 2 2 β (X β)) ∧
      ¬ ∃ τ : ℝ → ℝ → ℝ,
        (∀ β : ℝ, 1 < β → Continuous (τ β)) ∧
        (∀ β : ℝ, 1 < β → τ β 0 = 0 ∧ StrictMonoOn (τ β) (Set.Ici 0) ∧
          Tendsto (τ β) atTop atTop) ∧
        HasStaircaseProfile 2 2 X τ := by
  refine ⟨Perspective.sphereBasisConfig 2,
    fun β => Perspective.saFlow β (Perspective.sphereBasisConfig 2),
    fun β _ => Perspective.saFlow_spec β _, ?_⟩
  rintro ⟨τ, hτ, hclock, hprofile⟩
  exact not_hasStaircaseProfile_basis_pair τ hτ
    (fun β hβ => (hclock β hβ).1) (fun β hβ => (hclock β hβ).2.2) hprofile

/-- **The universal full-trajectory staircase assertion is false.**
This is the previous `saddle_to_saddle` statement with its complete
dimension, token, initial-value, dynamics, and clock quantifiers negated.
The counterexample above realizes all of them at `d=n=2`.

The conclusion concerns the normalized profile with its final plateau,
exactly as documented in `HasStaircaseProfile`. The printed problem and
a version with convergence away from jumps have different quantifiers.
Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`. -/
theorem not_saddle_to_saddle :
    ¬ ∀ (d n : ℕ), 2 ≤ d → 2 ≤ n →
      ∀ (X₀ : SphereTuple d n) (X : ℝ → ℝ → SphereTuple d n),
        (∀ β : ℝ, 1 < β → X β 0 = X₀ ∧ Perspective.SA d n β (X β)) →
        ∃ τ : ℝ → ℝ → ℝ,
          (∀ β : ℝ, 1 < β → Continuous (τ β)) ∧
          (∀ β : ℝ, 1 < β → τ β 0 = 0 ∧ StrictMonoOn (τ β) (Set.Ici 0) ∧
            Tendsto (τ β) atTop atTop) ∧
          HasStaircaseProfile d n X τ := by
  intro hall
  obtain ⟨X₀, X, hX, hno⟩ := saddle_to_saddle_counterexample
  exact hno (hall 2 2 le_rfl le_rfl X₀ X hX)

end Transformer.Metastability

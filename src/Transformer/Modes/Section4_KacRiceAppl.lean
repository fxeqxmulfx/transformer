import Transformer.Modes.Section2_MainForm
import Transformer.Modes.Section2_PhiTUniform
import Transformer.Modes.Section5_PtBddDensity

/-
# The number of modes of a Gaussian KDE — counterexamples to the application claims

§4.1 of arXiv:2412.09080v3, `sec: kac.rice.application`, asserts that the
Kac–Rice hypotheses hold for every `β > 0`, `n ≥ 5` on `T`. The claimed
continuous joint density does not exist throughout those parameters.

**What the source says and what is proved here.**

* `lem: pt.bdd` fixes any `β > 0`, `n ≥ 5`, `t ∈ T` and asserts a continuous
  joint density vanishing at infinity. It then says that `thm:kac-rice`
  applies. `not_pt_bdd` refutes precisely that conjunction when `β > 2`
  and `n < β + 2`. `n = 5`, `β = 10` is a concrete counterexample.

* `lem:kr-appl` writes the expected-upcrossing integral in terms of `p_t`.
  The version previously carried here also required the continuous density
  supplied by `lem: pt.bdd`, since the integral evaluates a density on the
  null line `x = 0`. `not_kr_appl` refutes that version: its density-existence
  clause is already impossible. The integral identity under other density
  hypotheses would be a separate statement.

* The density obstruction is probabilistic, not an artifact of Kac–Rice
  definitions. The remote Gaussian sampling event in §5.5 has more mass
  than any continuous planar density can put in a small rectangle.
  `not_continuous_density_fieldF` proves it from the formula for `F_n`.
  The test point `t = 0` belongs to every window `T`.

The two formerly unproved universal claims are replaced by counterexamples,
without retaining them as unproved inputs to any result.
All density laws below use the standard Gaussian product sampling measure
and the field with precisely the normalization of `eq:Fn`.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`, `lem:kr-appl`; §5.5.
-/

open MeasureTheory Filter
open scoped ENNReal Topology

namespace Transformer.Modes

/-- No continuous joint-density family exists on a nonempty observation
set in the counterexample range. The obstruction holds at each point.
Source: arXiv:2412.09080v3, §4.1, the density assertion of `lem: pt.bdd`. -/
theorem not_continuous_density_family {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) {S : Set ℝ} (hS : S.Nonempty) :
    ¬ ∃ p : ℝ → ℝ → ℝ → ℝ, ∀ t ∈ S,
      Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Measure.map (fun X => (fieldF β X t, deriv (fieldF β X) t)) (gaussianSample n) =
          volume.withDensity fun z : ℝ × ℝ => ENNReal.ofReal (p t z.1 z.2) := by
  rintro ⟨p, hp⟩
  obtain ⟨t, ht⟩ := hS
  obtain ⟨hcontinuous, hmap⟩ := hp t ht
  exact not_continuous_density_fieldF hn hβ hnβ t
    ⟨fun z => p t z.1 z.2, hcontinuous, hmap⟩

/-- All family-obstruction assumptions hold for a genuine observation set. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 10 ∧ (5 : ℝ) < 10 + 2 ∧
    (Set.Icc (-1 : ℝ) 1).Nonempty := by
  refine ⟨by norm_num, by norm_num, by norm_num, 0, ?_⟩
  norm_num

/-- **Counterexample to `lem: pt.bdd`.** For `n ≥ 5`, `β > 2`, `n < β + 2`,
the claimed Kac–Rice field and continuous joint density cannot exist on `T`.
These parameters satisfy the source's weaker requirements `β > 0`, `n ≥ 5`.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`. The source's universal
continuous-density assertion is false; this theorem proves its negation
on an explicit nonempty range of parameters. -/
theorem not_pt_bdd {β : ℝ} (hβ : 2 < β) {n : ℕ} (hn : 5 ≤ n)
    (hnβ : (n : ℝ) < β + 2) (w : ℝ) :
    ¬ ∃ (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ),
      IsKacRiceField (gaussianSample n) (fun X => fieldF β X) 0 (intervalT n β w) p1 p ∧
      ∀ t ∈ intervalT n β w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Tendsto (fun z : ℝ × ℝ => p t z.1 z.2) (cocompact (ℝ × ℝ)) (𝓝 0) := by
  rintro ⟨p1, p, hfield, hp⟩
  apply not_continuous_density_family (by omega : 0 < n) hβ hnβ
    (Set.nonempty_of_mem (zero_mem_intervalT n β w))
  refine ⟨p, ?_⟩
  intro t ht
  exact ⟨(hp t ht).1, hfield.map_joint_eq t ht⟩

/-- The counterexample satisfies both the source's assumptions and the
additional inequalities identifying the obstruction. -/
example : (0 : ℝ) < 10 ∧ (2 : ℝ) < 10 ∧ 5 ≤ (5 : ℕ) ∧ (5 : ℝ) < 10 + 2 := by
  norm_num

/-- **Counterexample to the continuous-density version of `lem:kr-appl`.**
For these parameters no density `p_t` can satisfy the displayed continuity
and law conditions, regardless of the expected-upcrossing identity.

Source: arXiv:2412.09080v3, §4.1, `lem:kr-appl` as deduced from
`lem: pt.bdd`. The density-existence assertion is refuted. -/
theorem not_kr_appl {β : ℝ} (hβ : 2 < β) {n : ℕ} (hn : 5 ≤ n)
    (hnβ : (n : ℝ) < β + 2) (w : ℝ) :
    ¬ ∃ p : ℝ → ℝ → ℝ → ℝ,
      (∀ t ∈ intervalT n β w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Measure.map (fun X => (fieldF β X t, deriv (fieldF β X) t)) (gaussianSample n) =
          volume.withDensity fun z : ℝ × ℝ => ENNReal.ofReal (p t z.1 z.2)) ∧
      expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 (intervalT n β w) =
        ∫⁻ t in intervalT n β w, ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal (y * p t 0 y) := by
  rintro ⟨p, hp, _⟩
  exact not_continuous_density_family (by omega : 0 < n) hβ hnβ
    (Set.nonempty_of_mem (zero_mem_intervalT n β w)) ⟨p, hp⟩

/-- The application counterexample has simultaneous numerical witnesses. -/
example : (2 : ℝ) < 10 ∧ 5 ≤ (5 : ℕ) ∧ (5 : ℝ) < 10 + 2 := by norm_num

/-- **`n = 5`, `β = 10` refutes `lem: pt.bdd` on every window.** All the
numerical assumptions are discharged, rather than carried as hypotheses.
Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`. -/
theorem not_pt_bdd_five_ten (w : ℝ) :
    ¬ ∃ (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ),
      IsKacRiceField (gaussianSample 5) (fun X => fieldF 10 X) 0 (intervalT 5 10 w) p1 p ∧
      ∀ t ∈ intervalT 5 10 w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Tendsto (fun z : ℝ × ℝ => p t z.1 z.2) (cocompact (ℝ × ℝ)) (𝓝 0) :=
  not_pt_bdd (by norm_num) (by norm_num) (by norm_num) w

/-- **The same concrete parameters refute the continuous-density
application claim `lem:kr-appl`.**
Source: arXiv:2412.09080v3, §4.1, `lem:kr-appl` and `lem: pt.bdd`. -/
theorem not_kr_appl_five_ten (w : ℝ) :
    ¬ ∃ p : ℝ → ℝ → ℝ → ℝ,
      (∀ t ∈ intervalT 5 10 w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Measure.map (fun X => (fieldF 10 X t, deriv (fieldF 10 X) t)) (gaussianSample 5) =
          volume.withDensity fun z : ℝ × ℝ => ENNReal.ofReal (p t z.1 z.2)) ∧
      expectedUpcrossings (gaussianSample 5) (fun X => fieldF 10 X) 0 (intervalT 5 10 w) =
        ∫⁻ t in intervalT 5 10 w, ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal (y * p t 0 y) :=
  not_kr_appl (by norm_num) (by norm_num) (by norm_num) w

/-- **The density obstruction persists in the regime `β = n`.** For every
sample count at least five, the source's continuous joint density already
fails when the precision parameter `β` equals the sample count, at any observation point.
Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`, and §1, the regime
`n^c ≲ β ≲ n^{2-c}` (which includes `β = n` for `0 < c < 1`). -/
theorem not_continuous_density_beta_eq_n {n : ℕ} (hn : 5 ≤ n) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      Measure.map (fun X => (fieldF (n : ℝ) X t, deriv (fieldF (n : ℝ) X) t))
        (gaussianSample n) = volume.withDensity fun z => ENNReal.ofReal (q z) := by
  apply not_continuous_density_fieldF (by omega : 0 < n)
  · exact_mod_cast (by omega : 2 < n)
  · linarith

/-- Five samples meet the sample-count hypothesis in the `β = n` regime. -/
example : 5 ≤ (5 : ℕ) := le_rfl

end Transformer.Modes

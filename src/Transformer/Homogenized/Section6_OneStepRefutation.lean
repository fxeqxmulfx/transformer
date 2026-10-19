module

public import Transformer.Homogenized.Section6_RadialError

/-!
# The source's modified one-step estimate is false

Source: arXiv:2604.01978v1, §6.1, `lem:stability_generator`.
The source claims a uniform error `C η² max(η,α)` between one
actual normalized layer and the modified SDE at time `η`.

The counterexample has one token, one head, dimension two,
positive temperature one, and deterministic head `(I+J,0)`.
All centered fluctuations vanish, so `ass:high_order_short`
and the variance-proxy condition hold with parameters zero.
The comparison is between genuine probability spaces.

Its projected drift is `Jx`, and its actual covariant
corrector vanishes. The modified equation is thus the
unit-speed rotation, with no step-size dependence.
The exact discrete layer instead rotates through
`arctan(η/(1+η))`. The second-coordinate observable is C⁴,
and its one-step discrepancy is at least `η²/3` for `η≤1`.

The source's proof replaces the unprojected attention field
inside normalization by its projected drift and fluctuation.
The missing radial part changes the second-order tangential
term. The obstruction therefore occurs at a full grid step;
restricting the comparison to grid times does not remove it.

Every process and differential operator in the counterexample
is the original one. The supporting modules verify the head,
law, full ambient derivative, normalization, exact chain,
modified SDE, and quantitative discrepancy independently.
-/

@[expose] public section

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- A single genuine model satisfies the source assumptions and violates
`lem:stability_generator` for every proposed positive constant.
The step size is chosen only after the constant, as required to refute
the uniform bound. Both processes have the same deterministic unit start.
Source: arXiv:2604.01978v1, §6.1, `lem:stability_generator`. -/
theorem stability_generator_counterexample :
    0 < (1 : ℕ) ∧ HasHighOrderLaw 2 0 0 (Measure.dirac radialRotHead) ∧
      ContDiff ℝ 4 (fun x : Idx 1 → EucSpace 2 => x 0 1) ∧
      IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧
      ∀ C : ℝ, 0 < C → ∃ η : ℝ, 0 < η ∧
        IsVarianceProxy 2 1 1 (Measure.dirac radialRotHead) 0 ∧
        (∀ _ : Idx 1, ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ))‖ = 1) ∧
        IsRandomChain (H := 1) η 1 (Measure.dirac radialRotHead) (Measure.dirac ())
          (fun _ _ _ => radialRotHead) (fun _ => radialRotChain η 1)
          (fun _ => EuclideanSpace.single 0 1) ∧
        IsModifiedSde η 1 (alphaOf η 0 1) 0 (Measure.dirac radialRotHead)
          (Measure.dirac ()) (fun t (_ : Unit) => rotFlow t) ∧
        (∀ _ : Unit, rotFlow 0 = fun _ => EuclideanSpace.single 0 1) ∧
        C * η ^ 2 * max η (alphaOf η 0 1) <
          |(∫ _ : Unit, (rotFlow η 0) 1 ∂Measure.dirac ()) -
            ∫ _ : Unit, radialRotChain η 1 1 0 1 ∂Measure.dirac ()| := by
  refine ⟨Nat.one_pos, (radialRotHead_law 1).1, contDiff_radialRotObservable,
    inferInstance, ?_⟩
  intro C hC
  obtain ⟨η, hη, _, hgap⟩ := radialRot_one_step_constant C hC
  refine ⟨η, hη, (radialRotHead_law 1).2, fun _ => by simp [PiLp.norm_single],
    isRandomChain_radialRotChain hη 1, ?_, ?_, hgap⟩
  · simpa only [alphaOf, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div] using
      isModifiedSde_radialRotHead η 1
  · intro ω
    funext i
    rw [rotFlow_apply]
    ext k
    fin_cases k <;> simp

/-- **Refutation of Lemma `lem:stability_generator`.** The negated statement
below is the original uniform claim, including its complete hypotheses.
Its proposed constant may depend on the model and C⁴ observable, but is
chosen before the step size, variance proxy, start, and both processes.

The SDE here is exactly the modified one: the source proof explicitly
writes its drift as `b-(η/2)∇_b b`. The discrete process is exactly
`eq:update_tokens`. Neither process is replaced by an arbitrary witness
with an assumed discrepancy. The preceding theorem constructs them.

The counterexample uses a positive temperature, a positive head count,
unit initial tokens, independent heads with the prescribed high-order law,
and probability measures for both expectations. The coordinate observable
is smooth on the ambient space, hence also on the sphere.

This refutes the source's displayed rate, rather than weakening that rate
under the original lemma's name. It does not refute the coarser `O(η)`
bound in `thm:weak_error_clean`.
Source: arXiv:2604.01978v1, §6.1, `lem:stability_generator`. -/
theorem not_stability_generator :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0)
      (ρ : Measure (HeadParam d)), HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
    ∃ C : ℝ, 0 < C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ x : Idx n → EucSpace d, (∀ i, ‖x i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x) →
        |(∫ ω', φ (X η ω') ∂P') - ∫ ω, φ (Xd ω 1) ∂P| ≤
          C * η ^ 2 * max η (alphaOf η s H) := by
  intro h
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) Nat.one_pos 1 0 0
    (Measure.dirac radialRotHead) (radialRotHead_law 1).1 _ contDiff_radialRotObservable
  obtain ⟨η, hη, _, hgap⟩ := radialRot_one_step_constant C hC
  have hsde : IsModifiedSde η 1 (alphaOf η 0 1) 0 (Measure.dirac radialRotHead)
      (Measure.dirac ()) (fun t (_ : Unit) => rotFlow t) := by
    simpa only [alphaOf, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div] using
      isModifiedSde_radialRotHead η 1
  have key := hb η 0 hη (radialRotHead_law 1).2
    (fun _ => EuclideanSpace.single 0 1) (fun _ => by simp [PiLp.norm_single])
    Unit (Measure.dirac ()) _ _ (isRandomChain_radialRotChain hη 1)
    Unit (Measure.dirac ()) (fun t _ => rotFlow t) hsde
    (fun _ => by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp)
  exact (not_lt_of_ge key) hgap

/-- A positive candidate constant realizes the sole numerical premise in
`stability_generator_counterexample`; all model assumptions are proved
inside that theorem rather than supplied as unproved inputs. -/
example : (0 : ℝ) < 1 := zero_lt_one

/-- At physical step one, the model premises hold jointly for the actual
chain and modified SDE, including their common deterministic unit start.
This example verifies satisfiability independently of the small-step
choice used to defeat each proposed approximation constant. -/
example :
    HasHighOrderLaw 2 0 0 (Measure.dirac radialRotHead) ∧
      IsVarianceProxy 2 1 1 (Measure.dirac radialRotHead) 0 ∧
      IsRandomChain (H := 1) 1 1 (Measure.dirac radialRotHead) (Measure.dirac ())
        (fun _ _ _ => radialRotHead) (fun _ => radialRotChain 1 1)
        (fun _ => EuclideanSpace.single 0 1) ∧
      IsModifiedSde 1 1 (alphaOf 1 0 1) 0 (Measure.dirac radialRotHead)
        (Measure.dirac ()) (fun t (_ : Unit) => rotFlow t) ∧
      (∀ _ : Idx 1, ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ))‖ = 1) ∧
      rotFlow 0 = (fun _ => EuclideanSpace.single 0 1) := by
  refine ⟨(radialRotHead_law 1).1, (radialRotHead_law 1).2,
    isRandomChain_radialRotChain zero_lt_one 1, ?_,
    fun _ => by simp [PiLp.norm_single], ?_⟩
  · simpa only [alphaOf, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div] using
      isModifiedSde_radialRotHead 1 1
  · funext i
    rw [rotFlow_apply]
    ext k
    fin_cases k <;> simp

end Transformer.Homogenized

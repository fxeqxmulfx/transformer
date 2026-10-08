/-
# The two inverse-time parameter bounds for neural interpolation are false

arXiv:2411.04551v3, §4, `prop: interpolation.neural.ode` and
`lem: induction.neural.ode` bound the norm of the entire tuple `(W,U,b)`
by `CM/(T min εᵢ)` and `C/(Tε)`. Fix one pair of antipodal unit vectors,
an orthogonal unit normal, and `ε = 1`. For every proposed constant `C`,
take `T = C² + 1`. The asserted bound permits displacement at most
`2C²/T < 2`, whereas the endpoints are distance `2` apart.

Both counterexamples allow arbitrary controls and require only the terminal
integral identity of the actual sphere-valued equation. They therefore
rule out the source's solutions before imposing its switch restrictions.
An essential bound on all three parameters suffices; isolated values at
switching times cannot evade the obstruction. The maximum of the operator
norms and vector norm is a norm on the finite-dimensional parameter space;
another norm changes the proposed constant, independent of `T`.
-/

import Transformer.Interpolation.Section4_NeuralSpeed

open scoped BigOperators
open MeasureTheory Real

namespace Transformer.Interpolation

/-- The normal `e₁` is perpendicular to the moving pair `e₀, -e₀`.
Source: arXiv:2411.04551v3, §4, the hypotheses of both neural propositions. -/
noncomputable def neuralBoundNormal : SSphere 3 :=
  ⟨EuclideanSpace.single (1 : Fin 3) (1 : ℝ), by simp [PiLp.norm_single]⟩

/-- The exact orthogonality required for the one-pair data.
Source: arXiv:2411.04551v3, §4, `prop: interpolation.neural.ode`. -/
theorem neuralBoundNormal_orthogonal :
    inner (𝕜 := ℝ) (neuralBoundNormal : EucSpace 3)
      ((basePoint 2 : EucSpace 3) - (antipode 3 (basePoint 2) : EucSpace 3)) = 0 := by
  simp [neuralBoundNormal, basePoint, antipode, EuclideanSpace.inner_single_left]

/-- The requested move is nonzero, with Euclidean length exactly `2`.
Source: arXiv:2411.04551v3, §4, the counterexample's interpolation data. -/
theorem neural_bound_endpoints_distance :
    ‖(antipode 3 (basePoint 2) : EucSpace 3) - (basePoint 2 : EucSpace 3)‖ = 2 := by
  have he : (antipode 3 (basePoint 2) : EucSpace 3) - (basePoint 2 : EucSpace 3) =
      (-2 : ℝ) • (basePoint 2 : EucSpace 3) := by
    dsimp [antipode]
    module
  rw [he, norm_smul, mem_sphere_zero_iff_norm.mp (basePoint 2).2]
  norm_num

/-- For every proposed inverse-time constant, a positive horizon excludes
even the terminal balance law between the antipodal endpoints. Every
source solution is sphere-valued and satisfies this law. No continuity,
switch count, or additional solution input is required for the obstruction.
Source: arXiv:2411.04551v3, §4, both neural norm estimates and
`eq: neural.ode.sphere`. -/
theorem neural_parameter_bound_counterexample (C : ℝ) :
    ∃ T : ℝ, 0 < T ∧ ∀ (W U : ℝ → ParamMatrix 3) (b : ℝ → EucSpace 3),
      (∀ᵐ s : ℝ, s ∈ Set.Icc 0 T → max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ C / T) →
      ¬ ∃ x : ℝ → SSphere 3, x 0 = basePoint 2 ∧ x T = antipode 3 (basePoint 2) ∧
        (x T : EucSpace 3) = (x 0 : EucSpace 3) +
          ∫ s in (0 : ℝ)..T, neuralVF 3 W U b s (x s) := by
  let T : ℝ := C ^ 2 + 1
  have hT : 0 < T := by dsimp [T]; positivity
  have hsmall : 2 * (C / T) ^ 2 * T < 2 := by
    have he : 2 * (C / T) ^ 2 * T = (2 * C ^ 2) / T := by
      field_simp
    rw [he, div_lt_iff₀ hT]
    dsimp [T]
    nlinarith
  refine ⟨T, hT, fun W U b hmax ⟨x, hx0, hxT, heq⟩ => ?_⟩
  have hd := neural_displacement_le W U b (fun s => (x s : EucSpace 3)) T (C / T)
    hT.le hmax (fun s _ => mem_sphere_zero_iff_norm.mp (x s).2) heq
  rw [hx0, hxT, neural_bound_endpoints_distance] at hd
  exact (not_le_of_gt hsmall) hd

/-- **The uniform bound in `prop: interpolation.neural.ode` is false.**
Already for `d = 3`, `M = 1`, and `ε₁ = 1`, no constant works for all horizons.
The conclusion denied here asks for only the source's endpoint balance and
an essential norm bound, allowing any controls without a switch restriction;
a solution promised by the source would satisfy both.
Source: arXiv:2411.04551v3, §4, `prop: interpolation.neural.ode`. -/
theorem not_prop_interpolation_neural_ode :
    ¬ ∃ C : ℝ, 0 < C ∧ ∀ (M : ℕ) (x₀ y γ : Idx M → SSphere 3) (ε : Idx M → ℝ),
      1 ≤ M →
      (∀ i j : Idx M, i ≠ j → x₀ i ≠ x₀ j) →
      (∀ i j : Idx M, i ≠ j → y i ≠ y j) →
      (∀ i, 0 < ε i) →
      (∀ i, inner (𝕜 := ℝ) (γ i : EucSpace 3)
        ((x₀ i : EucSpace 3) - (y i : EucSpace 3)) = 0) →
      (∀ i j : Idx M, i ≠ j → x₀ j ∉ Hε 3 (γ i) (ε i)) →
      ∀ T : ℝ, 0 < T →
        ∃ (W U : ℝ → ParamMatrix 3) (b : ℝ → EucSpace 3),
          (∀ᵐ s : ℝ, s ∈ Set.Icc 0 T →
            max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ C * M / (T * ⨅ i, ε i)) ∧
          ∃ x : ℝ → Idx M → SSphere 3,
            (∀ i, x 0 i = x₀ i) ∧ (∀ i, x T i = y i) ∧
            ∀ i, (x T i : EucSpace 3) = (x 0 i : EucSpace 3) +
              ∫ s in (0 : ℝ)..T, neuralVF 3 W U b s (x s i) := by
  rintro ⟨C, _, hs⟩
  rcases neural_parameter_bound_counterexample C with ⟨T, hT, hno⟩
  have hsingle : ∀ i j : Idx 1, i ≠ j → False :=
    fun i j h => h (Subsingleton.elim i j)
  rcases hs 1 (fun _ => basePoint 2) (fun _ => antipode 3 (basePoint 2))
    (fun _ => neuralBoundNormal) (fun _ => 1) le_rfl
    (fun i j h => False.elim (hsingle i j h))
    (fun i j h => False.elim (hsingle i j h)) (fun _ => one_pos)
    (fun _ => neuralBoundNormal_orthogonal) (fun i j h => False.elim (hsingle i j h))
    T hT with ⟨W, U, b, hmax, x, hx0, hxT, heq⟩
  apply hno W U b (by simpa using hmax)
  exact ⟨fun s => x s 0, hx0 0, hxT 0, heq 0⟩

/-- **The uniform bound in `lem: induction.neural.ode` is false.** One
active pair and no inactive points suffice. The `M+1` indexing below means
this case is `M = 0` here and `M = 1` in the source. All orthogonality and
separation hypotheses are retained; `ε = 1` is fixed before choosing `T`.
As above, even the terminal balance with arbitrary controls is impossible.
Source: arXiv:2411.04551v3, §4, `lem: induction.neural.ode`. -/
theorem not_lem_induction_neural_ode :
    ¬ ∃ C : ℝ, 0 < C ∧
      ∀ (M : ℕ) (x₀ y : Idx (M + 1) → SSphere 3) (γ : SSphere 3) (ε : ℝ),
        (∀ i j : Idx (M + 1), i ≠ j → x₀ i ≠ x₀ j) →
        (∀ i j : Idx (M + 1), i ≠ j → y i ≠ y j) →
        (∀ i : Idx (M + 1), i ≠ Fin.last M → x₀ i = y i) → 0 < ε →
        inner (𝕜 := ℝ) (γ : EucSpace 3)
          ((x₀ (Fin.last M) : EucSpace 3) - (y (Fin.last M) : EucSpace 3)) = 0 →
        (∀ i : Idx (M + 1), i ≠ Fin.last M → x₀ i ∉ Hε 3 γ ε) →
        ∀ T : ℝ, 0 < T →
          ∃ (W U : ℝ → ParamMatrix 3) (b : ℝ → EucSpace 3),
            (∀ᵐ s : ℝ, s ∈ Set.Icc 0 T → max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ C / (T * ε)) ∧
            ∃ x : ℝ → Idx (M + 1) → SSphere 3,
              (∀ i, x 0 i = x₀ i) ∧ (∀ i, x T i = y i) ∧
              ∀ i, (x T i : EucSpace 3) = (x 0 i : EucSpace 3) +
                ∫ s in (0 : ℝ)..T, neuralVF 3 W U b s (x s i) := by
  rintro ⟨C, _, hs⟩
  rcases neural_parameter_bound_counterexample C with ⟨T, hT, hno⟩
  have hsingle : ∀ i j : Idx 1, i ≠ j → False :=
    fun i j h => h (Subsingleton.elim i j)
  rcases hs 0 (fun _ => basePoint 2) (fun _ => antipode 3 (basePoint 2))
    neuralBoundNormal 1
    (fun i j h => False.elim (hsingle i j h))
    (fun i j h => False.elim (hsingle i j h))
    (fun i h => False.elim (hsingle i (Fin.last 0) h)) one_pos
    neuralBoundNormal_orthogonal (fun i h => False.elim (hsingle i (Fin.last 0) h))
    T hT with ⟨W, U, b, hmax, x, hx0, hxT, heq⟩
  apply hno W U b (by simpa using hmax)
  exact ⟨fun s => x s 0, hx0 0, hxT 0, heq 0⟩

/-- The source hypotheses hold for this nonzero move, with `d = 3`, one
pair, `ε = 1`, and `T = 2`. The normal is a unit vector by construction. -/
example : 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧
    (∀ i j : Idx 1, i ≠ j → basePoint 2 ≠ basePoint 2) ∧
    (∀ i j : Idx 1, i ≠ j → antipode 3 (basePoint 2) ≠ antipode 3 (basePoint 2)) ∧
    (∀ i : Idx 1, i ≠ Fin.last 0 → basePoint 2 = antipode 3 (basePoint 2)) ∧
    inner (𝕜 := ℝ) (neuralBoundNormal : EucSpace 3)
      ((basePoint 2 : EucSpace 3) - (antipode 3 (basePoint 2) : EucSpace 3)) = 0 ∧
    (∀ i : Idx 1, i ≠ Fin.last 0 → basePoint 2 ∉ Hε 3 neuralBoundNormal 1) := by
  have hs : ∀ i j : Idx 1, i ≠ j → False := fun i j h => h (Subsingleton.elim i j)
  exact ⟨le_rfl, one_pos, by norm_num, fun i j h => False.elim (hs i j h),
    fun i j h => False.elim (hs i j h), fun i h => False.elim (hs i (Fin.last 0) h),
    neuralBoundNormal_orthogonal, fun i h => False.elim (hs i (Fin.last 0) h)⟩

end Transformer.Interpolation

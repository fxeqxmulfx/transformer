module

public import Transformer.Basic
public import Transformer.Causal.Basic
public import Transformer.Causal.Interaction
public import Transformer.Causal.InteractionWindow
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Angle
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The original frozen angular field and its initialization space

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers` and its
proof in Appendix A. These definitions specify the field used
by the convergence and stability theorems.

There are `n` moving angular coordinates and `m` fixed centers.
Each angle is a real lift of a point on the circle. Adding an
integer multiple of `2π` changes its lift and represents the
same circular position. The distance `circDist` is the absolute
value of the principal circular representative, in `[0, π]`.
The centers have angles `θ` and masses `a` and contribute to
every moving row of the equation.

The score is `exp (β * (cos (φ k - φ j) - 1))`. The causal mask
keeps exactly the indices `j ≤ k`. In particular, the row keeps
its own score, which equals one. That self term contributes to
the partition `frozenZ`; its sine contribution to the velocity
vanishes. Each center contributes its score multiplied by its
own mass, in both the partition and the sine numerator.

The original field `frozenVelocity` divides the complete signed
sine numerator by that original partition. The sine sign is
`sin (other - current)`, as in the displayed source equation.
The source uses `β > 0` and `a j ≥ 1` in its clustering theorem;
the field itself is defined for arbitrary real parameters.
The proofs of regularity and global existence use nonnegative
center masses and the self score to bound the partition away
from zero.

The state space is a finite product of real lines. Its norm is
the maximum of the absolute coordinate values. The predicate
`IsAsymptoticallyStable` uses this norm and quantifies over every
original global ODE solution. It requires both Lyapunov
stability and convergence of all sufficiently nearby solutions.

Initialization measures live on these lifted angular tuples.
The reference measure is product Lebesgue volume. Absolute
continuity gives the original probability-one interpretation
of the exceptional initialization sets proved null later.
The final example realizes the original separation, parameter,
and mass hypotheses together with an actual probability measure:
restricted Lebesgue volume on the unit interval for one moving
token and two antipodal fixed centers.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (n m : ℕ)

/-- The geodesic distance on `𝕊^1` between angles whose difference is `x`. -/
noncomputable def circDist (x : ℝ) : ℝ := |(x : Real.Angle).toReal|

/-- **The normalization `Z_k`** of `thm:fixed_centers`:

  `Z_k = ∑_{j ≤ k} e^{β(cos(φ_k - φ_j) - 1)} + ∑_{j ≤ m} a_j e^{β(cos(φ_k - θ_j) - 1)}`.

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
noncomputable def frozenZ (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ) (k : Idx n) : ℝ :=
  (∑ j ∈ Finset.univ.filter (fun j : Idx n => j ≤ k),
      Real.exp (β * (Real.cos (φ k - φ j) - 1)))
    + ∑ j : Idx m, a j * Real.exp (β * (Real.cos (φ k - θ j) - 1))

/-- **The causal dynamics with frozen tokens**, the right-hand side of
`thm:fixed_centers`:

  `φ̇_k = Z_k⁻¹ (∑_{j ≤ k} e^{β(cos(φ_k - φ_j) - 1)} sin(φ_j - φ_k)
                + ∑_{j ≤ m} a_j e^{β(cos(φ_k - θ_j) - 1)} sin(θ_j - φ_k))`.

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
noncomputable def frozenVelocity (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ) :
    Idx n → ℝ := fun k =>
  (frozenZ n m β θ a φ k)⁻¹ *
    ((∑ j ∈ Finset.univ.filter (fun j : Idx n => j ≤ k),
        Real.exp (β * (Real.cos (φ k - φ j) - 1)) * Real.sin (φ j - φ k))
      + ∑ j : Idx m, a j * Real.exp (β * (Real.cos (φ k - θ j) - 1))
          * Real.sin (θ j - φ k))

/-- **Asymptotic stability** of a point `p` for the autonomous equation
`ẋ = F(x)`: solutions starting near `p` stay near it (Lyapunov stability), and
those starting near enough converge to it. -/
def IsAsymptoticallyStable (F : (Idx n → ℝ) → Idx n → ℝ) (p : Idx n → ℝ) : Prop :=
  (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ ∀ x : ℝ → Idx n → ℝ,
      (∀ t : ℝ, HasDerivAt x (F (x t)) t) → ‖x 0 - p‖ < δ →
        ∀ t : ℝ, 0 ≤ t → ‖x t - p‖ < ε) ∧
    ∃ η : ℝ, 0 < η ∧ ∀ x : ℝ → Idx n → ℝ,
      (∀ t : ℝ, HasDerivAt x (F (x t)) t) → ‖x 0 - p‖ < η →
        Filter.Tendsto x Filter.atTop (nhds p)

/-- The hypotheses of `fixed_centers` are satisfiable, and not vacuously so:
one particle, the two antipodal frozen tokens `θ = (0, π)` with unit weights,
`N = 3`, `β = 11`, `ε = 0.09`, `c = 5.68`, and `μ₀` uniform on `[0, 1)`. The
conditions on `h` and `g` are `lem:interaction`
(`Causal.interaction_inequalities`), whose bound on `N` is
`3 < e^{3·4.5²/8} · 0.09/4.68`. -/
example :
    (0 : ℕ) < 2 ∧ (0 : ℝ) < 11 ∧ (0 : ℝ) < 9 / 100 ∧ 2 + 2 * (9 / 100 : ℝ) < 142 / 25 ∧
      (∀ j : Idx 2, (1 : ℝ) ≤ (fun _ => 1 : Idx 2 → ℝ) j) ∧
      (∀ i j : Idx 2, i ≠ j → (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ))
        < circDist ((![0, π] : Idx 2 → ℝ) i - (![0, π] : Idx 2 → ℝ) j)) ∧
      (((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j)
          * h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))
        < h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
      -((((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j)
          * g_pot 11 ((142 / 25 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))))
        < g_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
      IsProbabilityMeasure (volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1))) ∧
      volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1)) ≪ volume := by
  have hN : ((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j = 3 := by
    simp; norm_num
  have hex : (156 : ℝ) < Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) := by
    have h1 := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 81 / 32 by norm_num)
    have h3 : Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) = Real.exp (81 / 32) ^ 3 := by
      rw [← Real.exp_nat_mul]; norm_num
    rw [h3]
    calc (156 : ℝ) < (1 + 81 / 32 + (81 / 32) ^ 2 / 2) ^ 3 := by norm_num
      _ ≤ Real.exp (81 / 32) ^ 3 := by gcongr
  have hwin := interaction_inequalities 3 (142 / 25) (9 / 100) 11 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by nlinarith)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, fun _ => le_rfl, ?_,
    by rw [hN]; exact hwin.1, by rw [hN]; exact hwin.2, ?_,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self⟩
  · have hs : (11 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 1 / 3 := by
      rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow, one_div]
      have : (3 : ℝ) ≤ Real.sqrt 11 := by
        rw [show (3 : ℝ) = Real.sqrt 9 by
          rw [show (9 : ℝ) = 3 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
        exact Real.sqrt_le_sqrt (by norm_num)
      exact inv_anti₀ (by norm_num) this
    have hpi : circDist π = π := Real.Angle.abs_toReal_coe_eq_self_iff.mpr ⟨pi_pos.le, le_rfl⟩
    have hpi' : circDist (-π) = π := by
      rw [circDist, Real.Angle.coe_neg, Real.Angle.neg_coe_pi]; exact hpi
    have hlt : (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) < π := by
      nlinarith [pi_gt_three]
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all
  · constructor
    rw [Measure.restrict_apply_univ, Real.volume_pi_Ico]
    simp
end Causal
end Transformer

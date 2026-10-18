/-
# Almost every solution of `eq:onangles` synchronizes

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §7.

After Open Problem `o:strictsaddle` the survey states: "By classical
arguments, recalled in Appendix A, a positive answer to Problem
`o:strictsaddle` would imply that for all initial conditions except a set of
measure zero, all `θ_i(t)` converge under the dynamics `eq:onangles` to a
common limit as `t → +∞`."

`strict_saddle` gives the positive answer for every `β > 0`: each critical
point of `-𝖤_β` is a global minimizer or a strict saddle
(`isMinOn_or_isStrictSaddle_neg_eucTorusEnergy`), and
`ae_tendsto_synchronized_of_angularUSA` draws the conclusion.  The solutions
of `eq:onangles` are the gradient flow of `-𝖤_β` on `EucSpace n` run at speed
`n` (`angularUSA_eq_gradientFlow`, `angularUSA_gradientFlow`).  `-𝖤_β` is
invariant under the periods `2π e_k`, which span the space (`span_periods`),
so `Landscape.SaddleAvoidance.ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top`
applies, and the minimizers of `-𝖤_β` are synchronized
(`synchronized_of_isGlobalMaxTorusEnergy`).

The classical arguments of Appendix A use the Łojasiewicz theorem for
real-analytic functions (`lojasiewicz1963propriete`): `-𝖤_β` is analytic
(`analyticAt_neg_eucTorusEnergy`), so it satisfies the Łojasiewicz inequality
at every point (`exists_isLojasiewiczAt_neg_eucTorusEnergy`, from
`Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`).  At the synchronized
states, and for two tokens at every critical point, the exponent is `1/2`
(`isLojasiewiczAt_of_synchronized`, `isLojasiewiczAt_of_isCritical_two`).
-/
import Transformer.Perspective.CircleLojasiewicz
import Transformer.Perspective.CircleSAFlow
import Transformer.Perspective.StrictSaddle
import Transformer.Landscape.SaddleAvoidance.FlowLandscape
import Transformer.Landscape.SaddleAvoidance.LocalInequality
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators Topology
open Real Filter MeasureTheory

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **Every critical point of `-𝖤_β` is a global minimizer or a strict saddle**,
for `β > 0`.  A critical point that is not a global maximum of `𝖤_β` has a
line of positive second derivative (`strict_saddle`), so it is a strict
saddle of `-𝖤_β` (`isStrictSaddle_of_isStrictSaddleTorusEnergy`). -/
theorem isMinOn_or_isStrictSaddle_neg_eucTorusEnergy {β : ℝ} (hβ : 0 < β) {z : EucSpace n}
    (hz : gradient (fun ψ => -eucTorusEnergy n β ψ) z = 0) :
    IsMinOn (fun ψ => -eucTorusEnergy n β ψ) Set.univ z ∨
      Landscape.SaddleAvoidance.IsStrictSaddle (fun ψ => -eucTorusEnergy n β ψ) z := by
  have hcrit := (gradient_neg_eucTorusEnergy_eq_zero_iff n hβ.ne' z).1 hz
  by_cases hmax : IsGlobalMaxTorusEnergy n β z.ofLp
  · exact Or.inl (isMinOn_univ_iff.2 fun y => neg_le_neg (hmax y.ofLp))
  · exact Or.inr (isStrictSaddle_of_isStrictSaddleTorusEnergy n β hβ.ne' hcrit
      (strict_saddle n β hβ z.ofLp hcrit hmax))

/-- **A solution of `eq:onangles` is the gradient flow of `-𝖤_β` run at speed
`n`**: `θ(t) = Φ^{n t}(θ(0))`.  The curve `s ↦ θ(s/n)` solves
`ẋ = ∇𝖤_β(x) = -∇(-𝖤_β)(x)`, as `angularUSA_is_gradient_flow` says up to the
factor `n`, and the gradient of `-𝖤_β` is Lipschitz and bounded
(`Landscape.SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic`), so this
equation has one solution through each point. -/
theorem angularUSA_eq_gradientFlow {β : ℝ} (hβ : β ≠ 0) (hn : 0 < n) {θ : ℝ → Angles n}
    (hθ : angularUSA n β θ) (t : ℝ) :
    WithLp.toLp 2 (θ t) = Landscape.SaddleAvoidance.gradientFlow (fun ψ => -eucTorusEnergy n β ψ) (n * t)
      (WithLp.toLp 2 (θ 0)) := by
  have hf : ContDiff ℝ 2 fun ψ => -eucTorusEnergy n β ψ := (contDiff_eucTorusEnergy n β).neg
  obtain ⟨L, hM, -, hL⟩ := Landscape.SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic hf
    (span_periods n) (by rintro _ ⟨k, rfl⟩ y; simp only [eucTorusEnergy_add_period])
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn.ne'
  have hγ : ∀ s, HasDerivAt (fun s => WithLp.toLp 2 (θ (s / n)))
      (-gradient (fun ψ => -eucTorusEnergy n β ψ) (WithLp.toLp 2 (θ (s / n)))) s := by
    intro s
    have hcomp : HasDerivAt (fun s => θ (s / n)) (fun i => (1 / (n : ℝ)) * (-(1 / (n : ℝ)) *
        ∑ j, exp (β * cos (θ (s / n) i - θ (s / n) j)) * sin (θ (s / n) i - θ (s / n) j))) s :=
      hasDerivAt_pi.2 fun i => by
        have h := (hθ (s / n) i).comp s ((hasDerivAt_id s).div_const (n : ℝ))
        exact h.congr_deriv (by ring)
    refine ((PiLp.hasFDerivAt_toLp 2 (θ (s / n))).comp_hasDerivAt s hcomp).congr_deriv ?_
    ext i
    have hodd : ∀ j, exp (β * cos (θ (s / n) j - θ (s / n) i)) * sin (θ (s / n) j - θ (s / n) i) =
        -(exp (β * cos (θ (s / n) i - θ (s / n) j)) * sin (θ (s / n) i - θ (s / n) j)) := fun j => by
      rw [← neg_sub, cos_neg, sin_neg, mul_neg]
    rw [gradient_neg_eucTorusEnergy, neg_neg, gradient_eucTorusEnergy_apply n β hβ,
      Finset.sum_congr rfl fun j _ => hodd j, Finset.sum_neg_distrib]
    simp
    ring
  have h := (Landscape.SaddleAvoidance.isFlowField_neg_gradient hL hM).flow_eq hγ 0 (n * t)
  simp only [zero_div, mul_div_cancel_left₀ _ hn'] at h
  exact h.symm

/-- **The gradient flow of `-𝖤_β` run at speed `n` solves `eq:onangles`**
from every initial angles `θ₀`: its velocity `n ∇𝖤_β` has the coordinates
`n · (1/n²) Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i)`, the right-hand side of
`eq:onangles`. -/
theorem angularUSA_gradientFlow {β : ℝ} (hβ : β ≠ 0) (θ₀ : Angles n) :
    angularUSA n β fun t => (Landscape.SaddleAvoidance.gradientFlow (fun ψ => -eucTorusEnergy n β ψ)
      (n * t) (WithLp.toLp 2 θ₀)).ofLp := by
  intro t i
  obtain ⟨L, hM, -, hL⟩ := Landscape.SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic
    (contDiff_eucTorusEnergy n β).neg (span_periods n)
    (by rintro _ ⟨k, rfl⟩ y; simp only [eucTorusEnergy_add_period])
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fin.pos i).ne'
  set y := Landscape.SaddleAvoidance.gradientFlow (fun ψ => -eucTorusEnergy n β ψ) (n * t) (WithLp.toLp 2 θ₀)
  have hodd : ∀ j, exp (β * cos (y.ofLp j - y.ofLp i)) * sin (y.ofLp j - y.ofLp i) =
      -(exp (β * cos (y.ofLp i - y.ofLp j)) * sin (y.ofLp i - y.ofLp j)) := fun j => by
    rw [← neg_sub, cos_neg, sin_neg, mul_neg]
  refine ((PiLp.hasFDerivAt_apply 2 _ i).comp_hasDerivAt t
    ((Landscape.SaddleAvoidance.hasDerivAt_gradientFlow hL hM (n * t) (WithLp.toLp 2 θ₀)).scomp t
      ((hasDerivAt_id t).const_mul (n : ℝ)))).congr_deriv ?_
  rw [PiLp.proj_apply, WithLp.ofLp_smul, Pi.smul_apply, gradient_neg_eucTorusEnergy, neg_neg,
    gradient_eucTorusEnergy_apply n β hβ, Finset.sum_congr rfl fun j _ => hodd j,
    Finset.sum_neg_distrib, smul_eq_mul]
  field_simp
  rfl

/-- **`-𝖤_β` is analytic**: a sum of exponentials of cosines of differences
of coordinates. -/
theorem analyticAt_neg_eucTorusEnergy (β : ℝ) (ψ : EucSpace n) :
    AnalyticAt ℝ (fun φ => -eucTorusEnergy n β φ) ψ := by
  have h : ∀ i : Idx n, AnalyticAt ℝ (fun φ : EucSpace n => φ.ofLp i) ψ := fun i =>
    (EuclideanSpace.proj i : EucSpace n →L[ℝ] ℝ).analyticAt ψ
  unfold eucTorusEnergy torusEnergy
  fun_prop

/-- **`-𝖤_β` satisfies the Łojasiewicz inequality at every point**, with an
exponent `0 ≤ a < 1`: it is analytic (`analyticAt_neg_eucTorusEnergy`), and
analytic functions do (`Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`,
Łojasiewicz's theorem). -/
theorem exists_isLojasiewiczAt_neg_eucTorusEnergy (β : ℝ) (z : EucSpace n) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ Landscape.SaddleAvoidance.IsLojasiewiczAt (fun φ => -eucTorusEnergy n β φ) z a :=
  Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt (analyticAt_neg_eucTorusEnergy n β z)

/-- **Almost every solution of `eq:onangles` synchronizes**, for every
`β > 0`: for Lebesgue-almost every initial angle vector `θ₀ ∈ ℝⁿ` there is a
`θ'` with `θ'_i - θ'_j ∈ 2πℤ` to which every solution with `θ(0) = θ₀`
converges.  A solution exists (`angularUSA_gradientFlow`).

Source: arXiv:2312.10794v5, §7, after `o:strictsaddle`: a positive answer to
that problem "would imply that for all initial conditions except a set of
measure zero, all `θ_i(t)` converge under the dynamics `eq:onangles` to a
common limit as `t → +∞`", by the arguments of Appendix A (`l:nosaddleconv`
and the Łojasiewicz theorem).  The answer is `strict_saddle` (arXiv:2507.22857v1,
§5), and the Łojasiewicz inequality is
`exists_isLojasiewiczAt_neg_eucTorusEnergy`.  The common limit on `𝕋` is a
limit `θ'` in `ℝⁿ` whose coordinates agree modulo `2π`. -/
theorem ae_tendsto_synchronized_of_angularUSA {β : ℝ} (hβ : 0 < β) :
    ∀ᵐ θ₀ : Angles n, ∃ θ' : Angles n, (∀ i j, ∃ k : ℤ, θ' i - θ' j = k * (2 * π)) ∧
      ∀ θ : ℝ → Angles n, angularUSA n β θ → θ 0 = θ₀ → Tendsto θ atTop (𝓝 θ') := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · refine Eventually.of_forall fun θ₀ => ⟨θ₀, fun i => i.elim0, fun θ _ _ => ?_⟩
    rw [show θ = fun _ => θ₀ from funext fun _ => Subsingleton.elim _ _]
    exact tendsto_const_nhds
  have hae := Landscape.SaddleAvoidance.ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top volume
    Measure.AbsolutelyContinuous.rfl (contDiff_eucTorusEnergy n β).neg (span_periods n)
    (by rintro _ ⟨k, rfl⟩ y; simp only [eucTorusEnergy_add_period])
    (fun z hz => isMinOn_or_isStrictSaddle_neg_eucTorusEnergy n hβ hz) fun z _ =>
      exists_isLojasiewiczAt_neg_eucTorusEnergy n β z
  filter_upwards [(PiLp.volume_preserving_toLp (Idx n)).quasiMeasurePreserving.ae hae]
    with θ₀ ⟨w, hw, hlim⟩
  refine ⟨w.ofLp, synchronized_of_isGlobalMaxTorusEnergy n hβ fun φ => ?_, fun θ hθ h0 => ?_⟩
  · simpa [eucTorusEnergy] using isMinOn_univ_iff.1 hw (WithLp.toLp 2 φ)
  · have hlim' := ((PiLp.continuous_ofLp 2 _).tendsto w).comp
      (hlim.comp (tendsto_id.const_mul_atTop (Nat.cast_pos.2 hn)))
    refine hlim'.congr fun t => ?_
    simp only [Function.comp_apply, id]
    rw [← h0, ← angularUSA_eq_gradientFlow n hβ.ne' hn hθ t, WithLp.ofLp_toLp]

/-- The hypotheses of `isMinOn_or_isStrictSaddle_neg_eucTorusEnergy`,
`angularUSA_eq_gradientFlow`, `angularUSA_gradientFlow` and
`ae_tendsto_synchronized_of_angularUSA` hold for two tokens at `β = 1`:
`∇(-𝖤_1)` vanishes at the synchronized state `0`, and `angularUSA_gradientFlow`
gives a solution of `eq:onangles` from `θ₀ = 0`. -/
example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ 0 < 2 ∧
    gradient (fun ψ => -eucTorusEnergy 2 1 ψ) 0 = 0 ∧ angularUSA 2 1 fun t =>
      (Landscape.SaddleAvoidance.gradientFlow (fun ψ => -eucTorusEnergy 2 1 ψ) (2 * t) (WithLp.toLp 2 0)).ofLp :=
  ⟨one_ne_zero, one_pos, two_pos, (gradient_neg_eucTorusEnergy_eq_zero_iff 2 one_ne_zero 0).2
    (isCriticalTorusEnergy_of_synchronized 2 one_ne_zero fun _ _ => ⟨0, by simp⟩),
    by simpa only [Nat.cast_ofNat] using angularUSA_gradientFlow 2 one_ne_zero 0⟩

end Perspective
end Transformer

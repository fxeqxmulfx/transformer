/-
# A single cluster on the circle for every β ≥ 0

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §4,
`thm: beta.tiny`, last sentence.

On `𝕊¹` the solutions of `USA` and `SA` in angles give solutions on the
sphere (`usa_anglesToSphere`, `sa_anglesToSphere`); they exist from every
start (`exists_angularUSA`, `exists_angularSA`), and the solution on the
sphere is unique (`USA_unique`, `SA_unique`).  If the angles converge to a
synchronized `θ'`, whose points `(cos θ'_i, sin θ'_i)` all coincide, every
token converges to that point (`exists_tendsto_of_angles`).  For `β > 0`
almost every solution in angles does (`ae_tendsto_synchronized_of_angularUSA`,
`ae_tendsto_synchronized_of_angularSA`), and a uniform tuple law on the circle
charges no Lebesgue-null set of angles (`ae_of_ae_volume`): almost every
initial tuple clusters (`ae_mem_clusteringSet_circle`).  At `β = 0` both
dynamics are that of `beta0_consensus` (`beta0Dynamics_of_SA`,
`beta0Dynamics_of_USA`).  So the conclusion of `thm: beta.tiny` holds on the
circle for every `β ≥ 0` (`clustering_circle`), in particular for `β ≤ 1`
(`beta_tiny_circle`).
-/

import Transformer.Perspective.AppendixA_Saddle
import Transformer.Perspective.CircleNull
import Transformer.Perspective.CircleSAConvergence
import Transformer.Perspective.Section2_SphereFlow

open scoped BigOperators Topology
open Real Filter MeasureTheory

namespace Transformer
namespace Perspective

/-- **The solution of `USA` from a given start is unique**: its field
`Y ↦ (Proj_{Y_i}(n⁻¹ Σ_j e^{β⟨Y_i, Y_j⟩} Y_j))_i` is smooth on all of
`(ℝ^d)ⁿ`, hence locally Lipschitz.

Source: arXiv:2312.10794v5, §2, "the solution to the Cauchy problem" for
`USA`, as `SA_unique` for `SA`. -/
theorem USA_unique {d n : ℕ} (β : ℝ) (X Y : ℝ → SphereTuple d n) (hX : USA d n β X)
    (hY : USA d n β Y) (h0 : X 0 = Y 0) : X = Y := by
  let F : (Idx n → EucSpace d) → Idx n → EucSpace d := fun Z i =>
    proj d (Z i) ((n : ℝ)⁻¹ • ∑ j, exp (β * inner ℝ (Z i) (Z j)) • Z j)
  have hF : ContDiff ℝ 1 F := by
    refine contDiff_pi.2 fun i => ?_
    have hc : ∀ j, ContDiff ℝ 1 fun Z : Idx n → EucSpace d => Z j := contDiff_apply ℝ _
    have hv : ContDiff ℝ 1 fun Z : Idx n → EucSpace d =>
        (n : ℝ)⁻¹ • ∑ j, exp (β * inner ℝ (Z i) (Z j)) • Z j :=
      (ContDiff.sum fun j _ =>
        (contDiff_const.mul ((hc i).inner ℝ (hc j))).exp.fun_smul (hc j)).const_smul _
    exact hv.sub (((hc i).inner ℝ hv).fun_smul (hc i))
  have hpath (W : ℝ → SphereTuple d n) (hW : USA d n β W) (t : ℝ) :
      HasDerivAt (fun s i => (W s i : EucSpace d)) (F fun i => (W t i : EucSpace d)) t :=
    hasDerivAt_pi.2 fun i => hW t i
  have heq := Landscape.GlobalFlow.eq_of_hasDerivAt hF.locallyLipschitz (hpath X hX) (hpath Y hY)
    (by rw [h0])
  funext t i
  exact Subtype.ext (congrFun (congrFun heq t) i)

/-- **At `β = 0`, `SA` is the dynamics of `beta0_consensus`**: every weight
`e^{0}` is `1`, so `Z_i = n`. -/
theorem beta0Dynamics_of_SA {d n : ℕ} {X : ℝ → SphereTuple d n} (hX : SA d n 0 X) :
    beta0Dynamics d n X := fun t i => by
  simpa [partitionSA] using hX t i

/-- **At `β = 0`, `USA` is the dynamics of `beta0_consensus`**: every weight
`e^{0}` is `1`. -/
theorem beta0Dynamics_of_USA {d n : ℕ} {X : ℝ → SphereTuple d n} (hX : USA d n 0 X) :
    beta0Dynamics d n X := fun t i => by
  simpa using hX t i

variable (n : ℕ)

/-- **`eq:onangles` has a solution from every start**, for `β ≠ 0`: the
gradient flow of `-𝖤_β` run at speed `n` (`angularUSA_gradientFlow`). -/
theorem exists_angularUSA {β : ℝ} (hβ : β ≠ 0) (θ₀ : Angles n) :
    ∃ θ : ℝ → Angles n, angularUSA n β θ ∧ θ 0 = θ₀ := by
  obtain ⟨L, hM, -, hL⟩ := Landscape.SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic
    (contDiff_eucTorusEnergy n β).neg (span_periods n)
    (by rintro _ ⟨k, rfl⟩ y; simp only [eucTorusEnergy_add_period])
  refine ⟨_, angularUSA_gradientFlow n hβ θ₀, ?_⟩
  simp [Landscape.SaddleAvoidance.gradientFlow, (Landscape.SaddleAvoidance.isFlowField_neg_gradient hL hM).flow_self]

/-- **Tokens whose angles synchronize converge to one point.**  If a dynamics
`S` on `(𝕊¹)ⁿ` has at most one solution from each start, and the angles `θ`
of a solution converge to a `θ'` whose coordinates agree modulo `2π`, then
every solution from `θ(0)` converges to the common point
`(cos θ'_i, sin θ'_i)`. -/
theorem exists_tendsto_of_angles {S : (ℝ → SphereTuple 2 n) → Prop}
    (hS : ∀ X Y, S X → S Y → X 0 = Y 0 → X = Y) {θ : ℝ → Angles n}
    (hθ : S fun t => anglesToSphere (θ t)) {θ' : Angles n}
    (hsync : ∀ i j, ∃ k : ℤ, θ' i - θ' j = k * (2 * π)) (hlim : Tendsto θ atTop (𝓝 θ')) :
    ∃ x_star : SSphere 2, ∀ X : ℝ → SphereTuple 2 n, X 0 = anglesToSphere (θ 0) → S X →
      ∀ i, Tendsto (fun t => (X t i : EucSpace 2) - x_star) atTop (𝓝 0) := by
  rcases isEmpty_or_nonempty (Idx n) with hn | ⟨⟨i₀⟩⟩
  · exact ⟨circlePoint 0, fun _ _ _ i => (IsEmpty.false i).elim⟩
  refine ⟨circlePoint (θ' i₀), fun X hX0 hX i => ?_⟩
  rw [hS X _ hX hθ hX0]
  obtain ⟨k, hk⟩ := hsync i i₀
  have hpt : circlePoint (θ' i) = circlePoint (θ' i₀) := by
    rw [show θ' i = θ' i₀ + k * (2 * π) by linarith]
    exact circlePoint_periodic.int_mul k _
  rw [← hpt]
  exact tendsto_sub_nhds_zero_iff.2 ((continuous_subtype_val.tendsto _).comp
    ((continuous_circlePoint.tendsto _).comp (((continuous_apply i).tendsto _).comp hlim)))

/-- **For `β > 0`, almost every initial tuple on the circle clusters**, for
`SA` and for `USA`: almost every solution in angles synchronizes
(`ae_tendsto_synchronized_of_angularSA`,
`ae_tendsto_synchronized_of_angularUSA`), and the tokens follow their angles
(`exists_tendsto_of_angles`). -/
theorem ae_mem_clusteringSet_circle {β : ℝ} (hβ : 0 < β) {P : Measure (SphereTuple 2 n)}
    (hP : UniformTuple 2 n P) :
    ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet 2 n β ∧ X₀ ∈ clusteringSetUSA 2 n β := by
  refine ae_of_ae_volume hP ?_
  filter_upwards [ae_tendsto_synchronized_of_angularSA n hβ,
    ae_tendsto_synchronized_of_angularUSA n hβ] with θ₀ ⟨θ', hs, hlim⟩ ⟨θ'', hs', hlim'⟩
  obtain ⟨θ, hθ, h0⟩ := exists_angularSA n hβ.ne' θ₀
  obtain ⟨ψ, hψ, hψ0⟩ := exists_angularUSA n hβ.ne' θ₀
  have hSA := exists_tendsto_of_angles n (fun X Y hX hY h => SA_unique β X Y hX hY h)
    (sa_anglesToSphere n hθ) hs (hlim θ hθ h0)
  have hUSA := exists_tendsto_of_angles n (fun X Y hX hY h => USA_unique β X Y hX hY h)
    (usa_anglesToSphere hψ) hs' (hlim' ψ hψ hψ0)
  rw [h0] at hSA
  rw [hψ0] at hUSA
  exact ⟨hSA, hUSA⟩

/-- **On the circle, almost every initial tuple clusters for every `β ≥ 0`**,
for `SA` and for `USA`: at `β = 0` by `beta0_consensus`, for `β > 0` by
`ae_mem_clusteringSet_circle`.

For `β ≥ -0.16` this is arXiv:2507.22857v1, §2, `cor:conjecture` (for
`eq:selfattention`, `USA` in angles up to a change of time scale) and
`cor:conjecturenormal` (for `eq:selfattentionnormal`, `SA` in angles): global
synchronization, that is convergence of all angles to one point modulo `2π`
from almost every start (`def:synchronization`). -/
theorem clustering_circle {β : ℝ} (hβ : 0 ≤ β) (P : Measure (SphereTuple 2 n))
    (hP : UniformTuple 2 n P) :
    ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet 2 n β ∧ X₀ ∈ clusteringSetUSA 2 n β := by
  rcases hβ.eq_or_lt with rfl | hβ
  · filter_upwards [beta0_consensus 2 n le_rfl P hP] with X₀ ⟨x_star, hx⟩
    exact ⟨⟨x_star, fun X hX0 hX => hx X hX0 (beta0Dynamics_of_SA hX)⟩,
      ⟨x_star, fun X hX0 hX => hx X hX0 (beta0Dynamics_of_USA hX)⟩⟩
  · exact ae_mem_clusteringSet_circle n hβ hP

/-- **`thm: beta.tiny` for `d = 2`** (Criscitiello–Boumal 2024).  "Moreover,
when `d = 2` one can take `β ≤ 1` and the same conclusion holds": for
Lebesgue-almost every initial tuple on `𝕊¹` all tokens of the solution of `SA`
(resp. `USA`) converge to one point.  It holds for every `β ≥ 0`
(`clustering_circle`).

Source: arXiv:2312.10794v5, §4, `thm: beta.tiny`, last sentence; credited to
criscitiello2024synchronization.  The source assumes `n ≥ 2`; it is dropped. -/
theorem beta_tiny_circle :
    ∀ β : ℝ, 0 ≤ β → β ≤ 1 →
      ∀ P : Measure (SphereTuple 2 n), UniformTuple 2 n P →
        ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet 2 n β ∧ X₀ ∈ clusteringSetUSA 2 n β :=
  fun _ hβ _ => clustering_circle n hβ

/-- The hypotheses of the theorems above hold for two tokens: the consensus
`θ_i = c` solves `eq:onangles` and SA in angles at every `β`, so the
consensus at `(cos c, sin c)` solves `USA` and `SA` at `β = 1` and at `β = 0`;
`SA` has at most one solution from each start; the constant angles are
synchronized and converge; and the product of circle laws is a uniform tuple
law. -/
example (c : ℝ) :
    let θ : ℝ → Angles 2 := fun _ _ => c
    USA 2 2 1 (fun t => anglesToSphere (θ t)) ∧ SA 2 2 1 (fun t => anglesToSphere (θ t)) ∧
      SA 2 2 0 (fun t => anglesToSphere (θ t)) ∧ USA 2 2 0 (fun t => anglesToSphere (θ t)) ∧
      (∀ X Y, SA 2 2 1 X → SA 2 2 1 Y → X 0 = Y 0 → X = Y) ∧
      (∀ i j, ∃ k : ℤ, θ 0 i - θ 0 j = k * (2 * π)) ∧ Tendsto θ atTop (𝓝 (θ 0)) ∧
      (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
      UniformTuple 2 2 (Measure.pi fun _ : Idx 2 => circleLaw) := by
  intro θ
  have hU : ∀ β, angularUSA 2 β θ := fun β t i => by simpa [θ] using hasDerivAt_const t c
  have hS : ∀ β, angularSA 2 β θ := fun β t i => by simpa [θ] using hasDerivAt_const t c
  exact ⟨usa_anglesToSphere (hU 1), sa_anglesToSphere 2 (hS 1), sa_anglesToSphere 2 (hS 0),
    usa_anglesToSphere (hU 0), fun X Y hX hY h => SA_unique 1 X Y hX hY h,
    fun _ _ => ⟨0, by simp [θ]⟩, tendsto_const_nhds, one_ne_zero, one_pos, zero_le_one, le_rfl,
    uniformTuple_circleLaw 2⟩

end Perspective
end Transformer

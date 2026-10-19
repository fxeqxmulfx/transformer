module

public import Transformer.Perspective.CircleAngles
public import Mathlib.Analysis.Calculus.ContDiff.WithLp

/-!
# SA on the circle in angles

At the points `x_k = (cos θ_k, sin θ_k)` of the circle the partition function
`Z_{β,i}` of `SA` is `Z_i(θ) = Σ_k e^{β cos(θ_i - θ_k)}` (`circlePartition`,
`partitionSA_anglesToSphere`), and the computation of §7.1 that turns `USA`
into `eq:onangles` turns `SA` into

  `θ̇_i = -(1/Z_i) Σ_j e^{β cos(θ_i - θ_j)} sin(θ_i - θ_j)`  (`angularSA`),

`eq:onangles` with `n` replaced by `Z_i`, as `USA` is `SA` with `Z_i` replaced
by `n`. Its solutions give solutions of `SA` on `(𝕊¹)ⁿ`
(`sa_anglesToSphere`), and its right-hand side is `n²/Z_i` times the `i`-th
partial derivative of the energy `𝖤_β` of `eq:energyF`
(`angularSA_is_gradient_flow`): SA in angles is the gradient ascent of
`n² 𝖤_β = (1/2β) Σ_i Σ_j e^{β cos(θ_i - θ_j)}`, the energy of §3.4, in the
metric `diag(Z_i)`, which is `e:scalarproduct` written in angles. The
inverse of the metric `diag(Z_i/n²)` of `𝖤_β`, the gain
`G(θ) = diag(n²/Z_i(θ))` on `EucSpace n` (`saGain`), is smooth, symmetric and
positive definite (`inner_saGain_self_pos`). `Perspective.CircleSAGradient`
states the gradient flow with the metric, and `Perspective.CircleSAFlow`
draws the consequence for strict saddles.

Source: arXiv:2312.10794v5, §3.4 (`e:scalarproduct`), §3.3 (`USA` "is obtained
by replacing the partition function `Z_{β,i}(t)` by `n`") and §7.1
(`eq:onangles`, `eq:energyF`).
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **The partition function of SA on the circle**, in angles:
`Z_i(θ) = Σ_k e^{β cos(θ_i - θ_k)}`. -/
noncomputable def circlePartition (β : ℝ) (θ : Angles n) (i : Idx n) : ℝ :=
  ∑ k : Idx n, Real.exp (β * Real.cos (θ i - θ k))

/-- The partition function is positive: the term `k = i` alone is. -/
theorem circlePartition_pos (β : ℝ) (θ : Angles n) (i : Idx n) :
    0 < circlePartition n β θ i :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩

/-- **SA in angles**:

  `θ̇_i(t) = -(1/Z_i(θ(t))) Σ_j e^{β cos(θ_i(t) - θ_j(t))} sin(θ_i(t) - θ_j(t))`,

the angular equation `eq:onangles` with the factor `1/n` replaced by
`1/Z_i`. -/
def angularSA (β : ℝ) (θ : ℝ → Angles n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => θ s i)
      (-(circlePartition n β (θ t) i)⁻¹ *
        ∑ j : Idx n, Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j)) t

/-- On the circle the partition function of `SA` is `circlePartition`:
`⟨x_i, x_k⟩ = cos (θ_i - θ_k)`. -/
theorem partitionSA_anglesToSphere (β : ℝ) (θ : ℝ → Angles n) (t : ℝ) (i : Idx n) :
    partitionSA 2 n β (fun t => anglesToSphere (θ t)) t i = circlePartition n β (θ t) i := by
  simp only [partitionSA, circlePartition, anglesToSphere, inner_circlePoint]

/-- **SA in angles gives SA on the circle.**  If the angles `θ(t)` solve
`angularSA` with parameter `β`, the points `(cos θ_i(t), sin θ_i(t))` solve
`SA` on `(𝕊¹)ⁿ` with the same `β`: the computation of `usa_anglesToSphere`,
with `1/n` replaced by `1/Z_i`. -/
theorem sa_anglesToSphere {β : ℝ} {θ : ℝ → Angles n} (hθ : angularSA n β θ) :
    SA 2 n β (fun t => anglesToSphere (θ t)) := by
  intro t i
  refine (hasDerivAt_circlePoint (hθ t i)).congr_deriv ?_
  rw [partitionSA_anglesToSphere]
  simp only [anglesToSphere, inner_circlePoint, proj_smul_sum, proj_circlePoint, smul_smul,
    ← Finset.sum_smul]
  congr 1
  simp only [mul_neg, Finset.sum_neg_distrib]
  ring

/-- **SA in angles is a gradient ascent of `𝖤_β` in the metric
`diag(Z_i/n²)`**: along a solution,

  `θ̇_i = (n²/Z_i) ∂_i 𝖤_β(θ)`,

with `∂_i 𝖤_β = (1/n²) Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i)`
(`hasDerivAt_torusEnergy`). Equivalently `Z_i θ̇_i = ∂_i (n² 𝖤_β)`: the
gradient of the energy of §3.4 in the metric `diag(Z_i)`. For `USA` the
factor is `n` (`angularUSA_is_gradient_flow`). The energy `𝖤_β` of
`torusEnergy` is `0` for `β = 0`, hence `hβ`. -/
theorem angularSA_is_gradient_flow (β : ℝ) (hβ : β ≠ 0) (θ : ℝ → Angles n)
    (hθ : angularSA n β θ) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => θ s i)
      ((n : ℝ) ^ 2 / circlePartition n β (θ t) i *
        deriv (fun x : ℝ => torusEnergy n β (Function.update (θ t) i x)) (θ t i)) t := by
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fin.pos i).ne'
  have hZ := (circlePartition_pos n β (θ t) i).ne'
  rw [(hasDerivAt_torusEnergy n β hβ (θ t) i).deriv]
  convert hθ t i using 1
  have hodd : ∀ j : Idx n,
      Real.exp (β * Real.cos (θ t j - θ t i)) * Real.sin (θ t j - θ t i)
        = -(Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j)) := by
    intro j
    rw [show θ t j - θ t i = -(θ t i - θ t j) by ring, Real.cos_neg, Real.sin_neg]
    ring
  rw [Finset.sum_congr rfl fun j _ => hodd j, Finset.sum_neg_distrib]
  field_simp

/-- **The gain `G(θ) = diag(n²/Z_i(θ))`**, the inverse of the metric
`diag(Z_i/n²)` in which SA is the gradient ascent of `𝖤_β`. -/
noncomputable def saGain (β : ℝ) (ψ : EucSpace n) : EucSpace n →L[ℝ] EucSpace n :=
  ∑ i : Idx n, ((n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i) •
    (innerSL ℝ (EuclideanSpace.single i (1 : ℝ))).smulRight (EuclideanSpace.single i (1 : ℝ))

theorem saGain_apply (β : ℝ) (ψ v : EucSpace n) (i : Idx n) :
    (saGain n β ψ v).ofLp i = (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i * v.ofLp i := by
  simp [saGain, EuclideanSpace.inner_single_left, Pi.single_apply]

/-- `⟨G u, v⟩ = Σ_i (n²/Z_i) u_i v_i`: `G` is symmetric, and positive definite. -/
theorem inner_saGain (β : ℝ) (ψ u v : EucSpace n) :
    inner ℝ (saGain n β ψ u) v =
      ∑ i : Idx n, (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i * u.ofLp i * v.ofLp i := by
  simp only [PiLp.inner_apply, saGain_apply, RCLike.inner_apply, conj_trivial]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- The gain is symmetric. -/
theorem isSymmetric_saGain (β : ℝ) (ψ : EucSpace n) :
    (saGain n β ψ : EucSpace n →ₗ[ℝ] EucSpace n).IsSymmetric := fun u v => by
  change inner ℝ (saGain n β ψ u) v = inner ℝ u (saGain n β ψ v)
  rw [inner_saGain, real_inner_comm, inner_saGain]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- The gain is positive definite: `⟨G v, v⟩ = Σ_i (n²/Z_i) v_i² > 0` for
`v ≠ 0`. -/
theorem inner_saGain_self_pos (β : ℝ) (ψ : EucSpace n) {v : EucSpace n} (hv : v ≠ 0) :
    0 < inner ℝ (saGain n β ψ v) v := by
  obtain ⟨i, hi⟩ : ∃ i, v.ofLp i ≠ 0 := by
    by_contra h
    push Not at h
    exact hv (PiLp.ext fun i => by simpa using h i)
  have hn : (0 : ℝ) < (n : ℝ) ^ 2 := pow_pos (Nat.cast_pos.2 (Fin.pos i)) 2
  have hc : ∀ j, 0 < (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp j := fun j =>
    div_pos hn (circlePartition_pos n β _ j)
  rw [inner_saGain]
  refine Finset.sum_pos' (fun j _ => ?_) ⟨i, Finset.mem_univ i, ?_⟩
  · rw [mul_assoc]
    exact mul_nonneg (hc j).le (mul_self_nonneg _)
  · rw [mul_assoc]
    exact mul_pos (hc i) (mul_self_pos.2 hi)

/-- The gain is smooth: `Z_i` is smooth and positive. -/
theorem contDiff_saGain (β : ℝ) {k : WithTop ℕ∞} : ContDiff ℝ k (saGain n β) := by
  have hZ : ∀ i : Idx n, ContDiff ℝ k fun ψ : EucSpace n => circlePartition n β ψ.ofLp i := by
    intro i
    unfold circlePartition
    fun_prop
  unfold saGain
  exact ContDiff.sum fun i _ => (contDiff_const.div (hZ i) fun ψ =>
    (circlePartition_pos n β _ i).ne').smul contDiff_const

/-- The hypotheses of `sa_anglesToSphere` and `angularSA_is_gradient_flow`
hold for every consensus `θ_i = c`, with `β = 1`: every sine `sin 0`
vanishes, so the constant curve solves `angularSA`. That of
`inner_saGain_self_pos` holds for `v = e_0` in `ℝ¹`. -/
example (c : ℝ) : angularSA n 1 (fun _ _ => c) ∧ (1 : ℝ) ≠ 0 ∧
    EuclideanSpace.single (0 : Fin 1) (1 : ℝ) ≠ 0 := by
  refine ⟨fun t i => ?_, one_ne_zero, fun h => ?_⟩
  · simpa using hasDerivAt_const t c
  · simpa using congrArg (fun v : EucSpace 1 => v.ofLp 0) h

end Perspective
end Transformer

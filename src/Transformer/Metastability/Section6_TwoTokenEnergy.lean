/-
# The actual two-token energy in the staircase problem

ArXiv:2410.06833v1, §6, `conj: saddle-to-saddle`, with the normalized
energy `2β Eβ` used in the meaningful full-trajectory formulation.
Start `SA` from the two orthogonal basis tokens on the circle. Its initial
normalized energy is `(1+exp(-β))/2`, tending to `1/2` as `β → ∞`.
For each fixed nonzero `β`, its long-time normalized energy tends to `1`.

The trajectories are the actual global solutions constructed in
`Perspective.Section2_SphereFlow`. The long-time limit follows from the
proved hemisphere-collapse theorem, without any staircase conjecture.
-/

import Transformer.Metastability.Section4_EnergyGap
import Transformer.Perspective.Section2_FlowContinuity
import Transformer.Perspective.Section6_GramFlow
import Transformer.Perspective.Section5_ExpRate

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Metastability

/-- The normalized interaction energy of two actual unit tokens is
`(1+exp(β(⟪x₀,x₁⟫-1)))/2`. Source: arXiv:2410.06833v1, §1,
`eq: interaction.energy`, applied to the two-token case of §6. -/
theorem normalized_energy_two {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (X : SphereTuple d 2) :
    2 * β * Eβ d 2 β X =
      (1 + Real.exp (β * (inner ℝ (X 0 : EucSpace d) (X 1 : EucSpace d) - 1))) / 2 := by
  have hdiag (i : Idx 2) : inner ℝ (X i : EucSpace d) (X i : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (X i).2]
    norm_num
  simp only [Eβ, Fin.sum_univ_two, hdiag, mul_one, Nat.cast_ofNat]
  rw [real_inner_comm (X 0 : EucSpace d) (X 1 : EucSpace d),
    show β * (inner ℝ (X 0 : EucSpace d) (X 1 : EucSpace d) - 1) =
      β * inner ℝ (X 0 : EucSpace d) (X 1 : EucSpace d) - β by ring,
    Real.exp_sub]
  field_simp
  ring

/-- The normalization hypothesis is satisfiable at positive temperature. -/
example : (1 : ℝ) ≠ 0 := by norm_num

/-- At consensus the normalized two-token energy is exactly one.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy`, and the last
energy level in the staircase diagram of §6. -/
theorem normalized_energy_consensus_two {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (x : SSphere d) : 2 * β * Eβ d 2 β (fun _ : Idx 2 => x) = 1 := by
  rw [normalized_energy_two β hβ, real_inner_self_eq_norm_sq,
    mem_sphere_zero_iff_norm.mp x.2]
  norm_num

/-- A nonzero temperature and a genuine consensus tuple satisfy its input. -/
example : (1 : ℝ) ≠ 0 ∧ ∃ x : SSphere 2, ‖(x : EucSpace 2)‖ = 1 :=
  ⟨by norm_num, basePoint 1, mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

/-- The basis pair's actual initial normalized energy is
`(1+exp(-β))/2`. Source: arXiv:2410.06833v1, §6,
`conj: saddle-to-saddle`, and its energy from §1. -/
theorem normalized_energy_basis_pair (β : ℝ) (hβ : β ≠ 0) :
    2 * β * Eβ 2 2 β (Perspective.sphereBasisConfig 2) = (1 + Real.exp (-β)) / 2 := by
  rw [normalized_energy_two β hβ,
    Perspective.sphereBasisConfig_orthogonal 2 0 1 (by decide)]
  simp

/-- The basis-pair formula's temperature restriction is satisfiable. -/
example : (1 : ℝ) ≠ 0 := by norm_num

/-- The two orthogonal basis tokens lie in the explicit open hemisphere
around `(3/5,4/5)`. Source: arXiv:2410.06833v1, §6, the initial tuples in
`conj: saddle-to-saddle`; arXiv:2312.10794v5, §6.1, hemisphere collapse. -/
theorem basis_pair_in_hemisphere : ∃ w : SSphere 2, ∀ i : Idx 2,
    0 < inner ℝ (Perspective.sphereBasisConfig 2 i : EucSpace 2) (w : EucSpace 2) := by
  let v : EucSpace 2 := EuclideanSpace.single 0 (3 / 5 : ℝ) +
    EuclideanSpace.single 1 (4 / 5 : ℝ)
  have hv : ‖v‖ = 1 := by
    have hs : ‖v‖ ^ 2 = 1 := by
      rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
      norm_num [v, PiLp.single_apply]
    nlinarith [norm_nonneg v]
  refine ⟨⟨v, mem_sphere_zero_iff_norm.mpr hv⟩, ?_⟩
  intro i
  fin_cases i <;>
    norm_num [Perspective.sphereBasisConfig, EuclideanSpace.inner_single_left,
      v, PiLp.single_apply]

/-- The normalized energy along every actual trajectory from the basis
pair is continuous. Source: arXiv:2410.06833v1, §6, the continuous curves
in `conj: saddle-to-saddle`, with the normalization `2β Eβ`. -/
theorem continuous_basis_pair_energy (β : ℝ) :
    Continuous (fun t => 2 * β * Eβ 2 2 β
      (Perspective.saFlow β (Perspective.sphereBasisConfig 2) t)) :=
  ((continuous_energy β).comp
    (Perspective.continuous_saFlow_path β (Perspective.sphereBasisConfig 2))).const_mul (2 * β)

/-- The actual initial normalized energy tends to `1/2` in temperature.
Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`; the two initial
tokens are orthogonal, so only the diagonal terms survive as `β → ∞`. -/
theorem tendsto_basis_pair_initial_energy :
    Tendsto (fun β : ℝ => 2 * β * Eβ 2 2 β
      (Perspective.saFlow β (Perspective.sphereBasisConfig 2) 0)) atTop (nhds (1 / 2)) := by
  have hbase : Tendsto (fun β : ℝ => (1 + Real.exp (-β)) / 2) atTop (nhds (1 / 2)) := by
    convert (tendsto_const_nhds.add tendsto_exp_neg_atTop_nhds_zero).div_const 2 using 1
    norm_num
  refine hbase.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with β hβ
  rw [Perspective.saFlow_zero, normalized_energy_basis_pair β hβ.ne']

/-- **The actual long-time normalized energy tends to `1`.** The initial
basis pair lies in an open hemisphere, so the proved attention dynamics
converges to consensus. Continuity of the finite energy gives its limit.

The source's positive temperatures are included; only `β ≠ 0` is needed
for this normalization. Source: arXiv:2410.06833v1, §6,
`conj: saddle-to-saddle`; arXiv:2312.10794v5, §6.1, hemisphere collapse. -/
theorem tendsto_basis_pair_energy (β : ℝ) (hβ : β ≠ 0) :
    Tendsto (fun t => 2 * β * Eβ 2 2 β
      (Perspective.saFlow β (Perspective.sphereBasisConfig 2) t)) atTop (nhds 1) := by
  obtain ⟨x, C, lam, -, hlam, hbound⟩ :=
    (Perspective.cone_collapse 2 2 β (Perspective.sphereBasisConfig 2)
      basis_pair_in_hemisphere).1
  have hdecay : Tendsto (fun t : ℝ => C * Real.exp (-(lam * t))) atTop (nhds 0) := by
    simpa using (tendsto_exp_neg_atTop_nhds_zero.comp
      (tendsto_id.const_mul_atTop hlam)).const_mul C
  have htokens : ∀ i : Idx 2,
      Tendsto (fun t => Perspective.saFlow β (Perspective.sphereBasisConfig 2) t i)
        atTop (nhds x) := by
    intro i
    apply tendsto_subtype_rng.mpr
    apply Metric.tendsto_nhds.mpr
    intro ε hε
    filter_upwards [hdecay.eventually (gt_mem_nhds hε), eventually_ge_atTop (0 : ℝ)] with t ht ht0
    rw [dist_eq_norm]
    exact (hbound (Perspective.saFlow β (Perspective.sphereBasisConfig 2))
      (Perspective.saFlow_zero β _) (Perspective.saFlow_spec β _).2 i t ht0).trans_lt ht
  have htuple : Tendsto (Perspective.saFlow β (Perspective.sphereBasisConfig 2))
      atTop (nhds (fun _ : Idx 2 => x)) := tendsto_pi_nhds.mpr htokens
  have henergy := (((continuous_energy β).tendsto (fun _ : Idx 2 => x)).comp htuple).const_mul
    (2 * β)
  simpa only [normalized_energy_consensus_two β hβ, Function.comp_apply] using henergy

/-- The long-time theorem's hypothesis is realized by `β=1`, and the actual
trajectory and its prescribed initial value are both available. -/
example : (1 : ℝ) ≠ 0 ∧
    Perspective.saFlow 1 (Perspective.sphereBasisConfig 2) 0 = Perspective.sphereBasisConfig 2 ∧
    Perspective.SA 2 2 1 (Perspective.saFlow 1 (Perspective.sphereBasisConfig 2)) :=
  ⟨by norm_num, Perspective.saFlow_spec 1 _⟩

end Transformer.Metastability

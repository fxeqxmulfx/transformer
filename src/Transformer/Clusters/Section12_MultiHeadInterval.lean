module

public import Transformer.Clusters.Section12_MultiHeadEscape

/-!
# The original multi-head layer factor and rescaled token interval

ArXiv:2305.05465v6, §12, `sec:conclusion`, and §3,
`r:discreterescaling`, the convex-hull interpretation of rescaling.
For scalar value coefficients v_h>0, the actual multi-head layer factor
is q=1+Δt Σv_h. A nonnegative step size makes q positive. Each original
head average stays in the current token interval, so the original
update sends [L q^k,U q^k] into [L q^(k+1),U q^(k+1)].

Consequently, every original coordinate divided by q^k remains in the
original interval, and initial absolute coordinate bounds grow by at
most q^k. The argument allows arbitrary original query-key matrices;
it uses their actual normalized attention averages and assumes no
limiting token configuration or convergence rate.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Clusters

variable {m H : ℕ}

/-- The actual scalar value coordinate is positive under the original value hypothesis.
Source: arXiv:2305.05465v6, §2, `t:boolean`, applied to an original head of §12. -/
theorem value_coordinate_pos (V : ParamMatrix 1) (hV : IsPosDefOp V) : 0 < (V unit1) 0 := by
  have hu : unit1 ≠ 0 := fun he => by simpa using congrArg norm he
  have hv := hV unit1 hu
  rw [eq_smul_unit1 (V unit1), real_inner_smul_left, real_inner_self_eq_norm_sq, norm_unit1] at hv
  simpa using hv

/-- The actual total value coefficient is nonnegative, including an empty sum of heads.
Source: arXiv:2305.05465v6, §12, the sum of the original scalar head values. -/
theorem multiHeadValueCoefficient_nonneg (V : Idx H → ParamMatrix 1) (hV : ∀ h, IsPosDefOp (V h)) :
    0 ≤ multiHeadValueCoefficient V :=
  Finset.sum_nonneg fun h _ => (value_coordinate_pos (V h) (hV h)).le

/-- The exact discrete rescaling factor of the original scalar multi-head update.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, for the head sum of §12. -/
noncomputable def multiHeadLayerFactor (Δt : ℝ) (V : Idx H → ParamMatrix 1) : ℝ :=
  1 + Δt * multiHeadValueCoefficient V

/-- The original layer factor is at least one for nonnegative steps and positive original values.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, with §12's head sum. -/
theorem multiHeadLayerFactor_ge_one {Δt : ℝ} (hΔt : 0 ≤ Δt) (V : Idx H → ParamMatrix 1)
    (hV : ∀ h, IsPosDefOp (V h)) : 1 ≤ multiHeadLayerFactor Δt V := by
  have hv := multiHeadValueCoefficient_nonneg V hV
  have hm := mul_nonneg hΔt hv
  dsimp [multiHeadLayerFactor]
  linarith

/-- A positive step and an actual original positive head give a strictly expanding layer factor.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, for the scalar head sum of §12. -/
theorem multiHeadLayerFactor_gt_one {Δt : ℝ} (hΔt : 0 < Δt) (V : Idx H → ParamMatrix 1)
    (hV : ∀ h, IsPosDefOp (V h)) (h : Idx H) : 1 < multiHeadLayerFactor Δt V := by
  have hsum : 0 < multiHeadValueCoefficient V :=
    (value_coordinate_pos (V h) (hV h)).trans_le
      (Finset.single_le_sum (fun a _ => (value_coordinate_pos (V a) (hV a)).le) (Finset.mem_univ h))
  have hm := mul_pos hΔt hsum
  dsimp [multiHeadLayerFactor]
  linarith

/-- The actual original update preserves the initial interval after its exact scalar layer rescaling.
Source: arXiv:2305.05465v6, §3's discrete convex-hull rescaling, for §12's original equation. -/
theorem multiHeadTransformer_interval_growth {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (L U : ℝ) (hini : ∀ i, X 0 i 0 ∈ Icc L U) :
    ∀ k i, X k i 0 ∈ Icc (L * multiHeadLayerFactor Δt V ^ k) (U * multiHeadLayerFactor Δt V ^ k) := by
  let q := multiHeadLayerFactor Δt V
  have hq : q = 1 + Δt * ∑ h, (V h unit1) 0 := rfl
  change ∀ k i, X k i 0 ∈ Icc (L * q ^ k) (U * q ^ k)
  intro k
  induction k with
  | zero => intro i; simpa only [pow_zero, mul_one] using hini i
  | succ k ih =>
      intro i
      have hd := multiHeadScalarDrift_mem_Icc (Nat.succ_pos m)
        (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1)) (fun h => (V h unit1) 0)
        (fun h => (value_coordinate_pos (V h) (hV h)).le)
        (fun j => X k j 0) (X k i 0) (L * q ^ k) (U * q ^ k) ih
      have hlo := mul_le_mul_of_nonneg_left hd.1 hΔt
      have hhi := mul_le_mul_of_nonneg_left hd.2 hΔt
      rw [multiHeadTransformer_coord hX k i]
      constructor
      · change L * q ^ (k + 1) ≤ _
        rw [pow_succ]
        have he : L * (q ^ k * q) = L * q ^ k + Δt * ((∑ h, (V h unit1) 0) * (L * q ^ k)) := by
          rw [hq]
          ring
        rw [he]
        linarith [(ih i).1]
      · change _ ≤ U * q ^ (k + 1)
        rw [pow_succ]
        have he : U * (q ^ k * q) = U * q ^ k + Δt * ((∑ h, (V h unit1) 0) * (U * q ^ k)) := by
          rw [hq]
          ring
        rw [he]
        linarith [(ih i).2]

/-- Every original normalized coordinate remains inside the original scalar token interval.
Source: arXiv:2305.05465v6, §3, `r:discreterescaling`, for §12's original positive values. -/
theorem multiHeadTransformer_normalized_mem_Icc {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (L U : ℝ) (hini : ∀ i, X 0 i 0 ∈ Icc L U) (k : ℕ) (i : Idx (m + 1)) :
    X k i 0 / multiHeadLayerFactor Δt V ^ k ∈ Icc L U := by
  have hq : 0 < multiHeadLayerFactor Δt V := by
    have hge := multiHeadLayerFactor_ge_one hΔt V hV
    linarith
  have hp : 0 < multiHeadLayerFactor Δt V ^ k := pow_pos hq k
  have hi := multiHeadTransformer_interval_growth hΔt hV hX L U hini k i
  exact ⟨(le_div_iff₀ hp).mpr hi.1, (div_le_iff₀ hp).mpr hi.2⟩

/-- The actual original coordinates obey the full layer-factor absolute bound.
Source: arXiv:2305.05465v6, §7's exponential upper bound, in §3's discrete convention and §12's head sum. -/
theorem multiHeadTransformer_abs_growth {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (R : ℝ) (hini : ∀ i, |X 0 i 0| ≤ R) (k : ℕ) (i : Idx (m + 1)) :
    |X k i 0| ≤ R * multiHeadLayerFactor Δt V ^ k := by
  have hi := multiHeadTransformer_interval_growth hΔt hV hX (-R) R
    (fun j => abs_le.mp (hini j)) k i
  exact abs_le.mpr ⟨by linarith [hi.1], hi.2⟩

/-- Every actual original trajectory has a positive finite prefactor in its full layer-factor bound.
Source: arXiv:2305.05465v6, §7's exponential upper bound, for the actual discrete head sum of §12. -/
theorem multiHeadTransformer_exists_abs_growth {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X) :
    ∃ R : ℝ, 0 < R ∧ ∀ k i, |X k i 0| ≤ R * multiHeadLayerFactor Δt V ^ k := by
  let R := 1 + ∑ j, |X 0 j 0|
  have hsum : 0 ≤ ∑ j, |X 0 j 0| := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hR : 0 < R := by dsimp [R]; linarith
  have hini : ∀ i, |X 0 i 0| ≤ R := by
    intro i
    have hs := Finset.single_le_sum (f := fun j => |X 0 j 0|) (fun j _ => abs_nonneg _) (Finset.mem_univ i)
    dsimp [R]
    linarith
  exact ⟨R, hR, fun k i => multiHeadTransformer_abs_growth hΔt hV hX R hini k i⟩

/-- Two positive original heads and a nonzero interval realize every value,
step, original vector equation, and initial interval premise jointly. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ i, X 0 i 0 ∈ Icc (-1 : ℝ) 1) ∧
    (∀ i, |X 0 i 0| ≤ 1) ∧ multiHeadLayerFactor 1 V = 3 := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  refine ⟨V, multiHeadTrajectory 1 V V V Y, one_pos, fun _ => isPosDefOp_id 1,
    multiHeadTrajectory_solution _ _ _ _ _, ?_, ?_, ?_⟩
  · intro i
    change Y i 0 ∈ Icc (-1 : ℝ) 1
    fin_cases i <;> norm_num [Y, unit1]
  · intro i
    change |Y i 0| ≤ 1
    fin_cases i <;> norm_num [Y, unit1]
  · norm_num [multiHeadLayerFactor, multiHeadValueCoefficient, V, unit1]

end Transformer.Clusters

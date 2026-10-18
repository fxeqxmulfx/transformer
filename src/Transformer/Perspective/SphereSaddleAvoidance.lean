/-
# Almost no rescaled sphere flow converges to an unsynchronized critical point

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

For a start `x ∈ (ℝ^d)^n` with nonzero vectors and a curve `X` of configurations from
`x/‖x‖`, lift `X` to `y(t) = (‖x_i‖ X_i(t))_i`, a curve in the shell of radii `‖x_i‖`.
If the lift solves `ẏ = M(y) ∇g(y)` for a block scaling `M` by positive `C¹` scales, as
`USA` and `SA` do, then for `d ≥ 3`, `β > 0` and almost every `x`, `X` converges to no
critical point of `𝖤_β` with two distinct tokens
(`ae_not_tendsto_nonTrivial_of_blockScale`).  There the lifted limit is a strict saddle
of `-g` (`exists_secondDerivEBetaAt_pos_of_nonTrivial`, `g` being constant along rays),
and the shells `1/(k+2) < ‖x_i‖² < k+2` cover the tuples of nonzero vectors with
countably many applications of `ae_not_tendsto_of_cutoff`.

Source: arXiv:2312.10794v5, §6, `thm: boumal`, and the outline of the gradient-flow
proofs at the end of §5; arXiv:2405.18273, `thm:spheres`, for the strict saddles.
-/

import Transformer.Perspective.SphereCutoffFlow
import Transformer.Perspective.SphereFlowLimit
import Transformer.Perspective.SphereSaddle
import Transformer.Perspective.SphereNull

open Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- Lebesgue measure of `(ℝ^d)^n`, carried to `ℓ²` tuples, is a Haar measure. -/
theorem isAddHaarMeasure_map_toLp :
    (Measure.map (WithLp.toLp 2) (volume : Measure (Idx n → EucSpace d)) :
      Measure (TupleSpace d n)).IsAddHaarMeasure :=
  ContinuousLinearEquiv.isAddHaarMeasure_map
    (PiLp.continuousLinearEquiv 2 ℝ fun _ : Idx n => EucSpace d).symm volume

/-- **Almost every tuple of `(ℝ^d)^n` has no zero vector**, for `d ≥ 1`. -/
theorem ae_forall_ne_zero (hd1 : 1 ≤ d) :
    ∀ᵐ x ∂(volume : Measure (Idx n → EucSpace d)), ∀ i, x i ≠ 0 := by
  have := nontrivial_eucSpace hd1
  refine ae_all_iff.2 fun i => ?_
  rw [ae_iff]
  simp only [not_not]
  rw [volume_pi]
  exact Measure.pi_eval_preimage_null _ (measure_singleton 0)

/-- The hypothesis of `ae_forall_ne_zero` holds for `d = 3`. -/
example : 1 ≤ 3 := by norm_num

/-- **Almost every start avoids the critical points with two distinct tokens.**  For
`d ≥ 3`, `β > 0` and almost every `x ∈ (ℝ^d)^n`: if a curve `X` of configurations from
`x/‖x‖` lifts to a solution of `ẏ = M(y) ∇g(y)` on the radii `‖x_i‖`, `M(y)` the block
scaling by `w(y)`, `w` positive and `C¹` off the zero vectors, then `X` converges to no
critical point of `𝖤_β` with two distinct tokens. -/
theorem ae_not_tendsto_nonTrivial_of_blockScale (hd1 : 1 ≤ d) (hd : 3 ≤ d) {β : ℝ}
    (hβ : 0 < β) {w : TupleSpace d n → Idx n → ℝ}
    (hw : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 1 w x)
    (hwpos : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ∀ i, 0 < w x i) :
    ∀ᵐ x ∂(volume : Measure (Idx n → EucSpace d)), ∀ X : ℝ → SphereTuple d n,
      X 0 = tupleNormalize hd1 x →
      (∀ t, HasDerivAt
        (fun s => (WithLp.toLp 2 fun i => ‖x i‖ • (X s i : EucSpace d) : TupleSpace d n))
        (blockScale (w (WithLp.toLp 2 fun i => ‖x i‖ • (X t i : EucSpace d)))
          (gradient (tupleEnergy hd1 β) (WithLp.toLp 2 fun i => ‖x i‖ • (X t i : EucSpace d))))
        t) →
      ∀ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar → NonTrivialTuple d n Xstar →
        ¬∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)) := by
  have := isAddHaarMeasure_map_toLp (d := d) (n := n)
  have hg : ∀ x : TupleSpace d n, (∀ i, x.ofLp i ≠ 0) → ContDiffAt ℝ 2 (tupleEnergy hd1 β) x :=
    fun x hx => (analyticOnNhd_tupleEnergy hd1 β x hx).contDiffAt
  have hε : ∀ k : ℕ, (0 : ℝ) < 1 / (k + 2) := fun k => by positivity
  have hK : ∀ k : ℕ, (1 : ℝ) / (k + 2) < k + 2 := fun k => by
    rw [div_lt_iff₀ (by positivity)]
    nlinarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)]
  have hall := ae_all_iff.2 fun k : ℕ => ae_not_tendsto_of_cutoff
    (Measure.map (WithLp.toLp 2) (volume : Measure (Idx n → EucSpace d)))
    (isOpen_blockShell _ _) (contDiff_blockCutoff (hε k) (hK k))
    (hasCompactSupport_blockCutoff (hε k) (hK k)) (tsupport_blockCutoff_subset (hε k) (hK k))
    (fun _ hx => blockCutoff_eq_one (hε k) (hK k) hx) hg hw hwpos
  have hvol := ae_of_ae_map (WithLp.measurable_toLp 2 _).aemeasurable hall
  filter_upwards [hvol, ae_forall_ne_zero hd1] with x hx hx0
  intro X hX0 hX Xstar hcrit hnt hlim
  have hr : ∀ i, 0 < ‖x i‖ := fun i => norm_pos_iff.2 (hx0 i)
  obtain ⟨k, hk⟩ : ∃ k : ℕ, ∀ i, 1 / ((k : ℝ) + 2) < ‖x i‖ ^ 2 ∧ ‖x i‖ ^ 2 < k + 2 := by
    obtain ⟨k, hk⟩ := exists_nat_gt (∑ i, (‖x i‖ ^ 2 + (‖x i‖ ^ 2)⁻¹))
    refine ⟨k, fun i => ?_⟩
    have hsq : 0 < ‖x i‖ ^ 2 := pow_pos (hr i) 2
    have hle : ‖x i‖ ^ 2 + (‖x i‖ ^ 2)⁻¹ ≤ ∑ j, (‖x j‖ ^ 2 + (‖x j‖ ^ 2)⁻¹) :=
      Finset.single_le_sum (f := fun j => ‖x j‖ ^ 2 + (‖x j‖ ^ 2)⁻¹)
        (fun j _ => by positivity) (Finset.mem_univ i)
    have hinv : 0 < (‖x i‖ ^ 2)⁻¹ := inv_pos.2 hsq
    refine ⟨?_, by linarith⟩
    rw [one_div]
    exact inv_lt_of_inv_lt₀ hsq (by linarith)
  have hshell : ∀ U : SphereTuple d n, (WithLp.toLp 2 fun i => ‖x i‖ • (U i : EucSpace d) :
      TupleSpace d n) ∈ blockShell (1 / ((k : ℝ) + 2)) (k + 2) := fun U i => by
    change _ < ‖‖x i‖ • (U i : EucSpace d)‖ ^ 2 ∧ ‖‖x i‖ • (U i : EucSpace d)‖ ^ 2 < _
    rw [norm_smul, norm_norm, mem_sphere_zero_iff_norm.mp (U i).2, mul_one]
    exact hk i
  have hY0 : (WithLp.toLp 2 fun i => ‖x i‖ • (X 0 i : EucSpace d) : TupleSpace d n) =
      WithLp.toLp 2 x := by
    rw [hX0]
    congr 1
    funext i
    change ‖x i‖ • (sphereNormalize hd1 (x i) : EucSpace d) = x i
    rw [coe_sphereNormalize hd1 (hx0 i), smul_smul, mul_inv_cancel₀ (hr i).ne', one_smul]
  set z : TupleSpace d n := WithLp.toLp 2 fun i => ‖x i‖ • (Xstar i : EucSpace d)
  have hgz : gradient (tupleEnergy hd1 β) z = 0 := by
    rw [gradient_tupleEnergy_eq_zero_iff hd1 hβ.ne' fun i =>
      smul_ne_zero (hr i).ne' (ne_zero_of_mem_unit_sphere _)]
    rwa [show z.ofLp = fun i => ‖x i‖ • (Xstar i : EucSpace d) from rfl,
      tupleNormalize_smul hd1 hr]
  obtain ⟨v, L, c, hv, hL, ⟨f', h1, h2⟩, hc⟩ :=
    exists_secondDerivEBetaAt_pos_of_nonTrivial d n hd β hβ Xstar hcrit hnt
  have hline : ∀ u : ℝ, tupleEnergy hd1 β (z + u • WithLp.toLp 2 fun i => ‖x i‖ • v i) =
      selfEnergy d n β (L u) := fun u => by
    simp only [tupleEnergy]
    congr 1
    funext i
    apply Subtype.ext
    have hne := add_smul_ne_zero d (mem_sphere_zero_iff_norm.mp (Xstar i).2) (hv i) u
    have heq : (z + u • WithLp.toLp 2 fun i => ‖x i‖ • v i).ofLp i =
        ‖x i‖ • ((Xstar i : EucSpace d) + u • v i) := by
      simp only [z, WithLp.ofLp_add, WithLp.ofLp_smul, Pi.add_apply, Pi.smul_apply, smul_add,
        smul_comm u (‖x i‖)]
    change (sphereNormalize hd1 _ : EucSpace d) = _
    rw [heq, coe_sphereNormalize hd1 (smul_ne_zero (hr i).ne' hne), hL u i, norm_smul,
      norm_norm, smul_smul]
    congr 1
    field_simp [(hr i).ne', norm_ne_zero_iff.2 hne]
  have hYlim : Tendsto (fun t => (WithLp.toLp 2 fun i => ‖x i‖ • (X t i : EucSpace d) :
      TupleSpace d n)) atTop (𝓝 z) :=
    ((PiLp.continuous_toLp 2 _).tendsto _).comp
      (tendsto_pi_nhds.2 fun i => (hlim i).const_smul (‖x i‖))
  exact hx k _ hY0 (fun t => hshell (X t)) hX z (hshell Xstar) hgz _ f' c hc
    (fun s => (h1 s).congr_of_eventuallyEq (Eventually.of_forall hline)) h2 hYlim

/-- The hypotheses of `ae_not_tendsto_nonTrivial_of_blockScale` hold for `d = 3`,
`β = 1` and the scales `w = 1`. -/
example := ae_not_tendsto_nonTrivial_of_blockScale (d := 3) (n := 2) (by norm_num) le_rfl
  one_pos (w := fun _ _ => 1) (fun _ _ => contDiffAt_const) fun _ _ _ => one_pos

end Perspective
end Transformer

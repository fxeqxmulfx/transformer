/-
# The emergence of clusters in self-attention dynamics — the attention kernel
  is Lipschitz in the measure

`e:lipinmu` of arXiv:2305.05465v6, §6: on `B̄(0,R)`,
`‖𝒳[μ] - 𝒳[ν]‖ ≤ C(R) W_2(μ, ν)` for probability measures carried by
`B̄(0,R)`.  The source bounds `W_1` by `W_2` and uses Lipschitz test functions;
that step is `Transformer.Wasserstein.norm_integral_sub_le_W2`.

The source's `L^∞(B(0,R))` is read on the closed ball, which is the same bound
for the continuous `𝒳[μ] - 𝒳[ν]`.

Source: arXiv:2305.05465v6, `lem: vectorfield.properties`, `e:lipinmu`.
-/

import Transformer.Clusters.Section6_Kernel

open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- `exp` is `e^M`-Lipschitz on `(-∞, M]`. -/
theorem abs_exp_sub_exp_le {a b M : ℝ} (ha : a ≤ M) (hb : b ≤ M) :
    |Real.exp a - Real.exp b| ≤ Real.exp M * |a - b| := by
  have key : ∀ a b : ℝ, b ≤ a → a ≤ M → Real.exp a - Real.exp b ≤ Real.exp M * (a - b) := by
    intro a b hba haM
    have h1 := Real.add_one_le_exp (b - a)
    have h2 : Real.exp a * Real.exp (b - a) = Real.exp b := by rw [← Real.exp_add]; ring_nf
    have h3 := Real.exp_le_exp.2 haM
    have h4 := Real.exp_pos a
    nlinarith
  rcases le_total b a with h | h
  · rw [abs_of_nonneg (sub_nonneg.2 (Real.exp_le_exp.2 h)), abs_of_nonneg (sub_nonneg.2 h)]
    exact key a b h ha
  · rw [abs_sub_comm, abs_of_nonneg (sub_nonneg.2 (Real.exp_le_exp.2 h)), abs_sub_comm,
      abs_of_nonneg (sub_nonneg.2 h)]
    exact key b a h hb

/-- On `B̄(0,R)` the exponent of `e:vectorfield` is at most `‖Q‖ ‖K‖ R²` in
absolute value. -/
theorem abs_inner_le_of_mem_closedBall (Q K : ParamMatrix d) {R : ℝ} {x y : EucSpace d}
    (hx : ‖x‖ ≤ R) (hy : ‖y‖ ≤ R) :
    |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖Q‖ * ‖K‖ * R ^ 2 := by
  have hR : 0 ≤ R := (norm_nonneg _).trans hx
  calc |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖Q x‖ * ‖K y‖ := abs_real_inner_le_norm _ _
    _ ≤ (‖Q‖ * R) * (‖K‖ * R) := mul_le_mul
        ((Q.le_opNorm x).trans (mul_le_mul_of_nonneg_left hx (norm_nonneg _)))
        ((K.le_opNorm y).trans (mul_le_mul_of_nonneg_left hy (norm_nonneg _)))
        (norm_nonneg _) (by positivity)
    _ = ‖Q‖ * ‖K‖ * R ^ 2 := by ring

/-- **Estimate (e:lipinmu).**  On `B(0,R)` the kernel is Lipschitz in the
measure for `W_2`, with a constant depending only on `R`.

As in the source: on `B̄(0,R)` both `y ↦ e^{⟨Qx, Ky⟩}` and
`y ↦ e^{⟨Qx, Ky⟩} V y` are Lipschitz, so their integrals move by at most a
constant times `W_2(μ, ν)` (`Wasserstein.norm_integral_sub_le_W2`); the
denominator is at least `e^{-‖Q‖‖K‖R²}`, and the quotient is taken apart as
`D_μ⁻¹ (N_μ - N_ν) + (D_μ⁻¹ - D_ν⁻¹) N_ν`, with `‖N_ν‖ ≤ ‖V‖ R D_ν`.

Source: arXiv:2305.05465v6, `e:lipinmu`. -/
theorem attentionKernel_lipschitz_in_measure (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R) :
    ∃ C : ℝ, 0 < C ∧ ∀ μ ν : Measure (EucSpace d), IsProbabilityMeasure μ →
      IsProbabilityMeasure ν → IsCarriedBy μ R → IsCarriedBy ν R →
        ∀ x ∈ Metric.closedBall (0 : EucSpace d) R,
          ‖attentionKernel Q K V μ x - attentionKernel Q K V ν x‖
            ≤ C * Wasserstein.W2 μ ν := by
  set M := ‖Q‖ * ‖K‖ * R ^ 2
  set E := Real.exp M
  set Lw := E * (‖Q‖ * ‖K‖ * R)
  set Lg := Lw * (‖V‖ * R) + E * ‖V‖
  have hM : 0 ≤ M := by positivity
  refine ⟨E * (Lg + ‖V‖ * R * Lw) + 1, by positivity, ?_⟩
  intro μ ν hμ hν hsμ hsν x hx
  have hx' : ‖x‖ ≤ R := by simpa using hx
  set s := Metric.closedBall (0 : EucSpace d) R
  have hs : MeasurableSet s := Metric.isClosed_closedBall.measurableSet
  set w := fun y => Real.exp (inner (𝕜 := ℝ) (Q x) (K y))
  set g := fun y => w y • V y
  have hwE : ∀ y ∈ s, w y ≤ E := fun y hy => Real.exp_le_exp.2
    ((le_abs_self _).trans (abs_inner_le_of_mem_closedBall Q K hx' (by simpa [s] using hy)))
  have hwL : ∀ y ∈ s, ∀ y' ∈ s, ‖w y - w y'‖ ≤ Lw * dist y y' := by
    intro y hy y' hy'
    rw [Real.norm_eq_abs]
    refine (abs_exp_sub_exp_le
      ((le_abs_self _).trans (abs_inner_le_of_mem_closedBall Q K hx' (by simpa [s] using hy)))
      ((le_abs_self _).trans (abs_inner_le_of_mem_closedBall Q K hx' (by simpa [s] using hy')))).trans ?_
    rw [← inner_sub_right, ← map_sub, dist_eq_norm]
    have hb : |inner (𝕜 := ℝ) (Q x) (K (y - y'))| ≤ ‖Q‖ * ‖K‖ * R * ‖y - y'‖ :=
      calc |inner (𝕜 := ℝ) (Q x) (K (y - y'))| ≤ ‖Q x‖ * ‖K (y - y')‖ := abs_real_inner_le_norm _ _
        _ ≤ (‖Q‖ * R) * (‖K‖ * ‖y - y'‖) := mul_le_mul
            ((Q.le_opNorm x).trans (mul_le_mul_of_nonneg_left hx' (norm_nonneg _)))
            (K.le_opNorm _) (norm_nonneg _) (by positivity)
        _ = ‖Q‖ * ‖K‖ * R * ‖y - y'‖ := by ring
    refine (mul_le_mul_of_nonneg_left hb (Real.exp_pos _).le).trans (le_of_eq ?_)
    simp only [Lw, E]; ring
  have hgL : ∀ y ∈ s, ∀ y' ∈ s, ‖g y - g y'‖ ≤ Lg * dist y y' := by
    intro y hy y' hy'
    have hVy : ‖V y‖ ≤ ‖V‖ * R :=
      (V.le_opNorm y).trans (mul_le_mul_of_nonneg_left (by simpa [s] using hy) (norm_nonneg _))
    have e : g y - g y' = (w y - w y') • V y + w y' • (V y - V y') := by
      simp only [g, sub_smul, smul_sub]; abel
    rw [e]
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, ← map_sub]
    have h1 := mul_le_mul (hwL y hy y' hy') hVy (norm_nonneg _) (by positivity)
    have h2 := mul_le_mul (hwE y' hy') (V.le_opNorm (y - y')) (norm_nonneg _)
      (Real.exp_pos _).le
    rw [dist_eq_norm] at h1 ⊢
    simp only [Lg]
    nlinarith [norm_nonneg (y - y')]
  have hsb : Bornology.IsBounded s := Metric.isBounded_closedBall
  have hgi : ∀ ρ : Measure (EucSpace d), IsProbabilityMeasure ρ → IsCarriedBy ρ R →
      Integrable g ρ := fun ρ _ hρ => by
    refine Integrable.of_bound (by fun_prop) (E * (‖V‖ * R)) ?_
    filter_upwards [ae_mem_closedBall hρ] with y hy
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
    exact mul_le_mul (hwE y hy) ((V.le_opNorm y).trans
      (mul_le_mul_of_nonneg_left (by simpa using hy) (norm_nonneg _))) (norm_nonneg _)
      (Real.exp_pos _).le
  have hwi := integrable_attentionWeight Q K R μ hsμ x
  have hwi' := integrable_attentionWeight Q K R ν hsν x
  set Dμ := ∫ y, w y ∂μ
  set Dν := ∫ y, w y ∂ν
  set Nμ := ∫ y, g y ∂μ
  set Nν := ∫ y, g y ∂ν
  have hD0 : Real.exp (-M) ≤ Dμ := by
    have : ∫ _y, Real.exp (-M) ∂μ = Real.exp (-M) := by simp
    rw [← this]
    refine integral_mono_ae (integrable_const _) hwi ?_
    filter_upwards [ae_mem_closedBall hsμ] with y hy
    refine Real.exp_le_exp.2 (neg_le.1 ?_)
    exact (neg_le_abs _).trans (abs_inner_le_of_mem_closedBall Q K hx' (by simpa using hy))
  have hDpos : 0 < Dμ := (Real.exp_pos _).trans_le hD0
  have hDν : 0 < Dν := by
    have : ∫ _y, Real.exp (-M) ∂ν = Real.exp (-M) := by simp
    refine (Real.exp_pos (-M)).trans_le ?_
    rw [← this]
    refine integral_mono_ae (integrable_const _) hwi' ?_
    filter_upwards [ae_mem_closedBall hsν] with y hy
    refine Real.exp_le_exp.2 (neg_le.1 ?_)
    exact (neg_le_abs _).trans (abs_inner_le_of_mem_closedBall Q K hx' (by simpa using hy))
  have hDinv : Dμ⁻¹ ≤ E := by
    rw [inv_le_comm₀ hDpos (Real.exp_pos _), ← Real.exp_neg]; exact hD0
  have hNν : ‖Nν‖ ≤ ‖V‖ * R * Dν := by
    rw [← integral_const_mul]
    refine norm_integral_le_of_norm_le (hwi'.const_mul _) ?_
    filter_upwards [ae_mem_closedBall hsν] with y hy
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
    exact mul_le_mul_of_nonneg_right ((V.le_opNorm y).trans
      (mul_le_mul_of_nonneg_left (by simpa using hy) (norm_nonneg _))) (Real.exp_pos _).le
  have hΔD : ‖Dμ - Dν‖ ≤ Lw * Wasserstein.W2 μ ν :=
    Wasserstein.norm_integral_sub_le_W2 μ ν hs hsb hsμ hsν w hwi hwi' (by positivity) hwL
  have hΔN : ‖Nμ - Nν‖ ≤ Lg * Wasserstein.W2 μ ν :=
    Wasserstein.norm_integral_sub_le_W2 μ ν hs hsb hsμ hsν g (hgi μ hμ hsμ) (hgi ν hν hsν)
      (by positivity) hgL
  have hW := Wasserstein.W2_nonneg μ ν
  have e : attentionKernel Q K V μ x - attentionKernel Q K V ν x
      = Dμ⁻¹ • ((Nμ - Nν) + ((Dν - Dμ) / Dν) • Nν) := by
    simp only [attentionKernel, smul_add, smul_sub, smul_smul]
    rw [show Dμ⁻¹ * ((Dν - Dμ) / Dν) = Dμ⁻¹ - Dν⁻¹ by field_simp, sub_smul]
    abel
  rw [e, norm_smul, Real.norm_of_nonneg (inv_pos.2 hDpos).le]
  have h2 : ‖((Dν - Dμ) / Dν) • Nν‖ ≤ ‖V‖ * R * (Lw * Wasserstein.W2 μ ν) := by
    rw [norm_smul, norm_div, Real.norm_of_nonneg hDν.le, norm_sub_rev]
    calc ‖Dμ - Dν‖ / Dν * ‖Nν‖ ≤ ‖Dμ - Dν‖ / Dν * (‖V‖ * R * Dν) :=
          mul_le_mul_of_nonneg_left hNν (by positivity)
      _ = ‖V‖ * R * ‖Dμ - Dν‖ := by field_simp
      _ ≤ _ := mul_le_mul_of_nonneg_left hΔD (by positivity)
  have hsum := (norm_add_le _ _).trans (add_le_add hΔN h2)
  calc Dμ⁻¹ * ‖(Nμ - Nν) + ((Dν - Dμ) / Dν) • Nν‖
      ≤ E * (Lg * Wasserstein.W2 μ ν + ‖V‖ * R * (Lw * Wasserstein.W2 μ ν)) :=
        mul_le_mul hDinv hsum (norm_nonneg _) (Real.exp_pos _).le
    _ ≤ _ := by nlinarith

/-- The hypotheses of `attentionKernel_lipschitz_in_measure` are satisfiable:
the Dirac mass at the origin is a probability measure carried by the unit
ball, which contains the origin. -/
example : (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 ∧
    (0 : EucSpace d) ∈ Metric.closedBall (0 : EucSpace d) 1 :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one, by simp⟩

end Clusters
end Transformer

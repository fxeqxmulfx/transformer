/-
# Caveats and patterns of the noise scale

arXiv:1812.06162, §2.4 and §2.5.

§2.4, caveat 3, "Simplified noise scale": `B_simple` "may be inaccurate to the extent that the
Hessian is not well-conditioned".  For `l |v|² ≤ vᵀHv ≤ u |v|²` with `0 < l` and `Σ` the
covariance of a random vector, `B_noise` lies within the factor `u/l` of `B_simple`
(`noiseScale_mem_Icc`), and the factor is attained: `G = (1, 0)`, `H = diag(1, κ)`,
`Σ = diag(0, 1)` give `B_noise = κ B_simple` (`noiseScale_diagonal`).

§2.5, "Growth over training", read for `B_simple`, whose noise is `tr(Σ)`, with "roughly
constant" read as constant and `tr(Σ) > 0`, which the paper leaves implicit: the noise scale
grows when `|G|` decreases (`simpleNoiseScale_lt_of_lt`) and tends to infinity as `G → 0`
(`tendsto_simpleNoiseScale`), as at a local minimum of a continuously differentiable loss
(`tendsto_simpleNoiseScale_of_isLocalMin`).  "Weak dependence on model size": for a model of
two parts, `B_simple` is the mediant `(tr Σ₁ + tr Σ₂)/(|G₁|² + |G₂|²)` whatever the covariance
between the parts (`simpleNoiseScale_sumElim`), between the noise scales of the parts
(`simpleNoiseScale_sumElim_mem_Icc`), so that parts of equal noise scale leave it unchanged.

The other caveats and patterns are qualitative, except the dependence on the learning rate,
the temperature of Appendix C.
-/

import Transformer.NoiseScale.Section2_Implications

open MeasureTheory ProbabilityTheory Filter Topology
open scoped Matrix

namespace Transformer.NoiseScale

variable {Ω ι : Type*} [Fintype ι] [MeasurableSpace Ω] {P : Measure Ω}

/-- `tr(HΣ) = E[(Y - E[Y])ᵀ H (Y - E[Y])]` for `Σ` the covariance of `Y`. -/
theorem integral_quadForm_sub [IsProbabilityMeasure P] {Y : Ω → ι → ℝ}
    (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P) (H : Matrix ι ι ℝ) :
    ∫ ω, (Y ω - fun a => ∫ ω, Y ω a ∂P) ⬝ᵥ H *ᵥ (Y ω - fun a => ∫ ω, Y ω a ∂P) ∂P =
      (H * covMatrix Y P).trace := by
  have hY' : ∀ a, MemLp (fun ω => (Y ω - fun a => ∫ ω, Y ω a ∂P) a) 2 P := fun a =>
    (hY a).sub (memLp_const _)
  have hm : (fun a => ∫ ω, (Y ω - fun a => ∫ ω, Y ω a ∂P) a ∂P) = 0 := funext fun a => by
    simp [integral_sub ((hY a).integrable one_le_two) (integrable_const _)]
  have hcov : covMatrix (fun ω => Y ω - fun a => ∫ ω, Y ω a ∂P) P = covMatrix Y P := by
    ext a b
    simp [covMatrix, covariance_sub_const_left ((hY a).integrable one_le_two),
      covariance_sub_const_right ((hY b).integrable one_le_two)]
  rw [integral_quadForm hY' H, hm, hcov]
  simp

/-- The hypotheses of `integral_quadForm_sub` are satisfiable: a constant. -/
example (G : ι → ℝ) (H : Matrix ι ι ℝ) :=
  integral_quadForm_sub (P := Measure.dirac ()) (Y := fun _ : Unit => G)
    (fun a => memLp_const (G a)) H

/-- **§2.4, caveat 3**: for `l |v|² ≤ vᵀHv ≤ u |v|²`, `0 < l`, and `Σ` the covariance of a
random vector, `B_noise` lies between `(l/u) B_simple` and `(u/l) B_simple`. -/
theorem noiseScale_mem_Icc [IsProbabilityMeasure P] {Y : Ω → ι → ℝ}
    (hY : ∀ a, MemLp (fun ω => Y ω a) 2 P) {H : Matrix ι ι ℝ} {l u : ℝ} (hl : 0 < l)
    (hlH : ∀ v, l * (v ⬝ᵥ v) ≤ v ⬝ᵥ H *ᵥ v) (hHu : ∀ v, v ⬝ᵥ H *ᵥ v ≤ u * (v ⬝ᵥ v))
    {G : ι → ℝ} (hG : G ≠ 0) :
    noiseScale G H (covMatrix Y P) ∈ Set.Icc (l / u * simpleNoiseScale G (covMatrix Y P))
      (u / l * simpleNoiseScale G (covMatrix Y P)) := by
  classical
  have hY' : ∀ a, MemLp (fun ω => (Y ω - fun a => ∫ ω, Y ω a ∂P) a) 2 P := fun a =>
    (hY a).sub (memLp_const _)
  have hI1 := integrable_quadForm hY' 1
  have hI2 := integrable_quadForm hY' H
  simp only [Matrix.one_mulVec] at hI1
  have hlo : l * (covMatrix Y P).trace ≤ (H * covMatrix Y P).trace := by
    rw [← integral_dotProduct_sub hY, ← integral_quadForm_sub hY, ← integral_const_mul]
    exact integral_mono (hI1.const_mul l) hI2 fun ω => hlH _
  have hhi : (H * covMatrix Y P).trace ≤ u * (covMatrix Y P).trace := by
    rw [← integral_dotProduct_sub hY, ← integral_quadForm_sub hY, ← integral_const_mul]
    exact integral_mono hI2 (hI1.const_mul u) fun ω => hHu _
  have hT : 0 ≤ (covMatrix Y P).trace := by
    rw [← integral_dotProduct_sub hY]
    exact integral_nonneg fun ω => Finset.sum_nonneg fun a _ => mul_self_nonneg _
  have hg : 0 < G ⬝ᵥ G := lt_of_le_of_ne (Finset.sum_nonneg fun a _ => mul_self_nonneg (G a))
    (Ne.symm (dotProduct_self_eq_zero.not.2 hG))
  have hq1 := hlH G
  have hq2 := hHu G
  have hq : 0 < G ⬝ᵥ H *ᵥ G := (mul_pos hl hg).trans_le hq1
  have hu : 0 < u := pos_of_mul_pos_left (hq.trans_le hq2) hg.le
  unfold noiseScale simpleNoiseScale
  constructor
  · rw [div_mul_div_comm, div_le_div_iff₀ (by positivity) hq]
    nlinarith [mul_le_mul_of_nonneg_left hq2 (mul_nonneg hl.le hT)]
  · rw [div_mul_div_comm, div_le_div_iff₀ hq (by positivity)]
    nlinarith [mul_le_mul_of_nonneg_left hq1 (mul_nonneg hu.le hT),
      mul_le_mul_of_nonneg_right hhi (mul_nonneg hl.le hg.le)]

/-- The hypotheses of `noiseScale_mem_Icc` are satisfiable: `H = 1`, `l = u = 1`. -/
example := noiseScale_mem_Icc (P := Measure.dirac ()) (Y := fun _ : Unit => fun _ : Unit => 1)
  (fun _ => memLp_const _) (H := 1) (l := 1) (u := 1) one_pos (fun v => by simp)
  (fun v => by simp) (G := fun _ => 1) fun h => one_ne_zero (congr_fun h ())

/-- **§2.4, caveat 3**: the factor `u/l` is attained.  `H = diag(1, κ)`, `κ ≥ 1`, has
`|v|² ≤ vᵀHv ≤ κ |v|²`, and `G = (1, 0)`, `Σ = diag(0, 1)` give `B_simple = 1`,
`B_noise = κ`. -/
theorem noiseScale_diagonal {κ : ℝ} (hκ : 1 ≤ κ) :
    (∀ v : Fin 2 → ℝ, 1 * (v ⬝ᵥ v) ≤ v ⬝ᵥ Matrix.diagonal ![1, κ] *ᵥ v ∧
      v ⬝ᵥ Matrix.diagonal ![1, κ] *ᵥ v ≤ κ * (v ⬝ᵥ v)) ∧
      simpleNoiseScale ![1, 0] (Matrix.diagonal ![0, 1]) = 1 ∧
      noiseScale ![1, 0] (Matrix.diagonal ![1, κ]) (Matrix.diagonal ![0, 1]) = κ := by
  refine ⟨fun v => ⟨?_, ?_⟩, ?_, ?_⟩ <;>
    simp [simpleNoiseScale, noiseScale, dotProduct, Fin.sum_univ_two, Matrix.mulVec_diagonal,
      Matrix.diagonal_mul_diagonal, Matrix.trace_diagonal]
  · nlinarith [mul_self_nonneg (v 1)]
  · nlinarith [mul_self_nonneg (v 0)]

/-- The hypothesis of `noiseScale_diagonal` is satisfiable: `κ = 2`. -/
example := noiseScale_diagonal (κ := 2) one_le_two

/-- **§2.5, "Growth over training"**: "`B` will grow when the gradient decreases in magnitude,
as long as the noise `tr(Σ)` stays roughly constant", for `B_simple` at constant `tr(Σ)`; the
paper leaves `tr(Σ) > 0` and `G' ≠ 0` implicit. -/
theorem simpleNoiseScale_lt_of_lt {G G' : ι → ℝ} {S : Matrix ι ι ℝ} (hS : 0 < S.trace)
    (hG' : 0 < G' ⬝ᵥ G') (h : G' ⬝ᵥ G' < G ⬝ᵥ G) :
    simpleNoiseScale G S < simpleNoiseScale G' S :=
  div_lt_div_of_pos_left hS hG' h

/-- The hypotheses of `simpleNoiseScale_lt_of_lt` are satisfiable: `|G'|² = 1 < 4 = |G|²`. -/
example := simpleNoiseScale_lt_of_lt (G := fun _ : Unit => 2) (G' := fun _ => 1) (S := 1)
  (by simp) (by simp) (by simp [dotProduct]; norm_num)

/-- **§2.5, "Growth over training"**: as `G → 0`, as at the minimum of a smooth loss,
`B_simple → ∞` for `tr(Σ) > 0`. -/
theorem tendsto_simpleNoiseScale {S : Matrix ι ι ℝ} (hS : 0 < S.trace) :
    Tendsto (fun G => simpleNoiseScale G S) (𝓝[≠] 0) atTop := by
  have hc : Continuous fun G : ι → ℝ => G ⬝ᵥ G := continuous_id.dotProduct continuous_id
  have h : Tendsto (fun G : ι → ℝ => G ⬝ᵥ G) (𝓝[≠] 0) (𝓝[>] 0) := by
    refine tendsto_nhdsWithin_iff.2 ⟨?_, eventually_nhdsWithin_of_forall fun G hG => ?_⟩
    · simpa using (hc.tendsto 0).mono_left (nhdsWithin_le_nhds (s := {0}ᶜ))
    · exact lt_of_le_of_ne (Finset.sum_nonneg fun a _ => mul_self_nonneg (G a))
        (Ne.symm (dotProduct_self_eq_zero.not.2 hG))
  simpa only [simpleNoiseScale, div_eq_mul_inv, Function.comp_def] using
    (tendsto_inv_nhdsGT_zero.comp h).const_mul_atTop hS

/-- The hypothesis of `tendsto_simpleNoiseScale` is satisfiable: `Σ = 1`. -/
example := tendsto_simpleNoiseScale (ι := Unit) (S := 1) (by simp)

/-- **§2.5, "Growth over training"**: "`|G|` decreases as we approach the minimum of a smooth
loss", so `B_simple → ∞` there.  For a continuously differentiable `L` of gradient `G θ'` at
`θ'`, as `θ'` tends to a local minimum among the points of nonzero gradient. -/
theorem tendsto_simpleNoiseScale_of_isLocalMin {L : (ι → ℝ) → ℝ} (hL : ContDiff ℝ 1 L)
    {G : (ι → ℝ) → ι → ℝ} (hG : ∀ θ' v, fderiv ℝ L θ' v = G θ' ⬝ᵥ v) {θ : ι → ℝ}
    (hθ : IsLocalMin L θ) {S : Matrix ι ι ℝ} (hS : 0 < S.trace) :
    Tendsto (fun θ' => simpleNoiseScale (G θ') S) (𝓝[{θ' | G θ' ≠ 0}] θ) atTop := by
  classical
  have hG' : G = fun θ' a => fderiv ℝ L θ' (Pi.single a 1) := by
    ext θ' a
    simp [hG]
  have hc : Continuous G := hG' ▸ continuous_pi fun a =>
    (hL.continuous_fderiv one_ne_zero).clm_apply continuous_const
  have h0 : G θ = 0 := funext fun a => by simp [hG', hθ.fderiv_eq_zero]
  refine (tendsto_simpleNoiseScale hS).comp (tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩)
  · simpa [h0] using (hc.tendsto θ).mono_left nhdsWithin_le_nhds
  · exact eventually_nhdsWithin_of_forall fun θ' h => h

/-- The hypotheses of `tendsto_simpleNoiseScale_of_isLocalMin` are satisfiable:
`L(θ) = θ²` of gradient `2θ`, least at `0`. -/
example := tendsto_simpleNoiseScale_of_isLocalMin (ι := Unit) (L := fun θ => θ () * θ ())
  (by fun_prop) (G := fun θ _ => 2 * θ ()) (fun θ v => by
    rw [((hasFDerivAt_apply () θ).fun_mul (hasFDerivAt_apply () θ)).fderiv]
    simp [dotProduct]
    ring) (θ := 0) (Filter.Eventually.of_forall fun y => by simp [mul_self_nonneg])
  (S := 1) (by simp)

/-- **§2.5, "Weak dependence on model size"**: for a model of two parts, `B_simple` is the
mediant of the parts, whatever the covariance `C`, `D` between them. -/
theorem simpleNoiseScale_sumElim {κ : Type*} [Fintype κ] (G₁ : ι → ℝ) (G₂ : κ → ℝ)
    (S₁ : Matrix ι ι ℝ) (C : Matrix ι κ ℝ) (D : Matrix κ ι ℝ) (S₂ : Matrix κ κ ℝ) :
    simpleNoiseScale (Sum.elim G₁ G₂) (Matrix.fromBlocks S₁ C D S₂) =
      (S₁.trace + S₂.trace) / (G₁ ⬝ᵥ G₁ + G₂ ⬝ᵥ G₂) := by
  simp [simpleNoiseScale, Matrix.trace, Fintype.sum_sum_type, sumElim_dotProduct_sumElim]

/-- **§2.5, "Weak dependence on model size"**: the noise scale of a model of two parts lies
between those of the parts. -/
theorem simpleNoiseScale_sumElim_mem_Icc {κ : Type*} [Fintype κ] {G₁ : ι → ℝ} {G₂ : κ → ℝ}
    {S₁ : Matrix ι ι ℝ} {S₂ : Matrix κ κ ℝ} (hG₁ : 0 < G₁ ⬝ᵥ G₁) (hG₂ : 0 < G₂ ⬝ᵥ G₂)
    (h : simpleNoiseScale G₁ S₁ ≤ simpleNoiseScale G₂ S₂) (C : Matrix ι κ ℝ)
    (D : Matrix κ ι ℝ) :
    simpleNoiseScale (Sum.elim G₁ G₂) (Matrix.fromBlocks S₁ C D S₂) ∈
      Set.Icc (simpleNoiseScale G₁ S₁) (simpleNoiseScale G₂ S₂) := by
  rw [simpleNoiseScale_sumElim]
  unfold simpleNoiseScale at h ⊢
  rw [div_le_div_iff₀ hG₁ hG₂] at h
  constructor
  · rw [div_le_div_iff₀ hG₁ (by positivity)]
    nlinarith
  · rw [div_le_div_iff₀ (by positivity) hG₂]
    nlinarith

/-- The hypotheses of `simpleNoiseScale_sumElim_mem_Icc` are satisfiable: two equal parts. -/
example := simpleNoiseScale_sumElim_mem_Icc (G₁ := fun _ : Unit => 1) (G₂ := fun _ : Unit => 1)
  (S₁ := 1) (S₂ := 1) (by simp) (by simp) le_rfl 0 0

end Transformer.NoiseScale

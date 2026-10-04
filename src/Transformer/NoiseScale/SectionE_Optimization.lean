/-
# Comments on optimization

arXiv:1812.06162, Appendix E, with eq. (A.3) of Appendix A.2.

§E.1: "training often tends towards a regime where the optimal step size (determined by a line
search in the direction of the parameter update) is almost exactly half of the actual update
magnitude", interpreted as "large Hessian directions ... dominating the update ..., so that
training involves rapid oscillations".  In the model (2.4) along an update of positive
curvature, the best step of the line search is half the update exactly when the update leaves
the loss unchanged, crossing the valley to the same height on its other wall
(`quadModel_half_le_iff`).

§E.2: the central learning rate `ε(B) = ε_*/(1 + B_*/B)^α` of the grid searches, eq. (A.3)
(`lrCentral`), "generalizes" eq. (2.6): at `α = 1`, `ε_* = ε_max`, `B_* = B_noise` it is
`ε_opt(B)` (`lrCentral_one`).  "The SGD linear scaling rule (`α = 1`) means that the step size
per data example stays fixed up to `B_*`": `ε(B)/B` lies within a factor 2 of `ε_*/B_*` for
`B ≤ B_*` (`lrCentral_one_div_mem_Icc`).  §3.1's optimal learning rate that "initially obeys a
power law `ε(B) ∝ B^α` ..., then becomes roughly constant" is the shape of (A.3):
`ε(B)/B^α → ε_*/B_*^α` as `B → 0`, the limit `ε_*/B_*` of `ε(B)/B` at `α = 1`, and
`ε(B) → ε_*` as `B → ∞` (`tendsto_lrCentral_div_rpow`, `tendsto_lrCentral`).

For Adam, disregarding `β₁`, `β₂` and `ε_Adam`, the update `εE[G_i]/√E[G_i²]` is
`ε sign(E[G_i])/√(1 + s_i/E[G_i]²)`, `s_i` the variance of `G_i` (`adam_update_eq`).  At
`s_i = s/B`, as for the mean of a batch (`covMatrix_batchGrad`), its size `ε/√(1 + B_*/B)`,
`B_* = s/E[G_i]²`, follows the profile `1/(1 + B_*/B)` of eq. (2.6) exactly when `ε` follows
(A.3) at `α = 1/2`: "this implies a square-root scaling rule (`α = 0.5`)"
(`adam_lrCentral_half`).
-/

import Transformer.NoiseScale.Section2_NoiseScale
import Mathlib.Probability.Moments.Variance
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

open MeasureTheory ProbabilityTheory Filter Topology
open scoped Matrix

namespace Transformer.NoiseScale

/-- **Appendix E.1**: in the model (2.4) along an update `V` of curvature `VᵀHV > 0`, the line
search's best step is half the update, `ε = 1/2`, exactly when the update leaves the loss
unchanged: "the optimal step size ... is almost exactly half of the actual update magnitude",
read as the oscillation across a valley that the paper's interpretation names. -/
theorem quadModel_half_le_iff {ι : Type*} [Fintype ι] {G V : ι → ℝ} {H : Matrix ι ι ℝ}
    (hV : 0 < V ⬝ᵥ H *ᵥ V) (L : ℝ) :
    (∀ ε, quadModel L G H (1 / 2) V ≤ quadModel L G H ε V) ↔ quadModel L G H 1 V = L := by
  simp only [quadModel]
  generalize G ⬝ᵥ V = g
  generalize V ⬝ᵥ H *ᵥ V = a at hV
  constructor
  · intro h
    have h1 := h (g / a)
    have h2 : (a - 2 * g) ^ 2 ≤ 0 := by
      field_simp at h1
      nlinarith
    have := pow_eq_zero_iff two_ne_zero |>.1 (le_antisymm h2 (sq_nonneg _))
    linarith
  · intro h ε
    have hg : g = a / 2 := by linarith
    subst hg
    nlinarith [mul_nonneg hV.le (sq_nonneg (ε - 1 / 2))]

/-- The hypothesis of `quadModel_half_le_iff` is satisfiable: `V = H = 1` in one dimension. -/
example := quadModel_half_le_iff (G := fun _ : Unit => 1) (V := fun _ => 1) (H := 1) (by simp) 0

/-- The central learning rate `ε(B) = ε_*/(1 + B_*/B)^α` of a batch of `B`, eq. (A.3). -/
noncomputable def lrCentral (εs Bs α B : ℝ) : ℝ := εs / (1 + Bs / B) ^ α

/-- **Appendix E.2**: "Equation (A.3) generalizes Equation (2.6)": at `α = 1`, `ε_* = ε_max`
and `B_* = B_noise` it is `ε_opt(B)`. -/
theorem lrCentral_one {ι : Type*} [Fintype ι] (G : ι → ℝ) (H S : Matrix ι ι ℝ) (B : ℝ) :
    lrCentral (stepMax G H) (noiseScale G H S) 1 B = stepOpt G H S B := by
  rw [lrCentral, Real.rpow_one, stepOpt]

/-- **Appendix E.2**: "the SGD linear scaling rule (`α = 1`) means that the step size per data
example stays fixed up to `B_*`": for `B ≤ B_*`, `ε(B)/B` lies in `[ε_*/(2B_*), ε_*/B_*]`. -/
theorem lrCentral_one_div_mem_Icc {εs Bs B : ℝ} (hεs : 0 ≤ εs) (hB : 0 < B) (h : B ≤ Bs) :
    lrCentral εs Bs 1 B / B ∈ Set.Icc (εs / (2 * Bs)) (εs / Bs) := by
  have hBs := hB.trans_le h
  have e : lrCentral εs Bs 1 B / B = εs / (B + Bs) := by
    rw [lrCentral, Real.rpow_one]
    field_simp
  rw [e]
  exact ⟨div_le_div_of_nonneg_left hεs (by positivity) (by linarith),
    div_le_div_of_nonneg_left hεs hBs (by linarith)⟩

/-- The hypotheses of `lrCentral_one_div_mem_Icc` are satisfiable: `ε_* = B_* = B = 1`. -/
example := lrCentral_one_div_mem_Icc zero_le_one one_pos le_rfl

/-- Eq. (A.3) per `B^α`: `ε(B)/B^α = ε_*/(B + B_*)^α`. -/
theorem lrCentral_div_rpow {εs Bs α B : ℝ} (hBs : 0 ≤ Bs) (hB : 0 < B) :
    lrCentral εs Bs α B / B ^ α = εs / (B + Bs) ^ α := by
  rw [lrCentral, div_div, ← Real.mul_rpow (by positivity) hB.le]
  congr 2
  field_simp

/-- The hypotheses of `lrCentral_div_rpow` are satisfiable: `B_* = 0`, `B = 1`. -/
example (εs α : ℝ) := lrCentral_div_rpow (εs := εs) (α := α) le_rfl one_pos

/-- §3.1 and eq. (A.3): "initially obeys a power law `ε(B) ∝ B^α`": `ε(B)/B^α → ε_*/B_*^α` as
`B → 0`. -/
theorem tendsto_lrCentral_div_rpow {Bs : ℝ} (hBs : 0 < Bs) (εs α : ℝ) :
    Tendsto (fun B => lrCentral εs Bs α B / B ^ α) (𝓝[>] 0) (𝓝 (εs / Bs ^ α)) := by
  have h : ContinuousAt (fun B : ℝ => εs / (B + Bs) ^ α) 0 := by
    refine continuousAt_const.div ?_ (by rw [zero_add]; exact (Real.rpow_pos_of_pos hBs α).ne')
    exact (Real.continuousAt_rpow_const _ α (Or.inl (by simp [hBs.ne']))).comp
      (continuousAt_id.add continuousAt_const)
  refine (tendsto_nhdsWithin_of_tendsto_nhds (by simpa using h.tendsto)).congr' ?_
  exact eventually_nhdsWithin_of_forall fun B (hB : 0 < B) =>
    (lrCentral_div_rpow hBs.le hB).symm

/-- §3.1 and eq. (A.3): the learning rate "then becomes roughly constant": `ε(B) → ε_*` as
`B → ∞`. -/
theorem tendsto_lrCentral (εs Bs α : ℝ) :
    Tendsto (fun B => lrCentral εs Bs α B) atTop (𝓝 εs) := by
  have h : Tendsto (fun B : ℝ => 1 + Bs / B) atTop (𝓝 1) := by
    simpa using ((tendsto_const_nhds (x := Bs)).div_atTop tendsto_id).const_add 1
  have h1 : Tendsto (fun B : ℝ => (1 + Bs / B) ^ α) atTop (𝓝 1) := by
    have := (Real.continuousAt_rpow_const 1 α (Or.inl one_ne_zero)).tendsto.comp h
    rw [Real.one_rpow] at this
    exact this
  have h2 := (tendsto_const_nhds (x := εs)).div h1 one_ne_zero
  rw [div_one] at h2
  exact h2

/-- The hypothesis of `tendsto_lrCentral_div_rpow` is satisfiable: `B_* = 1`. -/
example (εs α : ℝ) := tendsto_lrCentral_div_rpow one_pos εs α

/-- **Appendix E.2**: disregarding `β₁`, `β₂` and `ε_Adam`, the Adam update
`εE[G_i]/√E[G_i²]` is `ε sign(E[G_i])/√(1 + s_i/E[G_i]²)`, `s_i` the variance of `G_i`.  The
paper's `≈` is an equality once its moving averages are read as means over the timesteps, a
probability measure `P`; at `E[G_i] = 0` both sides vanish. -/
theorem adam_update_eq {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    {G : Ω → ℝ} (hG : MemLp G 2 P) (ε : ℝ) :
    ε * (∫ ω, G ω ∂P) / √(∫ ω, G ω ^ 2 ∂P) =
      ε * SignType.sign (∫ ω, G ω ∂P) / √(1 + variance G P / (∫ ω, G ω ∂P) ^ 2) := by
  rcases eq_or_ne (∫ ω, G ω ∂P) 0 with hm | hm
  · simp [hm]
  have hv := variance_eq_sub hG
  simp only [Pi.pow_apply] at hv
  have e : ∫ ω, G ω ^ 2 ∂P = (∫ ω, G ω ∂P) ^ 2 * (1 + variance G P / (∫ ω, G ω ∂P) ^ 2) := by
    rw [hv]
    field_simp
    ring
  have hq : 0 < √(1 + variance G P / (∫ ω, G ω ∂P) ^ 2) :=
    Real.sqrt_pos.2 (by have := variance_nonneg G P; positivity)
  have hsm := sign_mul_abs (∫ ω, G ω ∂P)
  rw [e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs]
  generalize √(1 + variance G P / (∫ ω, G ω ∂P) ^ 2) = q at hq ⊢
  rw [div_eq_div_iff (mul_pos (abs_pos.2 hm) hq).ne' hq.ne']
  linear_combination -(ε * q) * hsm

/-- The hypotheses of `adam_update_eq` are satisfiable: `G = 1`. -/
example (ε : ℝ) := adam_update_eq (P := Measure.dirac ()) (G := fun _ : Unit => 1)
  (memLp_const 1) ε

/-- **Appendix E.2**: for a variance `s/B` of `G_i`, the Adam update of size
`ε/√(1 + (s/B)/E[G_i]²)` follows the profile `c/(1 + B_*/B)` of eq. (2.6),
`B_* = s/E[G_i]²`, exactly when `ε` is (A.3) at `α = 1/2`: "this implies a square-root scaling
rule (`α = 0.5`) to maintain a constant learning rate per data example". -/
theorem adam_lrCentral_half {m s B : ℝ} (hs : 0 ≤ s) (hB : 0 < B) (c ε : ℝ) :
    ε / √(1 + s / B / m ^ 2) = c / (1 + s / m ^ 2 / B) ↔
      ε = lrCentral c (s / m ^ 2) (1 / 2) B := by
  have e : s / B / m ^ 2 = s / m ^ 2 / B := by ring
  have hx : 0 < 1 + s / m ^ 2 / B := by positivity
  rw [e, lrCentral, ← Real.sqrt_eq_rpow]
  generalize 1 + s / m ^ 2 / B = x at hx ⊢
  have hq := Real.sqrt_pos.2 hx
  have hxq := Real.mul_self_sqrt hx.le
  rw [div_eq_div_iff hq.ne' hx.ne', eq_div_iff hq.ne']
  constructor
  · intro h
    apply mul_right_cancel₀ hq.ne'
    rwa [mul_assoc, hxq]
  · intro h
    rw [← h, mul_assoc, hxq]

/-- The hypotheses of `adam_lrCentral_half` are satisfiable: `E[G_i] = s = B = 1`. -/
example (c ε : ℝ) := adam_lrCentral_half (m := 1) zero_le_one one_pos c ε

end Transformer.NoiseScale

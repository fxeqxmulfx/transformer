/-
# Metastability — On the initial configuration, Gaussian mixtures
  (§4 of 2410.06833v1)

* `Proposition prop: mixture.of.gaussians`.

The corrected density is in `GaussianMixture`; centered configurations and
the deterministic radial-projection lemmas are in `InitialGeometry`.

The uniform-initialization half of §4 (`prop: concentration unif`,
`coro: cm`, the low-dimensional bound) is in
`Transformer.Metastability.InitialUniform`.

`prop: mixture.of.gaussians` bounds a probability, so it is stated against the
law `mixtureLaw` of the sample. The printed density has the one-dimensional
normalization `√(2πσ²)` even though the sample lives in `ℝ^d`; the denominator
is corrected to `(2πσ²)^(d/2)` in `GaussianMixture`. That the resulting law
is a probability measure for `r ≥ 1` and `σ > 0` is `mixtureLaw_probability`.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Transformer.Metastability.GaussianMixture
import Transformer.Metastability.InitialGeometry
import Transformer.Metastability.InitialMixtureBound
import Transformer.Metastability.InitialScale
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **Proposition (prop: mixture.of.gaussians).**

Let `(w_1,…,w_r)` be `(β, ε)`-centered, let `X_1,…,X_n` be i.i.d. with the
Gaussian-mixture density, and assume, with `δ = σ / √r`,

  `(6 δ √d)/(1 + δ √d) + δ √(2 d log n) ≤ ε`.

Then `(X_i / ‖X_i‖)_{i=1}^n` is `(β, ε)`-separated with probability at least
`1 - 2 e^{-d}`.

The source assumes `β, ε, σ > 0`, `d, n ≥ 2` and `1 ≤ r ≤ n`; those conditions
are explicit here. The density uses the corrected `d`-dimensional normalizer
described above. Without `σ > 0` the earlier Lean statement was false, as
`not_mixture_separation_without_scale_pos` demonstrates.

Source: arXiv:2410.06833v1, §4. -/
theorem mixture_separation
    (hd : 2 ≤ d) (hn : 2 ≤ n) (β ε σ : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hσ : 0 < σ) (r : ℕ) (hr : 1 ≤ r) (w : Idx r → SSphere d)
    (hcent : isCentered d n β ε r w)
    (hδ : (6 * (σ / Real.sqrt r) * Real.sqrt d)
            / (1 + (σ / Real.sqrt r) * Real.sqrt d)
          + (σ / Real.sqrt r) * Real.sqrt (2 * d * Real.log n) ≤ ε) :
    1 - 2 * Real.exp (-(d : ℝ))
      ≤ (mixtureLaw d n r σ fun q => ((w q : EucSpace d))).real
          (projectedSeparated d n β ε) := by
  let μ : Measure (Idx n → EucSpace d) :=
    mixtureLaw d n r σ fun q => (w q : EucSpace d)
  let _ : IsProbabilityMeasure μ :=
    mixtureLaw_probability d n r σ hσ hr (fun q => (w q : EucSpace d))
  by_cases hε2 : ε < 2
  · have hε4 := centered_epsilon_lt_quarter_of_lt_two d n r hn β ε hβ hε hε2 w hcent
    have hrℝ : (0 : ℝ) < r := by exact_mod_cast hr
    have hsqrtr : 0 < Real.sqrt (r : ℝ) := Real.sqrt_pos.mpr hrℝ
    have hδpos : 0 < σ / Real.sqrt (r : ℝ) := div_pos hσ hsqrtr
    let R : ℝ := ε / (8 * (σ / Real.sqrt (r : ℝ)) ^ 2)
    have hRlower : 2 * (d : ℝ) + Real.log n ≤ R :=
      mixture_scale_radius_lower d n hd hn _ ε hδpos hε hε4 hδ
    have hR : 8 * σ ^ 2 * R ≤ (r : ℝ) * ε := by
      dsimp [R]
      rw [div_pow, Real.sq_sqrt hrℝ.le]
      field_simp
      norm_num
    have hmarkov := mixtureLaw_projectedSeparated_markov d n r (by omega)
      β ε σ R hσ hr w hcent hR
    have hnpos : (0 : ℝ) < n := by exact_mod_cast (lt_of_lt_of_le (by norm_num : 0 < 2) hn)
    have hlog2 : Real.log (2 : ℝ) ≤ 1 := by
      linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)]
    have hpow : (2 : ℝ) ^ ((d : ℝ) / 2) ≤ Real.exp (d : ℝ) := by
      rw [Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2)]
      apply Real.exp_le_exp.mpr
      have hd0 : (0 : ℝ) ≤ d := by positivity
      nlinarith [mul_nonneg (sub_nonneg.mpr hlog2) hd0]
    have hexp : (n : ℝ) * Real.exp (d : ℝ) ≤
        Real.exp R * Real.exp (-(d : ℝ)) := by
      have he : Real.exp (2 * (d : ℝ) + Real.log n) ≤ Real.exp R :=
        Real.exp_le_exp.mpr hRlower
      rw [Real.exp_add, Real.exp_log hnpos] at he
      have hfact : Real.exp (2 * (d : ℝ)) * Real.exp (-(d : ℝ)) =
          Real.exp (d : ℝ) := by
        rw [← Real.exp_add]
        congr 1
        ring
      nlinarith [mul_nonneg (sub_nonneg.mpr he) (Real.exp_pos (-(d : ℝ))).le]
    change 1 - 2 * Real.exp (-(d : ℝ)) ≤ μ.real (projectedSeparated d n β ε)
    change Real.exp R * (1 - μ.real (projectedSeparated d n β ε)) ≤ _ at hmarkov
    have hbad : 1 - μ.real (projectedSeparated d n β ε) ≤
        Real.exp (-(d : ℝ)) := by
      have htail : Real.exp R *
          (1 - μ.real (projectedSeparated d n β ε)) ≤
          Real.exp R * Real.exp (-(d : ℝ)) :=
        hmarkov.trans ((mul_le_mul_of_nonneg_left hpow (by positivity)).trans hexp)
      exact le_of_mul_le_mul_left htail (Real.exp_pos R)
    linarith [Real.exp_pos (-(d : ℝ))]
  · have hε2' : 2 ≤ ε := by linarith
    let Z : Set (Idx n → EucSpace d) := {X | ∃ i : Idx n, X i = 0}
    have hZ : μ Z = 0 := by
      have hae := mixtureLaw_nonzero_ae d n r (by omega) σ hσ hr
        (fun q => (w q : EucSpace d))
      simpa only [Z, μ, not_forall, ne_eq, not_not] using (ae_iff).mp hae
    have hZmeas : MeasurableSet Z := by
      dsimp [Z]
      measurability
    have hsubset : Zᶜ ⊆ projectedSeparated d n β ε := by
      intro X hX
      let Y : SphereTuple d n := fun i =>
        ⟨‖X i‖⁻¹ • X i, mem_sphere_zero_iff_norm.mpr (by
          simpa using (norm_smul_inv_norm (𝕜 := ℝ) (by
            intro hi
            exact hX ⟨i, hi⟩ : X i ≠ 0)))⟩
      refine ⟨Y, fun _ => rfl, r, hcent.1, w, ?_, hcent.2⟩
      intro i
      refine ⟨⟨0, hr⟩, ?_⟩
      change 1 - ε ≤ inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((w ⟨0, hr⟩ : SSphere d) : EucSpace d)
      linarith [(inner_mem_Icc_sphere (Y i) (w ⟨0, hr⟩)).1]
    have hμone : μ.real Zᶜ = 1 := by
      rw [measureReal_compl hZmeas]
      simp [measureReal_def, hZ]
    have hbound : μ.real Zᶜ ≤ μ.real (projectedSeparated d n β ε) :=
      measureReal_mono hsubset (measure_lt_top μ _).ne
    change 1 - 2 * Real.exp (-(d : ℝ)) ≤ μ.real (projectedSeparated d n β ε)
    rw [hμone] at hbound
    linarith [Real.exp_pos (-(d : ℝ))]

/-- A single cap centre is `(β, ε)`-centered once `β` is large enough: with
`r = 1` no two centres exist, so `αDist` is the supremum of the empty set,
`γ_β` reduces to `1 - 8ε - β⁻¹ log(2n²/ε)`, and `β = 100`, `ε = 1/32` makes
it positive. At the degenerate scale `σ = 0` the numerical noise bound also
holds, exhibiting the omitted hypothesis refuted in `InitialCounterexample`.
The positive-scale witness below shows the corrected hypotheses are jointly
satisfiable.

Source: arXiv:2410.06833v1, §4, `d: separated_mixtures` and
`prop: mixture.of.gaussians`. -/
theorem centered_zero_scale_witness :
    isCentered 1 1 100 (1 / 32) 1 (fun _ => Transformer.basePoint 0) ∧
      (6 * ((0 : ℝ) / Real.sqrt 1) * Real.sqrt 1)
            / (1 + (0 / Real.sqrt 1) * Real.sqrt 1)
          + (0 / Real.sqrt 1) * Real.sqrt (2 * 1 * Real.log 1) ≤ 1 / 32 := by
  have hempty : { c : ℝ |
      ∃ i j : Idx 1, i ≠ j ∧
        ∃ x ∈ sphericalCap 1 ((fun _ => Transformer.basePoint 0) i) (2 * (1 / 32)),
        ∃ y ∈ sphericalCap 1 ((fun _ => Transformer.basePoint 0) j) (2 * (1 / 32)),
          c = inner (𝕜 := ℝ) ((x : EucSpace 1)) ((y : EucSpace 1)) } = ∅ := by
    ext c
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨i, j, hij, -⟩
    exact hij (Subsingleton.elim i j)
  have hα : αDist 1 1 (fun _ => Transformer.basePoint 0) (1 / 32) = 0 := by
    rw [αDist, hempty, Real.sSup_empty]
  refine ⟨⟨le_rfl, ?_⟩, by norm_num⟩
  rw [hα, γβ]
  have hlog : Real.log (2 * (1 : ℝ) ^ 2 / (1 / 32)) < 64 := by
    have h : Real.log (2 * (1 : ℝ) ^ 2 / (1 / 32)) < 2 * (1 : ℝ) ^ 2 / (1 / 32) - 1 :=
      Real.log_lt_sub_one_of_pos (by norm_num) (by norm_num)
    linarith [h]
  norm_num
  linarith

/-- The hypotheses of the deterministic radial-projection criteria are
satisfiable: take the one-centre witness and a noiseless sample at that
centre. Source: arXiv:2410.06833v1, §4,
`prop: mixture.of.gaussians`. -/
example :
    isCentered 1 1 100 (1 / 32) 1 (fun _ => Transformer.basePoint 0) ∧
      (∀ i : Idx 1,
        (fun _ : Idx 1 => ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)) i ≠ 0) ∧
      (∀ i : Idx 1, ∃ q : Idx 1,
        ‖(fun _ : Idx 1 => ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)) i -
          ((fun _ : Idx 1 => Transformer.basePoint 0) q : EucSpace 1)‖ ≤ (1 / 32 : ℝ) / 2) ∧
      (fun _ : Idx 1 => ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)) ∈
        projectedSeparated 1 1 100 (1 / 32) := by
  have hc := centered_zero_scale_witness.1
  have hw : ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1) ≠ 0 := by
    intro h
    have hn := mem_sphere_zero_iff_norm.mp (Transformer.basePoint 0).2
    simp [h] at hn
  have hnear : ∀ i : Idx 1, ∃ q : Idx 1,
      ‖((Transformer.basePoint 0 : SSphere 1) : EucSpace 1) -
        ((fun _ : Idx 1 => Transformer.basePoint 0) q : EucSpace 1)‖ ≤ (1 / 32 : ℝ) / 2 := by
    intro i
    refine ⟨0, ?_⟩
    norm_num
  refine ⟨hc, fun _ => hw, hnear, ?_⟩
  have hnearScaled : ∀ i : Idx 1, ∃ q : Idx 1,
      ‖(Real.sqrt (1 : ℝ))⁻¹ •
          ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1) -
        ((fun _ : Idx 1 => Transformer.basePoint 0) q : EucSpace 1)‖
          ≤ (1 / 32 : ℝ) / 2 := by simpa using hnear
  exact projectedSeparated_of_near_scaled_centers 1 1 100 (1 / 32) 1
    (by decide) (fun _ => Transformer.basePoint 0) hc _ (fun _ => hw)
    (by simpa using hnearScaled)

/-- The corrected proposition's numerical and geometric hypotheses are
satisfiable: one cap in `ℝ²`, two samples, and a small positive scale. -/
example :
    2 ≤ 2 ∧ 2 ≤ 2 ∧ (0 : ℝ) < 100 ∧ (0 : ℝ) < 1 / 32 ∧
      (0 : ℝ) < 1 / 1000 ∧ 1 ≤ 1 ∧
      isCentered 2 2 100 (1 / 32) 1 (fun _ => Transformer.basePoint 1) ∧
      (6 * ((1 / 1000 : ℝ) / Real.sqrt 1) * Real.sqrt 2)
            / (1 + ((1 / 1000 : ℝ) / Real.sqrt 1) * Real.sqrt 2)
          + ((1 / 1000 : ℝ) / Real.sqrt 1) * Real.sqrt (2 * 2 * Real.log 2)
            ≤ 1 / 32 := by
  have hempty : { c : ℝ |
      ∃ i j : Idx 1, i ≠ j ∧
        ∃ x ∈ sphericalCap 2 ((fun _ => Transformer.basePoint 1) i) (2 * (1 / 32)),
        ∃ y ∈ sphericalCap 2 ((fun _ => Transformer.basePoint 1) j) (2 * (1 / 32)),
          c = inner (𝕜 := ℝ) ((x : EucSpace 2)) ((y : EucSpace 2)) } = ∅ := by
    ext c
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨i, j, hij, -⟩
    exact hij (Subsingleton.elim i j)
  have hα : αDist 2 1 (fun _ => Transformer.basePoint 1) (1 / 32) = 0 := by
    rw [αDist, hempty, Real.sSup_empty]
  have hlog : Real.log (256 : ℝ) < 6 := by
    have heq : Real.log (256 : ℝ) = 8 * Real.log 2 := by
      rw [show (256 : ℝ) = 2 ^ 8 by norm_num, Real.log_pow]
      norm_num
    rw [heq]
    nlinarith [Real.log_two_lt_d9]
  have hcent : isCentered 2 2 100 (1 / 32) 1 (fun _ => Transformer.basePoint 1) := by
    constructor
    · omega
    · rw [hα, γβ]
      norm_num
      linarith
  have hsqrt2 : Real.sqrt (2 : ℝ) < 2 := by
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_nonneg (2 : ℝ)]
  have hlog2 : 0 ≤ Real.log (2 : ℝ) := Real.log_nonneg (by norm_num)
  have hsqrtlog : Real.sqrt (4 * Real.log 2) < 2 := by
    have hsq := Real.sq_sqrt (by positivity : (0 : ℝ) ≤ 4 * Real.log 2)
    nlinarith [Real.sqrt_nonneg (4 * Real.log 2), Real.log_two_lt_d9]
  have hden : 0 < 1 + (1 / 1000 : ℝ) * Real.sqrt 2 := by positivity
  have hbound : (6 / 1000 : ℝ) * Real.sqrt 2 /
      (1 + (1 / 1000 : ℝ) * Real.sqrt 2) ≤ (6 / 1000 : ℝ) * Real.sqrt 2 := by
    apply (div_le_iff₀ hden).mpr
    nlinarith [Real.sqrt_nonneg (2 : ℝ)]
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hcent, ?_⟩
  have htarget : (6 / 1000 : ℝ) * Real.sqrt 2 /
      (1 + (1 / 1000 : ℝ) * Real.sqrt 2) +
        (1 / 1000 : ℝ) * Real.sqrt (4 * Real.log 2) ≤ 1 / 32 := by
    linarith
  simp only [Real.sqrt_one, div_one,
    show (2 : ℝ) * 2 = 4 by norm_num]
  convert htarget using 1; ring


end Metastability
end Transformer

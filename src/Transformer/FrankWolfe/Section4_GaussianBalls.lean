/-
# Gaussian vertices: fixed balls lose their mass in high dimension

For a standard Gaussian vector in `ℝ^d`, membership in a fixed ball forces
every coordinate into a fixed interval. The probability of that interval is
strictly less than one, and independence bounds the ball's mass by its `d`-th
power. Thus the ball's mass tends to zero as `d` tends to infinity.

This is the qualitative input needed by the conditional tail argument for
`prop: d.to.infty` in §4 of arXiv:2508.09628v1. The manuscript instead uses
chi-square concentration about `d`. Both arguments give the same asserted
limit; the bound here is only for a radius fixed before taking that limit.
-/

import Transformer.FrankWolfe.Section4_GaussianTails
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Probability.Independence.Basic

open Real MeasureTheory ProbabilityTheory Filter
open scoped BigOperators NNReal ENNReal

namespace Transformer.FrankWolfe

/-- A finite interval has standard Gaussian mass strictly less than one;
arXiv:2508.09628v1, §4, the Gaussian input of `prop: d.to.infty`. -/
theorem gaussian_coordinate_ball_lt_one (R : ℝ) :
    (gaussianReal 0 1) {x : ℝ | ‖x‖ ≤ R} < 1 := by
  have hs : MeasurableSet {x : ℝ | ‖x‖ ≤ R} :=
    measurableSet_le (by fun_prop) measurable_const
  refine lt_of_le_of_ne prob_le_one ?_
  intro hmass
  have hzero : (gaussianReal 0 1) {x : ℝ | ‖x‖ ≤ R}ᶜ = 0 :=
    (prob_compl_eq_zero_iff hs).2 hmass
  have hvol : volume {x : ℝ | ‖x‖ ≤ R}ᶜ = 0 :=
    gaussianReal_absolutelyContinuous' 0 (by norm_num : (1 : ℝ≥0) ≠ 0) hzero
  have hinterval : volume (Set.Ioo (R + 1) (R + 2)) = 0 :=
    measure_mono_null (fun x hx =>
      show x ∉ {x : ℝ | ‖x‖ ≤ R} from fun hn => by
        change ‖x‖ ≤ R at hn
        have hxnorm : x ≤ ‖x‖ := by rw [Real.norm_eq_abs]; exact le_abs_self x
        linarith [hx.1]) hvol
  rw [Real.volume_Ioo, show R + 2 - (R + 1) = 1 by ring,
    ENNReal.ofReal_one] at hinterval
  exact one_ne_zero hinterval

/-- A Euclidean ball lies in its coordinate box; arXiv:2508.09628v1,
§4, the Gaussian input of `prop: d.to.infty`. -/
theorem euclidean_ball_preimage_subset_box (d : ℕ) (R : ℝ) :
    WithLp.toLp 2 ⁻¹' {x : EucSpace d | ‖x‖ ≤ R} ⊆
      Set.univ.pi (fun _ : Fin d => {x : ℝ | ‖x‖ ≤ R}) := by
  intro x hx i _
  exact (PiLp.norm_apply_le (WithLp.toLp 2 x) i).trans hx

/-- Independence bounds the mass of a fixed Euclidean ball by the `d`-th
power of the coordinate interval's mass; arXiv:2508.09628v1, §4,
`prop: d.to.infty`. This avoids any change in the claimed limit. -/
theorem stdGaussian_ball_le_pow (d : ℕ) (R : ℝ) :
    (stdGaussian (EucSpace d)) {x | ‖x‖ ≤ R} ≤
      ((gaussianReal 0 1) {x : ℝ | ‖x‖ ≤ R}) ^ d := by
  have hs : MeasurableSet {x : EucSpace d | ‖x‖ ≤ R} :=
    measurableSet_le (by fun_prop) measurable_const
  rw [← map_pi_eq_stdGaussian (ι := Fin d),
    Measure.map_apply (WithLp.measurable_toLp 2 (Fin d → ℝ)) hs]
  calc
    _ ≤ (Measure.pi fun _ : Fin d => gaussianReal 0 1)
        (Set.univ.pi fun _ : Fin d => {x : ℝ | ‖x‖ ≤ R}) :=
      measure_mono (euclidean_ball_preimage_subset_box d R)
    _ = _ := by simp

/-- The coordinate-box bound vanishes with the dimension;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
theorem tendsto_gaussian_box_mass (R : ℝ) :
    Tendsto (fun d : ℕ => ((gaussianReal 0 1) {x : ℝ | ‖x‖ ≤ R}) ^ d)
      atTop (nhds 0) :=
  ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one (gaussian_coordinate_ball_lt_one R)

/-- Standard Gaussian mass escapes every fixed ball as the dimension grows;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. The radius is fixed before the
limit, exactly as required by the conditional comparison estimate. -/
theorem tendsto_stdGaussian_ball_mass (R : ℝ) :
    Tendsto (fun d : ℕ => (stdGaussian (EucSpace d)) {x | ‖x‖ ≤ R})
      atTop (nhds 0) := by
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (tendsto_gaussian_box_mass R) (fun _ => bot_le)
    (fun d => stdGaussian_ball_le_pow d R)

/-- Eventually the fixed ball has mass smaller than any positive prescribed
probability; arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
theorem eventually_stdGaussian_ball_mass_lt (R : ℝ) {ε : ℝ≥0∞} (hε : 0 < ε) :
    ∀ᶠ d : ℕ in atTop, (stdGaussian (EucSpace d)) {x | ‖x‖ ≤ R} < ε :=
  (tendsto_order.mp (tendsto_stdGaussian_ball_mass R)).2 ε hε

/-- The positive tolerance can be `1/2`; arXiv:2508.09628v1, §4,
`prop: d.to.infty`. -/
example : (0 : ℝ≥0∞) < 1 / 2 := by norm_num

/-- The conditional comparison error tends to zero along positive radii
`R = n+1`; arXiv:2508.09628v1, §4, `prop: d.to.infty`.
This limit is taken after the fixed-radius dimension limit. -/
theorem tendsto_gaussian_cell_radius_error {l : ℝ} (hl : 0 < l) (M : ℝ) :
    Tendsto (fun n : ℕ => ENNReal.ofReal (M ^ 2 / (l * ((n : ℝ) + 1)) ^ 2))
      atTop (nhds 0) := by
  have heq : ∀ n : ℕ, M ^ 2 / (l * ((n : ℝ) + 1)) ^ 2 =
      (M ^ 2 / l ^ 2) * (1 / ((n : ℝ) + 1)) ^ 2 := by
    intro n
    have hn : (n : ℝ) + 1 ≠ 0 := by positivity
    field_simp [hl.ne', hn]
  simp_rw [heq]
  have hlim : Tendsto (fun n : ℕ => (M ^ 2 / l ^ 2) * (1 / ((n : ℝ) + 1)) ^ 2)
      atTop (nhds 0) := by
    simpa using (tendsto_const_nhds (x := M ^ 2 / l ^ 2)).mul
      ((tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).pow 2)
  simpa using ENNReal.tendsto_ofReal hlim

/-- The radius-error limit applies at `l = 1`; arXiv:2508.09628v1,
§4, `prop: d.to.infty`. -/
example : (0 : ℝ) < 1 := one_pos

/-- Choose a fixed positive radius with any prescribed positive conditional
error; arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
theorem exists_gaussian_cell_radius {l : ℝ} (hl : 0 < l) (M : ℝ)
    {ε : ℝ≥0∞} (hε : 0 < ε) :
    ∃ R : ℝ, 0 < R ∧ ENNReal.ofReal (M ^ 2 / (l * R) ^ 2) < ε := by
  obtain ⟨n, hn⟩ := ((tendsto_order.mp
    (tendsto_gaussian_cell_radius_error hl M)).2 ε hε).exists
  exact ⟨(n : ℝ) + 1, by positivity, hn⟩

/-- Both hypotheses of the radius choice hold at `l = 1`, `ε = 1/2`;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ≥0∞) < 1 / 2 := ⟨one_pos, by norm_num⟩

/-- Two distinct coordinates of the Gaussian vertex tuple have the product
law used by the conditional comparison argument; arXiv:2508.09628v1, §4,
`prop: d.to.infty`. -/
theorem map_pair_stdGaussian {d κ : ℕ} {i j : Fin κ} (hij : i ≠ j) :
    (Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d)).map
        (fun v => (v i, v j)) =
      (stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)) := by
  have hind : iIndepFun (fun i (v : Fin κ → EucSpace d) => v i)
      (Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d)) :=
    iIndepFun_pi (fun _ => measurable_id.aemeasurable)
  have hmap := (hind.indepFun hij).map_prod_eq_prod_map_map
    (measurable_pi_apply i).aemeasurable (measurable_pi_apply j).aemeasurable
  rw [(measurePreserving_eval (fun _ : Fin κ => stdGaussian (EucSpace d)) i).map_eq,
    (measurePreserving_eval (fun _ : Fin κ => stdGaussian (EucSpace d)) j).map_eq] at hmap
  exact hmap

/-- Distinct indices exist for two vertices; arXiv:2508.09628v1, §4,
`prop: d.to.infty`. -/
example : (0 : Fin 2) ≠ 1 := by decide

end Transformer.FrankWolfe

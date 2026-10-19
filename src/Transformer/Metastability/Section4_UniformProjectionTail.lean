module

public import Transformer.Metastability.Section4_OrthonormalCoordinates

/-!
# Uniform tails for the original finite-frame projection

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The original projection threshold is controlled uniformly over every
actual orthonormal frame. The fixed half-net on its coordinate sphere
reduces the actual projection event to finitely many original scalar
tail events with threshold `r/2` and actual unit ambient directions.

The even moments and Markov estimate already proved for the original
uniform law supply all probability inputs. The resulting bound is
the net cardinality times `d^(-1/32)` at an actual logarithmic degree.
A finite budget over all previous frame lengths at most `n` depends
only on `n`; the original large-dimension threshold can absorb it.
No conditional independence or projection estimate is assumed here.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Metastability

/-- The finite sum of actual net sizes for all previous frame lengths
up to `n`. Source: arXiv:2410.06833v1, §4, the dimension threshold
depending only on `n` in the original projection concentration. -/
noncomputable def sphericalNetBudget (n : ℕ) : ℕ :=
  ∑ m ∈ Finset.range (n + 1), (sphericalHalfNet m).card

/-- The actual net of each allowable previous frame is bounded by the
fixed budget. Source: arXiv:2410.06833v1, §4, the original uniform
projection estimate over all earlier frame lengths. -/
theorem sphericalHalfNet_card_le_budget {m n : ℕ} (hmn : m ≤ n) :
    (sphericalHalfNet m).card ≤ sphericalNetBudget n := by
  unfold sphericalNetBudget
  exact Finset.single_le_sum (s := Finset.range (n + 1))
    (f := fun j => (sphericalHalfNet j).card) (fun j hj => Nat.zero_le _)
    (Finset.mem_range.mpr (by omega))

/-- The original finite budget is positive for every positive maximum
frame length. Source: arXiv:2410.06833v1, §4, the nonempty directional
cover used in the original projection concentration bound. -/
theorem sphericalNetBudget_pos {n : ℕ} (hn : 0 < n) : 0 < sphericalNetBudget n :=
  (Finset.card_pos.mpr (sphericalHalfNet_nonempty (by decide : 0 < 1))).trans_le
    (sphericalHalfNet_card_le_budget (by omega : 1 ≤ n))

/-- The actual projection threshold event is measurable jointly in the
previous frame and the original sample point.
Source: arXiv:2410.06833v1, §4, conditioning the original projection
estimate on the previously constructed frame. No orthonormality is needed. -/
theorem measurableSet_frameProjection_threshold {d m : ℕ} (r : ℝ) :
    MeasurableSet {p : (Fin m → EucSpace d) × SSphere d |
      r ≤ ‖frameProjection p.1 (p.2 : EucSpace d)‖} := by
  have hc : Continuous (fun p : (Fin m → EucSpace d) × SSphere d =>
      frameProjection p.1 (p.2 : EucSpace d)) := by
    unfold frameProjection frameSynthesis frameCoordinates
    fun_prop
  exact (isClosed_le continuous_const hc.norm).measurableSet

/-- Every actual projection threshold event is contained in the finite
union of the actual net-direction threshold events.
Source: arXiv:2410.06833v1, §4, the original projection concentration
bound; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem frameProjection_threshold_subset {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (r : ℝ) (hr : 0 < r) :
    {x : SSphere d | r ≤ ‖frameProjection v (x : EucSpace d)‖} ⊆
      ⋃ b ∈ sphericalHalfNet m,
        {x : SSphere d | r / 2 ≤ |inner (𝕜 := ℝ) (frameSynthesis v (b : EucSpace m))
          (x : EucSpace d)|} := by
  intro x hx
  change r ≤ ‖frameProjection v (x : EucSpace d)‖ at hx
  rw [norm_frameProjection v hv] at hx
  obtain ⟨b, hb, hdetect⟩ := sphericalHalfNet_detects_threshold
    (frameCoordinates v (x : EucSpace d)) r hr hx
  apply mem_iUnion.mpr
  refine ⟨b, mem_iUnion.mpr ⟨hb, ?_⟩⟩
  change r / 2 ≤ |inner (𝕜 := ℝ) (frameSynthesis v (b : EucSpace m)) (x : EucSpace d)|
  rwa [inner_frameSynthesis]

/-- The original finite-frame projection tail is bounded by its actual
finite net size times the original even-moment tail.
Source: arXiv:2410.06833v1, §4, the projection concentration estimate
in `prop: concentration unif`. The frame is an actual orthonormal list. -/
theorem uniform_frameProjection_tail_le_pow {d m : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v)
    (r : ℝ) (hr : 0 < r) (k : ℕ) :
    ν.real {x : SSphere d | r ≤ ‖frameProjection v (x : EucSpace d)‖} ≤
      ((sphericalHalfNet m).card : ℝ) * ((k : ℝ) / ((d : ℝ) * (r / 2) ^ 2)) ^ k := by
  let : IsProbabilityMeasure ν := hν.1
  apply (measureReal_mono (frameProjection_threshold_subset v hv r hr) (measure_ne_top ν _)).trans
  apply (measureReal_biUnion_finset_le (sphericalHalfNet m) _).trans
  have hsum := Finset.sum_le_sum (s := sphericalHalfNet m) (fun b hb =>
    uniform_directional_tail_le_pow hd ν hν
      (by rw [norm_frameSynthesis v hv, norm_eq_of_mem_sphere b]) (r / 2) (by positivity) k)
  simpa only [Finset.sum_const, nsmul_eq_mul] using hsum

/-- At an actual logarithmic degree the original projection tail has
the dimension-power bound uniformly over all actual previous frames.
Source: arXiv:2410.06833v1, §4, the concentration estimate in
`prop: concentration unif`; no frame-dependent tail input is assumed. -/
theorem uniform_frameProjection_tail_le_dimension_power {d m : ℕ} (hd : 0 < d)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v)
    (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * (r / 2) ^ 2) (hlog : Real.log (d : ℝ) / 32 ≤ k) :
    ν.real {x : SSphere d | r ≤ ‖frameProjection v (x : EucSpace d)‖} ≤
      ((sphericalHalfNet m).card : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 32) := by
  let : IsProbabilityMeasure ν := hν.1
  apply (measureReal_mono (frameProjection_threshold_subset v hv r hr) (measure_ne_top ν _)).trans
  apply (measureReal_biUnion_finset_le (sphericalHalfNet m) _).trans
  have hsum := Finset.sum_le_sum (s := sphericalHalfNet m) (fun b hb =>
    uniform_directional_tail_le_dimension_power hd ν hν
      (by rw [norm_frameSynthesis v hv, norm_eq_of_mem_sphere b]) (r / 2) (by positivity) k hk hlog)
  simpa only [Finset.sum_const, nsmul_eq_mul] using hsum

/-- The original projection tail is bounded by a constant depending
only on the maximum previous frame length, at the original dimension
power. Source: arXiv:2410.06833v1, §4, the uniform estimate used
at every stage of the original independent-sample approximation. -/
theorem uniform_frameProjection_tail_le_budget {d m n : ℕ} (hd : 0 < d)
    (hmn : m ≤ n) (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v)
    (r : ℝ) (hr : 0 < r) (k : ℕ)
    (hk : 4 * (k : ℝ) ≤ (d : ℝ) * (r / 2) ^ 2) (hlog : Real.log (d : ℝ) / 32 ≤ k) :
    ν.real {x : SSphere d | r ≤ ‖frameProjection v (x : EucSpace d)‖} ≤
      (sphericalNetBudget n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 32) := by
  apply (uniform_frameProjection_tail_le_dimension_power hd ν hν v hv r hr k hk hlog).trans
  exact mul_le_mul_of_nonneg_right
    (by exact_mod_cast sphericalHalfNet_card_le_budget hmn) (Real.rpow_nonneg (by positivity) _)

/-- An actual two-vector frame in dimension 32, the original uniform
law, a positive radius below one, a logarithmic degree and the dimension
scale jointly satisfy the projection-tail hypotheses. -/
example : (0 : ℕ) < 32 ∧ (2 : ℕ) ≤ 2 ∧ IsUniformOn 32 (MeanField.uniformLaw 32) ∧
    Orthonormal ℝ (fun i : Fin 2 => (EuclideanSpace.single
      (Fin.castLE (by decide : 2 ≤ 32) i) 1 : EucSpace 32)) ∧
    (0 : ℝ) < 3 / 4 ∧ 4 * (1 : ℝ) ≤ 32 * ((3 / 4 : ℝ) / 2) ^ 2 ∧
    Real.log (32 : ℝ) / 32 ≤ 1 := by
  refine ⟨by decide, le_rfl, isUniformOn_uniformLaw (by norm_num), ?_,
    by norm_num, by norm_num, ?_⟩
  · rw [orthonormal_iff_ite]
    intro i j
    simp [EuclideanSpace.inner_single_left]
  · have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 32)
    linarith

end Transformer.Metastability

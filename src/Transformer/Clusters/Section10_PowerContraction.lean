module

public import Transformer.Clusters.Section10_PowerMetric
public import Transformer.Clusters.Section5_Mix

/-!
# Spectral gap gives a contracting quadratic metric

ArXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`, and §10.
The complementary spectral radius is strictly below the leading positive
eigenvalue. Its geometric power bound supplies a finite cutoff at which
the operator contracts at an intermediate rate. The finite-power energy
then gives the same contraction in an actual positive quadratic metric.

This derives the operator bound needed by the growth argument without a
diagonalizability assumption or an unproved choice of Jordan coordinates.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d : ℕ}

/-- A geometric spectral-radius bound supplies a positive finite cutoff
with power norm at most the corresponding scale power.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`;
this finite-power estimate replaces the small Jordan-block entries. -/
theorem IsSpectralRadiusLtOn.exists_power_cutoff (G : Submodule ℝ (EucSpace d))
    (V : ParamMatrix d) (R : ℝ) (h : IsSpectralRadiusLtOn G V R) :
    ∃ m : ℕ, 0 < m ∧ ∀ w : EucSpace d, w ∈ G →
      ‖(V ^ m) w‖ ≤ R ^ m * ‖w‖ := by
  obtain ⟨C, r, hC, hr, hrR, hpower⟩ := h
  have hR : 0 < R := hr.trans_lt hrR
  have hlim : Tendsto (fun k : ℕ => C * (r / R) ^ k) atTop (nhds 0) := by
    simpa only [mul_zero] using
      (tendsto_pow_atTop_nhds_zero_of_lt_one (div_nonneg hr hR.le)
        ((div_lt_one₀ hR).mpr hrR)).const_mul C
  obtain ⟨m, hsmall, hm⟩ :=
    ((hlim.eventually (eventually_lt_nhds (show (0 : ℝ) < 1 from one_pos))).and
      (eventually_ge_atTop (1 : ℕ))).exists
  refine ⟨m, hm, fun w hw => (hpower m w hw).trans ?_⟩
  have hscale : C * r ^ m ≤ R ^ m := by
    rw [div_pow, ← mul_div_assoc] at hsmall
    have hs := (div_le_iff₀ (pow_pos hR m)).mp hsmall.le
    simpa only [one_mul] using hs
  exact mul_le_mul_of_nonneg_right hscale (norm_nonneg w)

/-- A trivial complementary space satisfies the full spectral input.
Source: arXiv:2305.05465v6, §5, `d:goodmulti`, the identity value map. -/
example (d : ℕ) : IsSpectralRadiusLtOn (⊥ : Submodule ℝ (EucSpace d))
    (1 : ParamMatrix d) 1 :=
  (isGoodTripleMulti_one d).2.2.2.2.2.2

/-- The last power controls the boundary term in the energy shift, so the
whole quadratic energy contracts at the selected rate on the subspace.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`. -/
theorem powerEnergy_apply_le (V : ParamMatrix d) (G : Submodule ℝ (EucSpace d))
    (R : ℝ) (hR : 0 < R) (m : ℕ)
    (hpower : ∀ w : EucSpace d, w ∈ G → ‖(V ^ m) w‖ ≤ R ^ m * ‖w‖)
    (x : EucSpace d) (hx : x ∈ G) :
    powerEnergy V R m (V x) ≤ R ^ 2 * powerEnergy V R m x := by
  have hRm : 0 < R ^ m := pow_pos hR m
  have hscaled : (R ^ m)⁻¹ * ‖(V ^ m) x‖ ≤ ‖x‖ := by
    calc
      (R ^ m)⁻¹ * ‖(V ^ m) x‖ ≤ (R ^ m)⁻¹ * (R ^ m * ‖x‖) :=
        mul_le_mul_of_nonneg_left (hpower x hx) (inv_nonneg.mpr hRm.le)
      _ = ‖x‖ := by rw [← mul_assoc, inv_mul_cancel₀ hRm.ne', one_mul]
  have hlast := (pow_le_pow_iff_left₀
    (mul_nonneg (inv_nonneg.mpr hRm.le) (norm_nonneg _))
    (norm_nonneg x) two_ne_zero).mpr hscaled
  rw [mul_pow] at hlast
  rw [powerEnergy_apply V R hR.ne']
  exact mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg R)

/-- The positive rate, finite power bound, and subspace membership all
hold for the identity map on a zero complementary vector.
Source: arXiv:2305.05465v6, §9, `e:notdi`. -/
example (d : ℕ) : (0 : ℝ) < 1 ∧
    (∀ w : EucSpace d, w ∈ (⊥ : Submodule ℝ (EucSpace d)) →
      ‖((1 : ParamMatrix d) ^ 1) w‖ ≤ (1 : ℝ) ^ 1 * ‖w‖) ∧
    (0 : EucSpace d) ∈ (⊥ : Submodule ℝ (EucSpace d)) := by
  refine ⟨one_pos, ?_, Submodule.zero_mem _⟩
  intro w hw
  have hw0 : w = 0 := by simpa only [Submodule.mem_bot] using hw
  rw [hw0]
  simp only [map_zero, norm_zero, mul_zero, le_refl]

/-- The quadratic estimate becomes an operator-norm estimate for any
linear square root representing the finite-power energy.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`.
The representation is explicit so that the constructed root can be used. -/
theorem powerMetric_apply_le (V : ParamMatrix d) (G : Submodule ℝ (EucSpace d))
    (R : ℝ) (hR : 0 < R) (m : ℕ)
    (hpower : ∀ w : EucSpace d, w ∈ G → ‖(V ^ m) w‖ ≤ R ^ m * ‖w‖)
    (B : ParamMatrix d)
    (henergy : ∀ w : EucSpace d, ‖B w‖ ^ 2 = powerEnergy V R m w)
    (x : EucSpace d) (hx : x ∈ G) : ‖B (V x)‖ ≤ R * ‖B x‖ := by
  apply (pow_le_pow_iff_left₀ (norm_nonneg _) (mul_nonneg hR.le (norm_nonneg _))
    two_ne_zero).mp
  rw [mul_pow, henergy, henergy]
  exact powerEnergy_apply_le V G R hR m hpower x hx

/-- The representation input, rate, power bound, and membership hold
simultaneously at cutoff one with the identity square root.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`. -/
example (d : ℕ) : (0 : ℝ) < 1 ∧
    (∀ w : EucSpace d, w ∈ (⊥ : Submodule ℝ (EucSpace d)) →
      ‖((1 : ParamMatrix d) ^ 1) w‖ ≤ (1 : ℝ) ^ 1 * ‖w‖) ∧
    (∀ w : EucSpace d, ‖(1 : ParamMatrix d) w‖ ^ 2 =
      powerEnergy (1 : ParamMatrix d) 1 1 w) ∧
    (0 : EucSpace d) ∈ (⊥ : Submodule ℝ (EucSpace d)) := by
  refine ⟨one_pos, ?_, ?_, Submodule.zero_mem _⟩
  · intro w hw
    have hw0 : w = 0 := by simpa only [Submodule.mem_bot] using hw
    rw [hw0]
    simp only [map_zero, norm_zero, mul_zero, le_refl]
  · intro w
    rw [powerEnergy_succ]
    simp only [powerEnergy, Finset.range_zero, Finset.sum_empty, pow_zero,
      inv_one, one_pow, one_apply_eq_self, one_mul, zero_add]

/-- **A spectral gap constructs an actual contracting linear metric.**
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`, as used
in §10. The contraction rate is positive and strictly below the leading
eigenvalue. The metric controls the original norm on the whole space;
only its operator contraction is restricted to the complementary space. -/
theorem IsSpectralRadiusLtOn.exists_contracting_powerMetric
    (G : Submodule ℝ (EucSpace d)) (V : ParamMatrix d) (lam : ℝ)
    (h : IsSpectralRadiusLtOn G V lam) :
    ∃ (R : ℝ) (B : ParamMatrix d), 0 < R ∧ R < lam ∧
      (∀ x : EucSpace d, ‖x‖ ≤ ‖B x‖) ∧
      ∀ x : EucSpace d, x ∈ G → ‖B (V x)‖ ≤ R * ‖B x‖ := by
  obtain ⟨C, r, hC, hr, hrLam, hpower⟩ := h
  let R := (r + lam) / 2
  have hrR : r < R := by dsimp [R]; linarith
  have hR : 0 < R := hr.trans_lt hrR
  have hRlam : R < lam := by dsimp [R]; linarith
  have hspectral : IsSpectralRadiusLtOn G V R :=
    ⟨C, r, hC, hr, hrR, hpower⟩
  obtain ⟨m, hm, hmPower⟩ := hspectral.exists_power_cutoff G V R
  obtain ⟨B, _, henergy, hdom⟩ := exists_powerMetric V R m hm
  refine ⟨R, B, hR, hRlam, hdom, fun x hx => ?_⟩
  exact powerMetric_apply_le V G R hR m hmPower B henergy x hx

/-- The complete spectral hypothesis is realized by a good triple with
multiplicity. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example (d : ℕ) : IsSpectralRadiusLtOn (⊥ : Submodule ℝ (EucSpace d))
    (1 : ParamMatrix d) 1 :=
  (isGoodTripleMulti_one d).2.2.2.2.2.2

end Transformer.Clusters

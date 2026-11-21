module

public import Transformer.Grokking.FlatSpectrum.ContractionDual
public import Transformer.Grokking.FlatSpectrum.CubicInterpolant
public import Transformer.Grokking.FlatSpectrum.ProductCostSVD
public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# Theorem A for one product layer

Source: `review/flat_spectrum.md`, NV10. The function `σ ↦ (3/2) σ^{2/3}` is concave, so it lies
below its tangent at the flat value `s`. Summed over the singular values, the tangent turns the
nuclear norm into an upper bound on the product cost that is tight at a flat `K`. So whatever
beats `K` in nuclear norm also beats it in product cost.

Theorem A gives a matrix that agrees with a flat `K` on `Ω` and has a smaller nuclear norm. The
segment towards it stays on `Ω`, and by the triangle inequality its nuclear norm is below `K`'s
all the way. So `K` is not even a local minimum of the product cost among interpolants of `Ω`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Filter Topology
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The tangent of the concave `(3/2) σ^{2/3}` at `v³`: `(3/2) σ^{2/3} ≤ σ / v + v² / 2`. -/
lemma rpow_two_thirds_le {σ v : ℝ} (hσ : 0 ≤ σ) (hv : 0 < v) :
    3 / 2 * σ ^ (2 / 3 : ℝ) ≤ v⁻¹ * σ + 1 / 2 * v ^ 2 := by
  have hu0 : 0 ≤ σ ^ (1 / 3 : ℝ) := Real.rpow_nonneg hσ _
  have hu3 : (σ ^ (1 / 3 : ℝ)) ^ 3 = σ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hσ]
    norm_num
  have hu2 : σ ^ (2 / 3 : ℝ) = (σ ^ (1 / 3 : ℝ)) ^ 2 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hσ]
    norm_num
  rw [hu2]
  clear hu2
  generalize σ ^ (1 / 3 : ℝ) = u at hu0 hu3
  subst hu3
  have hE : v⁻¹ * u ^ 3 + 1 / 2 * v ^ 2 - 3 / 2 * u ^ 2 =
      (u - v) ^ 2 * (2 * u + v) / (2 * v) := by
    field_simp
    ring
  have : 0 ≤ (u - v) ^ 2 * (2 * u + v) / (2 * v) :=
    div_nonneg (mul_nonneg (sq_nonneg _) (by linarith)) (by linarith)
  linarith

example := rpow_two_thirds_le (σ := 1) (v := 1) zero_le_one one_pos

/-- Whatever beats a flat `K` in nuclear norm beats it in product cost. -/
theorem productCost_lt_of_nuclearNorm_lt {K X : Matrix n n ℂ} {s : ℝ} (hs : 0 < s)
    (hK : ∀ i, singularValues K i = s) (h : nuclearNorm X < nuclearNorm K) :
    productCost X < productCost K := by
  obtain ⟨v, hvdef⟩ : ∃ v : ℝ, v = s ^ (1 / 3 : ℝ) := ⟨_, rfl⟩
  have hv : 0 < v := hvdef ▸ Real.rpow_pos_of_pos hs _
  have hs3 : s = v ^ 3 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hs.le]
    norm_num
  have hs2 : s ^ (2 / 3 : ℝ) = v ^ 2 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hs.le]
    norm_num
  have hK1 : nuclearNorm K = Fintype.card n * v ^ 3 := by
    simp only [nuclearNorm, hK, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    rw [← hs3]
  have hK2 : productCost K = 3 / 2 * (Fintype.card n * v ^ 2) := by
    simp only [productCost, hK, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, hs2]
  have hX : productCost X ≤ v⁻¹ * nuclearNorm X + Fintype.card n * (1 / 2 * v ^ 2) := by
    rw [productCost, Finset.mul_sum, nuclearNorm, Finset.mul_sum]
    calc ∑ i, 3 / 2 * singularValues X i ^ (2 / 3 : ℝ)
        ≤ ∑ i, (v⁻¹ * singularValues X i + 1 / 2 * v ^ 2) :=
          Finset.sum_le_sum fun i _ => rpow_two_thirds_le (Real.sqrt_nonneg _) hv
      _ = _ := by
          rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hlt : v⁻¹ * nuclearNorm X < v⁻¹ * (Fintype.card n * v ^ 3) := by
    rw [← hK1]
    exact mul_lt_mul_of_pos_left h (inv_pos.mpr hv)
  have hv3 : v⁻¹ * (Fintype.card n * v ^ 3) = Fintype.card n * v ^ 2 := by
    field_simp
  rw [hK2]
  linarith

example := productCost_lt_of_nuclearNorm_lt (K := cubicKernel 3) (X := 0)
  (s := √((3 : ℕ) : ℝ)) (by positivity) (singularValues_cubicKernel 3)
  (by rw [nuclearNorm_zero, nuclearNorm_cubicKernel]; positivity)

/-- **Theorem A for one product layer** (NV10): if all singular values of `K` are equal and no
entry vanishes, then for every proper `Ω`, `K` is not a local minimum of the product cost among
the matrices that agree with it on `Ω`. -/
theorem not_isLocalMinOn_productCost {K : Matrix n n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) :
    ¬ IsLocalMinOn productCost {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} K := by
  intro hloc
  obtain ⟨⟨a, b⟩, -⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  have hc : 0 < c := by
    have hc0 : 0 ≤ c := by
      rw [← h a]
      exact Real.sqrt_nonneg _
    refine hc0.lt_of_ne' ?_
    rintro rfl
    have h2 : frobeniusSq K = 0 := by
      rw [← sum_singularValues_sq]
      simp [h]
    exact hK a b (by rw [eq_zero_of_frobeniusSq_eq_zero h2, Matrix.zero_apply])
  obtain ⟨X, hXS, hX⟩ : ∃ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2},
      nuclearNorm X < nuclearNorm K := by
    by_contra hcon
    push Not at hcon
    exact not_isMinOn_of_singularValues_eq h hK hΩ (isMinOn_iff.mpr hcon)
  obtain ⟨γ, hγdef⟩ : ∃ γ : ℝ → Matrix n n ℂ, γ = fun t : ℝ => K + (t : ℂ) • (X - K) :=
    ⟨_, rfl⟩
  have hcont : Continuous γ := by
    rw [hγdef]
    exact continuous_const.add (Complex.continuous_ofReal.smul continuous_const)
  have hγ0 : γ 0 = K := by simp [hγdef]
  have hlim : Tendsto γ (𝓝[>] 0)
      (𝓝[{X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2}] K) := by
    refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
    · have := hcont.tendsto 0
      rw [hγ0] at this
      exact this.mono_left nhdsWithin_le_nhds
    · refine Eventually.of_forall fun t ij hij => ?_
      simp only [hγdef, Matrix.add_apply, Matrix.smul_apply, Matrix.sub_apply, smul_eq_mul]
      rw [hXS ij hij]
      ring
  have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), productCost K ≤ productCost (γ t) :=
    hlim.eventually (p := fun x => productCost K ≤ productCost x) hloc
  have hev2 : ∀ᶠ t : ℝ in 𝓝[>] 0, t ∈ Set.Ioc (0 : ℝ) 1 := Ioc_mem_nhdsGT one_pos
  obtain ⟨t, ht1, ht2⟩ := (hev.and hev2).exists
  have hγt : γ t = ((1 - t : ℝ) : ℂ) • K + (t : ℂ) • X := by
    ext i j
    simp only [hγdef, Matrix.add_apply, Matrix.smul_apply, Matrix.sub_apply, smul_eq_mul]
    push_cast
    ring
  have hnorm : nuclearNorm (γ t) ≤ (1 - t) * nuclearNorm K + t * nuclearNorm X := by
    rw [hγt]
    refine (nuclearNorm_add_le _ _).trans ?_
    rw [nuclearNorm_smul, nuclearNorm_smul, Complex.norm_real, Complex.norm_real,
      Real.norm_of_nonneg (by linarith [ht2.2]), Real.norm_of_nonneg ht2.1.le]
  have hlt : nuclearNorm (γ t) < nuclearNorm K := by
    have := mul_pos ht2.1 (sub_pos.mpr hX)
    linarith
  exact absurd ht1 (not_le.mpr (productCost_lt_of_nuclearNorm_lt hc h hlt))

/-- No entry of the `a³ + a b` kernel vanishes. -/
lemma cubicKernel_apply_ne_zero (p : ℕ) [NeZero p] (a b : ZMod p) : cubicKernel p a b ≠ 0 := by
  intro h
  have h1 : ‖cubicKernel p a b‖ = 1 := by simp [cubicKernel, taskKernel]
  rw [h, norm_zero] at h1
  exact zero_ne_one h1

example := not_isLocalMinOn_productCost (K := cubicKernel 3) (singularValues_cubicKernel 3)
  (cubicKernel_apply_ne_zero 3) (Ω := ∅) Set.empty_ne_univ

end Transformer.Grokking.FlatSpectrum

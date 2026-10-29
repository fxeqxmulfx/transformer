module

public import Mathlib.Analysis.Convex.Cone.Extension
public import Transformer.Grokking.FlatSpectrum.ContractionDual

/-!
# A dual certificate for every minimum

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 5.1, Theorem A without its
invertibility: `K` is a minimum of the nuclear norm among the matrices equal to it on `Ω` exactly
when some contraction `Y` vanishing off `Ω` has `tr(Yᴴ K) = ‖K‖_S1`
(`isMinOn_iff_exists_isContraction`). If: `tr(Yᴴ X) = tr(Yᴴ K)` for every such `X`, and
`|tr(Yᴴ X)| ≤ ‖X‖_S1`. Only if (`exists_isContraction_of_isMinOn`), by the Hahn–Banach theorem
over `ℝ`: on the span of `K` and the subspace `L` of the matrices vanishing on `Ω`, the
functional `H + t K ↦ t ‖K‖_S1` lies below the nuclear norm, since
`‖H + t K‖_S1 = t ‖K + H / t‖_S1 ≥ t ‖K‖_S1` for `t > 0`. An extension `g ≤ ‖·‖_S1` to all
matrices is `Re tr(Yᴴ ·)` (`re_trace_rieszMatrix`), with `Y` a contraction
(`isContraction_of_re_trace_le`) vanishing off `Ω` (`g = 0` on `L`) and `Re tr(Yᴴ K) = ‖K‖_S1`.
If `K ∈ L`, the minimum is `0` and `Y = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **A minimum of M has a dual certificate**: if `K` is a minimum of the nuclear norm among the
matrices equal to it on `Ω`, some contraction `Y` vanishing off `Ω` has `tr(Yᴴ K) = ‖K‖_S1`. -/
theorem exists_isContraction_of_isMinOn {K : Matrix m n ℂ} {Ω : Set (m × n)}
    (hK : IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ Y : Matrix m n ℂ, (∀ i j, (i, j) ∉ Ω → Y i j = 0) ∧ IsContraction Y ∧
      (Yᴴ * K).trace = nuclearNorm K := by
  rw [isMinOn_iff] at hK
  -- the real subspace `L` of the matrices vanishing on `Ω`
  let L : Submodule ℝ (Matrix m n ℂ) :=
    { carrier := {H | ∀ ij ∈ Ω, H ij.1 ij.2 = 0}
      add_mem' := fun ha hb ij hij => by
        rw [Matrix.add_apply, ha ij hij, hb ij hij, add_zero]
      zero_mem' := fun _ _ => rfl
      smul_mem' := fun c H hH ij hij => by
        rw [Matrix.smul_apply, hH ij hij, smul_zero] }
  have hL (H : Matrix m n ℂ) : H ∈ L ↔ ∀ ij ∈ Ω, H ij.1 ij.2 = 0 := Iff.rfl
  -- `t ‖K‖_S1 ≤ ‖H + t K‖_S1` for `H ∈ L`
  have hbound (H : Matrix m n ℂ) (hH : H ∈ L) (t : ℝ) :
      t * nuclearNorm K ≤ nuclearNorm (H + t • K) := by
    rcases le_or_gt t 0 with ht | ht
    · exact (mul_nonpos_of_nonpos_of_nonneg ht (nuclearNorm_nonneg K)).trans
        (nuclearNorm_nonneg _)
    · have h1 := hK (K + t⁻¹ • H) fun ij hij => by
        rw [Matrix.add_apply, Matrix.smul_apply, (hL H).1 hH ij hij, smul_zero, add_zero]
      rw [show H + t • K = t • (K + t⁻¹ • H) by
          rw [smul_add, smul_smul, mul_inv_cancel₀ ht.ne', one_smul, add_comm],
        nuclearNorm_real_smul ht.le]
      exact mul_le_mul_of_nonneg_left h1 ht.le
  by_cases hKL : K ∈ L
  · -- `0` equals `K` on `Ω`, so `‖K‖_S1 = 0`
    have h0 := hK 0 fun ij hij => ((hL K).1 hKL ij hij).symm
    rw [nuclearNorm_zero] at h0
    refine ⟨0, fun _ _ _ => rfl, isContraction_zero, ?_⟩
    rw [conjTranspose_zero, Matrix.zero_mul, trace_zero,
      le_antisymm h0 (nuclearNorm_nonneg K), Complex.ofReal_zero]
  · set f₀ := (0 : Matrix m n ℂ →ₗ[ℝ] ℝ).toPMap L with hf₀
    have hK₀ : K ∉ f₀.domain := by rwa [hf₀, LinearMap.toPMap_domain]
    have hdom (x : Matrix m n ℂ) (hx : x ∈ L) : x ∈ f₀.domain := by
      rwa [hf₀, LinearMap.toPMap_domain]
    have hf0 (x : f₀.domain) : f₀ x = 0 := rfl
    obtain ⟨g, hgf, hgN⟩ := exists_extension_of_le_sublinear
      (f₀.supSpanSingleton K (nuclearNorm K) hK₀) nuclearNorm
      (fun c hc X => nuclearNorm_real_smul hc.le X) nuclearNorm_add_le (by
        rintro ⟨x, hx⟩
        have hx' := hx
        rw [LinearPMap.domain_supSpanSingleton, hf₀, LinearMap.toPMap_domain] at hx'
        obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.1 hx'
        obtain ⟨c, rfl⟩ := Submodule.mem_span_singleton.1 hz
        have hfv := LinearPMap.supSpanSingleton_apply_mk f₀ K (nuclearNorm K) hK₀ y
          (hdom y hy) c
        rw [hf0, zero_add, RingHom.id_apply, smul_eq_mul] at hfv
        rw [hfv]
        exact hbound y hy c)
    -- `g = 0` on `L` and `g K = ‖K‖_S1`
    have hgL (H : Matrix m n ℂ) (hH : H ∈ L) : g H = 0 := by
      have h1 := hgf ⟨H, Submodule.mem_sup_left (hdom H hH)⟩
      rw [LinearPMap.supSpanSingleton_apply_mk_of_mem _ _ _ (hdom H hH), hf0] at h1
      exact h1
    have hgK : g K = nuclearNorm K := by
      have h1 := hgf ⟨K, Submodule.mem_sup_right (Submodule.mem_span_singleton_self K)⟩
      rwa [LinearPMap.supSpanSingleton_apply_self] at h1
    have hY : IsContraction (rieszMatrix g) :=
      isContraction_of_re_trace_le fun X => by rw [re_trace_rieszMatrix]; exact hgN X
    refine ⟨rieszMatrix g, fun i j hij => ?_, hY, ?_⟩
    · -- the matrix units off `Ω` lie in `L`
      have hE (c : ℂ) : single i j c ∈ L := fun ij hij' => by
        rw [single_apply, ite_eq_right]
        rintro ⟨rfl, rfl⟩
        exact hij hij'
      rw [rieszMatrix, of_apply, hgL _ (hE 1), hgL _ (hE Complex.I)]
      rfl
    · have hre : ((rieszMatrix g)ᴴ * K).trace.re = nuclearNorm K := by
        rw [re_trace_rieszMatrix, hgK]
      have h0 : 0 ≤ ((rieszMatrix g)ᴴ * K).trace := Complex.re_eq_norm.1
        (le_antisymm (Complex.re_le_norm _) (hre ▸ norm_trace_le_nuclearNorm hY K))
      exact Complex.ext (by rw [hre, Complex.ofReal_re])
        (by rw [← (Complex.nonneg_iff.1 h0).2, Complex.ofReal_im])

example := exists_isContraction_of_isMinOn (K := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := ∅)
  (isMinOn_iff.2 fun X _ => by rw [nuclearNorm_zero]; exact nuclearNorm_nonneg X)

/-- **Theorem A without invertibility**: `K` is a minimum of the nuclear norm among the matrices
equal to it on `Ω` exactly when some contraction `Y` vanishing off `Ω` has
`tr(Yᴴ K) = ‖K‖_S1`. -/
theorem isMinOn_iff_exists_isContraction {K : Matrix m n ℂ} {Ω : Set (m × n)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∃ Y : Matrix m n ℂ, (∀ i j, (i, j) ∉ Ω → Y i j = 0) ∧ IsContraction Y ∧
        (Yᴴ * K).trace = nuclearNorm K := by
  refine ⟨exists_isContraction_of_isMinOn, ?_⟩
  rintro ⟨Y, hYΩ, hY, hYK⟩
  refine isMinOn_iff.2 fun X hX => ?_
  have h : (Yᴴ * X).trace = (Yᴴ * K).trace := by
    rw [trace_conjTranspose_mul, trace_conjTranspose_mul]
    refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
    by_cases hij : (i, j) ∈ Ω
    · rw [hX (i, j) hij]
    · rw [hYΩ i j hij, star_zero, zero_mul, zero_mul]
  have h1 := norm_trace_le_nuclearNorm hY X
  rwa [h, hYK, Complex.norm_real, Real.norm_of_nonneg (nuclearNorm_nonneg K)] at h1

end Transformer.Grokking.FlatSpectrum

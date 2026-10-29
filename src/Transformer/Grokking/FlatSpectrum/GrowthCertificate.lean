module

public import Mathlib.Analysis.Convex.Cone.Extension
public import Transformer.Grokking.FlatSpectrum.LinearGrowth

/-!
# Linear growth exactly when item 1.2's certificate exists

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 6.1. Let `K = U diag(s) Vᴴ` with
`s > 0`, `P = 1 − U Uᴴ`, `P' = 1 − V Vᴴ`, and let `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1`, `c > 0`, for
every `H` vanishing on `Ω`. The first-order bound (`exists_nuclearNorm_add_smul_le`) at `t → 0⁺`
gives `c ‖H‖_S1 ≤ Re tr((U Vᴴ)ᴴ H) + ‖P H P'‖_S1` for every such `H`. So for
`a = max(1 − c, 1/2) < 1` the sublinear `p(X) = Re tr((U Vᴴ)ᴴ X) + a ‖P X P'‖_S1` is nonnegative
on them, as `‖P H P'‖_S1 ≤ ‖H‖_S1` (`nuclearNorm_mul_mul_le`). By the Hahn–Banach theorem the
functional `0` on them extends to a real-linear `g ≤ p`, which is `Re tr(Yᴴ ·)`
(`re_trace_rieszMatrix`) with `Y` vanishing off `Ω`. Then `W = Y − U Vᴴ` has
`Re tr(Wᴴ X) ≤ a ‖P X P'‖_S1` for every `X`. At `X = U (Uᴴ W)` and at `X = (W V) Vᴴ`, where
`P X P' = 0`, this says `‖Uᴴ W‖²_S2 ≤ 0` and `‖W V‖²_S2 ≤ 0`; and it makes `W / a` a contraction
(`isContraction_of_re_trace_le`) (`exists_certificate_of_exists_pos_le_nuclearNorm_add`). With
`LinearGrowth`, linear growth holds exactly when item 1.2's hypotheses do
(`exists_pos_le_nuclearNorm_add_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

/-- **Linear growth gives item 1.2's certificate**: with `K = U diag(s) Vᴴ`, `s > 0`, if some
`c > 0` has `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`, then some `W` with
`Uᴴ W = 0`, `W V = 0` and `c' W` a contraction for some `c' > 1` makes `U Vᴴ + W` vanish off
`Ω`. -/
theorem exists_certificate_of_exists_pos_le_nuclearNorm_add {U : Matrix m r ℂ}
    {V : Matrix n r ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k)
    {K : Matrix m n ℂ} (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)}
    {c : ℝ} (hc : 0 < c) (hgrow : ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H)) :
    ∃ W : Matrix m n ℂ, (∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0) ∧ Uᴴ * W = 0 ∧
      W * V = 0 ∧ ∃ c' : ℝ, 1 < c' ∧ IsContraction ((c' : ℂ) • W) := by
  have hPc := isContraction_one_sub_mul_conjTranspose hU
  have hP'c := isContraction_one_sub_mul_conjTranspose hV
  set a : ℝ := max (1 - c) (1 / 2) with hadef
  have ha0 : 0 < a := lt_max_of_lt_right one_half_pos
  have ha1 : a < 1 := max_lt (by linarith) one_half_lt_one
  have hca : 1 - c ≤ a := le_max_left _ _
  -- the first-order bound, at `t → 0⁺`
  have hfirst (H : Matrix m n ℂ) (hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0) : c * nuclearNorm H ≤
      ((U * Vᴴ)ᴴ * H).trace.re + nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) := by
    obtain ⟨C, hC⟩ := exists_nuclearNorm_add_smul_le hU hV hs hK H
    refine le_of_forall_pos_lt_add fun ε hε => ?_
    have hC1 : 0 < |C| + 1 := by positivity
    have ht : 0 < ε / (|C| + 1) := div_pos hε hC1
    have h1 := (hgrow (((ε / (|C| + 1) : ℝ) : ℂ) • H) fun ij hij => by
      rw [Matrix.smul_apply, hH ij hij, smul_zero]).trans (hC _)
    rw [nuclearNorm_smul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos ht] at h1
    have h2 : C * (ε / (|C| + 1)) < ε := by
      calc C * (ε / (|C| + 1)) ≤ |C| * (ε / (|C| + 1)) :=
            mul_le_mul_of_nonneg_right (le_abs_self C) ht.le
        _ < (|C| + 1) * (ε / (|C| + 1)) := mul_lt_mul_of_pos_right (lt_add_one _) ht
        _ = ε := mul_div_cancel₀ ε hC1.ne'
    have h3 : ε / (|C| + 1) * (c * nuclearNorm H) ≤ ε / (|C| + 1) *
        (((U * Vᴴ)ᴴ * H).trace.re + nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) +
          C * (ε / (|C| + 1))) := by nlinarith
    linarith [le_of_mul_le_mul_left h3 ht]
  -- the sublinear functional `p`, nonnegative on the matrices vanishing on `Ω`
  let p : Matrix m n ℂ → ℝ := fun X =>
    ((U * Vᴴ)ᴴ * X).trace.re + a * nuclearNorm ((1 - U * Uᴴ) * X * (1 - V * Vᴴ))
  have hp_smul (t : ℝ) (ht : 0 < t) (X : Matrix m n ℂ) : p (t • X) = t * p X := by
    simp only [p, Matrix.mul_smul, Matrix.smul_mul, trace_smul, Complex.smul_re, smul_eq_mul]
    rw [nuclearNorm_real_smul ht.le]
    ring
  have hp_add (X X' : Matrix m n ℂ) : p (X + X') ≤ p X + p X' := by
    simp only [p, Matrix.mul_add, Matrix.add_mul, trace_add, Complex.add_re]
    linarith [mul_le_mul_of_nonneg_left (nuclearNorm_add_le
      ((1 - U * Uᴴ) * X * (1 - V * Vᴴ)) ((1 - U * Uᴴ) * X' * (1 - V * Vᴴ))) ha0.le]
  have hpL (H : Matrix m n ℂ) (hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0) : 0 ≤ p H := by
    have h1 := hfirst H hH
    have h2 := nuclearNorm_mul_mul_le hPc hP'c H
    show 0 ≤ ((U * Vᴴ)ᴴ * H).trace.re + a * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ))
    nlinarith [mul_nonneg (sub_nonneg.2 ha1.le) (sub_nonneg.2 h2),
      mul_nonneg (by linarith : 0 ≤ c + a - 1) (nuclearNorm_nonneg H)]
  -- the Hahn–Banach extension of `0` from the matrices vanishing on `Ω`
  let L : Submodule ℝ (Matrix m n ℂ) :=
    { carrier := {H | ∀ ij ∈ Ω, H ij.1 ij.2 = 0}
      add_mem' := fun ha hb ij hij => by
        rw [Matrix.add_apply, ha ij hij, hb ij hij, add_zero]
      zero_mem' := fun _ _ => rfl
      smul_mem' := fun c H hH ij hij => by
        rw [Matrix.smul_apply, hH ij hij, smul_zero] }
  have hL (H : Matrix m n ℂ) : H ∈ L ↔ ∀ ij ∈ Ω, H ij.1 ij.2 = 0 := Iff.rfl
  set f₀ := (0 : Matrix m n ℂ →ₗ[ℝ] ℝ).toPMap L with hf₀
  have hdom (x : Matrix m n ℂ) (hx : x ∈ L) : x ∈ f₀.domain := by
    rwa [hf₀, LinearMap.toPMap_domain]
  have hf0 (x : f₀.domain) : f₀ x = 0 := rfl
  obtain ⟨g, hgf, hgp⟩ := exists_extension_of_le_sublinear f₀ p hp_smul hp_add (by
    rintro ⟨x, hx⟩
    rw [hf0]
    rw [hf₀, LinearMap.toPMap_domain] at hx
    exact hpL x ((hL x).1 hx))
  have hgL (H : Matrix m n ℂ) (hH : H ∈ L) : g H = 0 := by
    have h1 := hgf ⟨H, hdom H hH⟩
    rw [hf0] at h1
    exact h1
  obtain ⟨W, hWdef⟩ : ∃ W, W = rieszMatrix g - U * Vᴴ := ⟨_, rfl⟩
  have hW (X : Matrix m n ℂ) : (Wᴴ * X).trace.re ≤
      a * nuclearNorm ((1 - U * Uᴴ) * X * (1 - V * Vᴴ)) := by
    have h : g X ≤ ((U * Vᴴ)ᴴ * X).trace.re +
        a * nuclearNorm ((1 - U * Uᴴ) * X * (1 - V * Vᴴ)) := hgp X
    rw [hWdef, conjTranspose_sub, Matrix.sub_mul, trace_sub, Complex.sub_re,
      re_trace_rieszMatrix]
    linarith
  -- `Uᴴ W = 0` and `W V = 0`
  have hUW : Uᴴ * W = 0 := by
    have h := hW (U * (Uᴴ * W))
    have hP0 : (1 - U * Uᴴ) * (U * (Uᴴ * W)) * (1 - V * Vᴴ) = 0 := by
      rw [← Matrix.mul_assoc, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc U Uᴴ U, hU,
        Matrix.mul_one, sub_self, Matrix.zero_mul, Matrix.zero_mul]
    have htr : (Wᴴ * (U * (Uᴴ * W))).trace.re = frobeniusSq (Uᴴ * W) := by
      rw [frobeniusSq_eq_re_trace, conjTranspose_mul, conjTranspose_conjTranspose,
        Matrix.mul_assoc]
    rw [hP0, nuclearNorm_zero, mul_zero, htr] at h
    exact eq_zero_of_frobeniusSq_eq_zero (le_antisymm h (frobeniusSq_nonneg _))
  have hWV : W * V = 0 := by
    have h := hW (W * V * Vᴴ)
    have e : V * Vᴴ * (1 - V * Vᴴ) = 0 := by
      rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_assoc, ← Matrix.mul_assoc Vᴴ V, hV,
        Matrix.one_mul, sub_self]
    have hP0 : (1 - U * Uᴴ) * (W * V * Vᴴ) * (1 - V * Vᴴ) = 0 := by
      simp only [Matrix.mul_assoc] at e ⊢
      rw [e, Matrix.mul_zero, Matrix.mul_zero]
    have htr : (Wᴴ * (W * V * Vᴴ)).trace.re = frobeniusSq (W * V) := by
      rw [frobeniusSq_eq_re_trace, conjTranspose_mul, ← Matrix.mul_assoc, Matrix.trace_mul_comm]
      simp only [Matrix.mul_assoc]
    rw [hP0, nuclearNorm_zero, mul_zero, htr] at h
    exact eq_zero_of_frobeniusSq_eq_zero (le_antisymm h (frobeniusSq_nonneg _))
  -- `W / a` is a contraction
  have hWc : IsContraction (((a⁻¹ : ℝ) : ℂ) • W) := isContraction_of_re_trace_le fun X => by
    rw [conjTranspose_smul, Complex.star_def, Complex.conj_ofReal, Matrix.smul_mul, trace_smul,
      smul_eq_mul, Complex.re_ofReal_mul]
    calc a⁻¹ * (Wᴴ * X).trace.re
        ≤ a⁻¹ * (a * nuclearNorm ((1 - U * Uᴴ) * X * (1 - V * Vᴴ))) :=
          mul_le_mul_of_nonneg_left (hW X) (inv_nonneg.2 ha0.le)
      _ = nuclearNorm ((1 - U * Uᴴ) * X * (1 - V * Vᴴ)) := by
          rw [← mul_assoc, inv_mul_cancel₀ ha0.ne', one_mul]
      _ ≤ nuclearNorm X := nuclearNorm_mul_mul_le hPc hP'c X
  refine ⟨W, fun i j hij => ?_, hUW, hWV, a⁻¹, (one_lt_inv₀ ha0).2 ha1, hWc⟩
  have hE (z : ℂ) : single i j z ∈ L := fun ij hij' => by
    rw [single_apply, ite_eq_right]
    rintro ⟨rfl, rfl⟩
    exact hij hij'
  rw [hWdef, add_sub_cancel, rieszMatrix, of_apply, hgL _ (hE 1), hgL _ (hE Complex.I)]
  rfl

example := exists_certificate_of_exists_pos_le_nuclearNorm_add
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp)
  (by simp) (s := ![1, 2]) (fun k => by fin_cases k <;> norm_num) rfl (Ω := Set.univ) one_pos
  fun H hH => by
    rw [show H = 0 from by ext i j; exact hH (i, j) (Set.mem_univ _), nuclearNorm_zero,
      mul_zero, add_zero, add_zero]

/-- **Linear growth exactly when item 1.2's hypotheses hold**: for `K = U diag(s) Vᴴ` with
orthonormal columns `U`, `V` and `s > 0`, some `c > 0` has `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1`
for every `H` vanishing on `Ω` exactly when some `W` with `Uᴴ W = 0`, `W V = 0` and `c' W` a
contraction for some `c' > 1` makes `U Vᴴ + W` vanish off `Ω`, and no nonzero `U Aᴴ + B Vᴴ`
vanishes on `Ω`. -/
theorem exists_pos_le_nuclearNorm_add_iff {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)} :
    (∃ c > 0, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
        nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H)) ↔
      (∃ W : Matrix m n ℂ, (∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0) ∧ Uᴴ * W = 0 ∧
          W * V = 0 ∧ ∃ c : ℝ, 1 < c ∧ IsContraction ((c : ℂ) • W)) ∧
        ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
          (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → U * Aᴴ + B * Vᴴ = 0 := by
  obtain ⟨hKU, hKtr, -⟩ := svd_hypotheses hU hV fun k => (hs k).le
  rw [← hK] at hKU hKtr
  constructor
  · rintro ⟨c, hc, hgrow⟩
    exact ⟨exists_certificate_of_exists_pos_le_nuclearNorm_add hU hV hs hK hc hgrow,
      eq_zero_of_exists_pos_le_nuclearNorm_add hU hV hs hK hc hgrow⟩
  · rintro ⟨⟨W, hYΩ, hWU, hWV, c, hc, hW⟩, hT⟩
    exact exists_pos_le_nuclearNorm_add_of_certificate hU hV hKU hKtr hYΩ hWU hWV hc hW hT

example := (exists_pos_le_nuclearNorm_add_iff (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
  (fun k => by fin_cases k <;> norm_num) rfl (Ω := Set.univ)).2
  ⟨⟨0, fun _ _ hij => absurd (Set.mem_univ _) hij, by simp, by simp, 2, one_lt_two,
    by rw [smul_zero]; exact isContraction_zero⟩,
    fun _ _ hT => by ext i j; exact hT (i, j) (Set.mem_univ _)⟩

end Transformer.Grokking.FlatSpectrum

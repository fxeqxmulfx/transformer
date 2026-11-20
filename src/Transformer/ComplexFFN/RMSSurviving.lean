module

public import Transformer.ComplexFFN.RMSGap

/-!
# Witnesses that survive the branch RMS

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`): the branch
`f = R_D(W_F (g ⊙ u)/√2)` of `research/main_model/README.txt`, with the parameter-free
`R_D = GPTMini.rmsNormEps`, and the products of `Defs`.

`R_D` is odd (`rmsNormEps_neg`), so plain and radial FFNs with complex-linear projections
stay negated by `J` after it. The target `R_D(|φ z|² w)`, the four-unit SwiGLU block of
`normSq_eq_swigluFFN` after `R_D`, is unchanged by `J` when `φ` is complex-linear. So
every such FFN after `R_D`, of any width, errs on it by at least its norm
(`plainFFN_rms_gap_normSq`, `radialFFN_rms_gap_normSq`): the error of the zero map, since
`R_D 0 = 0`. The norm is `√d |φ z|² ‖w‖ / √(|φ z|⁴ ‖w‖² + d eps)` (`norm_rmsNormEps`),
exactly `√d` when `eps = 0`, `φ z ≠ 0` and `w ≠ 0` (`norm_rmsNormEps_zero`).

The torus witness survives too. One SwiGLU unit after `R_D` stays on the line through its
column, while one plain unit `z₁ z₂` after `R_D` points in every direction on the torus;
so the SwiGLU unit errs by the target's norm (`swigluUnit_rms_gap_plainUnit_torus`).
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace Transformer.ComplexFFN

variable {d : ℕ}

/-- `R_D` scales by `√d / √(‖x‖² + d eps)`: the normalized vector has norm
`√d ‖x‖ / √(‖x‖² + d eps)`. -/
theorem norm_rmsNormEps (eps : ℝ) (x : EucSpace d) :
    ‖GPTMini.rmsNormEps eps x‖ = Real.sqrt d * ‖x‖ / Real.sqrt (‖x‖ ^ 2 + d * eps) := by
  rw [GPTMini.rmsNormEps, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)), div_mul_eq_mul_div]

/-- With `eps = 0`, `R_D` puts every nonzero vector on the sphere of radius `√d`. -/
theorem norm_rmsNormEps_zero {x : EucSpace d} (hx : x ≠ 0) :
    ‖GPTMini.rmsNormEps 0 x‖ = Real.sqrt d := by
  rw [norm_rmsNormEps, mul_zero, add_zero, Real.sqrt_sq (norm_nonneg x),
    mul_div_cancel_right₀ _ (norm_ne_zero_iff.mpr hx)]

example := norm_rmsNormEps_zero (x := EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
  (norm_ne_zero_iff.mp (by simp))

variable {V κ : Type*} [Fintype κ] [AddCommGroup V] [Module ℝ V]

/-- After `R_D`, an approximant negated by `J` errs on `R_D(|φ z|² w)` by at least its
norm, when `φ` is complex-linear and both `z` and `J z` lie in the set. -/
theorem anti_rms_gap_normSq (eps : ℝ) {f : V → EucSpace d} {J : V → V}
    (hf : ∀ z, f (J z) = -f z) (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z)
    (w : EucSpace d) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ) {z : V}
    (hz : z ∈ K) (hJz : J z ∈ K) :
    ‖GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ := by
  have h := gap_of_anti (T := fun z => GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w))
    hf hδ hz hJz
  simp only [hφ, Complex.normSq_mul, Complex.normSq_I, one_mul] at h
  rw [← two_smul ℝ, norm_smul, Real.norm_two] at h
  linarith

/-- After `R_D`, no plain FFN with complex-linear projections, of any width, approximates
the four-unit SwiGLU block `|φ z|² w` after `R_D` better than the zero map does at `z`. -/
theorem plainFFN_rms_gap_normSq (eps : ℝ) (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → EucSpace d) (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z)
    (w : EucSpace d) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (plainFFN g u c c' z) -
      GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hJz : J z ∈ K) :
    ‖GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ :=
  anti_rms_gap_normSq eps (fun z => by simp only [plainFFN_anti J g u hg hu, rmsNormEps_neg])
    φ hφ w hδ hz hJz

/-- The same for the radial product with complex-linear projections. -/
theorem radialFFN_rms_gap_normSq (eps ε : ℝ) (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → EucSpace d) (φ : V →ₗ[ℝ] ℂ) (hφ : ∀ z, φ (J z) = Complex.I * φ z)
    (w : EucSpace d) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (radialFFN ε g u c c' z) -
      GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hJz : J z ∈ K) :
    ‖GPTMini.rmsNormEps eps (Complex.normSq (φ z) • w)‖ ≤ δ :=
  anti_rms_gap_normSq eps
    (fun z => by simp only [radialFFN_anti ε J g u hg hu, rmsNormEps_neg]) φ hφ w hδ hz hJz

omit [AddCommGroup V] [Module ℝ V] in
/-- After `R_D`, a map with values on the line `ℝ w` misses a target by at least `‖T z‖`
wherever `T z ⊥ w`: as much as the zero map. -/
theorem rms_line_gap (eps : ℝ) {f : V → ℝ} (w : EucSpace d) {T : V → EucSpace d}
    {K : Set V} {δ : ℝ} (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (f z • w) - T z‖ ≤ δ)
    {z : V} (hz : z ∈ K) (hT : ⟪w, T z⟫ = 0) : ‖T z‖ ≤ δ := by
  have key : ∀ m : ℝ, ‖T z‖ ≤ ‖m • w - T z‖ := fun m => by
    have h := norm_sub_sq_real (m • w) (T z)
    rw [real_inner_smul_left, hT, mul_zero, mul_zero, sub_zero] at h
    exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1 (by nlinarith [sq_nonneg ‖m • w‖])
  have h := hδ z hz
  rw [GPTMini.rmsNormEps, smul_smul] at h
  exact (key _).trans h

/-- One plain unit with columns `√2 R 1` and `√2 R i`, for a real-linear `R` on `ℂ`,
writes `R (φ z ψ z)`. -/
theorem plainFFN_unit_map {W : Type*} [AddCommGroup W] [Module ℝ W] (R : ℂ →ₗ[ℝ] W)
    (φ ψ : V →ₗ[ℝ] ℂ) (z : V) :
    plainFFN (fun _ : Unit => φ) (fun _ => ψ) (fun _ => Real.sqrt 2 • R 1)
      (fun _ => Real.sqrt 2 • R Complex.I) z = R (φ z * ψ z) := by
  have h2 : Real.sqrt 2 ≠ 0 := (Real.sqrt_pos.2 two_pos).ne'
  have hx : ∀ x : ℂ, x = x.re • (1 : ℂ) + x.im • Complex.I := fun x => by
    rw [Complex.real_smul, Complex.real_smul, mul_one, Complex.re_add_im]
  rw [hx (φ z * ψ z), map_add, map_smul, map_smul]
  simp only [plainFFN, complexReadout, Fintype.sum_unique, plainHidden, smul_smul,
    Complex.div_ofReal_re, Complex.div_ofReal_im, div_mul_cancel₀ _ h2]

/-- One SwiGLU unit against one plain unit `(z₁, z₂) ↦ z₁ z₂` read out into `ℝ²` by the
columns `√2 e₀, √2 e₁`, both after `R_D`. The SwiGLU unit then points along its column
`w`, while on the torus `|z₁| = |z₂| = 1` the product points in every direction. So the
SwiGLU unit errs by at least `√2 / √(1 + 2 eps)`, the norm of every normalized target
value: as much as the zero map. -/
theorem swigluUnit_rms_gap_plainUnit_torus (eps : ℝ) (a b : ℂ × ℂ →ₗ[ℝ] ℝ)
    (w : EucSpace 2) {δ : ℝ}
    (hδ : ∀ z : ℂ × ℂ, ‖z.1‖ = 1 → ‖z.2‖ = 1 →
      ‖GPTMini.rmsNormEps eps (swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => w) z) -
        GPTMini.rmsNormEps eps (plainFFN (fun _ : Unit => LinearMap.fst ℝ ℂ ℂ)
          (fun _ => LinearMap.snd ℝ ℂ ℂ) (fun _ => Real.sqrt 2 • EuclideanSpace.single 0 1)
          (fun _ => Real.sqrt 2 • EuclideanSpace.single 1 1) z)‖ ≤ δ) :
    Real.sqrt 2 / Real.sqrt (1 + 2 * eps) ≤ δ := by
  set R := Complex.orthonormalBasisOneI.repr
  obtain ⟨h0, h1⟩ : R 1 = EuclideanSpace.single 0 1 ∧ R Complex.I = EuclideanSpace.single 1 1 :=
    ⟨by simpa using Complex.orthonormalBasisOneI.repr_self 0,
      by simpa using Complex.orthonormalBasisOneI.repr_self 1⟩
  have hP : ∀ z : ℂ × ℂ, plainFFN (fun _ : Unit => LinearMap.fst ℝ ℂ ℂ)
      (fun _ => LinearMap.snd ℝ ℂ ℂ) (fun _ => Real.sqrt 2 • EuclideanSpace.single 0 1)
      (fun _ => Real.sqrt 2 • EuclideanSpace.single 1 1) z = R (z.1 * z.2) := fun z => by
    rw [← h0, ← h1]
    exact plainFFN_unit_map R.toLinearEquiv.toLinearMap _ _ z
  obtain ⟨v, hv, hvw⟩ : ∃ v : EucSpace 2, ‖v‖ = 1 ∧ ⟪w, v⟫ = 0 := by
    by_cases hw : R.symm w = 0
    · refine ⟨R 1, by simp, ?_⟩
      rw [← R.apply_symm_apply w, hw, map_zero, inner_zero_left]
    · have hn : 0 < ‖R.symm w‖ := norm_pos_iff.2 hw
      refine ⟨R (Complex.I * R.symm w / (‖R.symm w‖ : ℂ)), ?_, ?_⟩
      · rw [LinearIsometryEquiv.norm_map, norm_div, Complex.norm_mul, Complex.norm_I,
          one_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hn, div_self hn.ne']
      · rw [← R.apply_symm_apply w, LinearIsometryEquiv.inner_map_map, R.symm_apply_apply,
          Complex.inner]
        simp only [Complex.mul_re, Complex.mul_im, Complex.div_ofReal_re,
          Complex.div_ofReal_im, Complex.I_re, Complex.I_im, Complex.conj_re, Complex.conj_im]
        ring
  have h := rms_line_gap eps
    (f := fun z : ℂ × ℂ => Transformer.Landscape.Neural.silu (a z) * b z) w
    (T := fun z => GPTMini.rmsNormEps eps (R (z.1 * z.2))) (K := {z | ‖z.1‖ = 1 ∧ ‖z.2‖ = 1})
    (fun z hz => by simpa only [swigluFFN, Fintype.sum_unique, hP] using hδ z hz.1 hz.2)
    (z := (R.symm v, 1)) ⟨by simpa using hv, by simp⟩
    (by simp [GPTMini.rmsNormEps, real_inner_smul_right, hvw])
  simpa [norm_rmsNormEps, hv] using h

section Examples

/-- The zero map with `δ = √1` from `rmsNormEps_norm_le`, on `ℂ` with `J = i ·`. -/
example (z : ℂ) :=
  anti_rms_gap_normSq 1 (f := fun _ => (0 : EucSpace 1)) (J := (Complex.I * ·))
    (fun _ => neg_zero.symm) LinearMap.id (fun _ => rfl) (EuclideanSpace.single 0 1)
    (K := Set.univ) (fun _ _ => by
      rw [zero_sub, norm_neg]
      exact GPTMini.rmsNormEps_norm_le 1 one_pos one_pos _) (z := z) trivial trivial

/-- Gate and value `id` on `ℂ`, any readout into `ℝ¹`, `δ = 2√1`. -/
example (c c' : Unit → EucSpace 1) (z : ℂ) :=
  plainFFN_rms_gap_normSq 1 (Complex.I * ·) (fun _ : Unit => LinearMap.id)
    (fun _ => LinearMap.id) (fun _ _ => rfl) (fun _ _ => rfl) c c' LinearMap.id (fun _ => rfl)
    (EuclideanSpace.single 0 1) (K := Set.univ)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos one_pos _ _) (z := z) trivial trivial

example (c c' : Unit → EucSpace 1) (z : ℂ) :=
  radialFFN_rms_gap_normSq 1 (2 ^ (-23 : ℤ)) (Complex.I * ·) (fun _ : Unit => LinearMap.id)
    (fun _ => LinearMap.id) (fun _ _ => rfl) (fun _ _ => rfl) c c' LinearMap.id (fun _ => rfl)
    (EuclideanSpace.single 0 1) (K := Set.univ)
    (fun _ _ => norm_rmsNormEps_sub_le one_pos one_pos _ _) (z := z) trivial trivial

/-- The zero map attains the bound: `R_D 0 = 0`, so its error is the target's norm. -/
example (eps : ℝ) (T : EucSpace 1) : ‖GPTMini.rmsNormEps eps 0 - T‖ = ‖T‖ := by
  simp [GPTMini.rmsNormEps]

/-- The zero map on the line through `e₀` against the constant `e₁`. -/
example : ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ ≤ 1 :=
  rms_line_gap 1 (f := fun _ : ℝ => 0) (EuclideanSpace.single 0 1)
    (T := fun _ => EuclideanSpace.single 1 1) (K := Set.univ)
    (fun _ _ => by simp [GPTMini.rmsNormEps]) (z := 0) trivial single_orthonormal.2.2

/-- Any SwiGLU unit, `δ = 2√2` from `norm_rmsNormEps_sub_le`. -/
example (a b : ℂ × ℂ →ₗ[ℝ] ℝ) (w : EucSpace 2) :=
  swigluUnit_rms_gap_plainUnit_torus 1 a b w fun _ _ _ =>
    norm_rmsNormEps_sub_le one_pos two_pos _ _

end Examples

end Transformer.ComplexFFN

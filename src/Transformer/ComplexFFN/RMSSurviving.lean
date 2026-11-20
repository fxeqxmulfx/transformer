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
-/

@[expose] public section

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

end Examples

end Transformer.ComplexFFN

module

public import Transformer.ComplexFFN.RMSSurviving

/-!
# What the branch RMS erases

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`): the branch
`f = R_D(W_F (g ⊙ u)/√2)` of `research/main_model/README.txt`, with the parameter-free
`R_D = GPTMini.rmsNormEps`, and the products of `Defs`.

With RMS `eps = 0`, `R_D` forgets positive scalars (`rmsNormEps_zero_smul`), and two
one-unit witnesses of the comparison before the RMS go with them.

* At `-z` one SwiGLU unit `SiLU(a z) (b z) c` changes by the positive factor
  `σ(-a z) / σ(a z)` (`swigluFFN_unit_neg`), so after `R_D` it is even
  (`rmsNormEps_zero_swigluUnit_neg`). Its odd part, the bound of `plainFFN_gap_swigluUnit`
  against the even complex products, is zero after `R_D`.
* The radial gate is positive (`radialGate_pos`), and one unit has one gate. So one radial
  unit after `R_D` is one plain unit after `R_D` (`rmsNormEps_zero_radialFFN_unit`), a
  SwiGLU FFN with six units after `R_D` (`rmsNormEps_zero_radialFFN_unit_swiglu`). One
  diagonal unit after `R_D` is a product network of two units after `R_D`
  (`rmsNormEps_zero_diagonalFFN_unit`), a SwiGLU FFN with four units after `R_D`
  (`rmsNormEps_zero_diagonalFFN_unit_swiglu`). The parallelogram witnesses
  `swigluFFN_gap_radial` and `swigluFFN_gap_diagonal` have no counterpart after `R_D`:
  SwiGLU attains both units exactly.

With `eps > 0` the RMS keeps scale: it maps `r x`, `r > 1`, to a longer vector than `x`
(`norm_rmsNormEps_lt_smul`), so the erasure is exact only at `eps = 0`. Two units on each
side separate the classes again at every `eps ≥ 0` (`RMSIncomparable`); the
complex-structure gap and the torus gap survive at one unit (`RMSSurviving`).
-/

@[expose] public section

namespace Transformer.ComplexFFN

variable {d : ℕ}

/-- With `eps = 0`, `R_D` forgets positive scalars. -/
theorem rmsNormEps_zero_smul {r : ℝ} (hr : 0 < r) (x : EucSpace d) :
    GPTMini.rmsNormEps 0 (r • x) = GPTMini.rmsNormEps 0 x := by
  simp only [GPTMini.rmsNormEps, mul_zero, add_zero, smul_smul, norm_smul, Real.norm_eq_abs,
    abs_of_pos hr, Real.sqrt_sq (norm_nonneg x), Real.sqrt_sq (mul_nonneg hr.le (norm_nonneg x))]
  rw [mul_comm r, div_mul_eq_mul_div, mul_div_mul_right _ _ hr.ne']

/-- With `eps > 0`, `R_D` keeps scale: for `r > 1` and `x ≠ 0` the normalized `r x` is
longer than the normalized `x`. -/
theorem norm_rmsNormEps_lt_smul {eps r : ℝ} (heps : 0 < eps) (hd : 0 < d) (hr : 1 < r)
    {x : EucSpace d} (hx : x ≠ 0) :
    ‖GPTMini.rmsNormEps eps x‖ < ‖GPTMini.rmsNormEps eps (r • x)‖ := by
  have hr0 : 0 < r := by linarith
  have hn : 0 < ‖x‖ := norm_pos_iff.2 hx
  have hD : 0 < (d : ℝ) * eps := mul_pos (Nat.cast_pos.2 hd) heps
  have hs : 0 < Real.sqrt d := Real.sqrt_pos.2 (Nat.cast_pos.2 hd)
  have h₁ : 0 < Real.sqrt (‖x‖ ^ 2 + d * eps) := Real.sqrt_pos.2 (by positivity)
  have h₂ : 0 < Real.sqrt ((r * ‖x‖) ^ 2 + d * eps) := Real.sqrt_pos.2 (by positivity)
  rw [norm_rmsNormEps, norm_rmsNormEps, norm_smul, Real.norm_eq_abs, abs_of_pos hr0,
    div_lt_div_iff₀ h₁ h₂]
  have key : Real.sqrt ((r * ‖x‖) ^ 2 + d * eps) < r * Real.sqrt (‖x‖ ^ 2 + d * eps) := by
    rw [Real.sqrt_lt' (mul_pos hr0 h₁), mul_pow r, mul_pow r, Real.sq_sqrt (by positivity)]
    nlinarith [mul_pos (mul_pos (sub_pos.2 hr) (add_pos hr0 one_pos)) hD]
  nlinarith [mul_lt_mul_of_pos_left key (mul_pos hs hn)]

/-- The radial gate is a sigmoid value, so it is positive. -/
theorem radialGate_pos (ε : ℝ) (w : ℂ) : 0 < radialGate ε w :=
  Real.sigmoid_pos _

variable {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- At `-z` one SwiGLU unit changes by the positive factor `σ(-a z) / σ(a z)`: both
`SiLU(-t) (-s)` and `SiLU(t) s` are `t s` times a sigmoid value. -/
theorem swigluFFN_unit_neg (a b : V →ₗ[ℝ] ℝ) (c : W) (z : V) :
    swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c) (-z) =
      (Real.sigmoid (-a z) / Real.sigmoid (a z)) •
        swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c) z := by
  simp only [swigluFFN, Fintype.sum_unique, map_neg, smul_smul, silu_eq_mul_sigmoid]
  congr 1
  rw [div_mul_eq_mul_div, eq_div_iff (Real.sigmoid_pos (a z)).ne']
  ring

/-- With `eps = 0`, one SwiGLU unit after `R_D` is even. -/
theorem rmsNormEps_zero_swigluUnit_neg (a b : V →ₗ[ℝ] ℝ) (c : EucSpace d) (z : V) :
    GPTMini.rmsNormEps 0 (swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c) (-z)) =
      GPTMini.rmsNormEps 0 (swigluFFN (fun _ : Unit => a) (fun _ => b) (fun _ => c) z) := by
  rw [swigluFFN_unit_neg,
    rmsNormEps_zero_smul (div_pos (Real.sigmoid_pos _) (Real.sigmoid_pos _))]

/-- One radial unit is its gate times one plain unit. -/
theorem radialFFN_unit_eq_smul (ε : ℝ) (g u : V →ₗ[ℝ] ℂ) (c c' : W) (z : V) :
    radialFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z =
      radialGate ε (g z) •
        plainFFN (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z := by
  simp only [radialFFN, plainFFN, complexReadout, Fintype.sum_unique, radialHidden,
    plainHidden, mul_div_assoc, Complex.re_ofReal_mul, Complex.im_ofReal_mul, smul_add,
    smul_smul]

/-- With `eps = 0`, one radial unit after `R_D` is one plain unit after `R_D`. -/
theorem rmsNormEps_zero_radialFFN_unit (ε : ℝ) (g u : V →ₗ[ℝ] ℂ) (c c' : EucSpace d)
    (z : V) :
    GPTMini.rmsNormEps 0
        (radialFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z) =
      GPTMini.rmsNormEps 0
        (plainFFN (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z) := by
  rw [radialFFN_unit_eq_smul, rmsNormEps_zero_smul (radialGate_pos ε _)]

/-- With `eps = 0`, one radial unit after `R_D` is a SwiGLU FFN with six units after `R_D`
(`plainFFN_eq_swigluFFN`). -/
theorem rmsNormEps_zero_radialFFN_unit_swiglu (ε : ℝ) (g u : V →ₗ[ℝ] ℂ)
    (c c' : EucSpace d) :
    ∃ (a b : Fin 6 → V →ₗ[ℝ] ℝ) (e : Fin 6 → EucSpace d), ∀ z,
      GPTMini.rmsNormEps 0
          (radialFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z) =
        GPTMini.rmsNormEps 0 (swigluFFN a b e z) := by
  obtain ⟨a, b, e, h⟩ := plainFFN_eq_swigluFFN (fun _ : Unit => g) (fun _ => u)
    (fun _ => c) (fun _ => c')
  exact ⟨a, b, e, fun z => (rmsNormEps_zero_radialFFN_unit ε g u c c' z).trans
    (congrArg (GPTMini.rmsNormEps 0) (congrFun h z))⟩

/-- One diagonal unit is its gate times the product network `g_R u_R c + g_I u_I c'`. -/
theorem diagonalFFN_unit_eq_smul (ε : ℝ) (g u : V →ₗ[ℝ] ℂ) (c c' : W) (z : V) :
    diagonalFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z =
      radialGate ε (g z) • productFFN ![Complex.reLm ∘ₗ g, Complex.imLm ∘ₗ g]
        ![Complex.reLm ∘ₗ u, Complex.imLm ∘ₗ u] ![c, c'] z := by
  simp [diagonalFFN, productFFN, complexReadout, diagonalHidden, Fin.sum_univ_two, smul_add,
    smul_smul]

/-- With `eps = 0`, one diagonal unit after `R_D` is a product network of two units after
`R_D`. -/
theorem rmsNormEps_zero_diagonalFFN_unit (ε : ℝ) (g u : V →ₗ[ℝ] ℂ) (c c' : EucSpace d)
    (z : V) :
    GPTMini.rmsNormEps 0
        (diagonalFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z) =
      GPTMini.rmsNormEps 0 (productFFN ![Complex.reLm ∘ₗ g, Complex.imLm ∘ₗ g]
        ![Complex.reLm ∘ₗ u, Complex.imLm ∘ₗ u] ![c, c'] z) := by
  rw [diagonalFFN_unit_eq_smul, rmsNormEps_zero_smul (radialGate_pos ε _)]

/-- With `eps = 0`, one diagonal unit after `R_D` is a SwiGLU FFN with four units after
`R_D` (`productFFN_eq_swigluFFN`). -/
theorem rmsNormEps_zero_diagonalFFN_unit_swiglu (ε : ℝ) (g u : V →ₗ[ℝ] ℂ)
    (c c' : EucSpace d) :
    ∃ (a b : Fin 2 ⊕ Fin 2 → V →ₗ[ℝ] ℝ) (e : Fin 2 ⊕ Fin 2 → EucSpace d), ∀ z,
      GPTMini.rmsNormEps 0
          (diagonalFFN ε (fun _ : Unit => g) (fun _ => u) (fun _ => c) (fun _ => c') z) =
        GPTMini.rmsNormEps 0 (swigluFFN a b e z) := by
  have h := productFFN_eq_swigluFFN ![Complex.reLm ∘ₗ g, Complex.imLm ∘ₗ g]
    ![Complex.reLm ∘ₗ u, Complex.imLm ∘ₗ u] ![c, c']
  exact ⟨_, _, _, fun z => (rmsNormEps_zero_diagonalFFN_unit ε g u c c' z).trans
    (congrArg (GPTMini.rmsNormEps 0) (congrFun h z))⟩

section Examples

example (x : EucSpace 2) := rmsNormEps_zero_smul two_pos x

/-- The model's `eps = 2^-23` keeps scale: `R_D(2 e₀)` is longer than `R_D(e₀)` in `ℝ¹`. -/
example := norm_rmsNormEps_lt_smul (eps := 2 ^ (-23 : ℤ)) (by positivity) one_pos one_lt_two
  (x := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) (norm_ne_zero_iff.mp (by simp))

/-- Gate and value `id` on `ℝ`: `R_D(SiLU(-z) (-z)) = R_D(SiLU(z) z)` in `ℝ¹`. -/
example (z : ℝ) := rmsNormEps_zero_swigluUnit_neg LinearMap.id LinearMap.id
  (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) z

/-- Gate and value `id` on `ℂ`, read out into `ℝ²` by `e₀, e₁`, gate `ε = 2^-23`. -/
example (z : ℂ) := rmsNormEps_zero_radialFFN_unit (2 ^ (-23 : ℤ)) LinearMap.id LinearMap.id
  (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single 1 1) z

example := rmsNormEps_zero_radialFFN_unit_swiglu (2 ^ (-23 : ℤ)) (V := ℂ) LinearMap.id
  LinearMap.id (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single 1 1)

example (z : ℂ) := rmsNormEps_zero_diagonalFFN_unit (2 ^ (-23 : ℤ)) LinearMap.id
  LinearMap.id (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single 1 1) z

example := rmsNormEps_zero_diagonalFFN_unit_swiglu (2 ^ (-23 : ℤ)) (V := ℂ) LinearMap.id
  LinearMap.id (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single 1 1)

end Examples

end Transformer.ComplexFFN

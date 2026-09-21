import Transformer.Modes.Section2_Gt

/-
# The number of modes of a Gaussian KDE — the proof of `lem: pt.bdd`

§5.5 of arXiv:2412.09080v3, `sec: proof.pt.bdd`: the curve `ψ = (g, g')` that
the one-summand vector `(G(t), G'(t))` runs along, and the non-degeneracy of
the phases `φ_θ` that the stationary-phase bound on its Fourier transform rests
on.

**What the source says and what is carried here.**

* `(G(t), G'(t)) = (g(Z), g'(Z))` with `Z = t - X ~ N(t, 1)` and
  `g(z) = z e^{-βz²/2}`: `bigG` and `bigG'` are `gPt` and `gPt1` at `t - x`.
  The three derivatives of `g` and the determinant
  `g'g''' - g''² = -β(β²z⁴ + 3)e^{-βz²} < 0` are proved.

* "Any stationary point `x₀` of `φ_θ` satisfies `g'(x₀) = 0`" is a slip —
  `φ_θ' = cos θ g' + sin θ g''`, not `g'` — but the argument that follows does
  not use it: a stationary point where also `φ_θ'' = 0` would make the
  determinant vanish.  That argument is proved, `phase_nondegenerate`.

Source: arXiv:2412.09080v3, §5.5, `eq: Gt-prime`.
-/

open Real MeasureTheory
open scoped ENNReal

namespace Transformer
namespace Modes

/-- `g(z) = z e^{-βz²/2}`.  arXiv:2412.09080v3, `eq: Gt-prime`. -/
noncomputable def gPt (β z : ℝ) : ℝ := Real.exp (-(β / 2) * z ^ 2) * z

/-- `g'(z) = (1 - βz²) e^{-βz²/2}`.  arXiv:2412.09080v3, §5.5. -/
noncomputable def gPt1 (β z : ℝ) : ℝ := Real.exp (-(β / 2) * z ^ 2) * (1 - β * z ^ 2)

/-- `g''(z) = βz(βz² - 3) e^{-βz²/2}`.  arXiv:2412.09080v3, §5.5. -/
noncomputable def gPt2 (β z : ℝ) : ℝ := Real.exp (-(β / 2) * z ^ 2) * (β * z * (β * z ^ 2 - 3))

/-- `g'''(z) = β(-β²z⁴ + 6βz² - 3) e^{-βz²/2}`.  arXiv:2412.09080v3, §5.5. -/
noncomputable def gPt3 (β z : ℝ) : ℝ :=
  Real.exp (-(β / 2) * z ^ 2) * (β * (-β ^ 2 * z ^ 4 + 6 * β * z ^ 2 - 3))

/-- **Equation (eq: Gt-prime).**  `(G(t), G'(t)) = (g(Z), g'(Z))` at `Z = t - X`. -/
theorem bigG_eq_gPt (β t x : ℝ) : bigG β t x = gPt β (t - x) ∧ bigG' β t x = gPt1 β (t - x) :=
  ⟨rfl, rfl⟩

/-- The derivative of the Gaussian factor. -/
theorem hasDerivAt_gaussFactor (β z : ℝ) :
    HasDerivAt (fun z => Real.exp (-(β / 2) * z ^ 2))
      (Real.exp (-(β / 2) * z ^ 2) * (-(β / 2) * (↑2 * z ^ (2 - 1)))) z :=
  ((hasDerivAt_pow 2 z).const_mul (-(β / 2))).exp

/-- `g'` is the derivative of `g`.  arXiv:2412.09080v3, §5.5. -/
theorem hasDerivAt_gPt (β z : ℝ) : HasDerivAt (gPt β) (gPt1 β z) z := by
  have h := (hasDerivAt_gaussFactor β z).fun_mul (hasDerivAt_id' z)
  have hf : gPt β = fun z => Real.exp (-(β / 2) * z ^ 2) * z := rfl
  rw [hf]
  convert h using 1
  simp only [gPt1]
  ring

/-- `g''` is the derivative of `g'`.  arXiv:2412.09080v3, §5.5. -/
theorem hasDerivAt_gPt1 (β z : ℝ) : HasDerivAt (gPt1 β) (gPt2 β z) z := by
  have h := (hasDerivAt_gaussFactor β z).fun_mul
    ((hasDerivAt_const z (1 : ℝ)).fun_sub ((hasDerivAt_pow 2 z).const_mul β))
  have hf : gPt1 β = fun z => Real.exp (-(β / 2) * z ^ 2) * (1 - β * z ^ 2) := rfl
  rw [hf]
  convert h using 1
  simp only [gPt2]
  ring

/-- `g'''` is the derivative of `g''`.  arXiv:2412.09080v3, §5.5. -/
theorem hasDerivAt_gPt2 (β z : ℝ) : HasDerivAt (gPt2 β) (gPt3 β z) z := by
  have h := (hasDerivAt_gaussFactor β z).fun_mul
    (((hasDerivAt_pow 3 z).const_mul (β ^ 2)).fun_sub ((hasDerivAt_id' z).const_mul (3 * β)))
  have hf : gPt2 β = fun z => Real.exp (-(β / 2) * z ^ 2) * (β ^ 2 * z ^ 3 - 3 * β * z) := by
    funext z; simp only [gPt2]; ring
  rw [hf]
  convert h using 1
  simp only [gPt3]
  ring

/-- **`det(ψ', ψ'') = g'g''' - g''² = -β(β²z⁴ + 3)e^{-βz²}`**, where
`ψ = (g, g')`.  arXiv:2412.09080v3, §5.5. -/
theorem det_psi (β z : ℝ) :
    gPt1 β z * gPt3 β z - gPt2 β z ^ 2 = -β * (β ^ 2 * z ^ 4 + 3) * Real.exp (-β * z ^ 2) := by
  have he : Real.exp (-β * z ^ 2)
      = Real.exp (-(β / 2) * z ^ 2) * Real.exp (-(β / 2) * z ^ 2) := by
    rw [← Real.exp_add]; ring_nf
  rw [he]
  simp only [gPt1, gPt2, gPt3]
  ring

/-- **The determinant never vanishes:** it is negative for `β > 0`.
arXiv:2412.09080v3, §5.5. -/
theorem det_psi_neg {β : ℝ} (hβ : 0 < β) (z : ℝ) :
    gPt1 β z * gPt3 β z - gPt2 β z ^ 2 < 0 := by
  rw [det_psi]
  have : 0 < β * (β ^ 2 * z ^ 4 + 3) * Real.exp (-β * z ^ 2) := by positivity
  linarith

/-- **Stationary points of `φ_θ = cos θ g + sin θ g'` are non-degenerate**,
for every `θ`: `φ_θ'(x₀) = 0` forces `φ_θ''(x₀) ≠ 0`.
arXiv:2412.09080v3, §5.5; see the module docstring for the slip it corrects. -/
theorem phase_nondegenerate {β : ℝ} (hβ : 0 < β) (θ z : ℝ)
    (h : Real.cos θ * gPt1 β z + Real.sin θ * gPt2 β z = 0) :
    Real.cos θ * gPt2 β z + Real.sin θ * gPt3 β z ≠ 0 := by
  intro h'
  have hdet : gPt1 β z * gPt3 β z - gPt2 β z ^ 2 = 0 := by
    linear_combination (Real.cos θ * gPt3 β z - Real.sin θ * gPt2 β z) * h
      + (Real.sin θ * gPt1 β z - Real.cos θ * gPt2 β z) * h'
      - (gPt1 β z * gPt3 β z - gPt2 β z ^ 2) * Real.sin_sq_add_cos_sq θ
  exact (det_psi_neg hβ z).ne hdet

/-- The hypotheses of `det_psi_neg` and `phase_nondegenerate` are satisfiable:
at `θ = 0`, `z = 1`, `β = 1`, `φ_θ' = g'(1) = 0`. -/
example : (0 : ℝ) < 1 ∧ Real.cos 0 * gPt1 1 1 + Real.sin 0 * gPt2 1 1 = 0 :=
  ⟨one_pos, by simp [gPt1]⟩

end Modes
end Transformer

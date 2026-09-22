/-
# The number of modes of a Gaussian KDE — the Gaussian proxy of `p_t`

§2.2 of arXiv:2412.09080v3, `eq:Yi`, `eq:qt`, `eq:approx` and the first half of
`lem:phi-t`: the normal density with the mean `μ_t` and covariance `Σ_t` of
`(F_n(t), F_n'(t))`, read at `(0, y)`, and its exponent with the square
completed.

**What the source says and what is carried here.**

* The source writes the proxy as `(det Σ_t)^{-1/2} φ(Σ_t^{-1/2}[(0,y) - μ_t])`,
  `φ` the standard normal density on `ℝ²`.  Since
  `‖Σ^{-1/2} z‖² = zᵀ Σ⁻¹ z`, no matrix square root is needed: for
  `Σ = [[a, b], [b, d]]` with `D = ad - b²`,
  `zᵀ Σ⁻¹ z = (d z₁² - 2b z₁z₂ + a z₂²)/D`.  That is `krQuad`, and
  `krPhi = φ(Σ_t^{-1/2}[(x,y) - μ_t])`.

* `eq:Yi` and `eq:qt` define the standardized vector `Y` and its density `q_t`,
  and `eq:approx` is the informal `q_t ≈ φ`; the source makes it precise only
  as `eq:error-goal` of §3, where it is carried.

* The proof of `lem:phi-t` completes the square in `y`.  Done exactly, it
  gives `zᵀΣ⁻¹z = A + α(y - δ)²` at `z = (-μ₁, y - μ₂)`, with
  `A = μ₁²/a`, `α = a/D`, `δ = μ₂ - bμ₁/a`: `quadForm_complete_square`, proved
  for every symmetric `Σ` with `a ≠ 0`, `D ≠ 0`.  `phiA`, `phiAlpha` and
  `phiDelta` are these three for `Σ_t`, `μ_t`.  The asymptotics the source
  gives them, with their corrections, are in `Section2_PhiTAsymp.lean` and
  `Section2_PhiTDelta.lean`.

Source: arXiv:2412.09080v3, `eq:Yi`, `eq:qt`, `eq:approx`, `lem:phi-t`.
-/

import Transformer.Modes.Section2_MomentsP

open Real

namespace Transformer
namespace Modes

/-! ### `μ_t` and `Σ_t` -/

/-- `μ_{t,1} = √n E G(t)`.  Source: arXiv:2412.09080v3, `eq:moments-p`. -/
noncomputable def muFst (n : ℕ) (β t : ℝ) : ℝ := Real.sqrt n * meanG β t

/-- `μ_{t,2} = √n E G'(t)`.  Source: arXiv:2412.09080v3, `eq:moments-p`. -/
noncomputable def muSnd (n : ℕ) (β t : ℝ) : ℝ := Real.sqrt n * meanG' β t

/-- `Σ_{t,11} = Var G(t)`.  Source: arXiv:2412.09080v3, `eq:moments-p`. -/
noncomputable def sigmaFst (β t : ℝ) : ℝ := sqMeanG β t - meanG β t ^ 2

/-- `Σ_{t,12} = Cov(G(t), G'(t))`.  Source: arXiv:2412.09080v3, `eq:moments-p`. -/
noncomputable def sigmaCov (β t : ℝ) : ℝ := mulMeanGG' β t - meanG β t * meanG' β t

/-- `Σ_{t,22} = Var G'(t)`.  Source: arXiv:2412.09080v3, `eq:moments-p`. -/
noncomputable def sigmaSnd (β t : ℝ) : ℝ := sqMeanG' β t - meanG' β t ^ 2

/-- `det Σ_t`. -/
noncomputable def sigmaDet (β t : ℝ) : ℝ := sigmaFst β t * sigmaSnd β t - sigmaCov β t ^ 2

/-! ### The quadratic form and the proxy density -/

/-- `zᵀ Σ⁻¹ z` for `Σ = [[a, b], [b, d]]`, in closed form. -/
noncomputable def quadForm (a b d z₁ z₂ : ℝ) : ℝ :=
  (d * z₁ ^ 2 - 2 * b * z₁ * z₂ + a * z₂ ^ 2) / (a * d - b ^ 2)

/-- `‖Σ_t^{-1/2}[(x, y) - μ_t]‖²`.

Source: arXiv:2412.09080v3, `eq:qt`, `lem:phi-t`. -/
noncomputable def krQuad (n : ℕ) (β t x y : ℝ) : ℝ :=
  quadForm (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t) (x - muFst n β t) (y - muSnd n β t)

/-- `φ(Σ_t^{-1/2}[(x, y) - μ_t])`, `φ(z) = (2π)⁻¹ e^{-‖z‖²/2}`.

Source: arXiv:2412.09080v3, `eq:approx`. -/
noncomputable def krPhi (n : ℕ) (β t x y : ℝ) : ℝ :=
  (2 * π)⁻¹ * Real.exp (-krQuad n β t x y / 2)

/-! ### Completing the square -/

/-- **The square completed in `z₂`.**  For `a ≠ 0` and `D = ad - b² ≠ 0`,
`quadForm a b d (-m) (y - μ) = m²/a + (a/D)(y - (μ - bm/a))²`.

Source: arXiv:2412.09080v3, proof of `lem:phi-t`. -/
theorem quadForm_complete_square {a b d : ℝ} (ha : a ≠ 0) (hD : a * d - b ^ 2 ≠ 0)
    (m μ y : ℝ) :
    quadForm a b d (-m) (y - μ)
      = m ^ 2 / a + a / (a * d - b ^ 2) * (y - (μ - b * m / a)) ^ 2 := by
  unfold quadForm
  generalize hDdef : a * d - b ^ 2 = D at hD ⊢
  have hd : d = (D + b ^ 2) / a := by rw [← hDdef]; field_simp; ring
  subst hd
  field_simp
  ring

/-- The hypotheses of `quadForm_complete_square` are satisfiable: `Σ = I₂`. -/
example : (1 : ℝ) ≠ 0 ∧ (1 : ℝ) * 1 - 0 ^ 2 ≠ 0 := by norm_num

/-- `A_t = μ_{t,1}²/Σ_{t,11}`, the constant term of `lem:phi-t`. -/
noncomputable def phiA (n : ℕ) (β t : ℝ) : ℝ := muFst n β t ^ 2 / sigmaFst β t

/-- `α_t = Σ_{t,11}/det Σ_t`, the curvature in `y` of `lem:phi-t`. -/
noncomputable def phiAlpha (β t : ℝ) : ℝ := sigmaFst β t / sigmaDet β t

/-- `δ_t = μ_{t,2} - Σ_{t,12}μ_{t,1}/Σ_{t,11}`, the centre in `y` of `lem:phi-t`. -/
noncomputable def phiDelta (n : ℕ) (β t : ℝ) : ℝ :=
  muSnd n β t - sigmaCov β t * muFst n β t / sigmaFst β t

/-- **Lemma (lem:phi-t), first display, exactly.**
`‖Σ_t^{-1/2}[(0, y) - μ_t]‖² = A_t + α_t(y - δ_t)²`, whenever `Σ_{t,11} ≠ 0`
and `det Σ_t ≠ 0`.  The source writes `~`, with `A_t, α_t, δ_t` known only up
to constants; with the exact coefficients it is an identity.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:phi-t`. -/
theorem krQuad_zero_eq {n : ℕ} {β t : ℝ} (h₁ : sigmaFst β t ≠ 0) (hD : sigmaDet β t ≠ 0)
    (y : ℝ) :
    krQuad n β t 0 y = phiA n β t + phiAlpha β t * (y - phiDelta n β t) ^ 2 := by
  rw [krQuad, zero_sub, quadForm_complete_square h₁ hD]
  rfl

end Modes
end Transformer

/-
# The number of modes of a Gaussian KDE — the moments as quotients times scales

The five entries of `μ_t` and `Σ_t` (`Section2_PhiT.lean`), each written as a
quotient times the leading term of `lem:moments-p`.  The quotients are
continuous functions of `p = (1/β, t²/β)`, the ones of `Σ_t` up to a bounded
function of `t` times a function vanishing at `p = 0`
(`Section5_CovAsymp.lean`).  This is the form the uniform bounds over `T` of
`lem:phi-t` are read from.

Source: arXiv:2412.09080v3, `lem:moments-p`, §5.2.
-/

import Transformer.Modes.Section2_PhiT
import Transformer.Modes.Section5_CovAsymp

open Real

namespace Transformer
namespace Modes

/-- The quotient of `E G` by `β^{-3/2}e^{-t²/2}t`, at `p = (1/β, t²/β)`. -/
noncomputable def gMean (p : ℝ × ℝ) : ℝ :=
  Real.exp (p.2 * (1 / (1 + p.1)) / 2) * (1 / (1 + p.1)) * √(1 / (1 + p.1))

/-- The quotient of `E G'` by `β^{-3/2}e^{-t²/2}(1 - t² + 1/β)`. -/
noncomputable def gMean' (p : ℝ × ℝ) : ℝ :=
  Real.exp (p.2 * (1 / (1 + p.1)) / 2) * (1 / (1 + p.1)) ^ 2 * √(1 / (1 + p.1))

/-- The second-moment part of the quotient of `Var G`. -/
noncomputable def gVarA (p : ℝ × ℝ) : ℝ :=
  Real.exp (p.2 * (1 / (2 + p.1)) / 2) * (1 + p.2 * (1 / (2 + p.1))) * (2 * √2)
    * (1 / (2 + p.1)) * √(1 / (2 + p.1))

/-- The product-of-means part of the quotient of `Var G`, without `t²e^{-t²/2}`. -/
noncomputable def gVarΨ (p : ℝ × ℝ) : ℝ :=
  -(2 * √2) * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 3 * p.1 * √p.1

/-- The mixed-moment part of the quotient of `Cov(G, G')`. -/
noncomputable def gCovA (p : ℝ × ℝ) : ℝ :=
  Real.exp (p.2 * (1 / (2 + p.1)) / 2) * (1 - p.1 / 2 + p.2 / 2 - p.1 ^ 2 / 2) * (8 * √2)
    * (1 / (2 + p.1)) ^ 3 * √(1 / (2 + p.1))

/-- The product-of-means part of the quotient of `Cov(G, G')`, without
`e^{-t²/2}(1 - t² + 1/β)`. -/
noncomputable def gCovΨ (p : ℝ × ℝ) : ℝ :=
  4 * √2 * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 4 * p.1 * √p.1

/-- The second-moment part of the quotient of `Var G'`. -/
noncomputable def gVar'A (p : ℝ × ℝ) : ℝ :=
  Real.exp (p.2 * (1 / (2 + p.1)) / 2)
    * (1 + (p.2 + 5 * p.1) / 3 + (p.2 ^ 2 - 2 * p.2 * p.1 + 15 * p.1 ^ 2) / 12
      - (p.2 * p.1 ^ 2 - 3 * p.1 ^ 3) / 6 + p.1 ^ 4 / 12)
    * (16 * √2) * (1 / (2 + p.1)) ^ 4 * √(1 / (2 + p.1))

/-- The product-of-means part of the quotient of `Var G'`, without
`e^{-t²/2}(1 + 1/β - t²)²`. -/
noncomputable def gVar'Ψ (p : ℝ × ℝ) : ℝ :=
  -(4 * √2 / 3) * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 5 * p.1 ^ 2
    * √p.1

/-- The quotient of `Var G` by `2^{-5/2}β^{-3/2}e^{-t²/2}·2`. -/
noncomputable def qVar (β t : ℝ) : ℝ :=
  gVarA (β⁻¹, t ^ 2 / β) + t ^ 2 * Real.exp (-(t ^ 2) / 2) * gVarΨ (β⁻¹, t ^ 2 / β)

/-- The quotient of `Cov(G, G')` by `2^{-5/2}β^{-3/2}e^{-t²/2}·(-t)`. -/
noncomputable def qCov (β t : ℝ) : ℝ :=
  gCovA (β⁻¹, t ^ 2 / β)
    + Real.exp (-(t ^ 2) / 2) * (1 - t ^ 2 + β⁻¹) * gCovΨ (β⁻¹, t ^ 2 / β)

/-- The quotient of `Var G'` by `2^{-5/2}β^{-3/2}e^{-t²/2}·3β`. -/
noncomputable def qVar' (β t : ℝ) : ℝ :=
  gVar'A (β⁻¹, t ^ 2 / β)
    + Real.exp (-(t ^ 2) / 2) * (1 + β⁻¹ - t ^ 2) ^ 2 * gVar'Ψ (β⁻¹, t ^ 2 / β)

/-- `E G(t) = gMean · β^{-3/2}e^{-t²/2}t`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem meanG_eq_gMean {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG β t = gMean (β⁻¹, t ^ 2 / β) * (momentScale β t * t) := by
  rw [meanG_eq_mul hβ, gMean, momentScale]

/-- `E G'(t) = gMean' · β^{-3/2}e^{-t²/2}(1 - t² + 1/β)`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem meanG'_eq_gMean' {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    meanG' β t = gMean' (β⁻¹, t ^ 2 / β) * (momentScale β t * (1 - t ^ 2 + β⁻¹)) := by
  rw [meanG'_eq_mul hβ, gMean', momentScale]

/-- `Σ_{t,11} = qVar · 2^{-5/2}β^{-3/2}e^{-t²/2}·2`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem sigmaFst_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sigmaFst β t = qVar β t * (2 ^ (-(5 : ℝ) / 2) * momentScale β t * 2) := by
  rw [sigmaFst, varG_eq_mul hβ, qVar, gVarA, gVarΨ, momentScale]
  ring

/-- `Σ_{t,12} = qCov · 2^{-5/2}β^{-3/2}e^{-t²/2}·(-t)`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem sigmaCov_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sigmaCov β t = qCov β t * (2 ^ (-(5 : ℝ) / 2) * momentScale β t * (-t)) := by
  rw [sigmaCov, covG_eq_mul hβ, qCov, gCovA, gCovΨ, momentScale]
  ring

/-- `Σ_{t,22} = qVar' · 2^{-5/2}β^{-3/2}e^{-t²/2}·3β`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem sigmaSnd_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    sigmaSnd β t = qVar' β t * (2 ^ (-(5 : ℝ) / 2) * momentScale β t * (3 * β)) := by
  rw [sigmaSnd, varG'_eq_mul hβ, qVar', gVar'A, gVar'Ψ, momentScale]
  ring

/-- The hypothesis of the five identities above is satisfiable: `β = 1`. -/
example (t : ℝ) : sigmaSnd 1 t = qVar' 1 t * (2 ^ (-(5 : ℝ) / 2) * momentScale 1 t * (3 * 1)) :=
  sigmaSnd_eq one_pos t

/-- `A_t = β^{-3/2} n t² e^{-t²/2} · gMean²/(qVar·2^{-3/2})`, exactly.

Source: arXiv:2412.09080v3, `lem:phi-t`; §5.2. -/
theorem phiA_eq {n : ℕ} {β : ℝ} (hβ : 0 < β) (t : ℝ) (h3 : qVar β t ≠ 0) :
    phiA n β t = (β ^ (-(3 : ℝ) / 2) * n * t ^ 2 * Real.exp (-(t ^ 2) / 2))
      * (gMean (β⁻¹, t ^ 2 / β) ^ 2 / (qVar β t * (2 ^ (-(5 : ℝ) / 2) * 2))) := by
  have hm : 0 < β ^ (-(3 : ℝ) / 2) := by positivity
  have hp : 0 < (2 : ℝ) ^ (-(5 : ℝ) / 2) := by positivity
  rw [phiA, muFst, sigmaFst_eq hβ, meanG_eq_gMean hβ, momentScale, mul_pow,
    Real.sq_sqrt (Nat.cast_nonneg _)]
  field_simp

/-- `α_t = β^{1/2} e^{t²/2} · 2 qVar/(2^{-5/2}(6 qVar qVar' - (t²/β) qCov²))`, exactly.

Source: arXiv:2412.09080v3, `lem:phi-t`; §5.2. -/
theorem phiAlpha_eq {β : ℝ} (hβ : 0 < β) (t : ℝ) (h3 : qVar β t ≠ 0)
    (hD : 6 * qVar β t * qVar' β t - t ^ 2 / β * qCov β t ^ 2 ≠ 0) :
    phiAlpha β t = (β ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2))
      * (2 * qVar β t / (2 ^ (-(5 : ℝ) / 2) * (6 * qVar β t * qVar' β t
        - t ^ 2 / β * qCov β t ^ 2))) := by
  have hp : 0 < (2 : ℝ) ^ (-(5 : ℝ) / 2) := by positivity
  have hs : 0 < β ^ ((1 : ℝ) / 2) := by positivity
  have hr : β ^ (-(3 : ℝ) / 2) = β ^ ((1 : ℝ) / 2) / β ^ 2 := by
    rw [eq_div_iff (by positivity), ← Real.rpow_natCast, ← Real.rpow_add hβ]; norm_num
  have he : Real.exp (-(t ^ 2) / 2) = (Real.exp (t ^ 2 / 2))⁻¹ := by
    rw [← Real.exp_neg]; ring_nf
  rw [phiAlpha, sigmaDet, sigmaFst_eq hβ, sigmaSnd_eq hβ, sigmaCov_eq hβ, momentScale, hr, he]
  generalize qVar β t = F at h3 hD ⊢
  generalize qVar' β t = G at hD ⊢
  generalize qCov β t = H at hD ⊢
  have hD' : 6 * F * G * β - t ^ 2 * H ^ 2 ≠ 0 := by
    contrapose! hD; field_simp; linarith
  have hsq : (β ^ ((1 : ℝ) / 2)) ^ 2 = β := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ.le]; norm_num
  field_simp
  rw [hsq]
  ring

/-- `δ_t = n^{1/2}β^{-3/2}e^{-t²/2}(gMean'(1 - t² + 1/β) + t² qCov gMean/(2 qVar))`,
exactly.

Source: arXiv:2412.09080v3, `lem:phi-t`; §5.2. -/
theorem phiDelta_eq {n : ℕ} {β : ℝ} (hβ : 0 < β) (t : ℝ) (h3 : qVar β t ≠ 0) :
    phiDelta n β t = √n * momentScale β t * (gMean' (β⁻¹, t ^ 2 / β) * (1 - t ^ 2 + β⁻¹)
      + t ^ 2 * (qCov β t * gMean (β⁻¹, t ^ 2 / β) / (2 * qVar β t))) := by
  have hm : momentScale β t ≠ 0 := by unfold momentScale; positivity
  have hp : (2 : ℝ) ^ (-(5 : ℝ) / 2) ≠ 0 := by positivity
  rw [phiDelta, muSnd, muFst, sigmaCov_eq hβ, sigmaFst_eq hβ, meanG_eq_gMean hβ,
    meanG'_eq_gMean' hβ]
  field_simp
  ring

/-- The hypothesis of `phiA_eq` and `phiDelta_eq` is satisfiable: at `β = 1`,
`t = 0`, `qVar = gVarA(1, 0) > 0`.  For `phiAlpha_eq`, see
`Section2_PhiTUniform.lean`. -/
example : qVar 1 0 ≠ 0 := by
  have h1 : 0 < qVar 1 0 := by simp [qVar, gVarA]; positivity
  exact h1.ne'

end Modes
end Transformer

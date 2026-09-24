/-
# The number of modes of a Gaussian KDE — covariance scales for the proxy Kac–Rice integral

For `eq:int-phi-final`, the covariance factor combines with the curvature
`α_t` of `lem:phi-t` to produce the `√β` scale.

Source: arXiv:2412.09080v3, `lem:phi-t`, `eq:int-phi-final`.
-/

import Transformer.Modes.Section2_PhiTAsymp

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer
namespace Modes

/-- Uniformly on `T`, the first covariance entry has the scale
`β^{-3/2}e^{-t²/2}`; this is the `qVar ∈ (1/2,2)` part of `lem:moments-p`.

Source: arXiv:2412.09080v3, `lem:moments-p`, §5.2. -/
theorem eventually_sigmaFst_bounds {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        C₁ * momentScale (B k) t ≤ sigmaFst (B k) t ∧
          sigmaFst (B k) t ≤ C₂ * momentScale (B k) t := by
  let q : ℝ := (2 : ℝ) ^ (-(5 : ℝ) / 2)
  have hq : 0 < q := by dsimp [q]; positivity
  refine ⟨q, 4 * q, hq, by positivity, ?_⟩
  filter_upwards [eventually_quot_mem_Ioo hreg hω] with k hk t ht
  have hβ := hreg.B_pos k
  have hm : 0 ≤ momentScale (B k) t := by unfold momentScale; positivity
  obtain ⟨-, -, ⟨hqlo, hqhi⟩, -⟩ := hk t ht
  rw [sigmaFst_eq hβ]
  constructor
  · calc
      q * momentScale (B k) t = (1 / 2) * (q * momentScale (B k) t * 2) := by ring
      _ ≤ qVar (B k) t * (q * momentScale (B k) t * 2) := by
        exact mul_le_mul_of_nonneg_right hqlo.le (by positivity)
  · calc
      qVar (B k) t * (q * momentScale (B k) t * 2)
          ≤ 2 * (q * momentScale (B k) t * 2) := by
            exact mul_le_mul_of_nonneg_right hqhi.le (by positivity)
      _ = (4 * q) * momentScale (B k) t := by ring

/-- On `T` the covariance determinant is eventually positive.  The global
version for every `β > 0` and `t` is proved later in `Section3_SigmaPos`; this
local version needs only the quotient bounds of §5.2.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:Yi`. -/
theorem eventually_sigmaDet_pos {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
      0 < sigmaDet (B k) t := by
  obtain ⟨C, _, hC, _, hF⟩ := eventually_sigmaFst_bounds hreg hω
  obtain ⟨L, _, hL, _, hα⟩ := phiAlpha_isTheta hreg hω
  filter_upwards [hF, hα] with k hFk hαk t ht
  have hβ := hreg.B_pos k
  have hm : 0 < momentScale (B k) t := by unfold momentScale; positivity
  have hfst : 0 < sigmaFst (B k) t :=
    lt_of_lt_of_le (mul_pos hC hm) (hFk t ht).1
  have hp : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  have halpha : 0 < phiAlpha (B k) t :=
    lt_of_lt_of_le (mul_pos hL hp) (hαk t ht).1
  exact (div_pos_iff_of_pos_left hfst).mp (by simpa [phiAlpha] using halpha)

/-- The scales of `Σ₁₁` and `α_t` multiply to `β⁻¹`. -/
theorem momentScale_mul_alphaBase {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    momentScale β t * (β ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)) = β⁻¹ := by
  unfold momentScale
  calc
    β ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) *
        (β ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2))
      = (β ^ (-(3 : ℝ) / 2) * β ^ ((1 : ℝ) / 2)) *
          (Real.exp (-(t ^ 2) / 2) * Real.exp (t ^ 2 / 2)) := by ring
    _ = β⁻¹ := by
      rw [← Real.rpow_add hβ, ← Real.exp_add]
      have he : -(3 : ℝ) / 2 + (1 : ℝ) / 2 = -1 := by ring
      rw [he]
      simp [Real.rpow_neg_one, show -(t ^ 2) / 2 + t ^ 2 / 2 = (0 : ℝ) by ring]

/-- Uniformly on `T`, `Σ₁₁ α_t` is a positive constant multiple of `β⁻¹`.
This is the scale calculation behind the `√β` in `eq:int-phi-final`.

Source: arXiv:2412.09080v3, `eq:int-phi-final`, `lem:moments-p`. -/
theorem eventually_sigmaFst_mul_phiAlpha_bounds {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        C₁ * (B k)⁻¹ ≤ sigmaFst (B k) t * phiAlpha (B k) t ∧
          sigmaFst (B k) t * phiAlpha (B k) t ≤ C₂ * (B k)⁻¹ := by
  obtain ⟨F₁, F₂, hF₁, hF₂, hF⟩ := eventually_sigmaFst_bounds hreg hω
  obtain ⟨A₁, A₂, hA₁, hA₂, hA⟩ := phiAlpha_isTheta hreg hω
  refine ⟨F₁ * A₁, F₂ * A₂, by positivity, by positivity, ?_⟩
  filter_upwards [hF, hA] with k hFk hAk t ht
  have hβ := hreg.B_pos k
  let m := momentScale (B k) t
  let b := B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2)
  have hm : 0 < m := by dsimp [m, momentScale]; positivity
  have hb : 0 < b := by dsimp [b]; positivity
  have hmb : m * b = (B k)⁻¹ := momentScale_mul_alphaBase hβ t
  have hf : 0 < sigmaFst (B k) t :=
    lt_of_lt_of_le (mul_pos hF₁ hm) (hFk t ht).1
  have ha : 0 < phiAlpha (B k) t :=
    lt_of_lt_of_le (mul_pos hA₁ hb) (hAk t ht).1
  constructor
  · calc
      (F₁ * A₁) * (B k)⁻¹ = (F₁ * m) * (A₁ * b) := by rw [← hmb]; ring
      _ ≤ sigmaFst (B k) t * (A₁ * b) :=
        mul_le_mul_of_nonneg_right (hFk t ht).1 (mul_pos hA₁ hb).le
      _ ≤ sigmaFst (B k) t * phiAlpha (B k) t :=
        mul_le_mul_of_nonneg_left (hAk t ht).1 hf.le
  · calc
      sigmaFst (B k) t * phiAlpha (B k) t
          ≤ (F₂ * m) * phiAlpha (B k) t :=
            mul_le_mul_of_nonneg_right (hFk t ht).2 ha.le
      _ ≤ (F₂ * m) * (A₂ * b) :=
        mul_le_mul_of_nonneg_left (hAk t ht).2 (mul_pos hF₂ hm).le
      _ = (F₂ * A₂) * (B k)⁻¹ := by rw [← hmb]; ring

/-- Squaring the factor left after `lem:phi-t` removes the square root of the
covariance determinant: `(det Σ)⁻¹ α⁻² = (Σ₁₁ α)⁻¹`. -/
theorem det_inv_sqrt_mul_alpha_inv_sq {β t : ℝ}
    (hF : 0 < sigmaFst β t) (hD : 0 < sigmaDet β t) :
    (sigmaDet β t ^ (-(1 : ℝ) / 2) * (phiAlpha β t)⁻¹) ^ 2 =
      (sigmaFst β t * phiAlpha β t)⁻¹ := by
  have hr : (sigmaDet β t ^ (-(1 : ℝ) / 2)) ^ 2 = (sigmaDet β t)⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hD.le]
    norm_num [Real.rpow_neg_one]
  rw [mul_pow, hr]
  unfold phiAlpha
  field_simp [hF.ne', hD.ne']

/-- The determinant factor times the Gaussian first-moment factor is
uniformly comparable to `√β` on `T`.

Source: arXiv:2412.09080v3, `eq:int-phi-final`, `lem:moments-p`. -/
theorem eventually_det_inv_sqrt_mul_alpha_inv_bounds {c : ℝ} {N : ℕ → ℕ}
    {B : ℕ → ℝ} (hreg : IsRegime c N B) {ω : ℝ → ℝ}
    (hω : IsSlowGrowth ω) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        C₁ * Real.sqrt (B k) ≤
          sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹ ∧
        sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹ ≤
          C₂ * Real.sqrt (B k) := by
  obtain ⟨L, U, hL, hU, hP⟩ := eventually_sigmaFst_mul_phiAlpha_bounds hreg hω
  obtain ⟨A, A₂, hA, -, hα⟩ := phiAlpha_isTheta hreg hω
  have hsqrtL : 0 < Real.sqrt L := Real.sqrt_pos.2 hL
  have hsqrtU : 0 < Real.sqrt U := Real.sqrt_pos.2 hU
  refine ⟨(Real.sqrt U)⁻¹, (Real.sqrt L)⁻¹,
    inv_pos.2 hsqrtU, inv_pos.2 hsqrtL, ?_⟩
  filter_upwards [hP, hα, eventually_sigmaDet_pos hreg hω] with k hPk hαk hDk t ht
  have hβ := hreg.B_pos k
  have hb : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  have ha : 0 < phiAlpha (B k) t :=
    lt_of_lt_of_le (mul_pos hA hb) (hαk t ht).1
  have hd : 0 < sigmaDet (B k) t := hDk t ht
  have hf : 0 < sigmaFst (B k) t :=
    (div_pos_iff_of_pos_right hd).mp (by simpa [phiAlpha] using ha)
  let p := sigmaFst (B k) t * phiAlpha (B k) t
  let x := sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹
  have hp : 0 < p := mul_pos hf ha
  have hx : 0 < x := mul_pos (Real.rpow_pos_of_pos hd _) (inv_pos.2 ha)
  have hsq : x ^ 2 = p⁻¹ := det_inv_sqrt_mul_alpha_inv_sq hf hd
  have hsqrt : Real.sqrt (p⁻¹) = x := by
    rw [← hsq, Real.sqrt_sq_eq_abs, abs_of_pos hx]
  have hlo : (U * (B k)⁻¹)⁻¹ ≤ p⁻¹ := inv_anti₀ hp (hPk t ht).2
  have hhi : p⁻¹ ≤ (L * (B k)⁻¹)⁻¹ :=
    inv_anti₀ (mul_pos hL (inv_pos.2 hβ)) (hPk t ht).1
  constructor
  · calc
      (Real.sqrt U)⁻¹ * Real.sqrt (B k) = Real.sqrt ((B k) / U) := by
        rw [Real.sqrt_div hβ.le]; ring
      _ = Real.sqrt ((U * (B k)⁻¹)⁻¹) := by
        congr 1
        field_simp [hβ.ne', hU.ne']
      _ ≤ Real.sqrt (p⁻¹) := Real.sqrt_le_sqrt hlo
      _ = x := hsqrt
  · calc
      x = Real.sqrt (p⁻¹) := hsqrt.symm
      _ ≤ Real.sqrt ((L * (B k)⁻¹)⁻¹) := Real.sqrt_le_sqrt hhi
      _ = Real.sqrt ((B k) / L) := by
        congr 1
        field_simp [hβ.ne', hL.ne']
      _ = (Real.sqrt L)⁻¹ * Real.sqrt (B k) := by
        rw [Real.sqrt_div hβ.le]; ring

/-- The regime and slow-growth hypotheses of the covariance bounds occur at
`β = n`, `ω(β) = √(log log β)`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Modes
end Transformer

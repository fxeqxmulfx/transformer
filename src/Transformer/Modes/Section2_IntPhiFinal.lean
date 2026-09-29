/-
# The number of modes of a Gaussian KDE — `eq:int-phi-final`

§2.3 of arXiv:2412.09080v3.  The proxy Kac-Rice integral
`∫_S ∫_0^∞ y (det Σ_t)^{-1/2} φ(Σ_t^{-1/2}[(0,y) - μ_t]) dy dt` is `proxyKR`,
written with `lintegral` like `eq:krf` in `kacRice`; `eq:int-phi-final` compares it
with `√β ∫_S e^{-A_t/2} dt`.

**`eq:int-phi-final` is carried for `S ⊆ T`, and with `n ≲ β^{5/2}`.**  The
source states it "for any measurable `S ⊂ ℝ`", but derives it from
`lem:phi-t`, which holds only on `T`, and only with `n ≲ β^{5/2}`
(`integral_krPhi_isTheta`).  Without that hypothesis it fails for `S` a small
interval around `0`, for the reason given in `Section2_PhiTDelta.lean`.  The
bound over `T'` that needs no such hypothesis is `main_int_phi_T'`
(`Section2_MainIntPhi.lean`), which handles the shift directly.

Source: arXiv:2412.09080v3, §2.3 (`sec: 2.3`), `eq:approx`, `eq:int-phi-final`.
-/

import Transformer.Modes.Section2_PhiTAsymp
import Transformer.Modes.Section2_ProxyKRInner

open Real Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer
namespace Modes

/-- `∫_S ∫_0^∞ y (det Σ_t)^{-1/2} φ(Σ_t^{-1/2}[(0,y) - μ_t]) dy dt`, the Kac-Rice
integral `eq:krf` with `p_t` replaced by its Gaussian proxy.

Source: arXiv:2412.09080v3, `eq:approx`, `eq:int-phi-final`. -/
noncomputable def proxyKR (n : ℕ) (β : ℝ) (S : Set ℝ) : ℝ≥0∞ :=
  ∫⁻ t in S, ∫⁻ y in Set.Ioi (0 : ℝ),
    ENNReal.ofReal (y * sigmaDet β t ^ (-(1 : ℝ) / 2) * krPhi n β t 0 y)

/-- **Equation (eq:int-phi-final), for `S ⊆ T` and `n ≲ β^{5/2}`.**
`∫_S ∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy dt ≍ √β ∫_S e^{-A_t/2} dt`, uniformly in
`S`.  The source states it for every measurable `S ⊂ ℝ` and no bound on `n`;
see the module docstring.

Source: arXiv:2412.09080v3, `eq:int-phi-final`. -/
theorem int_phi_final {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω)
    (hN : (fun k => (N k : ℝ)) =O[atTop] fun k => B k ^ ((5 : ℝ) / 2)) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ᶠ k in atTop, ∀ S ⊆ intervalT (N k) (B k) (ω (B k)),
      MeasurableSet S →
        ENNReal.ofReal (C₁ * Real.sqrt (B k))
            * ∫⁻ t in S, ENNReal.ofReal (Real.exp (-phiA (N k) (B k) t / 2))
          ≤ proxyKR (N k) (B k) S ∧
        proxyKR (N k) (B k) S
          ≤ ENNReal.ofReal (C₂ * Real.sqrt (B k))
            * ∫⁻ t in S, ENNReal.ofReal (Real.exp (-phiA (N k) (B k) t / 2)) := by
  obtain ⟨I₁, I₂, hI₁, hI₂, hI⟩ := integral_krPhi_isTheta hreg hω hN
  obtain ⟨F₁, F₂, hF₁, hF₂, hF⟩ :=
    eventually_det_inv_sqrt_mul_alpha_inv_bounds hreg hω
  obtain ⟨A₁, A₂, hA₁, hA₂, hα⟩ := phiAlpha_isTheta hreg hω
  refine ⟨F₁ * I₁, F₂ * I₂, by positivity, by positivity, ?_⟩
  filter_upwards [hI, hF, hα, eventually_sigmaDet_pos hreg hω] with
    k hIk hFk hαk hDk S hST hSm
  have hpoint (t : ℝ) (ht : t ∈ S) :
      ENNReal.ofReal ((F₁ * I₁) * Real.sqrt (B k)) *
          ENNReal.ofReal (Real.exp (-phiA (N k) (B k) t / 2))
        ≤ ∫⁻ y in Set.Ioi (0 : ℝ),
            ENNReal.ofReal (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y) ∧
      (∫⁻ y in Set.Ioi (0 : ℝ),
          ENNReal.ofReal (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y))
        ≤ ENNReal.ofReal ((F₂ * I₂) * Real.sqrt (B k)) *
          ENNReal.ofReal (Real.exp (-phiA (N k) (B k) t / 2)) := by
    have htT := hST ht
    have hβ := hreg.B_pos k
    have hp : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
    have ha : 0 < phiAlpha (B k) t :=
      lt_of_lt_of_le (mul_pos hA₁ hp) (hαk t htT).1
    have hd : 0 < sigmaDet (B k) t := hDk t htT
    have hfac : 0 < sigmaDet (B k) t ^ (-(1 : ℝ) / 2) :=
      Real.rpow_pos_of_pos hd _
    let e := Real.exp (-phiA (N k) (B k) t / 2)
    let f := sigmaDet (B k) t ^ (-(1 : ℝ) / 2)
    let a := (phiAlpha (B k) t)⁻¹
    let r := Real.sqrt (B k)
    let J := ∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y
    have he : 0 < e := Real.exp_pos _
    have hf : 0 < f := hfac
    have hInt : I₁ * (a * e) ≤ J ∧ J ≤ I₂ * (a * e) := hIk t htT
    have hScale : F₁ * r ≤ f * a ∧ f * a ≤ F₂ * r := hFk t htT
    have hreal : (F₁ * I₁) * r * e ≤ f * J ∧ f * J ≤ (F₂ * I₂) * r * e := by
      constructor
      · calc
          (F₁ * I₁) * r * e = (I₁ * e) * (F₁ * r) := by ring
          _ ≤ (I₁ * e) * (f * a) :=
            mul_le_mul_of_nonneg_left hScale.1 (mul_pos hI₁ he).le
          _ = f * (I₁ * (a * e)) := by ring
          _ ≤ f * J := mul_le_mul_of_nonneg_left hInt.1 hf.le
      · calc
          f * J ≤ f * (I₂ * (a * e)) := mul_le_mul_of_nonneg_left hInt.2 hf.le
          _ = (I₂ * e) * (f * a) := by ring
          _ ≤ (I₂ * e) * (F₂ * r) :=
            mul_le_mul_of_nonneg_left hScale.2 (mul_pos hI₂ he).le
          _ = (F₂ * I₂) * r * e := by ring
    have hbridge := lintegral_det_krPhi_eq ha hd (n := N k)
    constructor
    · calc
        ENNReal.ofReal ((F₁ * I₁) * r) * ENNReal.ofReal e
            = ENNReal.ofReal (((F₁ * I₁) * r) * e) :=
              (ENNReal.ofReal_mul (by positivity)).symm
        _ ≤ ENNReal.ofReal (f * J) := ENNReal.ofReal_le_ofReal hreal.1
        _ = ∫⁻ y in Set.Ioi (0 : ℝ),
            ENNReal.ofReal (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y) := by
              rw [hbridge]
              exact ENNReal.ofReal_mul hf.le
    · calc
        (∫⁻ y in Set.Ioi (0 : ℝ),
            ENNReal.ofReal (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y))
            = ENNReal.ofReal (f * J) := by
              rw [hbridge]
              exact (ENNReal.ofReal_mul hf.le).symm
        _ ≤ ENNReal.ofReal (((F₂ * I₂) * r) * e) := ENNReal.ofReal_le_ofReal hreal.2
        _ = ENNReal.ofReal ((F₂ * I₂) * r) * ENNReal.ofReal e :=
          ENNReal.ofReal_mul (by positivity)
  constructor
  · unfold proxyKR
    rw [← lintegral_const_mul' (ENNReal.ofReal ((F₁ * I₁) * Real.sqrt (B k)))
      _ ENNReal.ofReal_ne_top]
    apply lintegral_mono_ae
    filter_upwards [ae_restrict_mem hSm] with t ht
    exact (hpoint t ht).1
  · unfold proxyKR
    rw [← lintegral_const_mul' (ENNReal.ofReal ((F₂ * I₂) * Real.sqrt (B k)))
      _ ENNReal.ofReal_ne_top]
    apply lintegral_mono_ae
    filter_upwards [ae_restrict_mem hSm] with t ht
    exact (hpoint t ht).2

/-- The hypotheses of `int_phi_final` are satisfiable at `β = n`; compare the
witness following `integral_krPhi_isTheta`. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) ∧
    (fun k => ((k + 1 : ℕ) : ℝ)) =O[atTop]
      fun k => ((k + 1 : ℕ) : ℝ) ^ ((5 : ℝ) / 2) := by
  refine ⟨isRegime_succ, isSlowGrowth_sqrt_log_log, ?_⟩
  refine IsBigO.of_bound 1 ?_
  filter_upwards with k
  have h1 : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by
    push_cast; linarith [k.cast_nonneg (α := ℝ)]
  rw [one_mul, Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
  calc ((k + 1 : ℕ) : ℝ) = ((k + 1 : ℕ) : ℝ) ^ (1 : ℝ) := (Real.rpow_one _).symm
    _ ≤ _ := Real.rpow_le_rpow_of_exponent_le h1 (by norm_num)

end Modes
end Transformer

/-
# The number of modes of a Gaussian KDE — the Kac-Rice integral over `φ`

§2.3 of arXiv:2412.09080v3: `eq:int-phi-final`, `lem:main-int-phi` and the two
bounds `eq:int-phi-b` its proof reduces to.

**What the source says and what is carried here.**

* The proxy Kac-Rice integral `∫_S ∫_0^∞ y (det Σ_t)^{-1/2} φ(Σ_t^{-1/2}[(0,y) -
  μ_t]) dy dt` is `proxyKR`, written with `lintegral` like `eq:krf` in
  `kacRice`.

* **`eq:int-phi-final` is carried for `S ⊆ T`, and with `n ≲ β^{5/2}`.**  The
  source states it "for any measurable `S ⊂ ℝ`", but derives it from
  `lem:phi-t`, which holds only on `T`, and only with `n ≲ β^{5/2}`
  (`integral_krPhi_isTheta`).  Without that hypothesis it fails for `S` a small
  interval around `0`, for the reason given in `Section2_PhiTDelta.lean`.

* `lem:main-int-phi` is stated as the source has it.  Its proof goes through
  `eq:int-phi-final`, so for `c < 2/5` the source's proof does not cover it.
  The statement is still expected to hold: where `n ≫ β^{5/2}` the extra factor
  `√(α_t) δ_t` is paid for by the factor `e^{-A_t/2}`.

* `eq:int-phi-b` is carried in the form its proof uses: for every `C > 0`,
  `∫_T e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≍ √(log β)` and
  `∫_{T'} e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≲ 1`.  This form needs neither the
  moments nor `A_t`: `phiA_isTheta` is what ties it to `A_t`.  The bound on
  `T'` is proved here: on `T'` the exponent is at least `C t²`.  The bound on
  `T` is proved in `Section2_IntPhiB.lean`.

Source: arXiv:2412.09080v3, §2.3 (`sec: 2.3`), `eq:int-phi-final`,
`lem:main-int-phi`, `eq:int-phi-b`.
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

/-- The rate `β^{-3/2} n t² e^{-t²/2}` that `A_t` is `≍` to. -/
noncomputable def phiRate (n : ℕ) (β t : ℝ) : ℝ :=
  β ^ (-(3 : ℝ) / 2) * n * t ^ 2 * Real.exp (-(t ^ 2) / 2)

/-! ### `eq:int-phi-final` -/

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

/-! ### `eq:int-phi-b` -/

/-- On `T'`, `β^{-3/2} n t² e^{-t²/2} ≥ t²`: there `e^{-t²/2} ≥ β^{3/2}/n`. -/
theorem sq_le_phiRate {n : ℕ} {β t : ℝ} (hn : 1 ≤ n) (hβ : 1 ≤ β)
    (ht : t ∈ intervalT' n β) : t ^ 2 ≤ phiRate n β t := by
  unfold intervalT' at ht
  split_ifs at ht with h
  · rcases eq_or_ne t 0 with rfl | ht0
    · simp [phiRate]
    set a := 2 * Real.log n - 3 * Real.log β
    have ha : 0 < Real.sqrt a := lt_of_le_of_ne (Real.sqrt_nonneg _) fun h0 => by
      rw [← h0] at ht; exact ht0 (le_antisymm ht.2 (by linarith [ht.1]))
    have ha0 : 0 < a := Real.sqrt_pos.mp ha
    have hta : t ^ 2 ≤ a := by
      rw [← Real.sq_sqrt ha0.le, ← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (abs_le.mpr ⟨ht.1, ht.2⟩) 2
    have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
    have hβ0 : 0 < β := by linarith
    have hexp : β ^ ((3 : ℝ) / 2) / n ≤ Real.exp (-(t ^ 2) / 2) := by
      calc β ^ ((3 : ℝ) / 2) / n = Real.exp (-a / 2) := by
            rw [Real.rpow_def_of_pos hβ0, show -a / 2 = 3 / 2 * Real.log β - Real.log n by ring,
              Real.exp_sub, Real.exp_log hn0, mul_comm]
        _ ≤ Real.exp (-(t ^ 2) / 2) := Real.exp_le_exp.mpr (by linarith)
    have hone : 1 ≤ β ^ (-(3 : ℝ) / 2) * n * Real.exp (-(t ^ 2) / 2) := by
      calc (1 : ℝ) = β ^ (-(3 : ℝ) / 2) * n * (β ^ ((3 : ℝ) / 2) / n) := by
            rw [show -(3 : ℝ) / 2 = -((3 : ℝ) / 2) by ring, Real.rpow_neg hβ0.le]
            field_simp
        _ ≤ _ := by gcongr
    calc t ^ 2 = 1 * t ^ 2 := (one_mul _).symm
      _ ≤ β ^ (-(3 : ℝ) / 2) * n * Real.exp (-(t ^ 2) / 2) * t ^ 2 := by gcongr
      _ = phiRate n β t := by unfold phiRate; ring
  · exact absurd ht (Set.notMem_empty t)

/-- The hypotheses of `sq_le_phiRate` are satisfiable: `n = β = 1`, `t = 0`.
Those of the regime theorems below are witnessed in `Section2_PhiTDelta.lean`. -/
example : 1 ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ∈ intervalT' 1 1 := by
  refine ⟨le_rfl, le_rfl, ?_⟩
  unfold intervalT'
  simp only [show (1 : ℝ) ≤ ((1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) by simp, ite_true]
  exact ⟨by simp, Real.sqrt_nonneg _⟩

/-- **Equation (eq:int-phi-b), on `T'`.**  For every `C > 0`,
`∫_{T'} e^{-C β^{-3/2} n t² e^{-t²/2}} dt ≲ 1`: the integrand is at most
`e^{-Ct²}`, whose integral over `ℝ` is `√(π/C)`.

Source: arXiv:2412.09080v3, `eq:int-phi-b`, proof of `lem:main-int-phi`. -/
theorem integral_exp_phiRate_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {C : ℝ} (hC : 0 < C) :
    (fun k => ∫ t in intervalT' (N k) (B k), Real.exp (-C * phiRate (N k) (B k) t))
      =O[atTop] fun _ => (1 : ℝ) := by
  refine IsBigO.of_bound (Real.sqrt (π / C)) ?_
  have hN1 : ∀ᶠ k in atTop, (1 : ℝ) ≤ N k := hreg.tendsto_N.eventually_ge_atTop 1
  filter_upwards [hN1, hreg.tendsto_B.eventually_ge_atTop 1] with k hN hB
  have hN' : 1 ≤ N k := by exact_mod_cast hN
  have hmeas : MeasurableSet (intervalT' (N k) (B k)) := by
    unfold intervalT'; split_ifs
    · exact measurableSet_Icc
    · exact MeasurableSet.empty
  have hcont : Continuous fun t => Real.exp (-C * phiRate (N k) (B k) t) := by
    unfold phiRate; fun_prop
  have hint : IntegrableOn (fun t => Real.exp (-C * phiRate (N k) (B k) t))
      (intervalT' (N k) (B k)) := by
    unfold intervalT'; split_ifs
    · exact hcont.integrableOn_Icc
    · exact integrableOn_empty
  have hg := integrable_exp_neg_mul_sq hC
  rw [norm_one, mul_one, Real.norm_of_nonneg (setIntegral_nonneg hmeas fun t _ => (Real.exp_pos _).le)]
  calc ∫ t in intervalT' (N k) (B k), Real.exp (-C * phiRate (N k) (B k) t)
      ≤ ∫ t in intervalT' (N k) (B k), Real.exp (-C * t ^ 2) :=
        setIntegral_mono_on hint hg.integrableOn hmeas fun t ht =>
          Real.exp_le_exp.mpr (by nlinarith [sq_le_phiRate hN' hB ht])
    _ ≤ ∫ t, Real.exp (-C * t ^ 2) :=
        setIntegral_le_integral hg (Eventually.of_forall fun t => (Real.exp_pos _).le)
    _ = Real.sqrt (π / C) := integral_gaussian C

/-! ### `lem:main-int-phi` -/

/-- **Lemma (lem:main-int-phi), on `T`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_T ∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy dt ≍ √(β log β)`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:main-int-phi`. -/
theorem main_int_phi_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B)
    {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    (fun k => (proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))).toReal)
      =Θ[atTop] fun k => Real.sqrt (B k * Real.log (B k)) := by
  sorry

/-- **Lemma (lem:main-int-phi), on `T'`.**  In the regime `n^c ≲ β ≲ n^{2-c}`,
`∫_{T'} ∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy dt ≲ √β`.  Finiteness is stated,
since `toReal` reads `∞` as `0`.

Not proved here.

Source: arXiv:2412.09080v3, `lem:main-int-phi`. -/
theorem main_int_phi_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    (∀ᶠ k in atTop, proxyKR (N k) (B k) (intervalT' (N k) (B k)) ≠ ∞) ∧
    (fun k => (proxyKR (N k) (B k) (intervalT' (N k) (B k))).toReal)
      =O[atTop] fun k => Real.sqrt (B k) := by
  sorry

end Modes
end Transformer

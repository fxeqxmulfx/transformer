/-
# Cumulant bounds — `lem:correl`

ArXiv:2605.09213v1, §3, `lem:correl`: for independent initial data the
covariance of two tokens is of size `i⁻¹` and their third cumulant of size
`(ij)⁻¹`, both with the nonstandard factor `e^{√(CT log(i/j))}` that the
singular cumulative influence of small indices produces (`correlation_bounds`).

The random flow is the flow map of its independent initial angles
(`covariance_randomFlow`, `thirdCumulant_randomFlow`). Under their product law
the Glauber calculus bounds both cumulants through the envelopes of the first and
second differences of the flow (`abs_covariance_flow_le`,
`abs_thirdCumulant_flow_le`). Their amplitudes are powers of `e^{αT/γ}`
(`covariance_amp_le`, `thirdCumulant_amp_le`), and the choice
`γ = min(1/16, √(T/log(i/k)))` turns them into the source's factors
(`exists_gamma_exp_le`).
-/

import Transformer.Kinetic.Section3_RandomFlow
import Transformer.Kinetic.Section3_KappaBound
import Transformer.Kinetic.Section3_GammaChoice
import Transformer.Kinetic.Section3_Amplitudes
import Transformer.Homogenized.RademacherLaw

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Lemma (lem:correl).**  For independent initial data, `φ ∈ C²(𝕋)`,
`t,t',t'' ∈ [0,T]` and `i > j > k`,

  `|Cov(φ(θ_i(t)), φ(θ_j(t')))| ≲ i⁻¹ e^{√(CT log(i/j))} e^{CT} ‖φ'‖_∞²`,
  `|κ_{1,1,1}(φ(θ_i(t)), φ(θ_j(t')), φ(θ_k(t'')))|
      ≲ (ij)⁻¹ e^{√(CT log(i/k))} e^{CT} ‖φ'‖_∞² ‖φ'‖_{W^{1,∞}}`.

The source's `i, j, k` are `1`-based positions, so they are `i+1, j+1, k+1`
here.  The two norms of `φ'` are carried as bounds `K ≥ ‖φ'‖_∞` and
`K' ≥ ‖φ'‖_{W^{1,∞}} = max{‖φ'‖_∞, ‖φ''‖_∞}`, which is what `≲ ‖·‖` means and
avoids the junk value an unbounded supremum would take.

**What the source says and what is changed here.**

* *The test is periodic.*  The source takes `φ ∈ C²(𝕋)`.  The statement took
  every `φ : ℝ → ℝ` with bounded `φ'` and `φ''`, which reads the winding numbers
  of the angles; `correlation_bounds_aperiodic_false` refutes it.  `φ` is now
  `2π`-periodic.
* *The initial data are random variables.*  The source's initial data are
  random variables, hence measurable; this is now explicit.  Without it the
  integrals of the covariance take junk values.  Let the second token rest at
  `0` or `π` according to a set of outer measure one and inner measure zero.
  Take a test `A + ψ` whose values make the product of the two observables the
  constant `A²`.  Both observables are then non-measurable, so their means are
  `0`, and the covariance is `A²`, while `φ'` stays bounded independently of
  `A`.  This derivation is not checked in Lean.

The constant is `C = 3π²/2 + 9π³/2 + 96(27ω + 2)π⁴ + 272α + (17α + 6)²` with
`α = 2e^{2|λ|}L₁ + 1`, `ω = 2e^{2|λ|}L₂` and `L₁, L₂` Lipschitz constants of
`w_β'` and `w_β''`.

Source: arXiv:2605.09213v1, `lem:correl`, `eq:est-cov-th`, `eq:est-kap3-th`. -/
theorem correlation_bounds (lam β : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ (N : ℕ) (P : Measure Ω), IsProbabilityMeasure P →
      ∀ ϑ : Ω → ℝ → Idx N → ℝ,
        (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
        iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
        (∀ j : Idx N, Measurable fun ω => ϑ ω 0 j) →
      ∀ (φ dφ ddφ : ℝ → ℝ), Function.Periodic φ (2 * π) →
        (∀ x, HasDerivAt φ (dφ x) x) → (∀ x, HasDerivAt dφ (ddφ x) x) →
      ∀ K K' : ℝ, (∀ x, |dφ x| ≤ K) → (∀ x, |dφ x| ≤ K') → (∀ x, |ddφ x| ≤ K') →
      ∀ T : ℝ, 0 ≤ T → ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ t' ∈ Set.Icc (0 : ℝ) T,
        ∀ t'' ∈ Set.Icc (0 : ℝ) T, ∀ i j k : Idx N, k < j → j < i →
          |covariance (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j)) P|
              ≤ C * ((i : ℝ) + 1)⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((j : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 ∧
            |thirdCumulant P (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j))
                (fun ω => φ (ϑ ω t'' k))|
              ≤ C * (((i : ℝ) + 1) * ((j : ℝ) + 1))⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((k : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 * K' := by
  obtain ⟨L₁, L₂, hL₁, hL₂0, hd, hL, hL₂⟩ := exists_lipschitz_wBetaDeriv β
  obtain ⟨α, hαd⟩ : ∃ α : ℝ, α = 2 * Real.exp (2 * |lam|) * L₁ + 1 := ⟨_, rfl⟩
  obtain ⟨w, hwd⟩ : ∃ w : ℝ, w = 2 * Real.exp (2 * |lam|) * L₂ := ⟨_, rfl⟩
  have he1 : 1 ≤ Real.exp (2 * |lam|) := Real.one_le_exp (by positivity)
  have hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α := by rw [hαd]; linarith
  have hα1 : 1 ≤ α := by rw [hαd]; linarith [mul_nonneg (by positivity :
    (0 : ℝ) ≤ 2 * Real.exp (2 * |lam|)) hL₁]
  have hL₁α : L₁ ≤ α := by rw [hαd]; nlinarith
  have hα0 : 0 ≤ α := by linarith
  have hw : 0 ≤ w := by rw [hwd]; positivity
  obtain ⟨C, hCd⟩ : ∃ C : ℝ, C = 3 / 2 * π ^ 2 + (9 / 2 * π ^ 3 + 96 * (27 * w + 2) * π ^ 4) +
      272 * α + (17 * α + 6) ^ 2 := ⟨_, rfl⟩
  have hc₃ : 0 ≤ 9 / 2 * π ^ 3 + 96 * (27 * w + 2) * π ^ 4 := by positivity
  have hsq : (6 * α + 1) ^ 2 ≤ (17 * α + 6) ^ 2 :=
    pow_le_pow_left₀ (by linarith) (by linarith) 2
  have hC0 : 0 < C := by rw [hCd]; positivity
  have hC₁ : 3 / 2 * π ^ 2 ≤ C := by rw [hCd]; nlinarith [sq_nonneg (17 * α + 6)]
  have hC₂ : 9 / 2 * π ^ 3 + 96 * (27 * w + 2) * π ^ 4 ≤ C := by
    rw [hCd]; nlinarith [sq_nonneg (17 * α + 6), Real.pi_pos]
  have hC₃ : ((6 : ℕ) : ℝ) * α / (1 / 16) ≤ C := by
    rw [hCd]; push_cast; nlinarith [sq_nonneg (17 * α + 6), Real.pi_pos]
  have hC₄ : (((6 : ℕ) : ℝ) * α + ((1 : ℕ) : ℝ)) ^ 2 ≤ C := by
    rw [hCd]; push_cast; nlinarith [Real.pi_pos]
  have hC₅ : ((17 : ℕ) : ℝ) * α / (1 / 16) ≤ C := by
    rw [hCd]; push_cast; nlinarith [sq_nonneg (17 * α + 6), Real.pi_pos]
  have hC₆ : (((17 : ℕ) : ℝ) * α + ((6 : ℕ) : ℝ)) ^ 2 ≤ C := by
    rw [hCd]; push_cast; nlinarith [Real.pi_pos]
  refine ⟨C, hC0, ?_⟩
  intro N P hP ϑ hflow hind hmeas φ dφ ddφ hper hφ hdφ K K' hK hK' hK'' T hT t ht t' ht'
    t'' ht'' i j k hkj hji
  have := hP
  rw [covariance_randomFlow lam β hflow hind hmeas hφ ht.1 ht'.1 i j,
    thirdCumulant_randomFlow lam β hflow hind hmeas hφ ht.1 ht'.1 ht''.1 i j k]
  set μs : Idx N → Measure ℝ := fun l => P.map fun ω => ϑ ω 0 l
  have : ∀ l, IsProbabilityMeasure (μs l) := fun l =>
    (Measure.isProbabilityMeasure_map_iff (hmeas l).aemeasurable).2 hP
  have hK0 : 0 ≤ K := (abs_nonneg _).trans (hK 0)
  have hK'0 : 0 ≤ K' := (abs_nonneg _).trans (hK' 0)
  have hκ : ∀ x, |dφ x| ≤ min K K' := fun x => le_min (hK x) (hK' x)
  have hκ0 : 0 ≤ min K K' := le_min hK0 hK'0
  have hkj' : (k : ℕ) < j := hkj
  have hji' : (j : ℕ) < i := hji
  have hcij : ((j : ℕ) : ℝ) ≤ ((i : ℕ) : ℝ) := by exact_mod_cast hji'.le
  have hcki : ((k : ℕ) : ℝ) ≤ ((i : ℕ) : ℝ) := by exact_mod_cast (hkj'.trans hji').le
  have hℓij : 0 ≤ Real.log (((i : ℝ) + 1) / ((j : ℝ) + 1)) :=
    Real.log_nonneg (by rw [one_le_div (by positivity)]; linarith)
  have hℓik : 0 ≤ Real.log (((i : ℝ) + 1) / ((k : ℝ) + 1)) :=
    Real.log_nonneg (by rw [one_le_div (by positivity)]; linarith)
  have hv : (0 : ℝ) < ((i : ℝ) + 1) * ((j : ℝ) + 1) := by positivity
  rcases hT.eq_or_lt with hT0 | hT0
  · subst hT0
    have hτ : profileRate α (1 / 16) 0 = 0 := by simp [profileRate]
    refine ⟨(abs_covariance_flow_le μs lam β (α := α) hL hL₁ hα (γ := 1 / 16) (by norm_num)
      (by norm_num) hper hφ hK ht ht' hji).trans ?_,
      (abs_thirdCumulant_flow_le μs lam β (α := α) hd hL hL₁ hL₂ hL₂0 hα (γ := 1 / 16)
        (by norm_num) (by norm_num) hper hφ hdφ hκ hK' hK'' ht ht' ht'' hkj hji).trans ?_⟩
    · calc _ = (0 : ℝ) := by rw [hτ]; simp
        _ ≤ _ := by positivity
    · calc _ = (0 : ℝ) := by rw [hτ]; simp
        _ ≤ _ := by positivity
  constructor
  · obtain ⟨γ, hγ0, hγ, hE⟩ := exists_gamma_exp_le hα0 6 1 (γ₀ := 1 / 16) (by norm_num) hT0
      hℓij hC₃ hC₄
    refine (abs_covariance_flow_le μs lam β hL hL₁ hα hγ0 (by linarith) hper hφ hK ht ht'
      hji).trans ?_
    refine (covariance_amp_le hα1 hL₁α hγ0 (by linarith) hT (ratioPow_pos γ i j).le
      (by positivity : (0 : ℝ) < (i : ℝ) + 1)).trans ?_
    rw [ratioPow_eq_exp]
    calc _ ≤ C * ((i : ℝ) + 1)⁻¹ * K ^ 2 * (Real.exp (√(C * T *
          Real.log (((i : ℝ) + 1) / ((j : ℝ) + 1)))) * Real.exp (C * T)) := by gcongr
      _ = _ := by ring
  · obtain ⟨γ, hγ0, hγ, hE⟩ := exists_gamma_exp_le hα0 17 6 (γ₀ := 1 / 16) (by norm_num) hT0
      hℓik hC₅ hC₆
    refine (abs_thirdCumulant_flow_le μs lam β hd hL hL₁ hL₂ hL₂0 hα hγ0 (by linarith) hper
      hφ hdφ hκ hK' hK'' ht ht' ht'' hkj hji).trans ?_
    rw [← hwd]
    refine (thirdCumulant_amp_le hα1 hL₁α hγ0 (by linarith) hT hw hκ0 (min_le_left K K')
      (min_le_right K K') (one_le_ratioPow hγ0.le (hkj'.trans hji').le)
      (ratioPow_pos γ j k).le (ratioPow_le_ratioPow_left hγ0.le hji'.le k) hv).trans ?_
    rw [ratioPow_eq_exp]
    calc _ ≤ C * (((i : ℝ) + 1) * ((j : ℝ) + 1))⁻¹ * (K ^ 2 * K') * (Real.exp (√(C * T *
          Real.log (((i : ℝ) + 1) / ((k : ℝ) + 1)))) * Real.exp (C * T)) := by gcongr
      _ = _ := by ring

/-- The hypotheses of `correlation_bounds` are satisfiable: three tokens resting
at `0` on the one-point probability space, the periodic test `sin`, and the
indices `0 < 1 < 2`. -/
example (lam β : ℝ) : IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧
    IsGPTFlow lam β 3 (fun _ _ => 0) ∧
    iIndepFun (fun (_ : Idx 3) (_ : Unit) => (0 : ℝ)) (Measure.dirac ()) ∧
    (Measurable fun (_ : Unit) => (0 : ℝ)) ∧ Function.Periodic Real.sin (2 * π) ∧
    (∀ x, HasDerivAt Real.sin (Real.cos x) x) ∧ (∀ x, HasDerivAt Real.cos (-Real.sin x) x) ∧
    (∀ x, |Real.cos x| ≤ 1) ∧ (∀ x, |-Real.sin x| ≤ 1) ∧ (0 : Idx 3) < 1 ∧ (1 : Idx 3) < 2 :=
  ⟨inferInstance, isGPTFlow_constant_zero lam β 3, Homogenized.iIndepFun_of_unit _,
    measurable_const, Real.sin_periodic, Real.hasDerivAt_sin, Real.hasDerivAt_cos,
    Real.abs_cos_le_one, fun x => by rw [abs_neg]; exact Real.abs_sin_le_one x,
    by decide, by decide⟩

end Kinetic
end Transformer

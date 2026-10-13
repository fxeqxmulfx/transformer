/-
# Kinetic theory for Transformers — fluctuations

Formalization of `th:mf-gpt`(ii) and `lem:correl` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The printed quantitative mean-field estimate is refuted below by a smooth
position test. The Glauber-calculus correlation estimates remain open:
a two-point covariance of size `i⁻¹` and a
third cumulant of size `(ij)⁻¹`, both with the nonstandard factor
`e^{√(CT log(i/j))}` that the singular cumulative influence of small indices
produces.
-/

import Transformer.Kinetic.MeanField
import Transformer.Kinetic.Section1_GridQuadrature
import Transformer.Homogenized.RandomChain
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Moments.Covariance
import Mathlib.Analysis.Fourier.AddCircle

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The Japanese bracket `⟨n⟩ = √(1 + n²)`.

Source: arXiv:2605.09213v1, §1.3. -/
noncomputable def bracket (n : ℤ) : ℝ := Real.sqrt (1 + (n : ℝ) ^ 2)

/-- **Equation (eq:init-conv).**  The initial token laws converge to the
profile `f_∘` at rate `N^{-δ}` with polynomial loss in the frequency:

  `|E[e^{inθ⁰_{⌈Nσ⌉}}] - f̂_∘(σ,n)| ≤ C N^{-δ} ⟨n⟩^γ`  for all `σ ∈ (0,1]`, `n ∈ ℤ`.

**What the source says and what is changed here.**  The source pairs
`E[e^{+inθ⁰}]` with `f̂_∘(σ,n) = ∫_𝕋 e^{-inθ} f_∘(σ,dθ)`, whose limit is in fact
`f̂_∘(σ,-n)`.  Since the condition is imposed for every `n ∈ ℤ` and
`⟨n⟩ = ⟨-n⟩`, the two readings define the same assumption; the one written here
is `∫_𝕋 e^{inθ} f_∘(σ,dθ)`, which is `fourier n`.

The position `⌈Nσ⌉` of the source is `1`-based, so it is the token of index
`j : Idx N` with `j + 1 = ⌈Nσ⌉`.

Source: arXiv:2605.09213v1, `eq:init-conv`. -/
def InitConv (P : Measure Ω) (N : ℕ) (ϑ : Ω → Idx N → ℝ)
    (f₀ : ℝ → ProbabilityMeasure Torus) (δ γ Cst : ℝ) : Prop :=
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ n : ℤ, ∀ j : Idx N, (j : ℕ) + 1 = ⌈(N : ℝ) * σ⌉₊ →
    ‖(∫ ω, fourier (T := 2 * π) n ((ϑ ω j : ℝ) : Torus) ∂P) -
        ∫ x, fourier (T := 2 * π) n x ∂(f₀ σ : Measure Torus)‖
      ≤ Cst * (N : ℝ) ^ (-δ) * bracket n ^ γ

/-- The third joint cumulant `κ_{1,1,1}(Y,Y',Y'') = E[(Y-EY)(Y'-EY')(Y''-EY'')]`.

Source: arXiv:2605.09213v1, before `lem:correl`. -/
noncomputable def thirdCumulant (P : Measure Ω) (Y Y' Y'' : Ω → ℝ) : ℝ :=
  ∫ ω, (Y ω - ∫ ω', Y ω' ∂P) * (Y' ω - ∫ ω', Y' ω' ∂P) * (Y'' ω - ∫ ω', Y'' ω' ∂P) ∂P

/-- **Lemma (lem:correl).**  For independent initial data, `φ ∈ C²(𝕋)`,
`t,t',t'' ∈ [0,T]` and `i > j > k`,

  `|Cov(φ(θ_i(t)), φ(θ_j(t')))| ≲ i⁻¹ e^{√(CT log(i/j))} e^{CT} ‖φ'‖_∞²`,
  `|κ_{1,1,1}(φ(θ_i(t)), φ(θ_j(t')), φ(θ_k(t'')))|
      ≲ (ij)⁻¹ e^{√(CT log(i/k))} e^{CT} ‖φ'‖_∞² ‖φ'‖_{W^{1,∞}}`.

The source's `i, j, k` are `1`-based positions, so they are `i+1, j+1, k+1`
here.  The two norms of `φ'` are carried as bounds `K ≥ ‖φ'‖_∞` and
`K' ≥ ‖φ'‖_{W^{1,∞}} = max{‖φ'‖_∞, ‖φ''‖_∞}`, which is what `≲ ‖·‖` means and
avoids the junk value an unbounded supremum would take.

Not proved here.

Source: arXiv:2605.09213v1, `lem:correl`, `eq:est-cov-th`, `eq:est-kap3-th`. -/
theorem correlation_bounds (lam β : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ (N : ℕ) (P : Measure Ω), IsProbabilityMeasure P →
      ∀ ϑ : Ω → ℝ → Idx N → ℝ,
        (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
        iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
      ∀ (φ dφ ddφ : ℝ → ℝ), (∀ x, HasDerivAt φ (dφ x) x) → (∀ x, HasDerivAt dφ (ddφ x) x) →
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
  sorry

/-- The hypotheses of `correlation_bounds` are satisfiable: the system of a
single token, which never moves, on the one-point probability space.  The index
hypotheses `k < j < i` are then unsatisfiable in `Idx 1`, which is why they are
quantified inside the statement, where the source puts them, rather than in the
binders. -/
example : IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧
    IsGPTFlow (1 : ℝ) 1 1 (fun _ _ => 0) ∧
    iIndepFun (fun (_ : Idx 1) (_ : Unit) => (0 : ℝ)) (Measure.dirac ()) := by
  refine ⟨inferInstance, ?_, iIndepFun.of_subsingleton⟩
  intro t j
  have : gptField (1 : ℝ) 1 1 (fun _ => (0 : ℝ)) j = 0 := by
    simp [gptField, wBetaDeriv]
  rw [this]
  exact hasDerivAt_const t 0

/-- **The printed error estimate of `th:mf-gpt` (ii) is false.**
The actual constant Dirac profile solves the mean-field equation, but
no prefactors uniform over jointly smooth tests give the printed rate
using only their token Sobolev norm. The source's Step 4 itself retains
`N⁻¹‖∂σφ‖∞`, then drops position derivatives in `eq:conv-rate-muNf`.

At `t = 0`, deterministic zero tokens are independent and satisfy
`eq:init-conv` exactly with `δ = 1`, `γ = Cst = 0`. The smooth test
`φ_N(σ,θ) = cos(2πNσ)` has every token derivative bounded by one, yet
its actual empirical-versus-limit root mean-square error is one.
The proposed `K N⁻¹ᐟ²` vanishes. The counterexample retains the source's
smoothness assumption omitted in the previous statement, so it also
refutes that less restrictive formalization.

Source: arXiv:2605.09213v1, §1.1, `th:mf-gpt` (ii),
`eq:conv-rate-muNf`, and §4, Step 4 of its proof. -/
theorem mean_field_rate_false (lam β : ℝ) :
    IsMeanFieldSolution lam β (fun _ => ⟨Measure.dirac 0, inferInstance⟩)
      (fun _ _ => ⟨Measure.dirac 0, inferInstance⟩) ∧
    ¬ ∃ C : ℝ, 0 < C ∧ ∀ (δ γ Cst : ℝ), 0 < δ →
      ∀ (m : ℕ), (m : ℝ) > max γ 2 + 1 / 2 →
      ∃ K : ℝ, 0 < K ∧
        ∀ (P : Measure Unit), IsProbabilityMeasure P →
        ∀ (N : ℕ) (ϑ : Unit → ℝ → Idx N → ℝ),
          (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
          iIndepFun (fun (j : Idx N) (ω : Unit) => ϑ ω 0 j) P →
          InitConv P N (fun ω => ϑ ω 0) (fun _ => ⟨Measure.dirac 0, inferInstance⟩) δ γ Cst →
        ∀ φ : ℝ × ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
        ∀ hper : ∀ σ, Function.Periodic (fun u => φ (σ, u)) (2 * π),
          ∀ Kφ : ℝ, (∀ σ ∈ Set.Icc (0 : ℝ) 1, ∀ r ≤ m, ∀ u : ℝ,
              |iteratedDeriv r (fun u => φ (σ, u)) u| ≤ Kφ) →
        ∀ t ∈ Set.Ici (0 : ℝ),
          Real.sqrt (∫ ω, ((N : ℝ)⁻¹ * ∑ j : Idx N, φ (((j : ℝ) + 1) / N, ϑ ω t j)
              - ∫ σ in (0 : ℝ)..1, ∫ x, (hper σ).lift x ∂(Measure.dirac (0 : Torus))) ^ 2 ∂P)
            ≤ K * (N : ℝ) ^ (-(min δ (1 / 2))) * Real.exp (C * t) * Kφ := by
  refine ⟨isMeanFieldSolution_constant_zero lam β, ?_⟩
  rintro ⟨C, -, hC⟩
  obtain ⟨K, -, hK⟩ := hC 1 0 0 one_pos 3 (by norm_num)
  obtain ⟨N, hN, hbad⟩ := exists_cursorCos_error_gt_decay K (1 / 2) (by norm_num)
  let φ : ℝ × ℝ → ℝ := fun p => cursorCos N p.1
  let hper : ∀ σ, Function.Periodic (fun u => φ (σ, u)) (2 * π) := periodic_cursorCos_test N
  have hinit : InitConv (Measure.dirac ()) N (fun (_ : Unit) (_ : Idx N) => 0)
      (fun _ => ⟨Measure.dirac 0, inferInstance⟩) 1 0 0 := by simp [InitConv]
  have hb := hK (Measure.dirac ()) inferInstance N (fun _ _ _ => 0)
    (fun _ => isGPTFlow_constant_zero lam β N) (Homogenized.iIndepFun_of_unit _) hinit
    φ (contDiff_cursorCos_test N) hper 1
    (fun σ _ r _ u => abs_iteratedDeriv_cursorCos_test_le_one N r σ u) 0 (by simp)
  have hlim : (∫ σ in (0 : ℝ)..1, ∫ x, (hper σ).lift x
      ∂(Measure.dirac (0 : Torus))) = 0 := by
    calc
      _ = ∫ σ in (0 : ℝ)..1, cursorCos N σ := by
        apply intervalIntegral.integral_congr
        intro σ _
        change (∫ x, (hper σ).lift x ∂(Measure.dirac (0 : Torus))) = cursorCos N σ
        rw [integral_dirac]
        exact (hper σ).lift_coe (0 : ℝ)
      _ = 0 := integral_cursorCos N hN
  have havg : (N : ℝ)⁻¹ * ∑ j : Idx N, φ (((j : ℝ) + 1) / N, 0) = 1 :=
    rightRiemann_cursorCos N hN
  rw [hlim, havg] at hb
  simp only [integral_dirac, sub_zero, one_pow, Real.sqrt_one,
    min_eq_right (by norm_num : (1 / 2 : ℝ) ≤ 1), mul_zero, Real.exp_zero, mul_one] at hb
  rw [cursorCos_error_one N hN] at hbad
  exact not_lt_of_ge hb hbad

/-- The refutation's original flow, independence, weakly continuous profile,
exact initialization rate and Sobolev order hold jointly for every grid size.
Source: arXiv:2605.09213v1, §1.1, `th:mf-gpt` (ii) and `eq:init-conv`.
Joint smoothness and all token derivative bounds are witnessed in
`Section1_GridQuadrature`. -/
example (lam β : ℝ) : (0 : ℝ) < 1 ∧ ((3 : ℕ) : ℝ) > max 0 2 + 1 / 2 ∧
    (∀ ψ : Torus → ℝ, Continuous
      (fun _ : ℝ => ∫ x, ψ x ∂(Measure.dirac (0 : Torus)))) ∧
    IsMeanFieldSolution lam β (fun _ => ⟨Measure.dirac 0, inferInstance⟩)
      (fun _ _ => ⟨Measure.dirac 0, inferInstance⟩) ∧
    ∀ N : ℕ, IsGPTFlow lam β N (fun _ _ => 0) ∧
      iIndepFun (fun (_ : Idx N) (_ : Unit) => (0 : ℝ)) (Measure.dirac ()) ∧
      InitConv (Measure.dirac ()) N (fun (_ : Unit) (_ : Idx N) => 0)
        (fun _ => ⟨Measure.dirac 0, inferInstance⟩) 1 0 0 :=
  ⟨one_pos, by norm_num, fun _ => continuous_const, isMeanFieldSolution_constant_zero lam β,
    fun N => ⟨isGPTFlow_constant_zero lam β N,
    Homogenized.iIndepFun_of_unit _, by simp [InitConv]⟩⟩

end Kinetic
end Transformer

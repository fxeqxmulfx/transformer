module

public import Transformer.Homogenized.MvGenerator
public import Mathlib.MeasureTheory.Integral.DominatedConvergence

/-!
# Passing smooth logistic test martingales to bounded limits

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The source applies Itô's formula to arcsin and its square.
Those functions are not globally smooth at the absorbing endpoints,
whereas the original martingale problem tests globally smooth functions.
Their use therefore requires an approximation argument.

A uniformly bounded pointwise limit of actual finite-horizon
martingales remains a martingale. Adaptation and integrability of
the limit are proved, and conditional expectations pass to the limit
by dominated convergence. Every original pair of times is retained.

For compensated tests, bounds on their actual values and drift
densities give a common bound on the entire compensated process.
Continuous drift densities permit dominated convergence of the
original interval integrals. Thus the limiting martingale contains
the integral of the actual limiting drift density. No martingale
identity for the limit, or convergence of a compensator integral,
is assumed separately.

The finite-measure hypothesis makes the uniform bounds integrable;
the logistic application uses the source's probability measure.
The examples retain nonzero approximants and genuinely varying
test paths with their actual constant drift compensators.
-/

@[expose] public section

open MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- Bounded pointwise limits preserve the full conditional martingale identity.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, justifying the arcsin test.
Measurability and integrability of the limit are conclusions of the proof. -/
theorem isMartingaleOn_of_bounded_pointwise_limit
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T C : ℝ) (M : ℕ → ℝ → Ω → ℝ) (N : ℝ → Ω → ℝ)
    (hM : ∀ k, IsMartingaleOn T ℱ P (M k))
    (hbound : ∀ k t, t ∈ Icc (0 : ℝ) T → ∀ ω, ‖M k t ω‖ ≤ C)
    (hlim : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, Tendsto (fun k => M k t ω) atTop (𝓝 (N t ω))) :
    IsMartingaleOn T ℱ P N := by
  have hm (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (N t) :=
    stronglyMeasurable_of_tendsto atTop (fun k => (hM k).1 t ht)
      (tendsto_pi_nhds.mpr (hlim t ht))
  have hb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) : ‖N t ω‖ ≤ C :=
    le_of_tendsto (hlim t ht ω).norm (Eventually.of_forall (fun k => hbound k t ht ω))
  have hi (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : Integrable (N t) P :=
    Integrable.of_bound ((hm t ht).mono (ℱ.le t)).aestronglyMeasurable C (ae_of_all P (hb t ht))
  refine ⟨hm, hi, ?_⟩
  intro s hs t ht hst
  have hce : P[N t | ℱ s] =ᵐ[P] P[N s | ℱ s] :=
    tendsto_condExp_unique (fun k => M k t) (fun k => M k s) (N t) (N s)
      (fun k => (hM k).2.1 t ht) (fun k => (hM k).2.1 s hs)
      (ae_of_all P (hlim t ht)) (ae_of_all P (hlim s hs))
      (fun _ => C) (integrable_const C) (fun _ => C) (integrable_const C)
      (fun k => ae_of_all P (hbound k t ht)) (fun k => ae_of_all P (hbound k s hs)) (fun k => by
        rw [condExp_of_stronglyMeasurable (ℱ.le s) ((hM k).1 s hs) ((hM k).2.1 s hs)]
        exact (hM k).2.2 s hs t ht hst)
  rw [condExp_of_stronglyMeasurable (ℱ.le s) (hm s hs) (hi s hs)] at hce
  exact hce

/-- The actual bounded test and drift limits preserve their compensated martingale.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the arcsin and square tests.
The interval-integral convergence is proved from continuous bounded drift densities. -/
theorem isMartingaleOn_of_bounded_compensator_limits
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T A B : ℝ) (hT : 0 ≤ T) (hB : 0 ≤ B)
    (F b : ℕ → ℝ → Ω → ℝ) (f a : ℝ → Ω → ℝ)
    (hM : ∀ k, IsMartingaleOn T ℱ P
      (fun t ω => F k t ω - F k 0 ω - ∫ s in (0 : ℝ)..t, b k s ω))
    (hFbound : ∀ k t, t ∈ Icc (0 : ℝ) T → ∀ ω, ‖F k t ω‖ ≤ A)
    (hbbound : ∀ k t, t ∈ Icc (0 : ℝ) T → ∀ ω, ‖b k t ω‖ ≤ B)
    (hbc : ∀ k ω, ContinuousOn (fun s => b k s ω) (Icc (0 : ℝ) T))
    (hFlim : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, Tendsto (fun k => F k t ω) atTop (𝓝 (f t ω)))
    (hblim : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω, Tendsto (fun k => b k t ω) atTop (𝓝 (a t ω))) :
    IsMartingaleOn T ℱ P (fun t ω => f t ω - f 0 ω - ∫ s in (0 : ℝ)..t, a s ω) := by
  have hsub (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : uIoc (0 : ℝ) t ⊆ Icc (0 : ℝ) T := by
    rw [uIoc_of_le ht.1]
    exact fun s hs => ⟨hs.1.le, hs.2.trans ht.2⟩
  have hi (k : ℕ) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (ω : Ω) :
      ‖∫ s in (0 : ℝ)..t, b k s ω‖ ≤ B * T := by
    have h := intervalIntegral.norm_integral_le_of_norm_le_const
      (fun s hs => hbbound k s (hsub t ht hs) ω)
    rw [sub_zero, abs_of_nonneg ht.1] at h
    exact h.trans (mul_le_mul_of_nonneg_left ht.2 hB)
  apply isMartingaleOn_of_bounded_pointwise_limit P ℱ T (2 * A + B * T)
    (fun k t ω => F k t ω - F k 0 ω - ∫ s in (0 : ℝ)..t, b k s ω) _ hM
  · intro k t ht ω
    calc
      _ ≤ ‖F k t ω - F k 0 ω‖ + ‖∫ s in (0 : ℝ)..t, b k s ω‖ := norm_sub_le _ _
      _ ≤ (‖F k t ω‖ + ‖F k 0 ω‖) + B * T := add_le_add (norm_sub_le _ _) (hi k t ht ω)
      _ ≤ (A + A) + B * T := by
        linarith [hFbound k t ht ω, hFbound k 0 ⟨le_rfl, hT⟩ ω]
      _ = _ := by ring
  · intro t ht ω
    apply ((hFlim t ht ω).sub (hFlim 0 ⟨le_rfl, hT⟩ ω)).sub
    apply intervalIntegral.tendsto_integral_filter_of_dominated_convergence (fun _ => B)
    · apply Eventually.of_forall
      intro k
      exact ((hbc k ω).mono (hsub t ht)).aestronglyMeasurable measurableSet_uIoc
    · exact Eventually.of_forall (fun k => ae_of_all volume (fun s hs => hbbound k s (hsub t ht hs) ω))
    · exact intervalIntegrable_const
    · exact ae_of_all volume (fun s hs => hblim s (hsub t ht hs) ω)

/-- Both limit theorems have simultaneous concrete inputs: nonzero constants
cₖ=1/(k+1), and nonconstant paths 1+t cₖ with their actual drift cₖ.
The probability law, full martingale identities, uniform bounds, drift
continuity and pointwise limits are all exhibited on the closed horizon. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit)) (c : ℕ → ℝ),
    Tendsto c atTop (𝓝 0) ∧
    (∀ k, IsMartingaleOn 1 ℱ (Measure.dirac ()) (fun _ _ => c k)) ∧
    (∀ k t, t ∈ Icc (0 : ℝ) 1 → ∀ ω : Unit, ‖(fun _ : Unit => c k) ω‖ ≤ 1) ∧
    (∀ k, IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t _ => (1 + t * c k) - (1 + 0 * c k) - ∫ _ in (0 : ℝ)..t, c k)) ∧
    (∀ k t, t ∈ Icc (0 : ℝ) 1 → ∀ _ω : Unit, ‖1 + t * c k‖ ≤ 2) ∧
    (∀ k t, t ∈ Icc (0 : ℝ) 1 → ∀ _ω : Unit, ‖c k‖ ≤ 1) ∧
    (∀ (k : ℕ) (_ω : Unit), ContinuousOn (fun _ : ℝ => c k) (Icc (0 : ℝ) 1)) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ _ω : Unit, Tendsto (fun k => 1 + t * c k) atTop (𝓝 1)) ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let c : ℕ → ℝ := fun k => 1 / ((k : ℝ) + 1)
  have hc : Tendsto c atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  have h0 (k : ℕ) : 0 ≤ c k := by dsimp [c]; positivity
  have h1 (k : ℕ) : c k ≤ 1 := by
    dsimp [c]
    exact (div_le_one (show (0 : ℝ) < (k : ℝ) + 1 by positivity)).mpr
      (by linarith [show (0 : ℝ) ≤ (k : ℝ) from Nat.cast_nonneg k])
  refine ⟨ℱ, c, hc, fun k => isMartingaleOn_const 1 ℱ (Measure.dirac ()) _,
    fun k t ht ω => by rw [Real.norm_eq_abs, abs_of_nonneg (h0 k)]; exact h1 k, ?_, ?_,
    fun k t ht ω => by rw [Real.norm_eq_abs, abs_of_nonneg (h0 k)]; exact h1 k,
    fun k ω => continuousOn_const, ?_, zero_le_one, zero_le_one⟩
  · intro k
    have hz : (fun (t : ℝ) (_ : Unit) => (1 + t * c k) - (1 + 0 * c k) -
        ∫ _ in (0 : ℝ)..t, c k) = fun _ _ => (0 : ℝ) := by
      funext t ω
      simp only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, zero_mul, add_zero]
      ring
    rw [hz]
    exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0
  · intro k t ht ω
    rw [Real.norm_eq_abs, abs_of_nonneg (add_nonneg zero_le_one (mul_nonneg ht.1 (h0 k)))]
    have hmul := mul_le_mul ht.2 (h1 k) (h0 k) zero_le_one
    linarith
  · intro t ht ω
    simpa only [mul_zero, add_zero] using (hc.const_mul t).const_add 1

end Transformer.Homogenized

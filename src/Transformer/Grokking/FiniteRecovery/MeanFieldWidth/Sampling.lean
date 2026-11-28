module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling
public import Mathlib.Probability.Moments.Variance
public import Mathlib.MeasureTheory.Integral.Pi

/-! Sampling error of N independent particles. Under the product of N
copies of the initialization law, the empirical mean of a bounded
measurable feature misses its expectation by at most C/√N in mean
absolute value. The limiting particles are such a sample: their dynamics
is deterministic given the minibatches and continuous in the initial
weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory ProbabilityTheory Finset Transformer.AMSGradW.Kernite

/-- Mean absolute value is at most the root mean square. -/
theorem integral_abs_le_sqrt {Ω : Type*} [MeasurableSpace Ω] (ν : Measure Ω) [IsProbabilityMeasure ν]
    {Y : Ω → ℝ} (hY : MemLp Y 2 ν) : ∫ ω, |Y ω| ∂ν ≤ Real.sqrt (∫ ω, Y ω ^ 2 ∂ν) := by
  apply Real.le_sqrt_of_sq_le
  have h := variance_nonneg |Y| ν
  rw [variance_eq_sub hY.abs] at h
  have e : ∫ ω, (|Y| ^ 2) ω ∂ν = ∫ ω, Y ω ^ 2 ∂ν := by
    congr 1
    funext ω
    simp [sq_abs]
  have e' : ∫ ω, |Y| ω ∂ν = ∫ ω, |Y ω| ∂ν := rfl
  rw [e, e'] at h
  linarith

/-- The law of large numbers with rate: N independent draws of a feature
bounded by C have empirical mean within C/√N of the expectation, in mean. -/
theorem integral_abs_empirical_sub_le {α : Type*} [MeasurableSpace α] (μ : Measure α)
    [IsProbabilityMeasure μ] {N : ℕ} (hN : 0 < N) {f : α → ℝ} (hf : Measurable f) {C : ℝ}
    (hC : ∀ x, |f x| ≤ C) :
    ∫ z, |(∑ i, f (z i)) / N - ∫ x, f x ∂μ| ∂(Measure.pi fun _ : Fin N => μ) ≤ C / Real.sqrt N := by
  set π := Measure.pi fun _ : Fin N => μ
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure μ
  have hC0 : 0 ≤ C := (abs_nonneg _).trans (hC x0)
  have hNr : (0:ℝ) < N := Nat.cast_pos.2 hN
  have hfm : AEStronglyMeasurable f μ := hf.aestronglyMeasurable
  have hfL : MemLp f 2 μ :=
    MemLp.of_bound hfm C (ae_of_all _ fun x => by simpa [Real.norm_eq_abs] using hC x)
  set S : (Fin N → α) → ℝ := fun z => ∑ i, f (z i)
  have hSm : Measurable S := Finset.measurable_sum _ fun i _ => hf.comp (measurable_pi_apply i)
  have hSL : MemLp S 2 π := by
    refine MemLp.of_bound hSm.aestronglyMeasurable (N * C) (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs]
    refine (abs_sum_le_sum_abs _ _).trans ?_
    simpa using sum_le_sum fun i (_ : i ∈ univ) => hC (z i)
  have hES : ∫ z, S z ∂π = N * ∫ x, f x ∂μ := by
    rw [integral_finsetSum _ fun i _ => ?_]
    · have e : ∀ i : Fin N, ∫ z, f (z i) ∂π = ∫ x, f x ∂μ :=
        fun i => integral_comp_eval (μ := fun _ : Fin N => μ) (i := i) hfm
      simp only [e, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    · exact (hfL.comp_measurePreserving (measurePreserving_eval _ i)).integrable one_le_two
  have hvar : variance S π = N * variance f μ := by
    have h := variance_sum_pi (μ := fun _ : Fin N => μ) (X := fun _ => f) (fun _ => hfL)
    have e : (∑ i : Fin N, fun ω : Fin N → α => f (ω i)) = S := by
      funext z
      simp [S, Finset.sum_apply]
    rw [e] at h
    rw [h]
    simp
  have hvf : variance f μ ≤ C^2 := by
    refine (variance_le_expectation_sq hfm).trans ?_
    have hb : ∀ x, (f ^ 2) x ≤ C^2 := fun x => by
      simpa [sq_abs] using pow_le_pow_left₀ (abs_nonneg _) (hC x) 2
    calc ∫ x, (f ^ 2) x ∂μ ≤ ∫ _x, C^2 ∂μ :=
          integral_mono (hfL.integrable_sq) (integrable_const _) hb
      _ = C^2 := by simp
  have hY : ∀ z, S z / N - ∫ x, f x ∂μ = (S z - ∫ w, S w ∂π) / N := by
    intro z
    rw [hES]
    field_simp
  have hYL : MemLp (fun z => S z / N - ∫ x, f x ∂μ) 2 π := by
    have h := (hSL.const_mul (1/(N:ℝ))).sub (memLp_const (∫ x, f x ∂μ))
    convert h using 1
    funext z
    simp only [Pi.sub_apply]
    ring
  calc ∫ z, |S z / N - ∫ x, f x ∂μ| ∂π
      ≤ Real.sqrt (∫ z, (S z / N - ∫ x, f x ∂μ)^2 ∂π) := integral_abs_le_sqrt π hYL
    _ = Real.sqrt (variance S π / N^2) := by
        congr 1
        rw [variance_eq_integral hSm.aemeasurable]
        simp_rw [hY, div_pow]
        rw [integral_div]
    _ ≤ Real.sqrt (C^2 / N) := by
        apply Real.sqrt_le_sqrt
        rw [hvar, div_le_div_iff₀ (by positivity) hNr]
        nlinarith [mul_le_mul_of_nonneg_left hvf (by positivity : (0:ℝ) ≤ N * N)]
    _ = C / Real.sqrt N := by
        rw [Real.sqrt_div' _ hNr.le, Real.sqrt_sq hC0]

theorem integrable_abs_empirical_sub {α : Type*} [MeasurableSpace α] (μ : Measure α)
    [IsProbabilityMeasure μ] (N : ℕ) {f : α → ℝ} (hf : Measurable f) {C : ℝ}
    (hC : ∀ x, |f x| ≤ C) :
    Integrable (fun z : Fin N → α => |(∑ i, f (z i)) / N - ∫ x, f x ∂μ|)
      (Measure.pi fun _ : Fin N => μ) := by
  have hS : MemLp (fun z : Fin N → α => ∑ i, f (z i)) 2 (Measure.pi fun _ : Fin N => μ) := by
    refine MemLp.of_bound (Finset.measurable_sum _ fun i _ => hf.comp (measurable_pi_apply i)).aestronglyMeasurable
      (N * C) (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs]
    refine (abs_sum_le_sum_abs _ _).trans ?_
    simpa using sum_le_sum fun i (_ : i ∈ univ) => hC (z i)
  have h := ((hS.const_mul (1/(N:ℝ))).sub (memLp_const (∫ x, f x ∂μ))).integrable one_le_two
  refine h.abs.congr (ae_of_all _ fun z => ?_)
  simp only [Pi.sub_apply]
  ring_nf

variable {P K : Type*}

/-- A map that is Lipschitz between ℓ¹ sizes is continuous. -/
theorem continuous_of_l1 {Q R : Type*} [Fintype Q] [Fintype R] {f : (Q → ℝ) → R → ℝ} {L : ℝ}
    (hL : 0 ≤ L) (h : ∀ x y, l1 (f x - f y) ≤ L * l1 (x - y)) : Continuous f := by
  refine (LipschitzWith.of_dist_le_mul (K := ⟨L * Fintype.card Q, by positivity⟩) fun x y => ?_).continuous
  have hd : 0 ≤ L * Fintype.card Q * dist x y := by positivity
  refine (dist_pi_le_iff hd).2 fun r => ?_
  have hxy : l1 (x - y) ≤ Fintype.card Q * dist x y := by
    calc l1 (x - y) ≤ ∑ _q : Q, dist x y :=
          sum_le_sum fun q _ => by simpa [Real.dist_eq] using dist_le_pi_dist x y q
      _ = _ := by simp
  calc dist (f x r) (f y r) = |(f x - f y) r| := Real.dist_eq _ _
    _ ≤ l1 (f x - f y) := single_le_sum (f := fun r => |(f x - f y) r|) (fun _ _ => abs_nonneg _) (mem_univ r)
    _ ≤ L * (Fintype.card Q * dist x y) := (h x y).trans (mul_le_mul_of_nonneg_left hxy hL)
    _ = _ := by ring

theorem continuous_direction {eps : ℝ} (he : 0 < eps) (t : ℕ) :
    Continuous fun q : ℝ × ℝ => direction eps t q.1 q.2 := by
  have hb := firstBias_lower t
  have hc : 0 ≤ Real.sqrt (1 - (95/100 : ℝ)^(t+1)) := Real.sqrt_nonneg _
  refine continuous_fst.div ((continuous_const.mul ((Real.continuous_sqrt.comp continuous_snd).div_const _
    |>.add continuous_const))) fun q => ?_
  have : 0 < Real.sqrt q.2 / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) + eps :=
    add_pos_of_nonneg_of_pos (div_nonneg (Real.sqrt_nonneg _) hc) he
  exact (mul_pos (by linarith) this).ne'

/-- The limiting particle is continuous in its initial weights. -/
theorem continuous_flow (A : Training P K) (μ : Measure (P → ℝ)) (he : 0 < A.eps)
    (hG : ∀ t Φ, Continuous fun θ => A.grad t θ Φ) (t : ℕ) :
    Continuous (fun θ => (flow A μ t θ).weights) ∧ Continuous (fun θ => (flow A μ t θ).first) ∧
      Continuous (fun θ => (flow A μ t θ).second) := by
  induction t with
  | zero => exact ⟨continuous_id, continuous_const, continuous_const⟩
  | succ t ih =>
    obtain ⟨hw, hm, hv⟩ := ih
    have hg : Continuous fun θ => A.grad t (flow A μ t θ).weights (limitField A μ t) := (hG t _).comp hw
    have hm' : Continuous fun θ => (flow A μ (t+1) θ).first := continuous_pi fun p =>
      (continuous_const.mul ((continuous_apply p).comp hm)).add
        (continuous_const.mul ((continuous_apply p).comp hg))
    have hv' : Continuous fun θ => (flow A μ (t+1) θ).second := continuous_pi fun p =>
      (continuous_const.mul ((continuous_apply p).comp hv)).add
        (continuous_const.mul (((continuous_apply p).comp hg).pow 2))
    refine ⟨continuous_pi fun p => ?_, hm', hv'⟩
    exact (continuous_const.mul ((continuous_apply p).comp hw)).sub (continuous_const.mul
      ((continuous_direction he t).comp (((continuous_apply p).comp hm').prodMk ((continuous_apply p).comp hv'))))

/-- The expected sampling error of the limiting field after t steps. -/
theorem integral_fluctuation_le [Fintype P] [Fintype K] (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    {N : ℕ} (hN : 0 < N) (he : 0 < A.eps) (hG : ∀ t Φ, Continuous fun θ => A.grad t θ Φ)
    (hφ : Continuous A.field) {C : ℝ} (hC : ∀ θ k, |A.field θ k| ≤ C) (t : ℕ) :
    Integrable (fun z => fluctuation A μ z t) (Measure.pi fun _ : Fin N => μ) ∧
      ∫ z, fluctuation A μ z t ∂(Measure.pi fun _ : Fin N => μ) ≤ Fintype.card K * C / Real.sqrt N := by
  have hf k : Measurable fun θ => A.field (flow A μ t θ).weights k :=
    ((continuous_apply k).comp (hφ.comp (continuous_flow A μ he hG t).1)).measurable
  have hi k := integrable_abs_empirical_sub μ N (hf k) (fun θ => hC _ k)
  refine ⟨integrable_finsetSum _ fun k _ => hi k, ?_⟩
  change ∫ z, ∑ k, |(∑ i, A.field (flow A μ t (z i)).weights k) / N -
    ∫ x, A.field (flow A μ t x).weights k ∂μ| ∂(Measure.pi fun _ : Fin N => μ) ≤ _
  rw [integral_finsetSum _ fun k _ => hi k]
  calc _ ≤ ∑ _k : K, C / Real.sqrt N :=
        sum_le_sum fun k _ => integral_abs_empirical_sub_le μ hN (hf k) fun θ => hC _ k
    _ = _ := by simp [mul_div_assoc]

example := integral_abs_le_sqrt (Measure.dirac (0:ℝ)) (memLp_const (1:ℝ))
example := integral_abs_empirical_sub_le (Measure.dirac (0:ℝ)) (N := 3) (by norm_num)
  (measurable_const (a := (0:ℝ))) (C := 0) fun _ => by simp
example := integrable_abs_empirical_sub (Measure.dirac (0:ℝ)) 3 (measurable_const (a := (0:ℝ)))
  (C := 0) fun _ => by simp
example := continuous_of_l1 (f := fun x : Fin 1 → ℝ => x) zero_le_one fun _ _ => (one_mul _).ge
example := continuous_direction one_pos 0
example := continuous_flow identityTraining (Measure.dirac 0) one_pos (fun _ _ => continuous_id) 4
example := integral_fluctuation_le identityTraining (Measure.dirac 0) (N := 3) (by norm_num) one_pos
  (fun _ _ => continuous_id) continuous_const (C := 0) (fun _ _ => by simp [identityTraining]) 2

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth

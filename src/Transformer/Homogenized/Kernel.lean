module

public import Transformer.Homogenized.Section3_SparseKernelMoments
public import Transformer.Homogenized.Generator

/-!
# The quantitative kernel regularity claim has a sparse-law counterexample

Lemma `lem:Kernel_regularity` of arXiv:2604.01978v1, §3.1, asserts both
regularity and quantitative moment bounds under `ass:high_order_short`.
In particular, it gives `C₁ = O(1 + σ_A ‖𝔼V‖_op β / σ_V)` in the estimate
`max_X ‖G(X,·)‖_{L³} ≤ C₁ ς`.

The uniform quantitative reading fails with fixed
`d = n = 2`, `β = C = σ_V = 1`, `σ_A = 0`, and zero mean matrices.
Let `V₀₁` take values `±1`, each with probability `q⁶/2`, and zero otherwise.
Every stated subGaussian and independence hypothesis holds, but the exact
variance proxy is `ς = q³`, and the third moment norm at two coincident unit
tokens is `q²`. Their ratio is unbounded as `q ↓ 0`.

The proof in §6.3 uses isotropy and identifies actual variances with the
variance proxies; neither property is part of `ass:high_order_short`.
The counterexample addresses the uniform quantitative estimate. It does not
assert a failure of qualitative smoothness for a fixed law.

The final theorem negates the formerly admitted conjunction, with its
derivative bounds restricted to unit configurations, as in the source.
The contradiction uses only item (2), so it is independent of the choice of
ambient or covariant higher derivatives and of behavior away from the sphere.
Both token count and temperature satisfy the source's positive-parameter setup.
The proxy is the per-token maximum of `eq: defining.alpha`; using the product
Euclidean norm multiplies both moments at coincident tokens by the same factor.

Source: arXiv:2604.01978v1, §3.1, `lem:Kernel_regularity`,
`eq:G_def`, `eq: defining.alpha`; proof §6.3.
-/

@[expose] public section

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- Every proposed uniform constant fails for a genuine high-order law at
two unit tokens. All model scales and both matrix means are fixed.
Source: arXiv:2604.01978v1, §3.1, counterexample to the quantitative
third moment estimate in `lem:Kernel_regularity`(2). -/
theorem kernel_regularity_counterexample (K : ℝ) (hK : 0 < K) :
    ∃ q : ℝ, q ∈ Set.Ioo 0 1 ∧
      IsHighOrderLaw 2 1 1 0 (sparseHeadLaw (q ^ 6)) (sparseSource (q ^ 6))
        sparseMatrix 0 0 0 0 ∧
      IsVarianceProxy 2 2 1 (sparseHeadLaw (q ^ 6)) (q ^ 3) ∧
      K * q ^ 3 <
        (∫ θ, ‖Gfield 1 (sparseHeadLaw (q ^ 6))
          (fun _ : Idx 2 => EuclideanSpace.single 1 1) θ 0‖ ^ 3
            ∂sparseHeadLaw (q ^ 6)) ^ ((1 : ℝ) / 3) := by
  let q : ℝ := 1 / (K + 1)
  have hden : 0 < K + 1 := by linarith
  have hq₀ : 0 < q := by dsimp [q]; positivity
  have hq₁ : q < 1 := (div_lt_one hden).mpr (by linarith)
  have hq : q ∈ Set.Icc 0 1 := ⟨hq₀.le, hq₁.le⟩
  have hp : q ^ 6 ∈ Set.Icc 0 1 := ⟨by positivity, pow_le_one₀ hq.1 hq.2⟩
  have hKq : K * q < 1 := by
    dsimp [q]
    rw [mul_one_div]
    exact (div_lt_one hden).mpr (by linarith)
  refine ⟨q, ⟨hq₀, hq₁⟩, isHighOrderLaw_sparse hp, isVarianceProxy_sparse hq 1, ?_⟩
  rw [sparse_third_moment hq]
  calc
    K * q ^ 3 = (K * q) * q ^ 2 := by ring
    _ < 1 * q ^ 2 := mul_lt_mul_of_pos_right hKq (by positivity)
    _ = q ^ 2 := one_mul _

example : (0 : ℝ) < 1 := one_pos

/-- The counterexample uses two unit tokens, a genuine high-order law, and
the claimed operator bound for the zero mean value matrix. -/
example :
    IsHighOrderLaw 2 1 1 0 (sparseHeadLaw (1 / 64)) (sparseSource (1 / 64))
        sparseMatrix 0 0 0 0 ∧
      (∀ y : EucSpace 2,
        ‖Matrix.toEuclideanLin (0 : Matrix (Fin 2) (Fin 2) ℝ) y‖ ≤ 0 * ‖y‖) ∧
      ∀ _i : Idx 2, ‖(EuclideanSpace.single (1 : Fin 2) (1 : ℝ))‖ = 1 := by
  refine ⟨isHighOrderLaw_sparse (by norm_num), ?_, ?_⟩
  · intro y; simp
  · intro i; simp

/-- There is no uniform constant in the claimed third moment estimate.
This is a necessary consequence of `lem:Kernel_regularity`, already false
with `C = σ_V = β = 1`, `σ_A = ‖𝔼V‖_op = 0`, and `d = n = 2`.
Source: arXiv:2604.01978v1, §3.1, `lem:Kernel_regularity`(2). -/
theorem not_kernel_regularity_moment_bound :
    ¬ ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ) (β : ℝ) (C σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
        (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
        (mV mA : Matrix (Fin d) (Fin d) ℝ) (opV : ℝ),
      IsHighOrderLaw d C σV σA ρ P Vr Wr Wr' mV mA →
      (∀ y : EucSpace d, ‖Matrix.toEuclideanLin mV y‖ ≤ opV * ‖y‖) →
      ∃ s : ℝ, IsVarianceProxy d n β ρ s ∧
        ∀ x : Idx n → EucSpace d, (∀ i, ‖x i‖ = 1) → ∀ i : Idx n,
          (∫ θ, ‖Gfield β ρ x θ i‖ ^ 3 ∂ρ) ^ ((1 : ℝ) / 3) ≤
            K * (1 + (σA : ℝ) * opV * β / (σV : ℝ)) * s := by
  rintro ⟨K, hK, h⟩
  obtain ⟨q, _, hlaw, hproxy, hfail⟩ := kernel_regularity_counterexample K hK
  obtain ⟨s, hs, hbound⟩ := h 2 2 1 1 1 0 (sparseHeadLaw (q ^ 6)) (Fin 3)
    (sparseSource (q ^ 6)) sparseMatrix 0 0 0 0 0 hlaw (fun y => by simp)
  have hse : s = q ^ 3 := hs.unique hproxy
  have hb := hbound (fun _ : Idx 2 => EuclideanSpace.single 1 1)
    (fun _ => by simp) 0
  have hle :
      (∫ θ, ‖Gfield 1 (sparseHeadLaw (q ^ 6))
        (fun _ : Idx 2 => EuclideanSpace.single 1 1) θ 0‖ ^ 3
          ∂sparseHeadLaw (q ^ 6)) ^ ((1 : ℝ) / 3) ≤ K * q ^ 3 := by
    simpa [hse] using hb
  exact (not_lt_of_ge hle) hfail

/-- **Counterexample to the quantitative conjunction in
`lem:Kernel_regularity`.** The third moment bound alone fails under the
stated assumption. The derivative bounds below range over unit configurations,
matching the source's domain rather than imposing global ambient bounds.
Source: arXiv:2604.01978v1, §3.1, `lem:Kernel_regularity`. -/
theorem not_kernel_regularity :
    ¬ ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ) (β : ℝ) (C σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
        (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
        (mV mA : Matrix (Fin d) (Fin d) ℝ) (opV : ℝ),
      IsHighOrderLaw d C σV σA ρ P Vr Wr Wr' mV mA →
      (∀ y : EucSpace d, ‖Matrix.toEuclideanLin mV y‖ ≤ opV * ‖y‖) →
      ContDiff ℝ 4 (fun x : Idx n → EucSpace d => bField β ρ x) ∧
      ∃ s : ℝ, IsVarianceProxy d n β ρ s ∧
        (∀ x : Idx n → EucSpace d, (∀ i, ‖x i‖ = 1) → ∀ i : Idx n,
          (∫ θ, ‖Gfield β ρ x θ i‖ ^ 3 ∂ρ) ^ ((1 : ℝ) / 3) ≤
            K * (1 + (σA : ℝ) * opV * β / (σV : ℝ)) * s) ∧
        (∀ᵐ θ ∂ρ, ContDiff ℝ 5 (fun x : Idx n → EucSpace d => Gfield β ρ x θ)) ∧
        ∃ N₁ N₂ : HeadParam d → ℝ,
          (∀ θ : HeadParam d, ∀ k ≤ 5, ∀ x : Idx n → EucSpace d,
            (∀ i, ‖x i‖ = 1) →
            ‖iteratedFDeriv ℝ k (fun y : Idx n → EucSpace d => Gfield β ρ y θ) x‖ ≤ N₁ θ) ∧
          (∀ θ : HeadParam d, ∀ k ≤ 5, ∀ x : Idx n → EucSpace d,
            (∀ i, ‖x i‖ = 1) →
            ‖iteratedFDeriv ℝ k
              (covDeriv (fun y : Idx n → EucSpace d => Gfield β ρ y θ)) x‖ ≤ N₂ θ) ∧
          Real.sqrt (∫ θ, N₁ θ ^ 2 ∂ρ) + Real.sqrt (∫ θ, N₂ θ ^ 2 ∂ρ) ≤
            K * (1 + β ^ 10 * (σA : ℝ) ^ 10 * (d : ℝ) ^ 5) * s := by
  rintro ⟨K, hK, h⟩
  apply not_kernel_regularity_moment_bound
  refine ⟨K, hK, ?_⟩
  intro d n β C σV σA ρ Ω _ P Vr Wr Wr' mV mA opV hlaw hop
  obtain ⟨_, s, hs, hbound, _⟩ :=
    h d n β C σV σA ρ Ω P Vr Wr Wr' mV mA opV hlaw hop
  exact ⟨s, hs, hbound⟩

end Transformer.Homogenized

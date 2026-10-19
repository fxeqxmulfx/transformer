module

public import Transformer.Interpolation.Section3_SeparationPhases

/-!
# Original weak support separation with a uniform control bound

ArXiv:2411.04551v3, §3.2, `prop: separation`.
Realize the actual finite separating phase sequence at any prescribed
positive horizon. Every original weak endpoint is identified, so the
geodesic support hulls are separated for every pair of original weak
solutions. The score matrix vanishes at every real time.

Small ridge phases refine the initial mean-direction classes. A common
transverse axis then isolates each selected law; reversal restores all
other laws. Disjoint caps retain the mean-direction separation during
the finite contraction induction.

In dimension at least two the number of constant pieces is at most
4N+1. The empty zero-dimensional sphere has no probability law. In
dimension one every quadrant law is the same positive atom, and input
injectivity rules out distinct indices. These cases give an absolute
constant five in the original O(d·N) bound.

As in the corrected source proposition, distinct initial laws and N≥1
are explicit hypotheses. Equal inputs refute the printed statement;
N=0 is incompatible with a positive-time partition having zero pieces.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

/-- Realization of the separating phases gives at most 4N+1 constant
pieces, zero scores and separation for every original weak solution.
Source: arXiv:2411.04551v3, §3.2, `prop: separation`, with distinct inputs. -/
theorem exists_separation_controls_ge_two {d N : ℕ} (hd : 2 ≤ d)
    (T : ℝ) (hT : 0 < T) (μ₀ : Idx N → Perspective.ProbSphere d)
    (hQ : ∀ i, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ₀) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 4 * N + 1 ∧
      (∀ s : ℝ, (θ s).B = 0) ∧ PiecewiseConstant d θ T K ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ i j : Idx N, i ≠ j → ∀ μ ν : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ i → cauchyPB d θ μ → ν 0 = μ₀ j → cauchyPB d θ ν →
        Disjoint (convG d (μ T : Measure (SSphere d)).support)
          (convG d (ν T : Measure (SSphere d)).support) := by
  obtain ⟨L, hlen, hL, hsep⟩ := exists_separation_phases hd μ₀ hQ hinj
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization ⟨0, by omega⟩ L hL T hT
  refine ⟨θ, L.length + 1, by omega, hB, hpc, hex, ?_⟩
  intro i j hij μ ν hμ0 hμ hν0 hν
  rw [hend μ hμ, hend ν hν, hμ0, hν0]
  exact hsep i j hij

/-- The corrected original support-separation proposition has uniform
constant five. The source omits distinctness and N≥1, both required by
its induction and control-piece bound. All remaining original data and
weak-solution conclusions are retained. Source: arXiv:2411.04551v3,
§3.2, `prop: separation` and its proof. -/
theorem exists_separation_controls :
    ∀ (d N : ℕ) (T : ℝ) (μ₀ : Idx N → Perspective.ProbSphere d), 1 ≤ N → 0 < T →
      (∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) →
      Function.Injective μ₀ →
      ∃ (θ : TimeParams d) (K : ℕ),
        K ≤ 5 * (d * N) ∧ (∀ s : ℝ, (θ s).B = 0) ∧ PiecewiseConstant d θ T K ∧
        (∀ i : Idx N, ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
        ∀ i j : Idx N, i ≠ j → ∀ μ ν : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ → ν 0 = μ₀ j → cauchyPB d θ ν →
          Disjoint (convG d (μ T : Measure (SSphere d)).support)
            (convG d (ν T : Measure (SSphere d)).support) := by
  intro d N T μ₀ hN hT hQ hinj
  by_cases hd : 2 ≤ d
  · obtain ⟨θ, K, hK, hB, hpc, hex, hsep⟩ :=
      exists_separation_controls_ge_two hd T hT μ₀ hQ hinj
    have hNN : N ≤ d * N := by
      simpa only [one_mul] using Nat.mul_le_mul_right N (by omega : 1 ≤ d)
    have hbound : K ≤ 5 * (d * N) := calc
      K ≤ 4 * N + 1 := hK
      _ ≤ 5 * N := by omega
      _ ≤ 5 * (d * N) := Nat.mul_le_mul_left 5 hNN
    exact ⟨θ, K, hbound, hB, hpc, (fun i => hex (μ₀ i)), hsep⟩
  · have hd' : d ≤ 1 := by omega
    interval_cases d
    · let i : Idx N := ⟨0, by omega⟩
      obtain ⟨x, -⟩ := nonempty_of_measure_ne_zero (μ := (μ₀ i : Measure (SSphere 0)))
        (s := Set.univ) (by rw [measure_univ]; exact one_ne_zero)
      have hx0 : (x : EucSpace 0) = 0 := by
        ext k
        exact Fin.elim0 k
      have hn := mem_sphere_zero_iff_norm.mp x.2
      rw [hx0, norm_zero] at hn
      exact (zero_ne_one hn).elim
    · obtain ⟨θ, hpc, hB, hex, -⟩ := exists_phase_sequence_realization (0 : Fin 1) []
        (by simp) T hT
      refine ⟨θ, 1, by omega, hB, hpc, (fun i => hex (μ₀ i)), ?_⟩
      intro i j hij
      have he : μ₀ i = μ₀ j :=
        (eq_dirac_of_positiveQuadrant_one (μ₀ i) (hQ i)).trans
          (eq_dirac_of_positiveQuadrant_one (μ₀ j) (hQ j)).symm
      exact (hij (hinj he)).elim

/-- Distinct equal-mean atomic laws in dimension three and time one
satisfy every hypothesis of both control theorems, including N≥1.
The actual construction gives at most nine pieces for these two laws. -/
example : ∃ μ₀ : Idx 2 → Perspective.ProbSphere 3,
    (2 : ℕ) ≤ 3 ∧ (1 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i, (μ₀ i : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
    Function.Injective μ₀ ∧ barycenter 3 (μ₀ 0) = barycenter 3 (μ₀ 1) ∧
    ∃ (θ : TimeParams 3) (K : ℕ), K ≤ 9 ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      PiecewiseConstant 3 θ 1 K ∧
      (∀ i : Idx 2, ∃ μ : ℝ → Perspective.ProbSphere 3, μ 0 = μ₀ i ∧ cauchyPB 3 θ μ) ∧
      ∀ i j : Idx 2, i ≠ j → ∀ μ ν : ℝ → Perspective.ProbSphere 3,
        μ 0 = μ₀ i → cauchyPB 3 θ μ → ν 0 = μ₀ j → cauchyPB 3 θ ν →
        Disjoint (convG 3 (μ 1 : Measure (SSphere 3)).support)
          (convG 3 (ν 1 : Measure (SSphere 3)).support) := by
  obtain ⟨ν, η, hne, hν, hη, hmean, -⟩ := perturbation_equal_barycenter_counterexample
  let μ₀ : Idx 2 → Perspective.ProbSphere 3 := ![ν, η]
  have hQ (i : Idx 2) : (μ₀ i : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 := by
    fin_cases i
    · exact hν
    · exact hη
  have hinj : Function.Injective μ₀ := by
    intro i j he
    fin_cases i <;> fin_cases j
    · rfl
    · exact (hne he).elim
    · exact (hne he.symm).elim
    · rfl
  obtain ⟨θ, K, hK, hB, hpc, hex, hsep⟩ := exists_separation_controls_ge_two
    (by norm_num : 2 ≤ (3 : ℕ)) 1 zero_lt_one μ₀ hQ hinj
  refine ⟨μ₀, by norm_num, by norm_num, zero_lt_one, hQ, hinj, hmean,
    θ, K, ?_, hB, hpc, ?_, ?_⟩
  · exact hK
  · intro i
    exact hex (μ₀ i)
  · intro i j hij μ ν hμ0 hμ hν0 hν
    exact hsep i j hij μ ν hμ0 hμ hν0 hν

/-- The positive atom in ambient dimension one supplies a genuine
input for the low-dimensional branch of the uniform proposition. -/
example : ∃ μ₀ : Idx 1 → Perspective.ProbSphere 1,
    (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (∀ i, (μ₀ i : Measure (SSphere 1)).support ⊆ positiveQuadrant 1) ∧
    Function.Injective μ₀ := by
  let μ₀ : Idx 1 → Perspective.ProbSphere 1 := fun _ => Perspective.diracProb 1 (basePoint 0)
  have hQ (i : Idx 1) : (μ₀ i : Measure (SSphere 1)).support ⊆ positiveQuadrant 1 := by
    intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact basePoint_mem_positiveQuadrant
  exact ⟨μ₀, le_rfl, zero_lt_one, hQ, fun i j _ => Subsingleton.elim i j⟩

end Transformer.Interpolation

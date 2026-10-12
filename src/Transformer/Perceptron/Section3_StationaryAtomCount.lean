/-
# The full atom-count corollary for circular-arc covers

ArXiv:2601.21366v2, §3.2, `cor: bound`, and its proof in §5.4.
The cluster-mass theorem now supplies the mass estimate for every
cluster of the actual stationary atomic configuration. No mass
estimate is an unproved input to this corollary.

The source assumes SOPD and `L<2π`. The proof here applies already
to stationary measures and to any finite circular-arc cover, which
strengthens the statement. Each arc is represented by endpoints
and an integer-turn membership condition, allowing it to cross zero.
The source's exact factor `M/ε (1+2L sqrt β)` and printed `0.5742`
are retained, with the same coefficient and threshold as the mass theorem.

Both the pointwise corollary and its uniform quantifier order are
proved. The actual quadratic two-atom stationary configuration at
`β=4` has a complete joint witness using one arc across angle zero.
-/

import Transformer.Perceptron.Section3_AtomCount

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- The actual mass-bound coefficient is positive before temperature,
activation, configuration, cover and atom threshold are chosen.
Source: arXiv:2601.21366v2, §3.2, the uniform error in `cor: bound`;
the explicit value is established by `bound_cluster_mass`. -/
theorem cluster_count_coefficient_pos (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) :
    0 < 2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1 := by positivity

/-- **The full pointwise atom-count corollary.** Source:
arXiv:2601.21366v2, §3.2, `cor: bound`, and §5.4, its proof.
Stationarity supplies the proved cluster-mass estimate, including
every cluster of size at least two. Circular cover membership
allows independent integer turns for each actual atom. -/
theorem stationary_atom_count_bound (β : ℝ) (hβ : 4 ≤ β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : LipschitzWith 1 σ) (hzero : σ 0 = 0)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (N : ℕ) (m θ : Idx N → ℝ)
    (μ : Perspective.ProbSphere 2) (hatom : IsAtomicOnCircle N m θ μ)
    (hμ : IsStationary β σ ω a μ) (M : ℕ) (hM : 1 ≤ M)
    (l r : Idx M → ℝ) (L : ℝ) (hlength : ∀ j, r j - l j ≤ L)
    (hcover : ∀ i : Idx N, ∃ j : Idx M, ∃ k : ℤ,
      θ i + 2 * π * (k : ℝ) ∈ Set.Icc (l j) (r j)) (ε : ℝ) (hε : 0 < ε) :
    (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) ≤
      (M / ε) * (1 + 2 * L * Real.sqrt β) *
        (0.5742 + (2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1) * Real.exp (-β)) := by
  apply bound_atom_count (2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1) β
    (cluster_count_coefficient_pos ω a).le (by linarith) N m θ μ hatom
    (fun S hcard hpair => cluster_mass_bound_of_stationary hatom β hβ φ σ hφ hσ hzero
      ω a hμ S hcard hpair) M hM l r L hlength hcover ε hε

/-- **The source's atom-count corollary with uniform constants.**
Source: arXiv:2601.21366v2, §3.2, `cor: bound`, and §5.4, its proof.
Constants are selected after the fixed perceptron and before the
temperature, activation, atomic configuration, arc cover, and `ε`.
The same `β₀=4` and `C=2Σ|ω_j|‖a_j‖²+1` work throughout. -/
theorem bound_stationary_atom_count :
    ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2),
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2),
        IsAtomicOnCircle N m θ μ → IsStationary β σ ω a μ →
      ∀ (M : ℕ), 1 ≤ M → ∀ (l r : Idx M → ℝ) (L : ℝ),
        (∀ j, r j - l j ≤ L) →
        (∀ i : Idx N, ∃ j : Idx M, ∃ k : ℤ,
          θ i + 2 * π * (k : ℝ) ∈ Set.Icc (l j) (r j)) →
      ∀ ε : ℝ, 0 < ε →
        (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) ≤
          (M / ε) * (1 + 2 * L * Real.sqrt β) * (0.5742 + C * Real.exp (-β)) := by
  intro ω a
  refine ⟨2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1, 4,
    cluster_count_coefficient_pos ω a, by norm_num, ?_⟩
  intro β hβ φ σ hφ hσ hzero N m θ μ hatom hμ M hM l r L hlength hcover ε hε
  exact stationary_atom_count_bound β hβ φ σ hφ hσ hzero ω a N m θ μ hatom hμ
    M hM l r L hlength hcover ε hε

/-- **One actual circular arc covers the complete quadratic cluster.**
Source: arXiv:2601.21366v2, §3.2, the arc cover in `cor: bound`.
The source representatives lie on opposite sides of angle zero,
and the second atom uses integer turn `-1` to enter `[-1/8,1/8]`. -/
theorem quadraticCluster_periodic_arc_cover :
    ∀ i : Idx 2, ∃ j : Idx 1, ∃ k : ℤ,
      quadraticClusterAngles i + 2 * π * (k : ℝ) ∈
        Set.Icc ((fun _ : Idx 1 => -(1 / 8 : ℝ)) j) ((fun _ : Idx 1 => (1 / 8 : ℝ)) j) := by
  intro i
  fin_cases i
  · exact ⟨0, 0, by norm_num [quadraticClusterAngles]⟩
  · refine ⟨0, -1, ?_⟩
    change (2 * π - 1 / 8 : ℝ) + 2 * π * (-1 : ℤ) ∈
      Set.Icc (-(1 / 8 : ℝ)) (1 / 8 : ℝ)
    rw [show (2 * π - 1 / 8 : ℝ) + 2 * π * (-1 : ℤ) = -(1 / 8 : ℝ) by ring]
    norm_num

/-- Both actual quadratic-cluster atoms are heavy at threshold `1/2`.
Source: arXiv:2601.21366v2, §3.2, the definition of `N_ε` in
`cor: bound`. Their actual masses are the two equal positive halves. -/
theorem quadraticCluster_heavy_atom_count :
    (({i : Idx 2 | (1 / 2 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i} :
      Set (Idx 2)).ncard : ℝ) = 2 := by
  have hset : ({i : Idx 2 | (1 / 2 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i} :
      Set (Idx 2)) = (Finset.univ : Finset (Idx 2)) := by ext i; simp
  rw [hset, Set.ncard_coe_finset]
  norm_num

/-- The printed estimate holds for this actual nonempty heavy-atom
configuration with one arc across zero. Source: arXiv:2601.21366v2,
§3.2, `cor: bound`. The conclusion uses its actual count and cover. -/
theorem quadraticCluster_count_estimate :
    (({i : Idx 2 | (1 / 2 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i} :
      Set (Idx 2)).ncard : ℝ) ≤
      (1 / (1 / 2 : ℝ)) * (1 + 2 * (1 / 4 : ℝ) * Real.sqrt 4) *
        (0.5742 + (2 * ∑ j : Idx 2, |quadraticClusterWeights j| *
          ‖quadraticClusterNeuron j‖ ^ 2 + 1) * Real.exp (-4)) := by
  simpa only [Nat.cast_one] using stationary_atom_count_bound 4 le_rfl (fun t : ℝ => t ^ 2) id
    (fun s => by simpa using hasDerivAt_pow 2 s) LipschitzWith.id rfl
    quadraticClusterWeights quadraticClusterNeuron 2 (fun _ => 1 / 2)
    quadraticClusterAngles quadraticClusterProb isAtomicOnCircle_quadraticCluster
    isStationary_quadraticCluster 1 le_rfl (fun _ : Idx 1 => -(1 / 8 : ℝ))
    (fun _ : Idx 1 => (1 / 8 : ℝ)) (1 / 4) (fun j => by norm_num)
    quadraticCluster_periodic_arc_cover (1 / 2) (by norm_num)

/-- Every regularity, atomicity, stationarity, threshold, cover and
positive-atom-threshold input of both corollary forms is jointly
realized by the actual quadratic stationary pair at `β=4`.
Its one circular arc crosses zero and has actual length `1/4`.
Source: arXiv:2601.21366v2, §3.2, `cor: bound`, and §5.4, its proof. -/
example : (4 : ℝ) ≤ 4 ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id s) s) ∧
    LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
    IsAtomicOnCircle 2 (fun _ => 1 / 2) quadraticClusterAngles quadraticClusterProb ∧
    IsStationary 4 id quadraticClusterWeights quadraticClusterNeuron quadraticClusterProb ∧
    (1 : ℕ) ≤ 1 ∧
    (∀ j : Idx 1, (fun _ : Idx 1 => (1 / 8 : ℝ)) j -
      (fun _ : Idx 1 => -(1 / 8 : ℝ)) j ≤ 1 / 4) ∧
    (∀ i : Idx 2, ∃ j : Idx 1, ∃ k : ℤ,
      quadraticClusterAngles i + 2 * π * (k : ℝ) ∈
        Set.Icc ((fun _ : Idx 1 => -(1 / 8 : ℝ)) j) ((fun _ : Idx 1 => (1 / 8 : ℝ)) j)) ∧
    (0 : ℝ) < 1 / 2 := by
  obtain ⟨hβ, hφ, hσ, hzero, hatom, hμ, -, -⟩ := quadraticCluster_stationary_witness
  exact ⟨hβ, hφ, hσ, hzero, hatom, hμ, le_rfl, fun j => by norm_num,
    quadraticCluster_periodic_arc_cover, by norm_num⟩

/-- The cover data satisfy the source's original strict length
restriction as well as the strengthened corollary's weaker input.
Source: arXiv:2601.21366v2, §3.2, `L<2π` in `cor: bound`. -/
example : (1 / 4 : ℝ) < 2 * π ∧ (0 : ℝ) < 1 / 4 := by
  exact ⟨by nlinarith [Real.pi_gt_three], by norm_num⟩

end Transformer.Perceptron

/-
# Perceptrons and attention's mean-field landscape — the anti-concentration bound

Formalization of `thm: bound` and `cor: bound` of arXiv:2601.21366v2, §3.2:
the strict concavity of `θ ↦ e^{β cos θ}` on `(-β^{-1/2}, β^{-1/2})` stops the
atoms of a SOPD critical point from concentrating — a cluster of angular width
`1/(2√β)` carries at most `0.5742 + O(e^{-β})` of the mass — and, when the
weights are small enough, the whole configuration cannot be one such cluster.

**What the source says and what is carried here.**

* "for `β` sufficiently large … `≤ 0.5742 + O(e^{-β})`" is one statement:
  constants `C, β₀` produced *before* the activation, the weights and the
  configuration, with the bound `0.5742 + C e^{-β}` for every `β ≥ β₀`.  Read
  the other way round — `C` chosen after the configuration — the claim would
  say nothing, since `C` could absorb any cluster mass.

* `max_{i,j ∈ S} min_{k ∈ ℤ} |θ_i - θ_j + 2πk| ≤ 1/(2√β)` is carried as: for
  every `i, j ∈ S` there is a `k` with `|θ_i - θ_j + 2πk| ≤ 1/(2√β)`.  The
  minimum over `k` is attained and the maximum over a finite set is a bound on
  every entry, so this is the same condition.

* The arcs `I_j` of `cor: bound` are carried by their endpoints, with
  `L` a bound on their lengths, and `supp μ ⊂ ⋃_j I_j` as "every angle lies in
  some `I_j`" — the support of `Σ m_i δ_{θ_i}` with `m_i > 0` is exactly the
  set of its atoms.

* `σ` globally Lipschitz with constant `1` is `LipschitzWith 1 σ`.

**What is not witnessed.**  The examples below exhibit every hypothesis except
`IsSOPD`: a Lipschitz activation with its primitive, and a three-atom
configuration two of whose atoms are `1/(2√β)` apart.  That such a cluster
occurs *at a SOPD critical point* is exhibited neither here nor in the source —
the source's evidence for it is numerical — and it is precisely what the bound
constrains.  `Antipodal.lean` exhibits the two-atom SOPD critical points, for
which the cluster condition fails.

Source: arXiv:2601.21366v2, `thm: bound`, `cor: bound`.
-/

import Transformer.Perceptron.Atoms
import Transformer.Perceptron.Geodesic
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

/-- **Theorem (thm: bound).**  Let `d = 2`, `β > 0` and `σ` globally Lipschitz
with constant `1` and `σ(0) = 0`.  Let `μ = Σ_i m_i δ_{θ_i}` be a SOPD
Wasserstein critical point as in `eq: atomic.thm.bound`, and let
`S ⊆ ⟦1,N⟧` with `|S| ≥ 2` satisfy the cluster condition

  `max_{i,j ∈ S} min_{k ∈ ℤ} |θ_i - θ_j + 2πk| ≤ 1/(2√β)`.

Then, for `β` large enough, `Σ_{i ∈ S} m_i ≤ 0.5742 + O(e^{-β})`.

Not proved here.

Source: arXiv:2601.21366v2, `thm: bound`, `eq: conclusion`. -/
theorem bound_cluster_mass :
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (N : ℕ) (m θ : Idx N → ℝ)
        (μ : Perspective.ProbSphere 2),
        IsAtomicOnCircle N m θ μ → IsSOPD β φ σ ω a μ →
      ∀ S : Finset (Idx N), 2 ≤ S.card →
        (∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ,
          |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β)) →
        ∑ i ∈ S, m i ≤ 0.5742 + C * Real.exp (-β) := by
  sorry

/-- The hypotheses of `bound_cluster_mass` are satisfiable apart from
`IsSOPD`: `σ = id` is `1`-Lipschitz with `σ(0) = 0` and has the primitive
`φ(s) = s²`, and the three angles `0, δ, 2δ` at `δ = 1/(2√β)` carry equal
masses and put the first two atoms at distance exactly `δ`. -/
example (β : ℝ) (hβ : 1 ≤ β) :
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id s) s) ∧
      LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
      IsAtomicOnCircle 3 (fun _ => 1 / 3)
        (fun i : Idx 3 => (i : ℕ) * (1 / (2 * Real.sqrt β)))
        (atomicProb (fun _ => 1 / 3)
          (fun i : Idx 3 => circlePoint ((i : ℕ) * (1 / (2 * Real.sqrt β))))
          (fun _ => by norm_num) (by norm_num)) ∧
      ∀ i ∈ ({0, 1} : Finset (Idx 3)), ∀ j ∈ ({0, 1} : Finset (Idx 3)), ∃ k : ℤ,
        |(i : ℕ) * (1 / (2 * Real.sqrt β)) - (j : ℕ) * (1 / (2 * Real.sqrt β))
          + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β) := by
  have hs1 : (1 : ℝ) ≤ Real.sqrt β := by
    simpa using Real.sqrt_le_sqrt hβ
  have hs : (0 : ℝ) < Real.sqrt β := lt_of_lt_of_le one_pos hs1
  set δ : ℝ := 1 / (2 * Real.sqrt β) with hδdef
  have hδ : 0 < δ := by positivity
  have hδhalf : δ ≤ 1 / 2 := by
    rw [hδdef]
    exact one_div_le_one_div_of_le (by norm_num) (by linarith)
  refine ⟨fun s => by simpa using hasDerivAt_pow 2 s, LipschitzWith.id, rfl,
    isAtomicOnCircle_atomicProb (by norm_num) _ _ (fun _ => by norm_num) (by norm_num)
      (fun i => ?_) (fun i j hij => ?_), fun i hi j hj => ⟨0, ?_⟩⟩
  · have hi2 : ((i : ℕ) : ℝ) ≤ 2 := by exact_mod_cast i.is_le
    constructor
    · positivity
    · have : ((i : ℕ) : ℝ) * δ ≤ 2 * δ := by nlinarith [Nat.cast_nonneg (α := ℝ) (i : ℕ)]
      nlinarith [Real.pi_gt_three]
  · have : ((i : ℕ) : ℝ) = ((j : ℕ) : ℝ) := mul_right_cancel₀ hδ.ne' hij
    exact Fin.ext (by exact_mod_cast this)
  · have hij : ((i : ℕ) : ℝ) * δ - ((j : ℕ) : ℝ) * δ = (((i : ℕ) : ℝ) - (j : ℕ)) * δ := by ring
    have hi1 : (i : ℕ) ≤ 1 := by fin_cases hi <;> simp
    have hj1 : (j : ℕ) ≤ 1 := by fin_cases hj <;> simp
    have hib : ((i : ℕ) : ℝ) ≤ 1 := by exact_mod_cast hi1
    have hjb : ((j : ℕ) : ℝ) ≤ 1 := by exact_mod_cast hj1
    have hi0 : (0 : ℝ) ≤ (i : ℕ) := Nat.cast_nonneg _
    have hj0 : (0 : ℝ) ≤ (j : ℕ) := Nat.cast_nonneg _
    rw [Int.cast_zero, mul_zero, add_zero, hij, abs_mul, abs_of_pos hδ]
    have : |((i : ℕ) : ℝ) - (j : ℕ)| ≤ 1 := abs_le.mpr ⟨by linarith, by linarith⟩
    nlinarith

/-- **Theorem (thm: bound), second part.**  If the weights satisfy
`|ω₁|‖a₁‖² + |ω₂|‖a₂‖² < 0.16547`, then the whole index set `⟦1,N⟧` cannot
satisfy the cluster condition `eq: pairwise.distance`, for any `β > 0`.

Not proved here.

Source: arXiv:2601.21366v2, `thm: bound`, `eq: theta.bound`. -/
theorem not_cluster_univ_of_weights_small (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hlip : LipschitzWith 1 σ)
    (hσ0 : σ 0 = 0) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hω : |ω 0| * ‖a 0‖ ^ 2 + |ω 1| * ‖a 1‖ ^ 2 < 0.16547)
    (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2)
    (hatom : IsAtomicOnCircle N m θ μ) (hSOPD : IsSOPD β φ σ ω a μ) :
    ¬ ∀ i j : Idx N, ∃ k : ℤ,
        |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β) := by
  sorry

/-- The weight hypothesis of `not_cluster_univ_of_weights_small` is
satisfiable: the vanishing perceptron `ω = 0`, whose activation `σ = 0` is
`1`-Lipschitz with `σ(0) = 0` and has the primitive `φ = 0`. -/
example (a : Idx 2 → EucSpace 2) :
    (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
      LipschitzWith 1 (0 : ℝ → ℝ) ∧ (0 : ℝ → ℝ) 0 = 0 ∧
      |(0 : Idx 2 → ℝ) 0| * ‖a 0‖ ^ 2 + |(0 : Idx 2 → ℝ) 1| * ‖a 1‖ ^ 2 < 0.16547 :=
  ⟨fun s => by simpa using hasDerivAt_const s (0 : ℝ),
    LipschitzWith.const' 0, rfl, by norm_num⟩

/-- **Corollary (cor: bound).**  If the support of `μ` is covered by `M` arcs
of length at most `L < 2π`, the number `N_ε` of atoms of mass at least `ε`
satisfies, for `β` large enough,

  `N_ε ≤ (M/ε) (1 + 2L√β) (0.5742 + O(e^{-β}))`.

Not proved here.

Source: arXiv:2601.21366v2, `cor: bound`. -/
theorem bound_atom_count :
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (N : ℕ) (m θ : Idx N → ℝ)
        (μ : Perspective.ProbSphere 2),
        IsAtomicOnCircle N m θ μ → IsSOPD β φ σ ω a μ →
      ∀ (M : ℕ), 1 ≤ M → ∀ (l r : Idx M → ℝ) (L : ℝ), L < 2 * π →
        (∀ j : Idx M, r j - l j ≤ L) →
        (∀ i : Idx N, ∃ j : Idx M, θ i ∈ Set.Icc (l j) (r j)) →
      ∀ ε : ℝ, 0 < ε →
        (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ)
          ≤ (M / ε) * (1 + 2 * L * Real.sqrt β) * (0.5742 + C * Real.exp (-β)) := by
  sorry

/-- The covering hypotheses of `bound_atom_count` are satisfiable: one arc of
length `1` containing the three angles `0, δ, 2δ` of the configuration above,
and any positive `ε`. -/
example (β : ℝ) (hβ : 1 ≤ β) :
    (1 : ℕ) ≤ 1 ∧ (1 : ℝ) < 2 * π ∧
      (∀ j : Idx 1, (fun _ : Idx 1 => (1 : ℝ)) j - (fun _ : Idx 1 => (0 : ℝ)) j ≤ 1) ∧
      (∀ i : Idx 3, ∃ j : Idx 1, ((i : ℕ) : ℝ) * (1 / (2 * Real.sqrt β)) ∈
        Set.Icc ((fun _ : Idx 1 => (0 : ℝ)) j) ((fun _ : Idx 1 => (1 : ℝ)) j)) ∧
      (0 : ℝ) < 1 := by
  have hs1 : (1 : ℝ) ≤ Real.sqrt β := by simpa using Real.sqrt_le_sqrt hβ
  have hs : (0 : ℝ) < Real.sqrt β := lt_of_lt_of_le one_pos hs1
  have hδ : 0 < 1 / (2 * Real.sqrt β) := by positivity
  have hδhalf : 1 / (2 * Real.sqrt β) ≤ 1 / 2 :=
    one_div_le_one_div_of_le (by norm_num) (by linarith)
  refine ⟨le_rfl, by linarith [Real.pi_gt_three], fun _ => by norm_num, fun i => ⟨0, ?_, ?_⟩,
    one_pos⟩
  · positivity
  · have hi2 : ((i : ℕ) : ℝ) ≤ 2 := by exact_mod_cast i.is_le
    nlinarith [Nat.cast_nonneg (α := ℝ) (i : ℕ)]

end Perceptron
end Transformer

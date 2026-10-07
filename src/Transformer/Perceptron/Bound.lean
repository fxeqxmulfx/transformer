/-
# Perceptrons and attention's mean-field landscape — the anti-concentration bound

Formalization of `thm: bound` and `cor: bound` of arXiv:2601.21366v2, §3.2:
the source's local concavity estimate for `θ ↦ e^{β cos θ}` bounds the mass
of a cluster in a SOPD critical point — a cluster of angular width
`1/(2√β)` carries at most `0.5742 + O(e^{-β})` of the mass at large `β`.
The source additionally excludes a single such cluster for every `β > 0`
when the weights are small. That universal claim is false:
`Section3_ClusterCounterexample` proves a counterexample and its negation.

**What the source says and what is carried here.**

* "for `β` sufficiently large … `≤ 0.5742 + O(e^{-β})`" is one statement about
  the fixed perceptron `ϑ = (ω_j, a_j)_j` of the theorem: constants `C, β₀`
  produced after `ϑ` but *before* `β`, the activation and the configuration,
  with the bound `0.5742 + C e^{-β}` for every `β ≥ β₀`.  `C` may depend on
  `ϑ`, and in the source's proof it does: the bound derived there is
  `(2e^{β-3/2} + C_ϑ)/(2e^{β-3/2} + (3/8)e^{β-1/8})` with
  `C_ϑ = 2 Σ_j |ω_j| ‖a_j‖²`, whose `O(e^{-β})` part is `C_ϑ e^{-β}` up to a
  numerical factor.  A `C` uniform in `ϑ` would be a stronger claim than the
  source makes or proves.  Read the other way round — `C` chosen after the
  configuration — the claim would say nothing, since `C` could absorb any
  cluster mass.

* `max_{i,j ∈ S} min_{k ∈ ℤ} |θ_i - θ_j + 2πk| ≤ 1/(2√β)` is carried as: for
  every `i, j ∈ S` there is a `k` with `|θ_i - θ_j + 2πk| ≤ 1/(2√β)`.  The
  minimum over `k` is attained and the maximum over a finite set is a bound on
  every entry, so this is the same condition.

* The arcs `I_j` of `cor: bound` are carried by their endpoints, with
  `L` a bound on their lengths, and `supp μ ⊂ ⋃_j I_j` as "every angle lies in
  some `I_j`" — the support of `Σ m_i δ_{θ_i}` with `m_i > 0` is exactly the
  set of its atoms.

* `σ` globally Lipschitz with constant `1` is `LipschitzWith 1 σ`.

* `rem: thm.bound.multid` states no theorem: it says that the restriction to
  `d = 2` is an artefact of the proof — the angular parametrization of `𝕊¹` and
  the scalar concavity estimate `lem: concavity` — and that `d ≥ 3` would need
  a Hessian estimate on small geodesic caps.  It names no such estimate and
  claims no bound in `d ≥ 3`, so there is nothing to put on the books; the
  dimension hypothesis of both theorems above is the record of it.

**What is not witnessed.**  The examples below exhibit every hypothesis except
`IsSOPD`: a Lipschitz activation with its primitive, and a three-atom
configuration two of whose atoms are `1/(2√β)` apart. The antipodal SOPD
family in `Section3_Antipodal` satisfies the cluster condition exactly when
`0 < β ≤ 1/(4π²)`, as proved in `Section3_ClusterCounterexample`; it does not
witness a cluster at large `β`.

Source: arXiv:2601.21366v2, `thm: bound`, `cor: bound`.
-/

import Transformer.Perceptron.Atoms
import Transformer.Perceptron.Geodesic
import Transformer.Perceptron.Section3_ClusterCounterexample
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

Then, for `β` large enough, `Σ_{i ∈ S} m_i ≤ 0.5742 + O(e^{-β})`, the
constant of `O(e^{-β})` depending on the weights `ϑ` — see the module
docstring.

Not proved here.

Source: arXiv:2601.21366v2, `thm: bound`, `eq: conclusion`. -/
theorem bound_cluster_mass :
    ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2),
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2),
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

/-- **Corollary (cor: bound).**  If the support of `μ` is covered by `M` arcs
of length at most `L < 2π`, the number `N_ε` of atoms of mass at least `ε`
satisfies, for `β` large enough,

  `N_ε ≤ (M/ε) (1 + 2L√β) (0.5742 + O(e^{-β}))`,

the constant of `O(e^{-β})` depending on the weights `ϑ`, as in
`bound_cluster_mass`.

Not proved here.

Source: arXiv:2601.21366v2, `cor: bound`. -/
theorem bound_atom_count :
    ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2),
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2),
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

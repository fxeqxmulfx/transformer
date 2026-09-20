/-
# Perceptrons and attention's mean-field landscape — the non-degeneracy
  condition

`eq: nondeg.normalized` of arXiv:2601.21366v2, the hypothesis
`prop: unified.log` puts on the perceptron weights: for every index subset
`J ⊆ ⟦1,d⟧` the matrix

  `M = Σ_{j ∈ J} ω_j a_j a_jᵀ - Σ_{j ∉ J} ω_j a_j a_jᵀ`

is not a scalar multiple of the identity.

**What the source says and what is carried here.**

* `M` appears in the proof only through its quadratic form `x ↦ xᵀMx`, whose
  gradient is `2Mx`; `signedGram` is the map `x ↦ Mx`, written with
  `(a_j a_jᵀ)x = (a_j·x) a_j` so that no matrix type is needed.  "Is a scalar
  multiple of the identity" is `∃ c, ∀ x, Mx = cx`, which for a linear map is
  the same statement.

* The condition is a predicate of `(ω_j, a_j)_j` alone, and it is satisfiable:
  `isNonDegenerate_single` shows that a single non-zero neuron already has it,
  because a rank-one `±(a·x)a` is never a multiple of the identity in `d ≥ 2`
  — it kills a vector orthogonal to `a` without killing `a`.

Source: arXiv:2601.21366v2, `eq: nondeg.normalized`.
-/

import Transformer.Perceptron.Atomicity

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- **`Mx` for the matrix `M` of `eq: nondeg.normalized`**: the signed Gram
combination `Σ_{j ∈ J} ω_j (a_j·x) a_j - Σ_{j ∉ J} ω_j (a_j·x) a_j`.

Source: arXiv:2601.21366v2, `eq: nondeg.normalized`. -/
noncomputable def signedGram (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (J : Finset (Idx d))
    (x : EucSpace d) : EucSpace d :=
  (∑ j ∈ J, (ω j * inner (𝕜 := ℝ) (a j) x) • a j)
    - ∑ j ∈ Jᶜ, (ω j * inner (𝕜 := ℝ) (a j) x) • a j

/-- **`eq: nondeg.normalized`.**  The weights are non-degenerate: for no index
subset is the signed Gram matrix a scalar multiple of the identity.

Source: arXiv:2601.21366v2, `eq: nondeg.normalized`. -/
def IsNonDegenerate (ω : Idx d → ℝ) (a : Idx d → EucSpace d) : Prop :=
  ∀ J : Finset (Idx d), ¬ ∃ c : ℝ, ∀ x : EucSpace d, signedGram ω a J x = c • x

/-- The one-neuron perceptron `ω = e_{j₀}`, `a_{j₀} = -u` contributes
`(u·x) u` to a sum over any index set containing `j₀`, and nothing to one
that does not. -/
theorem sum_single_neuron (j₀ : Idx d) (u : EucSpace d) (S : Finset (Idx d))
    (x : EucSpace d) :
    ∑ j ∈ S, ((Pi.single j₀ (1 : ℝ) : Idx d → ℝ) j *
          inner (𝕜 := ℝ) ((Pi.single j₀ (-u) : Idx d → EucSpace d) j) x) •
        (Pi.single j₀ (-u) : Idx d → EucSpace d) j
      = if j₀ ∈ S then (inner (𝕜 := ℝ) u x) • u else 0 := by
  have hterm : ∀ j : Idx d, j ≠ j₀ →
      ((Pi.single j₀ (1 : ℝ) : Idx d → ℝ) j *
          inner (𝕜 := ℝ) ((Pi.single j₀ (-u) : Idx d → EucSpace d) j) x) •
        (Pi.single j₀ (-u) : Idx d → EucSpace d) j = 0 := by
    intro j hj
    simp [Pi.single_eq_of_ne hj]
  by_cases h : j₀ ∈ S
  · rw [ite_eq_left h, ← Finset.add_sum_erase _ _ h,
      Finset.sum_eq_zero fun j hj => hterm j (Finset.ne_of_mem_erase hj), add_zero]
    simp [inner_neg_left, neg_smul, smul_neg]
  · rw [ite_eq_right h]
    exact Finset.sum_eq_zero fun j hj => hterm j fun e => h (e ▸ hj)

/-- The signed Gram map of a single non-zero neuron is `±(u·x)u`: the sign is
`+` when the neuron is active for `J` and `-` when it is not. -/
theorem signedGram_single (j₀ : Idx d) (u : EucSpace d) (J : Finset (Idx d))
    (x : EucSpace d) :
    signedGram (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-u)) J x
      = (if j₀ ∈ J then (1 : ℝ) else -1) • ((inner (𝕜 := ℝ) u x) • u) := by
  rw [signedGram, sum_single_neuron, sum_single_neuron]
  by_cases h : j₀ ∈ J
  · simp [h, Finset.mem_compl]
  · simp [h, Finset.mem_compl]

/-- **A single non-zero neuron is already non-degenerate.**  Its signed Gram
matrix is `±(u·x)u`, which kills a unit vector `w ⊥ u` but not `u`, so it is
no multiple of the identity.

This is the witness for the non-degeneracy hypothesis of
`prop: unified.log`. -/
theorem isNonDegenerate_single (j₀ : Idx d) (u w : EucSpace d) (hu : ‖u‖ = 1)
    (hw : ‖w‖ = 1) (huw : inner (𝕜 := ℝ) u w = 0) :
    IsNonDegenerate (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-u)) := by
  have hw0 : w ≠ 0 := fun h => one_ne_zero (by rw [← hw, h, norm_zero])
  rintro J ⟨c, hc⟩
  have hgen : ∀ ε : ℝ, ε ≠ 0 →
      (∀ x : EucSpace d, ε • ((inner (𝕜 := ℝ) u x) • u) = c • x) → False := by
    intro ε hε h
    have h1 := h w
    rw [huw, zero_smul, smul_zero] at h1
    have hc0 : c = 0 := (smul_eq_zero.mp h1.symm).resolve_right hw0
    have h2 := h u
    rw [real_inner_self_eq_norm_sq, hu, hc0, zero_smul] at h2
    have hu0 : u = 0 := by simpa [hε] using h2
    exact one_ne_zero (by rw [← hu, hu0, norm_zero])
  simp only [signedGram_single] at hc
  by_cases hJ : j₀ ∈ J
  · exact hgen 1 one_ne_zero fun x => by rw [← hc x, ite_eq_left hJ]
  · exact hgen (-1) (by norm_num) fun x => by rw [← hc x, ite_eq_right hJ]

/-- The hypotheses of `isNonDegenerate_single` are satisfiable: the first two
standard basis vectors of `ℝ²`. -/
example : ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ = 1 ∧ ‖secondAxis‖ = 1 ∧
    inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2) secondAxis = 0 :=
  ⟨norm_basePoint_one, norm_secondAxis, inner_basePoint_secondAxis⟩

/-- The one-neuron perceptron `a_0 = -basePoint 1` of the witnesses of §6 is
non-degenerate. -/
theorem isNonDegenerate_pin :
    IsNonDegenerate (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) :=
  isNonDegenerate_single 0 _ secondAxis norm_basePoint_one norm_secondAxis
    inner_basePoint_secondAxis

end Perceptron
end Transformer

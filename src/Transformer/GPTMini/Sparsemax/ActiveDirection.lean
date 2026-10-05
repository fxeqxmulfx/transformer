import Transformer.GPTMini.Sparsemax.NonSaturation
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Linear

/-!
# An actual nonzero sparsemax direction on two active coordinates

arXiv:1602.02068v2, §2.2, Proposition 1 and §2.5,
`sparsemax_gradient`. We derive the active-pair direction for the existing
variational projection instead of stipulating a Jacobian. Increasing one
active score while decreasing another preserves the threshold for small
steps. The same mass transfers between the two probabilities.

Two positive coordinates exclude a zero full score derivative of the
projection, including when other coordinates are at a support boundary.
This does not guarantee a nonzero derivative for an outer loss, or for
parameters whose score map annihilates this direction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open Filter
open scoped BigOperators Topology

/-- Transfer a score increment between two coordinates. Source:
arXiv:1602.02068v2, §2.5, the direction `e_j - e_k`; no weight or
derivative is built into this input perturbation. -/
def transferScores {T : ℕ} (scores : Fin T → ℝ) (j k : Fin T) (t : ℝ) : Fin T → ℝ :=
  fun n => scores n + t * (basis j n - basis k n)

/-- The input transfer curve has its stated tangent by ordinary calculus.
Source: arXiv:1602.02068v2, §2.5, the direction `e_j - e_k`. -/
theorem transferScores_hasDerivAt {T : ℕ} (scores : Fin T → ℝ) (j k : Fin T) :
    HasDerivAt (transferScores scores j k) (basis j - basis k) (0 : ℝ) := by
  have h := (hasDerivAt_const (0 : ℝ) scores).add
    ((hasDerivAt_id (0 : ℝ)).smul_const (basis j - basis k))
  unfold transferScores
  simpa [Pi.add_def, Pi.smul_def, Pi.sub_def, smul_eq_mul] using h

/-- The actual projection transfers precisely the same mass while both
chosen coordinates remain positive. Source: arXiv:1602.02068v2, §2.2,
`sparsemax_closedform`, using an unchanged normalized threshold.
Other coordinates, including inactive boundary coordinates, are unchanged. -/
theorem sparseWeights_transfer_active_pair {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (t : ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (htj : -sparseWeights scores i j < t) (htk : t < sparseWeights scores i k) :
    sparseWeights (transferScores scores j k t) i =
      fun n => sparseWeights scores i n + t * (basis j n - basis k n) := by
  classical
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold scores i
  have hactive (n : Fin T) (hp : 0 < sparseWeights scores i n) :
      sparseWeights scores i n = scores n - τ := by
    have hv := sparseWeights_positive_visible scores i n hp
    have hm : 0 < max (scores n - τ) 0 := by
      simpa [hτ, thresholdWeights, hv] using hp
    have hs : 0 ≤ scores n - τ := by
      by_contra hn
      rw [max_eq_right (le_of_not_ge hn)] at hm
      linarith
    simp [hτ, thresholdWeights, hv, max_eq_left hs]
  have hjτ := hactive j hj
  have hkτ := hactive k hk
  have hjv := sparseWeights_positive_visible scores i j hj
  have hkv := sparseWeights_positive_visible scores i k hk
  have hrow : thresholdWeights (transferScores scores j k t) i τ =
      fun n => sparseWeights scores i n + t * (basis j n - basis k n) := by
    funext n
    by_cases hnj : n = j
    · subst n
      have hs : 0 ≤ scores j + t - τ := by linarith
      have he : max (scores j + t - τ) 0 = sparseWeights scores i j + t := by
        rw [max_eq_left hs, hjτ]
        ring
      simpa [thresholdWeights, transferScores, basis, hne, hjv] using he
    · by_cases hnk : n = k
      · subst n
        have hs : 0 ≤ scores k - t - τ := by linarith
        have he : max (scores k - t - τ) 0 = sparseWeights scores i k - t := by
          rw [max_eq_left hs, hkτ]
          ring
        simpa [thresholdWeights, transferScores, basis, Ne.symm hne, hkv,
          sub_eq_add_neg] using he
      · simpa [thresholdWeights, transferScores, basis, hnj, hnk] using
          (congrFun hτ n).symm
  have hsum : ∑ n, thresholdWeights (transferScores scores j k t) i τ n = 1 := by
    rw [hrow, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib]
    rw [(sparseWeights_spec scores i).1.2.1,
      (basis_mem_simplex {n : Fin T | n ≤ i} j hjv).2.1,
      (basis_mem_simplex {n : Fin T | n ≤ i} k hkv).2.1]
    ring
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  exact hrow

/-- Active-pair and step hypotheses hold with a genuine inactive slot.
Source context: arXiv:1602.02068v2, §2.2, the bounded sparse example. -/
example : (0 : Fin 3) ≠ 1 ∧ 0 < sparseWeights twoActiveScores 2 0 ∧
    0 < sparseWeights twoActiveScores 2 1 ∧
    -sparseWeights twoActiveScores 2 0 < 0 ∧ 0 < sparseWeights twoActiveScores 2 1 := by
  norm_num [twoActiveScores_projection]

/-- The first active probability has derivative exactly one along the
transfer direction. Source: arXiv:1602.02068v2, §2.5,
`sparsemax_gradient`. This is derived from the actual projection and
requires no differentiability assumption on other support boundaries. -/
theorem sparseWeights_active_pair_hasDerivAt {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k) :
    HasDerivAt (fun t => sparseWeights (transferScores scores j k t) i j) 1 0 := by
  have hlow : ∀ᶠ t : ℝ in 𝓝 0, -sparseWeights scores i j < t :=
    eventually_gt_nhds (by linarith)
  have hhigh : ∀ᶠ t : ℝ in 𝓝 0, t < sparseWeights scores i k :=
    eventually_lt_nhds hk
  apply ((hasDerivAt_id 0).const_add (sparseWeights scores i j)).congr_of_eventuallyEq
  exact (hlow.and hhigh).mono fun t ht => by
    have he := congrFun (sparseWeights_transfer_active_pair scores i j k t hne hj hk
      ht.1 ht.2) j
    simpa [basis, hne] using he

/-- Two-active derivative premises hold without full support.
Source context: arXiv:1602.02068v2, §2.5, active-pair direction. -/
example : (0 : Fin 3) ≠ 1 ∧ 0 < sparseWeights twoActiveScores 2 0 ∧
    0 < sparseWeights twoActiveScores 2 1 := by norm_num [twoActiveScores_projection]

/-- The complete probability curve has the active-pair tangent.
Source: arXiv:1602.02068v2, §2.5, `sparsemax_gradient`, derived from
the unchanged-threshold formula for the actual variational projection. -/
theorem sparseWeights_active_pair_hasDerivAt_row {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k) :
    HasDerivAt (fun t => sparseWeights (transferScores scores j k t) i)
      (basis j - basis k) (0 : ℝ) := by
  have hlow : ∀ᶠ t : ℝ in 𝓝 0, -sparseWeights scores i j < t :=
    eventually_gt_nhds (by linarith)
  have hhigh : ∀ᶠ t : ℝ in 𝓝 0, t < sparseWeights scores i k := eventually_lt_nhds hk
  have hlinear : HasDerivAt
      (fun t : ℝ => fun n => sparseWeights scores i n + t * (basis j n - basis k n))
      (basis j - basis k) 0 := by
    have h := (hasDerivAt_const (0 : ℝ) (sparseWeights scores i)).add
      ((hasDerivAt_id (0 : ℝ)).smul_const (basis j - basis k))
    simpa [Pi.add_def, Pi.smul_def, Pi.sub_def, smul_eq_mul] using h
  apply hlinear.congr_of_eventuallyEq
  exact (hlow.and hhigh).mono fun t ht =>
    sparseWeights_transfer_active_pair scores i j k t hne hj hk ht.1 ht.2

/-- Vector-tangent hypotheses admit the same sparse three-slot example.
Source context: arXiv:1602.02068v2, §2.5. -/
example : (0 : Fin 3) ≠ 1 ∧ 0 < sparseWeights twoActiveScores 2 0 ∧
    0 < sparseWeights twoActiveScores 2 1 := by norm_num [twoActiveScores_projection]

/-- Two positive coordinates rule out a zero full score derivative for
the actual projection. Source: arXiv:1602.02068v2, §2.5,
`sparsemax_gradient`, proved by composition along an active-pair curve.
The conclusion is weaker than existence of a full derivative at boundaries. -/
theorem sparseWeights_not_hasFDerivAt_zero_of_two_active {T : ℕ}
    (scores : Fin T → ℝ) (i j k : Fin T) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin T → ℝ => sparseWeights z i) 0 scores := by
  intro hz
  have hline := transferScores_hasDerivAt scores j k
  have hpoint : transferScores scores j k 0 = scores := by
    funext n
    simp [transferScores]
  have hz' : HasFDerivAt (𝕜 := ℝ) (fun z : Fin T → ℝ => sparseWeights z i) 0
      (transferScores scores j k 0) := by rw [hpoint]; exact hz
  have hcurve := hz'.comp_hasDerivAt (0 : ℝ) hline
  have hcoordinate : HasDerivAt
      (fun t => sparseWeights (transferScores scores j k t) i j) 0 0 := by
    have h := (ContinuousLinearMap.proj j : (Fin T → ℝ) →L[ℝ] ℝ).hasFDerivAt
      (x := sparseWeights (transferScores scores j k 0) i)
    simpa [Function.comp_def] using h.comp_hasDerivAt (0 : ℝ) hcurve
  have hnonzero := sparseWeights_active_pair_hasDerivAt scores i j k hne hj hk
  have he := hnonzero.unique hcoordinate
  norm_num at he

/-- The nonzero-direction conclusion holds on the sparse bounded example.
Source context: arXiv:1602.02068v2, §2.5, active-pair direction. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin 3 → ℝ => sparseWeights z 2) 0
    twoActiveScores :=
  sparseWeights_not_hasFDerivAt_zero_of_two_active _ _ 0 1 (by decide)
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])

end Transformer.GPTMini.Sparsemax

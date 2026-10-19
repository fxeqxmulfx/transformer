module

public import Transformer.Clusters.Section8_EnergyGrowth
public import Transformer.Clusters.Section8_LimitPrinciples
public import Transformer.Clusters.Section8_PathBounds
public import Mathlib.Topology.UniformSpace.HeineCantor

/-!
# The maximal attention gain tends to zero

ArXiv:2305.05465v6, §8.1, Step 2 of `t:Idcase11` and `e:derivativenorm`.
The maximum of the gaps `a_j = <A z_i, A(z_j-z_i)>` is nonnegative,
continuous in the configuration, and uniformly continuous along the
bounded dynamics. Twice this maximum is dominated by the derivative
of the corrected energy. Energy convergence therefore forces the
maximum to tend to zero.

This gives the vanishing-gain conclusion directly, without assuming
the paper's geometric estimate `cl:gamma'`. It is valid for every
attention operator, even a singular one. Injectivity will be used to
make the candidate set finite in the subsequent convergence proof.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The maximum of the gaps `a_j(t)` in `e:derivativenorm`.
Source: arXiv:2305.05465v6, §8.1, Step 2 of `t:Idcase11`.
The token index gives a nonempty finite set of competing scores. -/
noncomputable def attentionGain (A : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i : Idx n) : ℝ :=
  Finset.univ.sup' ⟨i, Finset.mem_univ i⟩
    (fun j => inner (𝕜 := ℝ) (A (Y i)) (A (Y j - Y i)))

/-- Each individual gap is at most the maximum in Step 2 of `t:Idcase11`.
Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm`. -/
theorem inner_gap_le_attentionGain (A : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i j : Idx n) :
    inner (𝕜 := ℝ) (A (Y i)) (A (Y j - Y i)) ≤ attentionGain A Y i := by
  unfold attentionGain
  exact Finset.le_sup' (fun k : Idx n => inner (𝕜 := ℝ) (A (Y i)) (A (Y k - Y i)))
    (Finset.mem_univ j)

/-- The self-token gap is zero, hence the maximal gain is nonnegative.
Source: arXiv:2305.05465v6, §8.1, Step 2 of `t:Idcase11`. -/
theorem attentionGain_nonneg (A : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i : Idx n) : 0 ≤ attentionGain A Y i := by
  simpa only [sub_self, map_zero, inner_zero_right] using inner_gap_le_attentionGain A Y i i

/-- The maximal gain vanishes exactly when all token displacements have
nonpositive attention inner product. Source: arXiv:2305.05465v6, §8.1,
`e:derivativenorm` and the candidate equality `e:Ax15`. -/
theorem attentionGain_eq_zero_iff (A : ParamMatrix d) (Y : Idx n → EucSpace d) (i : Idx n) :
    attentionGain A Y i = 0 ↔
      ∀ j : Idx n, inner (𝕜 := ℝ) (A (Y i)) (A (Y j - Y i)) ≤ 0 := by
  constructor
  · intro hzero j
    exact (inner_gap_le_attentionGain A Y i j).trans_eq hzero
  · intro hnonpos
    apply le_antisymm _ (attentionGain_nonneg A Y i)
    exact (Finset.sup'_le_iff _ _).mpr fun j _ => hnonpos j

/-- A coincident configuration has zero maximal gain, matching its
stationarity in `e:dynIdzi`. Source: arXiv:2305.05465v6, §8.1. -/
theorem attentionGain_const (A : ParamMatrix d) (z : EucSpace d) (i : Idx n) :
    attentionGain A (fun _ => z) i = 0 := by
  simp only [attentionGain, sub_self, map_zero, inner_zero_right]
  apply Finset.sup'_const

/-- The finite maximum of the attention gaps is continuous in the token
tuple. Source: arXiv:2305.05465v6, §8.1, the compactness argument for
`cl:gamma'`, using the corresponding configuration-dependent maximum. -/
theorem continuous_attentionGain (A : ParamMatrix d) (i : Idx n) :
    Continuous (fun Y : Idx n → EucSpace d => attentionGain A Y i) := by
  unfold attentionGain
  apply Continuous.finset_sup'_apply
  intro j _
  fun_prop

/-- The maximal gain also bounds displacements to every point of the
current token hull. Source: arXiv:2305.05465v6, §8.1, proof of `cl:gamma'`,
the comparison of the token maximum with points in the limiting polytope. -/
theorem inner_gap_le_attentionGain_of_mem_hull (A : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) (y : EucSpace d) (hy : y ∈ tokenHull Y) :
    inner (𝕜 := ℝ) (A (Y i)) (A (y - Y i)) ≤ attentionGain A Y i := by
  let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A (Y i))).comp A
  let c := ‖A (Y i)‖ ^ 2 + attentionGain A Y i
  have hconv : Convex ℝ {y | f y ≤ c} := by
    intro y hy z hz a b ha hb hab
    change f (a • y + b • z) ≤ c
    simp only [map_add, map_smul, smul_eq_mul]
    calc
      a * f y + b * f z ≤ a * c + b * c :=
        add_le_add (mul_le_mul_of_nonneg_left hy ha) (mul_le_mul_of_nonneg_left hz hb)
      _ = c := by rw [← add_mul, hab, one_mul]
  have hgen : Set.range Y ⊆ {y | f y ≤ c} := by
    rintro _ ⟨j, rfl⟩
    have hj := inner_gap_le_attentionGain A Y i j
    simp only [map_sub, inner_sub_right, real_inner_self_eq_norm_sq] at hj
    change inner (𝕜 := ℝ) (A (Y i)) (A (Y j)) ≤ ‖A (Y i)‖ ^ 2 + attentionGain A Y i
    linarith
  have hle : f y ≤ c := convexHull_min hgen hconv hy
  rw [map_sub, inner_sub_right, real_inner_self_eq_norm_sq]
  change inner (𝕜 := ℝ) (A (Y i)) (A y) ≤
    ‖A (Y i)‖ ^ 2 + attentionGain A Y i at hle
  linarith

/-- The hull-membership input holds at every token of a genuine
two-token configuration. Source: arXiv:2305.05465v6, §8.1, `p:noninc`. -/
example (Y : Idx 2 → EucSpace 1) : Y 0 ∈ tokenHull Y := mem_tokenHull Y 0

/-- The gain is uniformly continuous along the actual future dynamics:
the initial hull confines the configuration to a compact product, and the
tuple is uniformly continuous in time. Source: arXiv:2305.05465v6, §8.1,
`c:theziunifbdd` and Steps 2--3 of `t:Idcase11`. -/
theorem IdRescaledDynamics.uniformContinuousOn_attentionGain (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    UniformContinuousOn (fun t => attentionGain A (Z t) i) (Set.Ici 0) := by
  have hcompact : IsCompact {Y : Idx n → EucSpace d | ∀ j, Y j ∈ tokenHull (Z 0)} :=
    isCompact_pi_infinite fun _ => isCompact_tokenHull (Z 0)
  exact (hcompact.uniformContinuousOn_of_continuous (continuous_attentionGain A i).continuousOn).comp
    (hZ.uniformContinuousOn_tuple A Z) (fun t ht j => mem_tokenHull_zero A Z hZ j ht)

/-- The uniform-continuity input is realized by two stationary unit
tokens. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

/-- **The maximal attention gain vanishes.** Source: arXiv:2305.05465v6,
§8.1, the growth argument in Step 2 of `t:Idcase11`.
The half-corrected energy has derivative at least the gain, is convergent,
and the gain is nonnegative and uniformly continuous on future times. -/
theorem IdRescaledDynamics.tendsto_attentionGain_zero (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    Tendsto (fun t => attentionGain A (Z t) i) atTop (nhds 0) := by
  choose v hv hbound using fun t => hZ.corrected_energy_deriv_bound A Z t i
  apply tendsto_zero_of_deriv_domination (fun t => correctedTokenEnergy A Z i t / 2)
    (fun t => v t / 2) (fun t => attentionGain A (Z t) i)
    ((⨆ t : ℝ, correctedTokenEnergy A Z i t) / 2) (fun t => (hv t).div_const 2)
  · intro t _
    unfold attentionGain
    apply (Finset.sup'_le_iff _ _).mpr
    intro j _
    have hj := hbound t j
    linarith
  · exact fun t _ => attentionGain_nonneg A (Z t) i
  · exact hZ.uniformContinuousOn_attentionGain A Z i
  · exact (hZ.tendsto_corrected_energy A Z i).div_const 2

/-- The vanishing-gain theorem has an actual stationary unit-token
instance. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

end Transformer.Clusters

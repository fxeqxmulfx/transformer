module

public import Transformer.Clusters.Section9_Growth
public import Transformer.Clusters.Section9_ThreeCoordinates

/-!
# The emergence of clusters in self-attention dynamics — the three
  hyperplanes

§9 of arXiv:2305.05465v6, `sec: clustering.hyperplanes`: the proof of
`l:3hyperplanes11`.

**What the source says and what is carried here.**

* The proof opens by naming what it will establish: "for any `i ∈ [n]`,
  `φ₁*(z_i(t))` converges as `t → +∞`", and "there exists a set of at most 3
  real numbers such that for any `i ∈ [n]` the limit belongs to this set",
  and then "Theorem `l:3hyperplanes11` directly follows from these facts".
  Both facts are proved: `exists_tendsto_eigenFunctional` obtains the
  coordinate limits, and `dist_tendsto_zero_of_tendsto_eigenFunctional`
  gives their affine hyperplanes. Together they prove the original
  `good_dist_tendsto_zero` of §4.

* The three numbers are `{0, a, b}` with `a`, `b` of `e:defab`
  (`exists_tendsto_maxCoord_minCoord`), and the hyperplanes are their level
  sets `H_c = {x : φ₁*(x) = c}`. A level set of `φ₁*` is a translate of
  `ker φ₁*`, which is `affineShift` of §4, and `c • v` for any `v` with
  `φ₁*(v) = 1` is a point on it.

* `φ₁*` is pinned as the leading eigenfunctional of a good triple by
  `IsGoodTripleWith Q K V lam φ G` together with `IsEigenFunctional V f lam`
  and the normalisation `f φ = 1`. The intermediate estimate
  `e:majorizeweights` of the proof is not carried on its own: it is a bound
  on `⟨Qe^{tV}z_i, Ke^{tV}z_j⟩` written in the full complex eigenbasis of `V`,
  which none of the statements of the paper needs.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes`, the proof of
`l:3hyperplanes11`, `e:defab`.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d m n : ℕ}

/-! ### From a converging coordinate to a limiting hyperplane -/

/-- **The distance to `H_c` is controlled by the coordinate.**  If
`f (x t) → c` then `x t` converges to the hyperplane `{f = c}`.

Source: arXiv:2305.05465v6, the proof of `l:3hyperplanes11`, "in which case we
deduce that `z_i(t)` converges toward `H_0`". -/
theorem tendsto_infDist_affineShift (f : EucSpace d →L[ℝ] ℝ) {v : EucSpace d} (hv : f v = 1)
    (x : ℝ → EucSpace d) {c : ℝ} (hx : Tendsto (fun t => f (x t)) atTop (nhds c))
    {p : EucSpace d} (hp : f p = c) :
    Tendsto (fun t => Metric.infDist (x t)
      (affineShift (LinearMap.ker (f : EucSpace d →ₗ[ℝ] ℝ)) p)) atTop (nhds 0) := by
  refine squeeze_zero (fun t => Metric.infDist_nonneg)
    (g := fun t => |f (x t) - c| * ‖v‖) (fun t => ?_) ?_
  · have hmem : x t - (f (x t) - c) • v ∈
        affineShift (LinearMap.ker (f : EucSpace d →ₗ[ℝ] ℝ)) p := by
      simp only [affineShift, Set.mem_ofPred_eq, LinearMap.mem_ker,
        ContinuousLinearMap.coe_coe, map_sub, map_smul, hv, hp, smul_eq_mul]
      ring
    refine (Metric.infDist_le_dist_of_mem hmem).trans_eq ?_
    rw [dist_eq_norm, sub_sub_cancel, norm_smul, Real.norm_eq_abs]
  · have := ((hx.sub_const c).abs).mul_const ‖v‖
    simpa using this

example (z : EucSpace 1) :
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
      Tendsto (fun _ : ℝ => (EuclideanSpace.proj (𝕜 := ℝ) 0) z) atTop (nhds (z 0)) ∧
      (EuclideanSpace.proj (𝕜 := ℝ) 0) z = z 0 :=
  ⟨by simp, tendsto_const_nhds, rfl⟩

/-- **Theorem `l:3hyperplanes11` "directly follows from these facts".**  Given
that every `φ₁*(z_i(t))` converges to one of at most three reals, every token
converges to one of at most three parallel hyperplanes — the level sets of
`φ₁*`, which have codimension `1`.

Source: arXiv:2305.05465v6, the proof of `l:3hyperplanes11`, first
paragraph. -/
theorem dist_tendsto_zero_of_tendsto_eigenFunctional (f : EucSpace d →L[ℝ] ℝ)
    {v : EucSpace d} (hv : f v = 1) (Z : ℝ → Idx n → EucSpace d) (S : Finset ℝ)
    (hS : S.card ≤ 3)
    (hlim : ∀ i : Idx n, ∃ c ∈ S, Tendsto (fun t => f (Z t i)) atTop (nhds c)) :
    ∃ (W : Submodule ℝ (EucSpace d)) (T : Finset (EucSpace d)),
      Module.finrank ℝ W + 1 = d ∧ T.card ≤ 3 ∧
        ∀ i : Idx n, ∃ p ∈ T,
          Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  refine ⟨LinearMap.ker (f : EucSpace d →ₗ[ℝ] ℝ), S.image fun c => c • v,
    finrank_ker_add_one f hv, Finset.card_image_le.trans hS, fun i => ?_⟩
  obtain ⟨c, hcS, hc⟩ := hlim i
  exact ⟨c • v, Finset.mem_image_of_mem _ hcS,
    tendsto_infDist_affineShift f hv _ hc (apply_smul_eq f hv c)⟩

/-- The hypotheses of `dist_tendsto_zero_of_tendsto_eigenFunctional` are
satisfiable: a configuration that does not move has every coordinate constant,
and a single value is a set of at most three. -/
example (z : EucSpace 1) :
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
      ({z 0} : Finset ℝ).card ≤ 3 ∧
      ∀ i : Idx n, ∃ c ∈ ({z 0} : Finset ℝ),
        Tendsto (fun t : ℝ => (EuclideanSpace.proj (𝕜 := ℝ) 0)
          ((fun (_ : ℝ) (_ : Idx n) => z) t i)) atTop (nhds c) := by
  refine ⟨by simp, by simp, fun _ => ⟨z 0, Finset.mem_singleton_self _, ?_⟩⟩
  exact tendsto_const_nhds

/-! ### The claim the proof of `l:3hyperplanes11` establishes -/

/-- **The two facts the proof of `l:3hyperplanes11` establishes.**  For a good
triple with leading eigenpair `(λ₁, φ₁)` and normalised eigenfunctional `φ₁*`,
every `φ₁*(z_i(t))` converges, and there is a set of at most three reals — the
`{0, a, b}` of `e:defab` — containing every limit.

The proof uses the actual leading score estimate, vanishing corrected-energy
gain, and finite cluster-point convergence from `Section9_ThreeCoordinates`.
It retains the original hypotheses, including non-diagonalizable value maps
and positivity only on the leading line.

Source: arXiv:2305.05465v6, the proof of `l:3hyperplanes11`. -/
theorem exists_tendsto_eigenFunctional (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hQKV : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ S : Finset ℝ, S.card ≤ 3 ∧
      ∀ i : Idx (m + 1), ∃ c ∈ S, Tendsto (fun t => f (Z t i)) atTop (nhds c) := by
  exact hZ.exists_three_leading_coordinate_limits Q K V lam φ G hQKV f hf hfφ Z

/-- The hypotheses of `exists_tendsto_eigenFunctional` are satisfiable: the
good triple `(I₁, I₁, I₁)` with its leading eigenpair `(1, e₀)`, the
coordinate functional, and the stationary configuration of
`rescaledDynamics_one_const`. -/
example (z : EucSpace 1) :
    IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1 (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
      IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
      (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
      RescaledDynamics (n := m + 1) (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp,
    rescaledDynamics_one_const _ _ z⟩

/-- **Theorem (l:3hyperplanes11), convergence toward `≤ 3` hyperplanes.**
For a good triple and any actual rescaled trajectory, every token's
distance to one of at most three parallel hyperplanes tends to zero.
The coordinate construction requires no diagonalizability of `V`.
Source: arXiv:2305.05465v6, §4, `l:3hyperplanes11`, and its proof in §9. -/
theorem good_dist_tendsto_zero (Q K V : ParamMatrix d) (hQKV : IsGoodTriple Q K V)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ (W : Submodule ℝ (EucSpace d)) (S : Finset (EucSpace d)),
      Module.finrank ℝ W + 1 = d ∧ S.card ≤ 3 ∧
        ∀ i : Idx n, ∃ p ∈ S,
          Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  obtain ⟨lam, φ, G, hgood⟩ := hQKV
  obtain ⟨f, hf, hfφ, _⟩ := hgood.exists_leading_eigenFunctional Q K V lam φ G
  cases n with
  | zero =>
    refine ⟨LinearMap.ker (f : EucSpace d →ₗ[ℝ] ℝ), ∅,
      finrank_ker_add_one f hfφ, by simp, fun i => ?_⟩
    exact Fin.elim0 i
  | succ m =>
    obtain ⟨S, hS, hlim⟩ := exists_tendsto_eigenFunctional Q K V lam φ G hgood f hf hfφ Z hZ
    exact dist_tendsto_zero_of_tendsto_eigenFunctional f hfφ Z S hS hlim

/-- The original good-triple and ODE hypotheses coexist for a nonzero
stationary scalar trajectory. Source: arXiv:2305.05465v6, §4, `l:3hyperplanes11`. -/
example (z : EucSpace 1) :
    IsGoodTriple (1 : ParamMatrix 1) 1 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨isGoodTriple_one, rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer

module

public import Transformer.Perspective.AppendixA_Convergence
public import Transformer.Perspective.Section3_Gronwall
public import Transformer.Perspective.Section5_ExpRate
public import Transformer.Perspective.Section5_HemisphereCone

/-!
# §4 — the sets `𝒮_0^m` and `𝒮_β` of `th:beta_small`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §4, proof of `th:beta_small`.

Two inclusions carry the proof of `th:beta_small`. A start whose `β = 0`
solution reaches consensus is `(3/4)`-clustered at every late enough integer
time (`eventually_mem_Sset0`), so, consensus being almost sure (`p:beta0`), the
measure of `𝒮_0^m` tends to that of the whole space
(`tendsto_measure_of_ae_eventually_mem`, for any sets almost every point
eventually enters). The source derives this limit, `e:Ps0n`, from
`𝒮_0^m ⊂ 𝒮_0^{m+1}` and `⋃_m 𝒮_0^m = 𝒮_0`; the monotonicity is not needed.
And `𝒮_0^m ⊂ 𝒮_β` for `β ≤ β_m`
(`exists_Sset0_subset_clusteringSet`): at time `m` the `SA` solution is within
`1/8` of the `β = 0` one (`distance_bound_at_time_m`), so it is
`(1/2)`-clustered, in the open hemisphere around its first particle, from which
it collapses to one point (`cone_collapse`).
-/

@[expose] public section

open Filter Set MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **`SA` at `β = 0` is `e:Snonres0`:** the attention weights are uniform. -/
theorem beta0Dynamics_of_SA_zero {X : ℝ → SphereTuple d n} (hX : SA d n 0 X) :
    beta0Dynamics d n X := fun t i =>
  (hX t i).congr_deriv (by rw [SA_zero_drift])

/-- The hypothesis of `beta0Dynamics_of_SA_zero` holds for a single particle at
rest. -/
example : SA 1 1 0 (fun _ _ => basePoint 0) := SA_const_consensus 1 1 one_pos 0 _

/-- **`e:Snonres0` has a solution from every start:** the `SA` flow at
`β = 0`. -/
theorem beta0Dynamics_saFlow (X₀ : SphereTuple d n) : beta0Dynamics d n (saFlow 0 X₀) :=
  beta0Dynamics_of_SA_zero (saFlow_spec 0 X₀).2

/-- **A start reaching consensus is `(3/4)`-clustered at every late enough
integer time:** along its `β = 0` solution `⟨x_i(t), x_j(t)⟩ → ⟨x⋆, x⋆⟩ = 1`. -/
theorem eventually_mem_Sset0 {X₀ : SphereTuple d n} (h : X₀ ∈ consensusSet0 d n) :
    ∀ᶠ m : ℕ in atTop, X₀ ∈ Sset0 d n m := by
  obtain ⟨x, hx⟩ := h
  have hlim := hx (saFlow 0 X₀) (saFlow_zero 0 X₀) (beta0Dynamics_saFlow X₀)
  have hxx : inner (𝕜 := ℝ) (x : EucSpace d) x = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2, one_pow]
  have hpair : ∀ᶠ t in atTop, ∀ i j : Idx n, (3 / 4 : ℝ) <
      inner (𝕜 := ℝ) (saFlow 0 X₀ t i : EucSpace d) (saFlow 0 X₀ t j : EucSpace d) := by
    refine eventually_all.mpr fun i => eventually_all.mpr fun j => ?_
    have h := Filter.Tendsto.inner (𝕜 := ℝ) (tendsto_sub_nhds_zero_iff.mp (hlim i))
      (tendsto_sub_nhds_zero_iff.mp (hlim j))
    rw [hxx] at h
    exact h.eventually (lt_mem_nhds (by norm_num))
  filter_upwards [tendsto_natCast_atTop_atTop.eventually hpair] with m hm
  exact ⟨saFlow 0 X₀, saFlow_zero 0 X₀, beta0Dynamics_saFlow X₀, hm⟩

/-- The hypothesis of `eventually_mem_Sset0` holds for a single particle: its
solution of `e:Snonres0` stands still. -/
example : (fun _ => basePoint 0 : SphereTuple 1 1) ∈ consensusSet0 1 1 := by
  refine ⟨basePoint 0, fun X hX0 hX i => ?_⟩
  have h := saFlow_eq_of_SA 0 X (SA_zero_of_beta0Dynamics hX)
  rw [hX0, saFlow_consensus 0 one_pos] at h
  simp [← h]

/-- **Sets almost every point eventually enters fill the space in the limit:**
`μ(S_m) → μ(univ)`, for sets `S_m` not necessarily measurable. The sets
`⋂_{m ≥ M} S_m` increase to a set of full measure. -/
theorem tendsto_measure_of_ae_eventually_mem {α : Type*} [MeasurableSpace α]
    {μ : Measure α} {S : ℕ → Set α} (h : ∀ᵐ x ∂μ, ∀ᶠ m in atTop, x ∈ S m) :
    Tendsto (fun m => μ (S m)) atTop (𝓝 (μ univ)) := by
  have hmono : Monotone fun M : ℕ => {x | ∀ m, M ≤ m → x ∈ S m} :=
    fun M M' hMM' x hx m hm => hx m (hMM'.trans hm)
  have hU : μ (⋃ M : ℕ, {x | ∀ m, M ≤ m → x ∈ S m}) = μ univ := by
    refine measure_congr (ae_eq_univ.mpr (measure_mono_null ?_ (ae_iff.mp h)))
    intro x hx hev
    obtain ⟨M, hM⟩ := eventually_atTop.mp hev
    exact hx (mem_iUnion.mpr ⟨M, hM⟩)
  have hlim := tendsto_measure_iUnion_atTop (μ := μ) hmono
  rw [hU] at hlim
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le hlim tendsto_const_nhds
    (fun M => measure_mono fun x hx => hx M le_rfl) (fun M => measure_mono (subset_univ _))

/-- The hypothesis of `tendsto_measure_of_ae_eventually_mem` holds for the
constant sequence `S_m = univ`. -/
example : ∀ᵐ x ∂(volume : Measure ℝ), ∀ᶠ m in atTop, x ∈ (fun _ : ℕ => (univ : Set ℝ)) m :=
  Eventually.of_forall fun _ => Eventually.of_forall fun _ => mem_univ _

/-- Moving two vectors by at most `1/8` each, the first from a unit vector and
the second to one, lowers their inner product by at most `1/4`. -/
theorem inner_sub_quarter_le {a b a' b' : EucSpace d} (ha : ‖a‖ = 1) (hb' : ‖b'‖ = 1)
    (h1 : ‖a' - a‖ ≤ 1 / 8) (h2 : ‖b' - b‖ ≤ 1 / 8) :
    inner (𝕜 := ℝ) a b - 1 / 4 ≤ inner (𝕜 := ℝ) a' b' := by
  have e : inner (𝕜 := ℝ) a' b'
      = inner (𝕜 := ℝ) (a' - a) b' + inner (𝕜 := ℝ) a (b' - b) + inner (𝕜 := ℝ) a b := by
    simp only [inner_sub_left, inner_sub_right]
    ring
  have h3 := abs_real_inner_le_norm (a' - a) b'
  have h4 := abs_real_inner_le_norm a (b' - b)
  rw [hb', mul_one] at h3
  rw [ha, one_mul] at h4
  linarith [neg_abs_le (inner (𝕜 := ℝ) (a' - a) b'), neg_abs_le (inner (𝕜 := ℝ) a (b' - b))]

/-- The hypotheses of `inner_sub_quarter_le` hold for four copies of a basis
vector of `ℝ²`. -/
example : ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ) - EuclideanSpace.single 0 1‖ ≤ 1 / 8 := by
  norm_num

/-- **`𝒮_0^m ⊂ 𝒮_β` for small `β`** (`n ≥ 1`). At time `m` the `SA` solution is
within `1/8` of the `β = 0` one (`distance_bound_at_time_m`), so it is
`(1/2)`-clustered and lies in the open hemisphere around its first particle;
from there it collapses to a single point (`cone_collapse`), and it is the only
solution of `SA` through its start. -/
theorem exists_Sset0_subset_clusteringSet (hn : 0 < n) (m : ℕ) :
    ∃ βm : ℝ, 0 < βm ∧ ∀ β : ℝ, 0 ≤ β → β ≤ βm → Sset0 d n m ⊆ clusteringSet d n β := by
  obtain ⟨βm, hβm, hclose⟩ := distance_bound_at_time_m d n m
  refine ⟨βm, hβm, fun β hβ0 hβ X₀ ⟨X0t, hX0t0, hX0t, hclus⟩ => ?_⟩
  have hSA : SA d n β (saFlow β X₀) := (saFlow_spec β X₀).2
  have hnear := hclose β hβ0 hβ (saFlow β X₀) X0t (by rw [saFlow_zero, hX0t0]) hSA hX0t
  have hhemi : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) (saFlow β X₀ m i : EucSpace d)
      (saFlow β X₀ m ⟨0, hn⟩ : EucSpace d) := fun i => by
    have h := inner_sub_quarter_le (mem_sphere_zero_iff_norm.mp (X0t m i).2)
      (mem_sphere_zero_iff_norm.mp (saFlow β X₀ m ⟨0, hn⟩).2) (hnear i) (hnear ⟨0, hn⟩)
    linarith [hclus i ⟨0, hn⟩]
  obtain ⟨x, C, lam, -, hlam, hexp⟩ := (cone_collapse d n β _ ⟨_, hhemi⟩).1
  have hbound := hexp _ (by simp) (SA_shift d n β _ hSA m)
  refine ⟨x, fun X hX0 hX i => ?_⟩
  have hXeq : X = saFlow β X₀ := by rw [← hX0, saFlow_eq_of_SA β X hX]
  rw [hXeq]
  have hdecay : Tendsto (fun t => C * Real.exp (-(lam * (t - m)))) atTop (𝓝 0) := by
    have h := (Real.tendsto_exp_neg_atTop_nhds_zero.comp
      ((tendsto_atTop_add_const_right _ (-(m : ℝ)) tendsto_id).const_mul_atTop hlam)).const_mul C
    simpa [Function.comp_def, sub_eq_add_neg] using h
  refine squeeze_zero_norm' ?_ hdecay
  filter_upwards [eventually_ge_atTop (m : ℝ)] with t ht
  simpa using hbound i (t - m) (sub_nonneg.mpr ht)

/-- The hypothesis of `exists_Sset0_subset_clusteringSet` holds for two
particles. -/
example : 0 < 2 := two_pos

end Perspective
end Transformer

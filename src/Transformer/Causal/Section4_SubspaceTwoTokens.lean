module

public import Transformer.Causal.Section4_TwoTokenPositiveEntry
public import Transformer.Causal.Section4_TwoTokenPositiveLimit
public import Transformer.Causal.Section4_SubspaceTransverse
public import Transformer.Perspective.SphereHyperplane
public import Transformer.Perspective.UniformAtomless
public import Transformer.Perspective.UniformShell

/-!
# The complete leading-subspace conclusion for two original tokens

ArXiv:2411.04990v2, §4, `thm2`, restricted to two tokens.
The original leader converges to its normalized initial leading
projection. A genuine transverse initial leading covector forces
the actual follower into the positive hemisphere, and the original
positive-overlap drift then gives its same limit. The original
uniform initialization supplies this covector almost surely.
All original spectral assumptions and the actual causal equation
are retained. The assertion for three or more tokens is separate.
-/

@[expose] public section

open Real Filter MeasureTheory
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- Almost every original uniform two-token initialization supplies
an actual leading covector orthogonal to the leader and positive on
the follower, whenever the leader has nonzero leading projection.
Source: arXiv:2411.04990v2, §4, the almost-everywhere qualification
in `thm2`, restricted to two original tokens. -/
theorem ae_leading_transverse_witness (L : Submodule ℝ (EucSpace d))
    (hdim : 2 ≤ Module.finrank ℝ L)
    (P : Measure (SphereTuple d 2)) (hP : Perspective.UniformTuple d 2 P) :
    ∀ᵐ X₀ ∂P, L.starProjection (X₀ 0 : EucSpace d) ≠ 0 →
      ∃ v ∈ L, inner ℝ v (X₀ 0 : EucSpace d) = 0 ∧
        0 < inner ℝ v (X₀ 1 : EucSpace d) := by
  obtain ⟨σ, hprob, hinv, rfl⟩ := hP
  have := hprob
  let B : Set (SSphere d × SSphere d) := {z |
    L.starProjection (z.1 : EucSpace d) ≠ 0 ∧
      leadingTransverse L (z.1 : EucSpace d) (z.2 : EucSpace d) = 0}
  have hx : Continuous (fun z : SSphere d × SSphere d =>
      L.starProjection (z.1 : EucSpace d)) :=
    L.starProjection.continuous.comp (continuous_subtype_val.comp continuous_fst)
  have hy : Continuous (fun z : SSphere d × SSphere d =>
      L.starProjection (z.2 : EucSpace d)) :=
    L.starProjection.continuous.comp (continuous_subtype_val.comp continuous_snd)
  have hT : Continuous (fun z : SSphere d × SSphere d =>
      leadingTransverse L (z.1 : EucSpace d) (z.2 : EucSpace d)) :=
    ((hx.inner (𝕜 := ℝ) hx).smul hy).sub ((hx.inner (𝕜 := ℝ) hy).smul hx)
  have hB : MeasurableSet B :=
    (isClosed_eq hx continuous_const).measurableSet.compl.inter
      (isClosed_eq hT continuous_const).measurableSet
  have hfiber (x : SSphere d) : σ (Prod.mk x ⁻¹' B) = 0 := by
    by_cases hp : L.starProjection (x : EucSpace d) = 0
    · simp only [B, Set.preimage_ofPred_eq, hp, ne_eq, not_true_eq_false, false_and,
        Set.ofPred_false, measure_empty]
    · obtain ⟨v, hv, hv0, hvx⟩ := exists_leading_transverse_normal L hdim x hp
      apply measure_mono_null (t := Perspective.greatSphere d v) ?_
        (Perspective.measure_greatSphere_eq_zero d σ hinv hv0)
      intro y hxy
      change inner ℝ (y : EucSpace d) v = 0
      rw [real_inner_comm v (y : EucSpace d)]
      exact inner_eq_zero_of_leadingTransverse_eq_zero L x y v hp hv hvx hxy.2
  have hprod : σ.prod σ B = 0 := by
    rw [Measure.prod_apply hB]
    simp only [hfiber, lintegral_zero]
  have hmeas : Measurable (fun X₀ : SphereTuple d 2 => (X₀ 0, X₀ 1)) :=
    (measurable_pi_apply 0).prodMk (measurable_pi_apply 1)
  have hnull : (Measure.pi (fun _ : Idx 2 => σ))
      ((fun X₀ : SphereTuple d 2 => (X₀ 0, X₀ 1)) ⁻¹' B) = 0 := by
    rw [← Measure.map_apply hmeas hB,
      Perspective.map_pair_pi d 2 σ (by decide : (0 : Idx 2) ≠ 1)]
    exact hprod
  have hgood : ∀ᵐ X₀ ∂(Measure.pi (fun _ : Idx 2 => σ)),
      L.starProjection (X₀ 0 : EucSpace d) ≠ 0 →
        leadingTransverse L (X₀ 0 : EucSpace d) (X₀ 1 : EucSpace d) ≠ 0 := by
    rw [ae_iff]
    simpa only [B, Set.preimage_ofPred_eq, not_imp, not_not] using hnull
  filter_upwards [hgood] with X₀ hX₀ hp
  exact leadingTransverse_witness L _ _ (hX₀ hp)

/-- The original two-token dynamics converges to the normalized
initial leading projection outside the explicit leading-line
exception. Source: arXiv:2411.04990v2, §4, `thm2`, restricted to two
original tokens. No follower limit is assumed. -/
theorem subspace_cluster_pair_of_transverse (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X)
    (hp : L.starProjection (X 0 0 : EucSpace d) ≠ 0)
    (htrans : ∃ v ∈ L, inner ℝ v (X 0 0 : EucSpace d) = 0 ∧
      0 < inner ℝ v (X 0 1 : EucSpace d)) :
    ∀ i : Idx 2, Tendsto (fun t => (X t i : EucSpace d)) atTop
      (nhds (‖L.starProjection (X 0 0 : EucSpace d)‖⁻¹ •
        L.starProjection (X 0 0 : EucSpace d))) := by
  let p := L.starProjection (X 0 0 : EucSpace d)
  let u : SSphere d := ⟨‖p‖⁻¹ • p, mem_sphere_zero_iff_norm.mpr
    (norm_smul_inv_norm (𝕜 := ℝ) hp)⟩
  have hu : (u : EucSpace d) ∈ L := L.smul_mem _ (L.starProjection_apply_mem _)
  have hleader : Tendsto (fun t => (X t 0 : EucSpace d)) atTop (nhds (u : EucSpace d)) :=
    csaLeader_tendsto_normalized_starProjection β Q K V L lam heig hinv hdecay X hX hp
  obtain ⟨v, hv, hv0, hvpos⟩ := htrans
  have hpos := csaSecond_eventually_pos_leading_of_transverse β Q K V L lam hlam heig hinv
    u hu v hv X hX hv0 hvpos hleader
  have hsecond := csaSecond_tendsto_of_eventually_nonneg_leading β Q K V L lam hlam
    heig hinv hdecay u hu X hX hleader (hpos.mono fun _ h => h.le)
  intro i
  fin_cases i
  · exact hleader
  · exact hsecond

/-- The complete original almost-everywhere leading-subspace
conclusion holds for two tokens, arbitrary original scores, and the
exact spectral hypotheses of the source. Source: arXiv:2411.04990v2,
§4, `thm2`, restricted to two original tokens. -/
theorem subspace_cluster_twoTokens (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hdim : 2 ≤ Module.finrank ℝ L)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2) :
    ∀ P : Measure (SphereTuple d 2), Perspective.UniformTuple d 2 P →
      ∀ᵐ X₀ ∂P, ∀ X : ℝ → SphereTuple d 2, X 0 = X₀ → CSA d 2 β Q K V X →
        L.starProjection (X₀ 0 : EucSpace d) ≠ 0 →
          ∀ i : Idx 2, Tendsto (fun t => (X t i : EucSpace d)) atTop
            (nhds (‖L.starProjection (X₀ 0 : EucSpace d)‖⁻¹ •
              L.starProjection (X₀ 0 : EucSpace d))) := by
  intro P hP
  filter_upwards [ae_leading_transverse_witness L hdim P hP] with X₀ hgood
  intro X h0 hX hp
  rw [← h0] at hgood hp ⊢
  exact subspace_cluster_pair_of_transverse β Q K V L lam hlam heig hinv hdecay X hX hp (hgood hp)

/-- A genuine orthogonal original pair under the identity value map
satisfies every pointwise premise, for arbitrary original query and
key matrices. Source: arXiv:2411.04990v2, §4, `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    ∃ X : ℝ → SphereTuple 2 2,
      CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) X ∧
      ∀ i : Idx 2, Tendsto (fun t => (X t i : EucSpace 2)) atTop
        (nhds (basePoint 1 : EucSpace 2)) := by
  let u : SSphere 2 := basePoint 1
  let v : SSphere 2 := ⟨EuclideanSpace.single (1 : Fin 2) 1, by simp⟩
  let X₀ : SphereTuple 2 2 := ![u, v]
  let X := csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀
  have hX : CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) X := (csaFlow_spec _ _ _ _ _).2
  have hp : (⊤ : Submodule ℝ (EucSpace 2)).starProjection (X 0 0 : EucSpace 2) ≠ 0 := by
    simp only [X, csaFlow_zero, X₀, Matrix.cons_val_zero, Submodule.starProjection_top,
      ContinuousLinearMap.id_apply]
    exact norm_ne_zero_iff.mp (by rw [mem_sphere_zero_iff_norm.mp u.2]; norm_num)
  have ht : ∃ w ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      inner ℝ w (X 0 0 : EucSpace 2) = 0 ∧ 0 < inner ℝ w (X 0 1 : EucSpace 2) := by
    refine ⟨v, Submodule.mem_top, ?_, ?_⟩
    · simp [X, csaFlow_zero, X₀, u, v, basePoint, EuclideanSpace.inner_single_left]
    · simp [X, csaFlow_zero, X₀, v]
  have h := subspace_cluster_pair_of_transverse β Q K (ContinuousLinearMap.id ℝ _)
    ⊤ 1 one_pos (fun z _ => by simp) (fun z hz => hz) (fun z hz hz0 => by
      exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim)
    X hX hp ht
  refine ⟨X, hX, ?_⟩
  simpa [X, csaFlow_zero, X₀, u, Submodule.starProjection_top,
    ContinuousLinearMap.id_apply, norm_smul, mem_sphere_zero_iff_norm.mp (basePoint 1).2] using h

/-- The actual rotation-invariant shell law on the two-dimensional
sphere and the full leading plane satisfy all measure and dimension
premises. Source: arXiv:2411.04990v2, §4, the uniform initialization
of `thm2`. -/
example : ∃ P : Measure (SphereTuple 2 2), Perspective.UniformTuple 2 2 P ∧
    2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧ (0 : ℝ) < 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
      inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) := by
  refine ⟨Measure.pi (fun _ : Idx 2 => Perspective.shellLaw (d := 2) (by norm_num)),
    Perspective.uniformTuple_shellLaw (by norm_num), by simp, one_pos,
    fun v _ => by simp, fun z hz => hz, ?_⟩
  intro z hz hz0
  exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim

end Transformer.Causal

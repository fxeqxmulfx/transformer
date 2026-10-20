module

public import Transformer.Causal.Section4_SubspaceFollowerLimits
public import Transformer.Causal.Section4_ProjectedGenericity

/-!
# Consensus from a common positive original leading covector

ArXiv:2411.04990v2, §4, a pointwise restriction of `thm2`.
For any number of original tokens, a common positive initial leading
covector is preserved by the actual triangular leading equations.
The actual leader limit then propagates through every original
prefix. This proves actual full-tuple convergence for the stated
initialization condition. It neither assumes follower convergence
nor claims that every initialization has a common positive covector.
-/

@[expose] public section

open Real Filter MeasureTheory
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- The actual original leader's leading limit propagates to every
original token when a common original leading covector is initially
positive. Source: arXiv:2411.04990v2, §4, the causal prefix argument of
`thm1` for the explicit leading-hemisphere case of `thm2`. -/
theorem csa_tendsto_of_leader_limit_positive_covector (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (w : EucSpace d) (hw : w ∈ L) :
    ∀ (m : ℕ) (X : ℝ → SphereTuple d (m + 1)), CSA d (m + 1) β Q K V X →
      (∀ i, 0 < inner ℝ w (X 0 i : EucSpace d)) →
      Tendsto (fun t => (X t 0 : EucSpace d)) atTop (nhds (u : EucSpace d)) →
        ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (nhds (u : EucSpace d)) := by
  intro m
  induction m with
  | zero =>
    intro X hX hpos hleader i
    have hi : i = (0 : Idx (0 + 1)) := by apply Fin.ext; omega
    rw [hi]
    exact hleader
  | succ m ih =>
    intro X hX hpos hleader i
    have hpre := ih (fun t j => X t j.castSucc) (CSA_castSucc β Q K V X hX)
      (fun j => hpos j.castSucc) (by simpa only [Fin.castSucc_zero] using hleader)
    refine Fin.lastCases ?_ (fun j => hpre j) i
    exact csaLast_tendsto_of_positive_leading_covector β Q K V L lam hlam heig hinv hdecay
      u hu w hw X hX (Nat.succ_le_succ (Nat.zero_le m)) hpos hpre

/-- The actual whole nonempty original tuple converges to the actual
leader's normalized initial leading projection under the explicit
common positive leading-covector condition. All source spectral
hypotheses are retained, and the leader limit is proved from its ODE.
Source: arXiv:2411.04990v2, §4, this pointwise restriction of `thm2`. -/
theorem subspace_cluster_of_positive_leading_covector (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (w : EucSpace d) (hw : w ∈ L) (m : ℕ)
    (X : ℝ → SphereTuple d (m + 1)) (hX : CSA d (m + 1) β Q K V X)
    (hpos : ∀ i, 0 < inner ℝ w (X 0 i : EucSpace d)) :
    ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop
      (nhds (‖L.starProjection (X 0 0 : EucSpace d)‖⁻¹ •
        L.starProjection (X 0 0 : EucSpace d))) := by
  have hp : L.starProjection (X 0 0 : EucSpace d) ≠ 0 := by
    intro hz
    have he : inner ℝ w (X 0 0 : EucSpace d) =
        inner ℝ w (L.starProjection (X 0 0 : EucSpace d)) := by
      rw [← L.inner_starProjection_left_eq_right, L.starProjection_eq_self_iff.mpr hw]
    rw [hz, inner_zero_right] at he
    exact (hpos 0).ne' he
  let p := L.starProjection (X 0 0 : EucSpace d)
  let u : SSphere d := ⟨‖p‖⁻¹ • p,
    mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm (𝕜 := ℝ) hp)⟩
  have hu : (u : EucSpace d) ∈ L := L.smul_mem _ (L.starProjection_apply_mem _)
  exact csa_tendsto_of_leader_limit_positive_covector β Q K V L lam hlam heig hinv hdecay u hu
    w hw m X hX hpos
    (csaLeader_tendsto_normalized_starProjection β Q K V L lam heig hinv hdecay X hX hp)

/-- The full original almost-everywhere conclusion holds when the
original token count is at most the leading dimension. This count
restriction is additional to `thm2` and is explicit in the name and
statement. All original spectral and dynamical hypotheses remain;
the common positive covector and every token limit are proved.
Source: arXiv:2411.04990v2, §4, this restricted case of `thm2`. -/
theorem subspace_cluster_of_count_le_finrank (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (n : ℕ) (hn : 1 ≤ n) (hcount : n ≤ Module.finrank ℝ L) :
    ∀ P : Measure (SphereTuple d n), Perspective.UniformTuple d n P →
      ∀ᵐ X₀ ∂P, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → CSA d n β Q K V X →
        ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop
          (nhds (‖L.starProjection (X₀ ⟨0, hn⟩ : EucSpace d)‖⁻¹ •
            L.starProjection (X₀ ⟨0, hn⟩ : EucSpace d))) := by
  intro P hP
  filter_upwards [ae_exists_positive_leading_covector L hn hcount P hP] with X₀ hgood
  intro X h0 hX
  obtain ⟨w, hw, hpos⟩ := hgood
  cases n with
  | zero => omega
  | succ m =>
    have hpos' : ∀ i, 0 < inner ℝ w (X 0 i : EucSpace d) := by simpa only [h0] using hpos
    have h := subspace_cluster_of_positive_leading_covector β Q K V L lam hlam heig hinv hdecay
      w hw m X hX hpos'
    have hi : (⟨0, hn⟩ : Idx (m + 1)) = 0 := rfl
    simpa only [h0, hi] using h

/-- Three distinct original unit coordinate vectors have a common
positive leading covector, and their actual three-token flow has the
asserted leader limit for arbitrary original scores.
Source: arXiv:2411.04990v2, §4, the leading-hemisphere case of `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 3) :
    ∃ X : ℝ → SphereTuple 3 3,
      CSA 3 3 β Q K (ContinuousLinearMap.id ℝ _) X ∧
      (∃ w : EucSpace 3, ∀ i, 0 < inner ℝ w (X 0 i : EucSpace 3)) ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace 3)) atTop (nhds (X 0 0 : EucSpace 3)) := by
  let X₀ : SphereTuple 3 3 := fun i => ⟨EuclideanSpace.single i 1, by simp⟩
  let w : EucSpace 3 := WithLp.toLp 2 (fun _ => (1 : ℝ))
  let X := csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀
  have hX : CSA 3 3 β Q K (ContinuousLinearMap.id ℝ _) X := (csaFlow_spec _ _ _ _ _).2
  have hp (i : Idx 3) : 0 < inner ℝ w (X 0 i : EucSpace 3) := by
    simp [X, csaFlow_zero, X₀, w, EuclideanSpace.inner_single_right]
  refine ⟨X, hX, ⟨w, hp⟩, ?_⟩
  have h := subspace_cluster_of_positive_leading_covector β Q K (ContinuousLinearMap.id ℝ _)
    ⊤ 1 one_pos (fun v _ => by simp) (fun z hz => hz) (fun z hz hz0 => by
      exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim)
    w Submodule.mem_top 2 X hX hp
  simpa only [Submodule.starProjection_top, ContinuousLinearMap.id_apply,
    mem_sphere_zero_iff_norm.mp (X 0 0).2, inv_one, one_smul] using h

/-- The actual three-token shell initialization and identity values in
the leading three-dimensional space jointly realize all almost-sure
count, measure and source spectral premises.
Source: arXiv:2411.04990v2, §4, the count-restricted case of `thm2`. -/
example : ∃ P : Measure (SphereTuple 3 3), Perspective.UniformTuple 3 3 P ∧
    (1 : ℕ) ≤ 3 ∧ 3 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 3)) ∧ (0 : ℝ) < 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 3)),
      (ContinuousLinearMap.id ℝ (EucSpace 3)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 3))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 3)) z ∈ (⊤ : Submodule ℝ (EucSpace 3))ᗮ) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 3))ᗮ, z ≠ 0 →
      inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 3)) z) z < (1 : ℝ) * ‖z‖ ^ 2) := by
  refine ⟨Measure.pi (fun _ : Idx 3 => Perspective.shellLaw (d := 3) (by norm_num)),
    Perspective.uniformTuple_shellLaw (by norm_num), by norm_num, by simp, one_pos,
    fun v _ => by simp, fun z hz => hz, ?_⟩
  intro z hz hz0
  exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim

end Transformer.Causal

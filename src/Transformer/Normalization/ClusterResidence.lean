/-
# Global residence in a small token cluster

Sphere residence, cap residence and nondecreasing radii used in
Appendix C, proof of Theorem 4.3 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.ClusterCap
import Transformer.Normalization.ClusterGeometry

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- On an interval of positive pairwise alignment and radii, every scheme
preserves unit directions, preserves a positive initial cap, and has
nondecreasing radii. Source: arXiv:2510.22026v2, equation (NA), Table 2 and
Appendix C, proof of Theorem 4.3. -/
theorem scheme_cluster_history (β m q τ T : ℝ) (hm : 0 < m) (hq : 0 < q)
    (hT : 0 ≤ T) (Q K : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (scheme : Scheme)
    (w : EucSpace d) (hα : ∀ t, 0 < α t)
    (hunit : ∀ j, ‖θ 0 j‖ = 1) (hcap : ∀ j, q ≤ inner (𝕜 := ℝ) w (θ 0 j))
    (hpair : ∀ t ∈ Icc 0 T, ∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k))
    (hr : ∀ t ∈ Icc 0 T, ∀ j, 0 < r t j)
    (hdyn : SchemeDynamics d n β Q K (idParams d) α τ scheme θ r) :
    (∀ t ∈ Icc 0 T, ∀ j, ‖θ t j‖ = 1) ∧
      (∀ j, q ≤ inner (𝕜 := ℝ) w (θ T j)) ∧ (∀ j, r 0 j ≤ r T j) := by
  let a : ℝ → Idx n → ℝ := fun t j =>
    (speedFactor d n β Q K (idParams d) α θ r τ scheme t j)⁻¹
  have ha : ∀ t ∈ Icc 0 T, ∀ j, 0 ≤ a t j := by
    intro t ht j
    exact (inv_pos.mpr (speedFactor_pos_of_positive_cluster β m τ t hm ht.1 Q K α θ r
      scheme (hα t) (hr t ht) (hpair t ht) j)).le
  have hnorm : ∀ t ∈ Icc 0 T, ∀ j, ‖θ t j‖ = 1 := by
    intro t ht j
    apply norm_eq_one_of_nonnegative_radial_flow t ht.1 (fun u => θ u j) (fun u => a u j)
      (fun u => attentionVec d n β (Q u) (K u) (idParams d u) (θ u) j)
    · exact fun s hs => (hdyn s hs.1 j).1
    · exact fun s hs => ha s ⟨hs.1, hs.2.trans ht.2⟩ j
    · intro s hs
      exact hm.le.trans (inner_attentionVec_id_ge β m (Q s) (K s) (θ s)
        (hpair s ⟨hs.1, hs.2.trans ht.2⟩) j)
    · exact hunit j
  refine ⟨hnorm, cap_invariant_of_unit_attention_flow β q T hq hT Q K θ a w hnorm
    (fun t ht j => (hdyn t ht.1 j).1) ha hcap, ?_⟩
  intro j
  have hc : ContinuousOn (fun t => r t j) (Icc 0 T) := fun t ht =>
    ((hdyn t ht.1 j).2.continuousWithinAt).mono Icc_subset_Ici_self
  have hd : ∀ t ∈ Ioo 0 T, HasDerivAt (fun t => r t j)
      (radialDerivative d n β Q K (idParams d) θ τ scheme t j) t :=
    fun t ht => (hdyn t ht.1.le j).2.hasDerivAt (Ici_mem_nhds ht.1)
  have hmono : MonotoneOn (fun t => r t j) (Icc 0 T) :=
    monotoneOn_of_deriv_nonneg (convex_Icc 0 T) hc
      (by
        rw [interior_Icc]
        exact fun t ht => (hd t ht).differentiableAt.differentiableWithinAt)
      (by
        rw [interior_Icc]
        intro t ht
        rw [(hd t ht).deriv]
        exact radialDerivative_nonneg_of_positive_cluster β m τ t hm.le Q K θ scheme
          (hpair t (Ioo_subset_Icc_self ht)) j)
  exact hmono (left_mem_Icc.mpr hT) (right_mem_Icc.mpr hT) hT

/-- A stationary unit singleton with Post-LN and unit radii satisfies all
the interval assumptions. Source: arXiv:2510.22026v2, equation (NA). -/
example :
    let θ : ℝ → Idx 1 → EucSpace 1 := fun _ _ => EuclideanSpace.single (0 : Fin 1) 1
    SchemeDynamics 1 1 1 (idParams 1) (idParams 1) (idParams 1) (fun _ => 1) 0 .post θ
      (fun _ _ => 1) ∧ (∀ j, ‖θ 0 j‖ = 1) ∧
      (∀ t ∈ Icc (0 : ℝ) 1, ∀ j k, (1 : ℝ) ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) := by
  dsimp
  refine ⟨fun t _ j => ⟨?_, ?_⟩, fun _ => by simp, fun _ _ _ _ => by simp⟩
  · simpa [speedFactor, attentionVec, idParams, proj] using
      (hasDerivAt_const t (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))).hasDerivWithinAt
  · exact (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt

/-- Small-cone initial data stays on the sphere, in a positive cluster,
with nondecreasing positive radii under all six schemes. The common initial
cap gives the uniform bound `1-4δ` along the trajectory.
Source: arXiv:2510.22026v2, equation (NA), Table 2 and Appendix C, proof of
Theorem 4.3. -/
theorem scheme_cluster_invariant (hn : 0 < n) (β δ τ : ℝ) (hδ : 0 ≤ δ)
    (hsmall : δ < 1 / 4) (Q K : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (scheme : Scheme)
    (hα : ∀ t, 0 < α t) (hunit : ∀ j, ‖θ 0 j‖ = 1) (hr : ∀ j, 0 < r 0 j)
    (hcone : ∀ j k, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k))
    (hdyn : SchemeDynamics d n β Q K (idParams d) α τ scheme θ r) :
    ∀ T, 0 ≤ T → (∀ j, ‖θ T j‖ = 1) ∧
      (∀ j k, 1 - 4 * δ ≤ inner (𝕜 := ℝ) (θ T j) (θ T k)) ∧
      (∀ j, r 0 j ≤ r T j) := by
  let w := θ 0 ⟨0, hn⟩
  let m : ℝ := 1 - 4 * δ
  have hm : 0 < m := by dsimp [m]; linarith
  have hq : 0 < 1 - δ := by linarith
  intro T hT
  let f : (Idx n × Idx n) ⊕ Idx n → ℝ → ℝ := fun i t =>
    match i with
    | .inl (j, k) => inner (𝕜 := ℝ) (θ t j) (θ t k) - m / 2
    | .inr j => r t j - r 0 j / 2
  have hcθ : ∀ j, ContinuousOn (fun t => θ t j) (Icc 0 T) := fun j t ht =>
    ((hdyn t ht.1 j).1.continuousWithinAt).mono Icc_subset_Ici_self
  have hcr : ∀ j, ContinuousOn (fun t => r t j) (Icc 0 T) := fun j t ht =>
    ((hdyn t ht.1 j).2.continuousWithinAt).mono Icc_subset_Ici_self
  have hc : ∀ i, ContinuousOn (f i) (Icc 0 T) := by
    intro i
    cases i with
    | inl p => exact ((hcθ p.1).inner (hcθ p.2)).sub continuousOn_const
    | inr j => exact (hcr j).sub continuousOn_const
  have h0 : ∀ i, 0 < f i 0 := by
    intro i
    cases i with
    | inl p =>
      dsimp [f, m]
      linarith [hcone p.1 p.2]
    | inr j => dsimp [f]; linarith [hr j]
  have hest : ∀ t ∈ Icc 0 T, (∀ s ∈ Icc 0 t, ∀ i, 0 ≤ f i s) →
      (∀ j, ‖θ t j‖ = 1) ∧ (∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) ∧
        (∀ j, r 0 j ≤ r t j) := by
    intro t ht hpast
    have hp : ∀ s ∈ Icc 0 t, ∀ j k, m / 2 ≤ inner (𝕜 := ℝ) (θ s j) (θ s k) := by
      intro s hs j k
      have h := hpast s hs (.inl (j, k))
      dsimp [f] at h
      linarith
    have hrad : ∀ s ∈ Icc 0 t, ∀ j, 0 < r s j := by
      intro s hs j
      have h := hpast s hs (.inr j)
      dsimp [f] at h
      linarith [hr j]
    obtain ⟨hu, hcap, hrmono⟩ := scheme_cluster_history β (m / 2) (1 - δ) τ t
      (by positivity) hq ht.1 Q K α θ r scheme w hα hunit (hcone ⟨0, hn⟩) hp hrad hdyn
    refine ⟨hu t (right_mem_Icc.mpr ht.1), ?_, hrmono⟩
    intro j k
    exact inner_ge_of_common_cap δ (θ t j) (θ t k) w
      (hu t (right_mem_Icc.mpr ht.1) j) (hu t (right_mem_Icc.mpr ht.1) k)
      (hunit ⟨0, hn⟩) (hcap j) (hcap k)
  have hstep : ∀ t ∈ Icc 0 T, (∀ s ∈ Icc 0 t, ∀ i, 0 ≤ f i s) →
      ∀ i, 0 < f i t := by
    intro t ht hpast i
    obtain ⟨_, hp, hrmono⟩ := hest t ht hpast
    cases i with
    | inl p => dsimp [f]; linarith [hp p.1 p.2]
    | inr j => dsimp [f]; linarith [hrmono j, hr j]
  have hall := finite_pos_of_history_pos f T hc h0 hstep
  exact hest T (right_mem_Icc.mpr hT) (fun s hs i => (hall s hs i).le)

/-- A stationary Post-LN singleton provides initial unit directions,
positive radii and a zero-width local cone. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    let θ : ℝ → Idx 1 → EucSpace 1 := fun _ _ => u
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 4 ∧ (∀ j, ‖θ 0 j‖ = 1) ∧
      (0 : ℝ) < 1 ∧
      (∀ j k, 1 - (0 : ℝ) ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) ∧
      SchemeDynamics 1 1 1 (idParams 1) (idParams 1) (idParams 1) (fun _ => 1) 0
        .post θ (fun _ _ => 1) := by
  dsimp
  refine ⟨le_rfl, by norm_num, fun _ => by simp, one_pos,
    fun _ _ => by simp, fun t _ j => ⟨?_, ?_⟩⟩
  · simpa [speedFactor, attentionVec, idParams, proj] using
      (hasDerivAt_const t (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))).hasDerivWithinAt
  · exact (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt

end Transformer.Normalization

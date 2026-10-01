/-
# Finite barriers and spherical cap invariance

The invariant geometry used in Appendix C, proof of Theorem 4.3 of
arXiv:2510.22026v2. Continuity and a receding strict boundary avoid
regularity assumptions on the attention parameters.
-/

import Transformer.Normalization.ClusterFlow
import Transformer.Normalization.Velocities
import Transformer.Metastability.PropagationBarrier
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.Order.IntermediateValue

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- If every possible first contact with the boundary is excluded by an
estimate on the preceding interval, finitely many initially positive
constraints remain positive. Source: arXiv:2510.22026v2, Appendix C,
the local-cone hypothesis used along the proof of Theorem 4.3. -/
theorem finite_pos_of_history_pos {ι : Type*} [Fintype ι]
    (f : ι → ℝ → ℝ) (T : ℝ)
    (hf : ∀ i, ContinuousOn (f i) (Icc 0 T)) (h0 : ∀ i, 0 < f i 0)
    (hstep : ∀ t ∈ Icc 0 T, (∀ s ∈ Icc 0 t, ∀ i, 0 ≤ f i s) →
      ∀ i, 0 < f i t) :
    ∀ t ∈ Icc 0 T, ∀ i, 0 < f i t := by
  let bad : Set ℝ := ⋃ i, Icc 0 T ∩ (f i) ⁻¹' Iic 0
  have hclosed : IsClosed bad :=
    isClosed_iUnion_of_finite fun i =>
      (hf i).preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic
  have hsub : bad ⊆ Icc 0 T := by
    intro t ht
    obtain ⟨i, hi⟩ := mem_iUnion.mp ht
    exact hi.1
  intro t ht i
  by_contra hbad
  have hne : bad.Nonempty := ⟨t, mem_iUnion.mpr ⟨i, ht, not_lt.mp hbad⟩⟩
  obtain ⟨a, ha, hleast⟩ :=
    (isCompact_Icc.of_isClosed_subset hclosed hsub).exists_isLeast hne
  have haI := hsub ha
  have hpast : ∀ s ∈ Icc 0 a, ∀ j, 0 ≤ f j s := by
    intro s hs j
    by_contra hneg
    have hsI : s ∈ Icc 0 T := ⟨hs.1, hs.2.trans haI.2⟩
    have hc : ContinuousOn (f j) (Icc 0 s) :=
      (hf j).mono (Icc_subset_Icc le_rfl hsI.2)
    obtain ⟨u, hu, huz⟩ :=
      intermediate_value_Icc' hs.1 hc ⟨(not_le.mp hneg).le, (h0 j).le⟩
    have hus : u < s := lt_of_le_of_ne hu.2 (by
      intro heq
      rw [heq] at huz
      exact (not_le.mp hneg).ne huz)
    have huI : u ∈ Icc 0 T := ⟨hu.1, hu.2.trans hsI.2⟩
    have hleastu := hleast (mem_iUnion.mpr ⟨j, huI, huz.le⟩)
    linarith [hs.2]
  obtain ⟨j, hj⟩ := mem_iUnion.mp ha
  exact (not_le_of_gt (hstep a haI hpast j)) hj.2

/-- Constant positive constraints satisfy the hypotheses on `[0,1]`. -/
example :
    (∀ i : Fin 1, ContinuousOn (fun t : ℝ => (fun _ : Fin 1 => fun _ : ℝ => (1 : ℝ)) i t)
      (Icc 0 1)) ∧
    (∀ i : Fin 1, 0 < (fun _ : Fin 1 => fun _ : ℝ => (1 : ℝ)) i 0) ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      (∀ s ∈ Icc 0 t, ∀ i : Fin 1, 0 ≤ (fun _ : Fin 1 => fun _ : ℝ => (1 : ℝ)) i s) →
        ∀ i : Fin 1, 0 < (fun _ : Fin 1 => fun _ : ℝ => (1 : ℝ)) i t) := by
  exact ⟨fun _ => continuousOn_const, fun _ => one_pos,
    fun _ _ _ _ => one_pos⟩

/-- A positive spherical cap containing the initial unit directions is
invariant under identity-value attention with nonnegative token speeds.
Source: arXiv:2510.22026v2, equation (NA) and the local-cone initialization
in Appendix C, proof of Theorem 4.3. -/
theorem cap_invariant_of_unit_attention_flow (β q T : ℝ) (hq : 0 < q) (hT : 0 ≤ T)
    (Q K : ℝ → ParamMatrix d) (θ : ℝ → Idx n → EucSpace d) (a : ℝ → Idx n → ℝ)
    (w : EucSpace d)
    (hθ : ∀ t ∈ Icc 0 T, ∀ j, ‖θ t j‖ = 1)
    (hD : ∀ t ∈ Icc 0 T, ∀ j, HasDerivWithinAt (fun u => θ u j)
      (a t j • proj d (θ t j)
        (attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j)) (Ici 0) t)
    (ha : ∀ t ∈ Icc 0 T, ∀ j, 0 ≤ a t j)
    (h0 : ∀ j, q ≤ inner (𝕜 := ℝ) w (θ 0 j)) :
    ∀ j, q ≤ inner (𝕜 := ℝ) w (θ T j) := by
  have hmargin : ∀ ε : ℝ, 0 < ε → ε * (T + 1) < q →
      ∀ j, q ≤ inner (𝕜 := ℝ) w (θ T j) + ε * (T + 1) := by
    intro ε hε hεq
    let f : Idx n → ℝ → ℝ := fun j t => inner (𝕜 := ℝ) w (θ t j) - q + ε * (t + 1)
    let A : ℝ → Idx n → EucSpace d := fun t j =>
      attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j
    let v : Idx n → ℝ → ℝ := fun j t =>
      a t j * (inner (𝕜 := ℝ) w (A t j) -
        inner (𝕜 := ℝ) (θ t j) (A t j) * inner (𝕜 := ℝ) w (θ t j)) + ε
    have hd : ∀ t ∈ Icc 0 T, ∀ j, HasDerivWithinAt (f j) (v j t) (Ici 0) t := by
      intro t ht j
      have h := (((hasDerivAt_const t w).hasDerivWithinAt).inner ℝ (hD t ht j)).sub_const q
      have he := (((hasDerivAt_id t).add_const 1).const_mul ε).hasDerivWithinAt (s := Ici 0)
      convert h.add he using 1
      · rfl
      · simp only [real_inner_smul_right, proj, inner_sub_right, inner_zero_left, add_zero]
        dsimp [v, A]
        ring
    have hc : ∀ j, ContinuousOn (f j) (Icc 0 T) := fun j t ht =>
      ((hd t ht j).continuousWithinAt).mono Icc_subset_Ici_self
    have hinit : ∀ j, 0 ≤ f j 0 := by
      intro j
      dsimp [f]
      linarith [h0 j]
    have hboundary : ∀ t ∈ Ico 0 T, (∀ k, 0 ≤ f k t) →
        ∀ j, f j t = 0 → 0 < deriv (f j) t := by
      intro t ht hnonneg j heq
      have ht0 : 0 < t := by
        by_contra h
        have htzero : t = 0 := le_antisymm (not_lt.mp h) ht.1
        rw [htzero] at heq
        dsimp [f] at heq
        linarith [h0 j]
      have htI := Ico_subset_Icc_self ht
      let B : ℝ := q - ε * (t + 1)
      have hB : 0 < B := by
        dsimp [B]
        have hmul := mul_le_mul_of_nonneg_left (show t + 1 ≤ T + 1 by linarith [ht.2]) hε.le
        linarith
      have hjB : inner (𝕜 := ℝ) w (θ t j) = B := by
        dsimp [f] at heq
        dsimp [B]
        linarith
      have hkB : ∀ k, B ≤ inner (𝕜 := ℝ) w (θ t k) := by
        intro k
        have hk := hnonneg k
        dsimp [f] at hk
        dsimp [B]
        linarith
      have hwA : B ≤ inner (𝕜 := ℝ) w (A t j) := by
        change B ≤ inner (𝕜 := ℝ) w
          (attentionVec d n β (Q t) (K t) (ContinuousLinearMap.id ℝ _) (θ t) j)
        rw [attentionVec_id_eq_sum, inner_sum]
        have h := Finset.sum_le_sum (s := Finset.univ) fun k _ =>
          mul_le_mul_of_nonneg_left (hkB k) (attentionWeight_pos β (Q t) (K t) (θ t) j k).le
        simpa only [← Finset.sum_mul, sum_attentionWeight, one_mul, real_inner_smul_right] using h
      have hθA : inner (𝕜 := ℝ) (θ t j) (A t j) ≤ 1 := by
        have hnorm : ‖A t j‖ ≤ 1 := norm_attentionVec_le_one d n β (Q t) (K t)
          (ContinuousLinearMap.id ℝ _) ContinuousLinearMap.norm_id_le (θ t) (hθ t htI) j
        have h := real_inner_le_norm (θ t j) (A t j)
        rw [hθ t htI j, one_mul] at h
        exact h.trans hnorm
      rw [(hd t htI j).hasDerivAt (Ici_mem_nhds ht0) |>.deriv]
      dsimp [v]
      rw [hjB]
      have hnonneg' : 0 ≤ inner (𝕜 := ℝ) w (A t j) - inner (𝕜 := ℝ) (θ t j) (A t j) * B := by
        nlinarith
      linarith [mul_nonneg (ha t htI j) hnonneg']
    have hall := Transformer.Metastability.finite_nonneg_of_boundary_deriv_on f 0 T hc
      ⟨hinit, hT⟩ hboundary T (right_mem_Icc.mpr hT)
    intro j
    have h := hall j
    dsimp [f] at h
    linarith
  intro j
  apply le_of_forall_pos_le_add
  intro η hη
  have hT1 : 0 < T + 1 := by linarith
  let ε := min (q / (2 * (T + 1))) (η / (T + 1))
  have hε : 0 < ε := lt_min (by positivity) (by positivity)
  have hεq : ε * (T + 1) < q := by
    have h := mul_le_mul_of_nonneg_right (min_le_left (q / (2 * (T + 1))) (η / (T + 1))) hT1.le
    have he : q / (2 * (T + 1)) * (T + 1) = q / 2 := by field_simp
    rw [he] at h
    dsimp [ε]
    linarith
  have hεη : ε * (T + 1) ≤ η := by
    have h := mul_le_mul_of_nonneg_right (min_le_right (q / (2 * (T + 1))) (η / (T + 1))) hT1.le
    simpa [ε, div_mul_cancel₀ _ hT1.ne'] using h
  linarith [hmargin ε hε hεq j]

/-- A constant unit singleton with zero speed stays in its initial cap. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ j : Idx 1,
      ‖(fun _ : ℝ => fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) t j‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ j : Idx 1,
      HasDerivWithinAt (fun _ : ℝ => EuclideanSpace.single j (1 : ℝ))
        ((0 : ℝ) • proj 1 (EuclideanSpace.single j (1 : ℝ)) 0) (Ici 0) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ j : Idx 1, 0 ≤ (fun _ : ℝ => fun _ : Idx 1 => (0 : ℝ)) t j) ∧
    ∀ j : Idx 1, (1 : ℝ) ≤ inner (𝕜 := ℝ) (EuclideanSpace.single j (1 : ℝ))
      (EuclideanSpace.single j (1 : ℝ)) := by
  refine ⟨one_pos, zero_le_one, fun _ _ _ => by simp, fun t _ j => ?_,
    fun _ _ _ => le_rfl, fun _ => by simp⟩
  simpa using (hasDerivAt_const t (EuclideanSpace.single j (1 : ℝ))).hasDerivWithinAt

end Transformer.Normalization

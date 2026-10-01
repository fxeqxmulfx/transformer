/-
# Radius bounds inside a token cluster

The radial estimates used with Table 2 in Appendix C, proof of
Theorem 4.3 of arXiv:2510.22026v2, including the initial radius and the
Mix-LN switching time.
-/

import Transformer.Normalization.ClusterResidence
import Mathlib.Analysis.Calculus.MeanValue

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- Integrating constant derivative bounds needs only derivatives of the
trajectory, not continuity of the vector field. Source: arXiv:2510.22026v2,
Appendix C, the estimates on radii in the proof of Theorem 4.3. -/
theorem linear_bounds_of_derivWithinAt (f v : ℝ → ℝ) (a b L U : ℝ)
    (ha : 0 ≤ a) (hab : a ≤ b)
    (hd : ∀ t ∈ Icc a b, HasDerivWithinAt f (v t) (Ici 0) t)
    (hv : ∀ t ∈ Ico a b, L ≤ v t ∧ v t ≤ U) :
    f a + L * (b - a) ≤ f b ∧ f b ≤ f a + U * (b - a) := by
  have hupper : ∀ (g w : ℝ → ℝ) (C : ℝ),
      (∀ t ∈ Icc a b, HasDerivWithinAt g (w t) (Ici 0) t) →
      (∀ t ∈ Ico a b, w t ≤ C) → g b ≤ g a + C * (b - a) := by
    intro g w C hg hw
    have hc : ContinuousOn g (Icc a b) := fun t ht =>
      ((hg t ht).continuousWithinAt).mono (fun _ hu => ha.trans hu.1)
    have hB : ∀ t ∈ Ico a b, HasDerivWithinAt (fun u => g a + C * (u - a)) C
        (Ici t) t := by
      intro t _
      convert ((hasDerivAt_const t (g a)).add
        (((hasDerivAt_id t).sub_const a).const_mul C)).hasDerivWithinAt using 1
      · rfl
      · simp
    exact image_le_of_deriv_right_le_deriv_boundary hc
      (fun t ht => (hg t (Ico_subset_Icc_self ht)).mono (Ici_subset_Ici.mpr (ha.trans ht.1)))
      (by simp) (by fun_prop) hB hw (right_mem_Icc.mpr hab)
  have hlo := hupper (fun t => -f t) (fun t => -v t) (-L)
    (fun t ht => (hd t ht).neg) (fun t ht => neg_le_neg (hv t ht).1)
  exact ⟨by linarith, hupper f v U hd (fun t ht => (hv t ht).2)⟩

/-- The affine function `f(t)=t` satisfies the derivative bounds. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, HasDerivWithinAt (fun t : ℝ => t) 1 (Ici 0) t) ∧
    ∀ t ∈ Ico (0 : ℝ) 1, (1 : ℝ) ≤ (fun _ : ℝ => (1 : ℝ)) t ∧
      (fun _ : ℝ => (1 : ℝ)) t ≤ 1 :=
  ⟨le_rfl, zero_le_one, fun t _ => (hasDerivAt_id t).hasDerivWithinAt,
    fun _ _ => ⟨le_rfl, le_rfl⟩⟩

/-- On the sphere inside a positive cluster, all radial derivatives lie
in `[0,1]`; those of Pre-LN and Peri-LN, and of Mix-LN after the switch,
lie in `[m,1]`. Source: arXiv:2510.22026v2, Table 2 and Appendix C, proof
of Theorem 4.3. -/
theorem radialDerivative_bounds (β m τ t : ℝ) (hm : 0 < m)
    (Q K : ℝ → ParamMatrix d) (θ : ℝ → Idx n → EucSpace d) (scheme : Scheme)
    (hunit : ∀ j, ‖θ t j‖ = 1)
    (hpair : ∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) (j : Idx n) :
    let v := radialDerivative d n β Q K (idParams d) θ τ scheme t j
    (0 ≤ v ∧ v ≤ 1) ∧
      ((scheme = .pre ∨ scheme = .peri ∨ (scheme = .mix ∧ τ < t)) → m ≤ v) := by
  let A := attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j
  have hinner : m ≤ inner (𝕜 := ℝ) (θ t j) A :=
    inner_attentionVec_id_ge β m (Q t) (K t) (θ t) hpair j
  have hA : ‖A‖ ≤ 1 := norm_attentionVec_le_one d n β (Q t) (K t)
    (ContinuousLinearMap.id ℝ _) ContinuousLinearMap.norm_id_le (θ t) hunit j
  have hcs : inner (𝕜 := ℝ) (θ t j) A ≤ ‖A‖ := by
    simpa [hunit j] using real_inner_le_norm (θ t j) A
  have hpos : 0 < ‖A‖ := hm.trans_le (hinner.trans hcs)
  have hdiv : m ≤ inner (𝕜 := ℝ) (θ t j) A / ‖A‖ := by
    apply (le_div_iff₀ hpos).mpr
    exact (mul_le_mul_of_nonneg_left hA hm.le).trans (by simpa using hinner)
  have hdiv1 : inner (𝕜 := ℝ) (θ t j) A / ‖A‖ ≤ 1 := by
    exact (div_le_iff₀ hpos).mpr (by simpa using hcs)
  cases scheme <;> simp only [radialDerivative]
  · exact ⟨⟨le_rfl, zero_le_one⟩, by simp⟩
  · exact ⟨⟨hm.le.trans hinner, hcs.trans hA⟩, fun _ => hinner⟩
  · split_ifs with hswitch
    · exact ⟨⟨hm.le.trans hinner, hcs.trans hA⟩, fun _ => hinner⟩
    · exact ⟨⟨le_rfl, zero_le_one⟩, by simp [hswitch]⟩
  · exact ⟨⟨hm.le.trans hdiv, hdiv1⟩, fun _ => hdiv⟩
  · exact ⟨⟨le_rfl, zero_le_one⟩, by simp⟩
  · exact ⟨⟨le_rfl, zero_le_one⟩, by simp⟩

/-- A unit singleton supplies the unit and positive-cluster assumptions. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    (0 : ℝ) < 1 ∧ (∀ j : Idx 1, ‖(fun _ : Idx 1 => u) j‖ = 1) ∧
      ∀ j k : Idx 1, (1 : ℝ) ≤ inner (𝕜 := ℝ) ((fun _ : Idx 1 => u) j)
        ((fun _ : Idx 1 => u) k) := by simp

/-- Every radius grows at most linearly. In Pre-LN and Peri-LN it grows
at least as `m t`, and in Mix-LN at least as `m(t-a)` after
`a=max(τ,0)+1`. Source: arXiv:2510.22026v2, Table 2 and Appendix C, proof
of Theorem 4.3. -/
theorem scheme_radius_growth (β m τ : ℝ) (hm : 0 < m)
    (Q K : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (scheme : Scheme)
    (hunit : ∀ t, 0 ≤ t → ∀ j, ‖θ t j‖ = 1)
    (hpair : ∀ t, 0 ≤ t → ∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k))
    (hrmono : ∀ t, 0 ≤ t → ∀ j, r 0 j ≤ r t j)
    (hdyn : SchemeDynamics d n β Q K (idParams d) α τ scheme θ r) :
    ∀ t, 0 ≤ t → ∀ j,
      r t j ≤ r 0 j + t ∧
      ((scheme = .pre ∨ scheme = .peri) → r 0 j + m * t ≤ r t j) ∧
      ((scheme = .mix ∧ max τ 0 + 1 ≤ t) →
        r 0 j + m * (t - (max τ 0 + 1)) ≤ r t j) := by
  intro t ht j
  have hupper := (linear_bounds_of_derivWithinAt (fun u => r u j)
    (fun u => radialDerivative d n β Q K (idParams d) θ τ scheme u j) 0 t 0 1 le_rfl ht
    (fun u hu => (hdyn u hu.1 j).2)
    (fun u hu => (radialDerivative_bounds β m τ u hm Q K θ scheme
      (hunit u hu.1) (hpair u hu.1) j).1)).2
  refine ⟨by simpa using hupper, ?_, ?_⟩
  · intro hs
    have hlow := (linear_bounds_of_derivWithinAt (fun u => r u j)
      (fun u => radialDerivative d n β Q K (idParams d) θ τ scheme u j) 0 t m 1 le_rfl ht
      (fun u hu => (hdyn u hu.1 j).2)
      (fun u hu => by
        have hb := radialDerivative_bounds β m τ u hm Q K θ scheme
          (hunit u hu.1) (hpair u hu.1) j
        exact ⟨hb.2 (hs.elim Or.inl (fun h => Or.inr (Or.inl h))), hb.1.2⟩)).1
    simpa using hlow
  · rintro ⟨hs, hswitch⟩
    have ha : 0 ≤ max τ 0 + 1 := by linarith [le_max_right τ 0]
    have hlow := (linear_bounds_of_derivWithinAt (fun u => r u j)
      (fun u => radialDerivative d n β Q K (idParams d) θ τ scheme u j)
      (max τ 0 + 1) t m 1 ha hswitch
      (fun u hu => (hdyn u (ha.trans hu.1) j).2)
      (fun u hu => by
        have hu0 := ha.trans hu.1
        have hb := radialDerivative_bounds β m τ u hm Q K θ scheme
          (hunit u hu0) (hpair u hu0) j
        exact ⟨hb.2 (Or.inr (Or.inr ⟨hs, by linarith [le_max_left τ 0, hu.1]⟩)), hb.1.2⟩)).1
    linarith [hrmono (max τ 0 + 1) ha j]

/-- Unit directions, unit radii and Post-LN give a stationary admissible
trajectory; the lower-growth implications then concern the other schemes. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    let θ : ℝ → Idx 1 → EucSpace 1 := fun _ _ => u
    (∀ t : ℝ, 0 ≤ t → ∀ j, ‖θ t j‖ = 1) ∧
      (∀ t : ℝ, 0 ≤ t → ∀ j k, (1 : ℝ) ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) ∧
      SchemeDynamics 1 1 1 (idParams 1) (idParams 1) (idParams 1) (fun _ => 1) 0
        .post θ (fun _ _ => 1) := by
  dsimp
  refine ⟨fun _ _ _ => by simp, fun _ _ _ _ => by simp, fun t _ j => ⟨?_, ?_⟩⟩
  · simpa [speedFactor, attentionVec, idParams, proj] using
      (hasDerivAt_const t (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))).hasDerivWithinAt
  · exact (hasDerivAt_const t (1 : ℝ)).hasDerivWithinAt

end Transformer.Normalization

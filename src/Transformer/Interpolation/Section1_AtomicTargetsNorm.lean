/-
# The norm estimate for original atomic-target interpolation

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, with the norm estimate
of its proof in §5: `‖θ₁‖ ≲ dN/T` "where the implicit constant depends
only on the supports of the initial measures", `‖θ₂‖ ≲ log 1/δ` for
clustering to accuracy `δ`, and `‖θ₃‖ ≲ N/T` for matching.

The quadrant preprocessing of inputs and targets, the matching of every
law into a cap of one fixed radius `r` around its quadrant target, and
the inverse quadrant phases do not depend on the accuracy. Between
matching and inversion, one cap contraction per law brings it within
`δ ≍ ε` of its quadrant target at a clock `O(1 + log 1/δ)`.

At a horizon `T` the whole sequence costs at most a data-dependent
constant times `1/T + 1 + log(1/δ)`: the fixed phases scale linearly
and the contractions with the square root of their clock. For
`ε ≤ 1/2` this is a data-dependent constant times `dN/T + log(1/ε)`.
There are at most `14N + 5 ≤ 7dN` pieces.
-/

import Transformer.Interpolation.Section1_PhaseRealizationCosts
import Transformer.Interpolation.Section1_AtomicTargetsExistence
import Transformer.Interpolation.Section1_InverseQuadrantLipschitz

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Distinct original input laws with a support hole are sent within
`ε` of distinct unit targets at every horizon `T`, in at most `7dN`
pieces, with complete parameter norm at most a constant times
`dN/T + log(1/ε)` when `ε ≤ 1/2`; the constant depends on the data
only. A solution from each input exists, and every solution ends
within `ε` of its target. Source: arXiv:2411.04551v3, §1,
`thm: targets.atoms`, and its norm estimate in §5. -/
theorem exists_atomic_target_norm_controls (hd : 3 ≤ d) (hN : 1 ≤ N)
    (μ : Idx N → Perspective.ProbSphere d) (hinj : Function.Injective μ)
    (hhole : (⋃ k, (μ k : Measure (SSphere d)).support) ≠ Set.univ)
    (Y : Idx N → SSphere d) (hYinj : Function.Injective Y) :
    ∃ Cnorm : ℝ, ∀ T ε : ℝ, 0 < T → 0 < ε →
      ∃ (θ : TimeParams d) (K : ℕ),
        K ≤ 7 * d * N ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
        (ε ≤ 1 / 2 → ∀ s, parameterCost (θ s) ≤ Cnorm * ((d * N : ℝ) / T + log (1 / ε))) ∧
        (∀ k, ∃ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ k ∧ cauchyPB d θ ν) ∧
        ∀ k (ν : ℝ → Perspective.ProbSphere d), ν 0 = μ k → cauchyPB d θ ν →
          W2 d (ν T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨A, hAlen, hA, hAQ⟩ := exists_quadrant_preprocessing_phases (by omega : 2 ≤ d) i μ hhole
  let ν₀ (k : Idx N) := phaseSequenceLaw A (μ k)
  have hν₀inj : Function.Injective ν₀ := fun k l he =>
    hinj ((bijective_phaseSequenceLaw A (fun P hP => (hA P hP).1)).1 he)
  obtain ⟨w, _, hw⟩ := exists_finite_atomic_missing_point hd Y
  obtain ⟨w₁, w₂, τ₁, τ₂, _, _, hYQ⟩ := exists_two_projection_quadrant (by omega : 2 ≤ d)
    (Set.range Y) (Set.finite_range Y).isCompact fun he => hw (he.symm ▸ Set.mem_univ _)
  let Y' (k : Idx N) := sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))
  have hY'inj : Function.Injective Y' := fun k l he =>
    hYinj (injective_twoProjection_map w₁ w₂ τ₁ τ₂ he)
  obtain ⟨r, hr, hr1, hsep⟩ := exists_cap_separation_radius Y' hY'inj
  obtain ⟨M, hMlen, hM, hMfinal⟩ := exists_quadrant_target_phases hd hN i ν₀ hAQ hν₀inj
    Y' hY'inj (fun k => hYQ (Y k) ⟨k, rfl⟩) r hr
  let B := inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)
  have hBlen : B.length = 2 := by rw [inversePhaseList_length]; rfl
  obtain ⟨A₁, hA₁, hinv⟩ := exists_linear_inverse_twoProjection_radius i w₁ w₂ τ₁ τ₂
  have hc : 0 ≤ 1 - r := by linarith
  obtain ⟨C₀, hC₀, hclock⟩ :=
    exists_logarithmic_center_clock (d := d) (1 - r ^ 2 / 2) (1 - r) hc (by nlinarith)
  have hF : ∀ P ∈ A ++ M ++ B, IsAutonomousPhase P ∧ P.1.B = 0 := by
    intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · rcases List.mem_append.mp hP with hP | hP
      · exact hA P hP
      · exact hM P hP
    · exact inversePhaseList_valid _ (twoProjectionPhaseList_valid i w₁ w₂ τ₁ τ₂) P hP
  obtain ⟨C₁, hC₁, hreal⟩ :=
    exists_fixed_and_cap_realization_constant i (A ++ M ++ B) hF Y' (1 - r) C₀ hc hC₀.le
  obtain ⟨G, hG⟩ : ∃ G : ℝ, G = 1 + log A₁ + log (1 / r) := ⟨_, rfl⟩
  refine ⟨2 ^ (A.length + M.length + N + B.length) * C₁ * (G / log 2 + 1), ?_⟩
  intro T ε hT hε
  obtain ⟨ε', hε'⟩ : ∃ e : ℝ, e = min ε (1 / 2) := ⟨_, rfl⟩
  have hε'0 : 0 < ε' := hε' ▸ lt_min hε (by norm_num)
  have hε'ε : ε' ≤ ε := hε' ▸ min_le_left _ _
  obtain ⟨hδ₁, hδ₁1, _, hback⟩ := hinv ε' hε'0 (by linarith [hε' ▸ min_le_right ε (1 / 2)])
  obtain ⟨δ, hδ⟩ : ∃ x : ℝ, x = min (ε' / A₁) r := ⟨_, rfl⟩
  have hδ0 : 0 < δ := hδ ▸ lt_min hδ₁ hr
  have hδr : δ ≤ r := hδ ▸ min_le_right _ _
  have hlogδ : 0 ≤ log (1 / δ) := log_nonneg (one_le_one_div hδ0 (hδr.trans hr1))
  obtain ⟨τ, hτ, hτle, hcontract⟩ := hclock δ hδ0 (hδr.trans hr1)
  let L := A ++ M ++ capContractionPhases i Y' (1 - r) τ (List.finRange N) ++ B
  have hLreal : ∀ P ∈ L, ∀ H : ℝ, 0 < H → ∃ R : AutonomousPhase d,
      IsAutonomousPhase R ∧ R.1.B = 0 ∧ (∀ ρ, R.2 ρ H = P.2 ρ 1) ∧
        parameterCost R.1 ≤ C₁ * (1 / H + 1 + log (1 / δ)) := by
    intro P hP
    refine hreal τ _ hτ.le hlogδ hτle (List.finRange N) P ?_
    rcases List.mem_append.mp hP with hP | hP
    · rcases List.mem_append.mp hP with hP | hP
      · exact Or.inl (List.mem_append_left B hP)
      · exact Or.inr hP
    · exact Or.inl (List.mem_append_right (A ++ M) hP)
  obtain ⟨_, θ, _, _, _, hpc, hBθ, hnorm, hex, _, hend⟩ :=
    exists_bounded_phase_sequence_controls i L C₁ (log (1 / δ)) hC₁ hlogδ hLreal T hT
  have hLlen : L.length = A.length + M.length + N + B.length := by
    simp only [L, List.length_append, capContractionPhases_length, List.length_finRange]
  refine ⟨θ, L.length + 1, ?_, hpc, hBθ, ?_, fun k => hex (μ k), ?_⟩
  · rw [hLlen, hAlen, hBlen]
    nlinarith [Nat.mul_le_mul_right N hd]
  · intro hε2 s
    have hεe : ε' = ε := hε'.trans (min_eq_left hε2)
    rw [hεe] at hδ₁ hδ₁1 hδ
    have hlog2 : 0 < log (2 : ℝ) := log_pos (by norm_num)
    have hlogε : log 2 ≤ log (1 / ε) := log_le_log two_pos (by rw [le_div_iff₀ hε]; linarith)
    have hGnn : 0 ≤ G := by
      rw [hG]
      linarith [log_nonneg hA₁, log_nonneg (one_le_one_div hr hr1)]
    have hA₁0 : 0 < A₁ := by linarith
    have hpos : 0 < ε / A₁ * r := mul_pos hδ₁ hr
    have hδlow : ε / A₁ * r ≤ δ := by
      rw [hδ]
      exact le_min (mul_le_of_le_one_right hδ₁.le hr1) (mul_le_of_le_one_left hr.le hδ₁1)
    have hlogδle : log (1 / δ) ≤ log A₁ + log (1 / ε) + log (1 / r) := by
      have h1 : 1 / δ ≤ A₁ * (1 / ε) * (1 / r) := by
        calc 1 / δ ≤ 1 / (ε / A₁ * r) := one_div_le_one_div_of_le hpos hδlow
          _ = A₁ * (1 / ε) * (1 / r) := by field_simp
      calc log (1 / δ) ≤ log (A₁ * (1 / ε) * (1 / r)) := log_le_log (one_div_pos.mpr hδ0) h1
        _ = log A₁ + log (1 / ε) + log (1 / r) := by
          rw [log_mul (by positivity) (by positivity), log_mul (by positivity) (by positivity)]
    have hdN : (1 : ℝ) ≤ d * N := by
      have h1 : (3 : ℝ) ≤ d := by exact_mod_cast hd
      have h2 : (1 : ℝ) ≤ N := by exact_mod_cast hN
      nlinarith
    have hT1 : 1 / T ≤ (d * N : ℝ) / T := div_le_div_of_nonneg_right hdN hT.le
    have hGε : G ≤ G / log 2 * log (1 / ε) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hlog2]
      exact mul_le_mul_of_nonneg_left hlogε hGnn
    have hdNT : 0 ≤ (d * N : ℝ) / T := by positivity
    have hGl : 0 ≤ G / log 2 := div_nonneg hGnn hlog2.le
    have hsum : 1 / T + 1 + log (1 / δ) ≤ (G / log 2 + 1) * ((d * N : ℝ) / T + log (1 / ε)) := by
      nlinarith [mul_nonneg hGl hdNT]
    calc parameterCost (θ s) ≤ 2 ^ L.length * C₁ * (1 / T + 1 + log (1 / δ)) := hnorm s
      _ ≤ 2 ^ L.length * C₁ * ((G / log 2 + 1) * ((d * N : ℝ) / T + log (1 / ε))) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = _ := by rw [hLlen]; ring
  · intro k ν h0 hν
    rw [hend ν hν, h0]
    change W2 d (phaseSequenceLaw (A ++ M ++ capContractionPhases i Y' (1 - r) τ (List.finRange N)
      ++ B) (μ k) : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε
    rw [phaseSequenceLaw_append, phaseSequenceLaw_append, phaseSequenceLaw_append]
    have hCs := support_capContractionPhases i Y' r δ (1 - r ^ 2 / 2) (1 - r) τ hτ.le hδr hsep
      (fun k x hx => le_inner_of_dist_le x (Y' k) r hx) (fun k ν hν => hcontract (Y' k) ν hν)
      (List.finRange N) k _ (hMfinal k) (Or.inl (List.mem_finRange k))
    have hfin := hback (Y k) _ (hCs.trans (Metric.closedBall_subset_closedBall
      (hδ ▸ min_le_left _ _)))
    exact (W2_le_radius_of_support _ (Y k) ε' hε'0.le hfin).trans hε'ε

/-- One original Dirac input and one unit target satisfy every
hypothesis: the dimension and count bounds, injectivity of both
families, and a support hole. -/
example : ∃ (μ : Idx 1 → Perspective.ProbSphere 3) (Y : Idx 1 → SSphere 3),
    3 ≤ 3 ∧ 1 ≤ 1 ∧ Function.Injective μ ∧
      (⋃ k, (μ k : Measure (SSphere 3)).support) ≠ Set.univ ∧ Function.Injective Y :=
  ⟨fun _ => Perspective.diracProb 3 (basePoint 2), fun _ => basePoint 2, le_rfl, le_rfl,
    Function.injective_of_subsingleton _,
    finite_atomic_support_union_ne_univ (d := 3) le_rfl (fun _ : Idx 1 => basePoint 2),
    Function.injective_of_subsingleton _⟩

end Transformer.Interpolation

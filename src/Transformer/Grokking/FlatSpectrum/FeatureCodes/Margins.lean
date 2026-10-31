module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Basic

/-!
# Finite separation margins and approximate features

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*, items 3–4.
Joint separation of a finite answer set gives a uniform positive margin. Score errors
bounded by epsilon preserve the unique winner if twice epsilon is smaller than that
margin. Unit-modulus linear readouts transfer weighted feature error into score error.
These are conditional robustness results, not claims about training convergence.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open scoped BigOperators

variable {A R : Type*} [Fintype R]

/-- **Uniform positive margin**, source item 3. Finite answer sets include the empty and
singleton cases; an inserted positive sentinel handles both without extra assumptions. -/
theorem exists_positive_margin [Fintype A] {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1)
    (hsep : ∀ a b, a ≠ b → ∃ r, 0 < w r ∧ C a r ≠ C b r) :
    ∃ δ > 0, ∀ a b, a ≠ b → δ ≤ codeScore w C a a - codeScore w C a b := by
  classical
  let gap : A × A → ℝ := fun ab =>
    if ab.1 = ab.2 then 1 else codeScore w C ab.1 ab.1 - codeScore w C ab.1 ab.2
  have hpos (ab : A × A) : 0 < gap ab := by
    dsimp [gap]
    split_ifs with h
    · norm_num
    · exact sub_pos.mpr ((codeScore_lt_self_iff hw C hC _ _).mpr (hsep _ _ h))
  let s := insert (1 : ℝ) (Finset.univ.image gap)
  have hs : s.Nonempty := ⟨1, Finset.mem_insert_self _ _⟩
  refine ⟨s.min' hs, ?_, ?_⟩
  · have hmem := s.min'_mem hs
    rcases Finset.mem_insert.mp hmem with h | h
    · rw [h]
      norm_num
    · obtain ⟨ab, _, hab⟩ := Finset.mem_image.mp h
      rw [← hab]
      exact hpos ab
  · intro a b hab
    have hmem : gap (a, b) ∈ s := Finset.mem_insert_of_mem
      (Finset.mem_image_of_mem gap (Finset.mem_univ _))
    simpa [gap, hab] using s.min'_le _ hmem

/-- **Score-error tolerance**, source item 3. The ideal scores may come from any model.
The strict factor-two condition is needed because both competing scores can move. -/
theorem strict_winner_of_score_error {s L : A → ℝ} {a : A} {δ ε : ℝ}
    (hgap : ∀ b, b ≠ a → δ ≤ s a - s b)
    (herr : ∀ b, |L b - s b| ≤ ε) (hδ : 2 * ε < δ) :
    ∀ b, b ≠ a → L b < L a := by
  intro b hb
  have ha := (abs_le.mp (herr a)).1
  have hc := (abs_le.mp (herr b)).2
  have hmargin := hgap b hb
  linarith

/-- A unit-modulus code transfers weighted feature error to score error, source item 4. -/
theorem score_error_le {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1)
    (z : R → ℂ) (a b : A) :
    |score w C z b - codeScore w C a b| ≤ ∑ r, w r * ‖z r - C a r‖ := by
  have hn (r : R) : ‖C b r‖ = 1 := by
    have h := hC b r
    rw [Complex.normSq_eq_norm_sq] at h
    nlinarith [norm_nonneg (C b r)]
  unfold score codeScore score
  rw [← Finset.sum_sub_distrib]
  have hterm (r : R) : w r * (z r * star (C b r)).re -
      w r * (C a r * star (C b r)).re =
      w r * ((z r - C a r) * star (C b r)).re := by
    rw [sub_mul, Complex.sub_re]
    ring
  simp_rw [hterm]
  calc
    _ ≤ ∑ r, |w r * ((z r - C a r) * star (C b r)).re| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ r, w r * ‖z r - C a r‖ := by
      apply Finset.sum_le_sum
      intro r _
      rw [abs_mul, abs_of_nonneg (hw r)]
      apply mul_le_mul_of_nonneg_left _ (hw r)
      simpa only [norm_mul, norm_star, hn, mul_one] using
        Complex.abs_re_le_norm ((z r - C a r) * star (C b r))

/-- For weights of total mass one, uniform coordinate error epsilon gives the same
score-error bound. Source item 4: normalization fixes the scale of the margin. -/
theorem score_error_le_of_coordinate_error {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (hmass : ∑ r, w r = 1) (C : A → R → ℂ)
    (hC : ∀ a r, Complex.normSq (C a r) = 1) (z : R → ℂ) (a b : A) {ε : ℝ}
    (herr : ∀ r, ‖z r - C a r‖ ≤ ε) :
    |score w C z b - codeScore w C a b| ≤ ε := by
  apply (score_error_le hw C hC z a b).trans
  calc
    _ ≤ ∑ r, w r * ε := Finset.sum_le_sum fun r _ =>
      mul_le_mul_of_nonneg_left (herr r) (hw r)
    _ = ε := by rw [← Finset.sum_mul, hmass, one_mul]

/-- **Approximate feature classification**, source item 4. A recovered or learned vector
classifies correctly whenever its weighted error is less than half the ideal score gap. -/
theorem strict_winner_of_feature_error {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1)
    (z : R → ℂ) (a : A) {δ ε : ℝ}
    (hgap : ∀ b, b ≠ a → δ ≤ codeScore w C a a - codeScore w C a b)
    (herr : ∑ r, w r * ‖z r - C a r‖ ≤ ε) (hδ : 2 * ε < δ) :
    ∀ b, b ≠ a → score w C z b < score w C z a :=
  strict_winner_of_score_error hgap
    (fun b => (score_error_le hw C hC z a b).trans herr) hδ

example := exists_positive_margin
  (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) binaryCode binaryCode_unit
  (fun a b hab => (codeScore_lt_self_iff
    (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) binaryCode binaryCode_unit a b).mp
      ((unique_classification_iff_injective
        (fun _ : Bool => (by norm_num : (0 : ℝ) < 1 / 2)) binaryCode binaryCode_unit).mpr
          binaryCode_injective a b hab))

example : ∀ b : Bool, b ≠ true →
    (if b then (3 / 4 : ℝ) else 1 / 4) < (3 / 4 : ℝ) := by
  apply strict_winner_of_score_error (s := fun b => if b then 1 else 0)
    (a := true) (δ := 1) (ε := 1 / 4)
  · intro b hb
    cases b <;> norm_num at *
  · intro b
    cases b <;> norm_num
  · norm_num

example := score_error_le (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2))
  binaryCode binaryCode_unit (binaryCode (false, true)) (false, true) (true, false)

example := score_error_le_of_coordinate_error
  (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) (by norm_num)
  binaryCode binaryCode_unit (binaryCode (false, true)) (false, true) (true, false)
  (ε := 1 / 4) (fun _ => by norm_num)

example : ∀ b : Bool × Bool, b ≠ (false, false) →
    score (fun _ => (1 / 2 : ℝ)) binaryCode (binaryCode (false, false)) b <
    score (fun _ => (1 / 2 : ℝ)) binaryCode (binaryCode (false, false)) (false, false) := by
  apply strict_winner_of_feature_error
    (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) binaryCode binaryCode_unit
    (binaryCode (false, false)) (false, false) (δ := 1) (ε := 1 / 4)
  · rintro ⟨a, b⟩ hab
    cases a <;> cases b <;> norm_num [codeScore, score, binaryCode] at *
  · norm_num
  · norm_num

end Transformer.Grokking.FlatSpectrum.FeatureCodes

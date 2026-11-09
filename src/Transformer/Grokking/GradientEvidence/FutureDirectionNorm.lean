module

public import Transformer.Grokking.GradientEvidence.FutureMomentCone
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ClassDecomposition
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Future whole-block direction bounds

Source: `review/flat_spectrum.md`, *Future forcing from checkpoint moments*,
FF2–FF3. Summing coordinate cones controls a whole readout block without future
gradient observations. Centering within the active classes cannot increase
this budget. These are upper bounds, not statements of favorable alignment.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset
variable {K : Type*} [Fintype K]

/-- FF2: the actual normalized direction after the next h+1 updates. -/
def warmDirection (m v : ℝ) (g : ℕ → ℝ) (eps : ℝ) (start h : ℕ) : ℝ :=
  ((warmMoments m v g (h+1)).1 / (1-(9/10 : ℝ)^(start+h+1))) /
    (Real.sqrt (warmMoments m v g (h+1)).2 /
      Real.sqrt (1-(19/20 : ℝ)^(start+h+1)) + eps)

/-- FF2: the forecast is a bound on the main model's actual resumed direction. -/
theorem main_direction_resume (g : ℕ → ℝ) (eps : ℝ) (start h : ℕ) :
    mainDirection g eps (start+h) =
      warmDirection (mainMoments g start).1 (mainMoments g start).2
        (fun j => g (start+j)) eps start h := by
  unfold mainDirection warmDirection
  rw [Nat.add_assoc start h 1, main_moments_resume g start (h+1)]

/-- FF2: normalization preserves the moment sign symmetry, with unchanged
variance, bias corrections and epsilon. Magnitudes do not select alignment. -/
theorem warm_direction_sign_symmetry (m v : ℝ) (g : ℕ → ℝ) (eps : ℝ) (start h : ℕ) :
    warmDirection (-m) v (fun j => -g j) eps start h =
      -warmDirection m v g eps start h := by
  unfold warmDirection
  rw [warm_moments_sign_symmetry]
  simp only [neg_div]

/-- FF2: a checkpoint-only magnitude envelope with the actual absolute-time
bias corrections and positive epsilon, valid for arbitrary future gradients. -/
theorem warm_direction_bound (m v r : ℝ) (g : ℕ → ℝ) (eps : ℝ)
    (hv : 0 ≤ v) (hr : 0 ≤ r) (hcone : m ^ 2 ≤ r * v) (heps : 0 < eps)
    (start h : ℕ) :
    |warmDirection m v g eps start h| ≤ Real.sqrt (futureCone r (h+1)) *
      Real.sqrt (1-(19/20 : ℝ)^(start+h+1)) / (1-(9/10 : ℝ)^(start+h+1)) := by
  have h1 : 0 < 1-(9/10 : ℝ)^(start+h+1) := by
    have hp := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 9/10)
      (by norm_num : (9/10 : ℝ) < 1) (by omega : start+h+1 ≠ 0)
    linarith
  have h2 : 0 < Real.sqrt (1-(19/20 : ℝ)^(start+h+1)) := by
    apply Real.sqrt_pos.mpr
    have hp := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 19/20)
      (by norm_num : (19/20 : ℝ) < 1) (by omega : start+h+1 ≠ 0)
    linarith
  apply corrected_moment_bound _ _ _ _ _ _ (warm_second_nonneg m v g hv (h+1))
    (Real.sqrt_nonneg _) h1 h2 heps
  rw [Real.sq_sqrt (future_cone_nonneg r hr (h+1))]
  exact warm_moment_cone m v r g hv hr hcone (h+1)

example (g : ℕ → ℝ) (start h : ℕ) := warm_direction_bound (1/10) (1/100) 1 g
  (1/100000000) (by norm_num) (by norm_num) (by norm_num) (by norm_num) start h


/-- FF3: summing cones needs only the initial sum of ratios and the dimension. -/
theorem sum_future_cone (r : K → ℝ) (h : ℕ) :
    (∑ k, futureCone (r k) h) = (Fintype.card K : ℝ) * (19/14) +
      ((∑ k, r k) - (Fintype.card K : ℝ) * (19/14)) * (81/95 : ℝ)^h := by
  simp only [futureCone, sum_add_distrib, ← sum_mul, sum_sub_distrib,
    sum_const, card_univ, nsmul_eq_mul]

/-- FF3: retain both bias corrections at absolute time start+h+1. -/
def futureBias (start h : ℕ) : ℝ :=
  Real.sqrt (1-(19/20 : ℝ)^(start+h+1)) / (1-(9/10 : ℝ)^(start+h+1))

/-- FF3: the bias factor is positive at every actual update. -/
theorem future_bias_pos (start h : ℕ) : 0 < futureBias start h := by
  have h1 := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 9/10)
    (by norm_num : (9/10 : ℝ) < 1) (by omega : start+h+1 ≠ 0)
  have h2 := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 19/20)
    (by norm_num : (19/20 : ℝ) < 1) (by omega : start+h+1 ≠ 0)
  unfold futureBias
  apply div_pos
  · exact Real.sqrt_pos.mpr (sub_pos.mpr h2)
  · exact sub_pos.mpr h1

/-- FF3: all coordinates of the resumed buffered direction in Euclidean norm. -/
def warmDirections (m v : K → ℝ) (g : K → ℕ → ℝ) (eps : ℝ)
    (start h : ℕ) : EuclideanSpace ℝ K :=
  WithLp.toLp 2 (fun k => warmDirection (m k) (v k) (g k) eps start h)

/-- FF3: sum of the actual future direction energies, with no gradient bound. -/
theorem warm_directions_energy (m v r : K → ℝ) (g : K → ℕ → ℝ) (eps : ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps) (start h : ℕ) :
    (∑ k, (warmDirection (m k) (v k) (g k) eps start h)^2) ≤
      (∑ k, futureCone (r k) (h+1)) * (futureBias start h)^2 := by
  rw [sum_mul]
  apply sum_le_sum
  intro k hk
  have hu := warm_direction_bound (m k) (v k) (r k) (g k) eps
    (hv k) (hr k) (hc k) heps start h
  have he : Real.sqrt (futureCone (r k) (h+1)) *
      Real.sqrt (1-(19/20 : ℝ)^(start+h+1)) / (1-(9/10 : ℝ)^(start+h+1)) =
      Real.sqrt (futureCone (r k) (h+1)) * futureBias start h := by
    unfold futureBias
    ring
  rw [he] at hu
  have hs := (sq_le_sq₀ (abs_nonneg _) ((abs_nonneg _).trans hu)).mpr hu
  simpa only [sq_abs, mul_pow, Real.sq_sqrt (future_cone_nonneg _ (hr k) _)] using hs

/-- FF3: an explicit block norm bound, usable in DP1's damped forcing tube. -/
theorem warm_directions_norm (m v r : K → ℝ) (g : K → ℕ → ℝ) (eps : ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps) (start h : ℕ) :
    ‖warmDirections m v g eps start h‖ ≤
      Real.sqrt (∑ k, futureCone (r k) (h+1)) * futureBias start h := by
  have hz : 0 ≤ ∑ k, futureCone (r k) (h+1) :=
    sum_nonneg (fun k hk => future_cone_nonneg _ (hr k) _)
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _)
    (future_bias_pos start h).le)).mp
  rw [EuclideanSpace.real_norm_sq_eq, mul_pow, Real.sq_sqrt hz]
  exact warm_directions_energy m v r g eps hv hr hc heps start h

/-- FF3: subtracting a group's computed mean does not increase its energy.
Thus the active-row block bound also bounds active class contrasts. -/
theorem class_center_energy_le {C : Type*} [Fintype C] [Nonempty C] (x : C → ℝ) :
    (∑ c, (x c - FlatSpectrum.FeatureCodes.classMean x)^2) ≤ ∑ c, (x c)^2 := by
  have he := FlatSpectrum.FeatureCodes.class_translation_energy x 0
  have hp : 0 ≤ (Fintype.card C : ℝ) *
      (FlatSpectrum.FeatureCodes.classMean x)^2 := by positivity
  simp only [sub_zero] at he
  linarith

example (g : Fin 3 → ℕ → ℝ) (start h : ℕ) :=
  warm_directions_energy (fun _ => 1/10) (fun _ => 1/100) (fun _ => 1) g
    (1/100000000) (by intro k; norm_num) (by intro k; norm_num)
    (by intro k; norm_num) (by norm_num) start h
example (g : Fin 3 → ℕ → ℝ) (start h : ℕ) :=
  warm_directions_norm (fun _ => 1/10) (fun _ => 1/100) (fun _ => 1) g
    (1/100000000) (by intro k; norm_num) (by intro k; norm_num)
    (by intro k; norm_num) (by norm_num) start h
example := class_center_energy_le (fun c : Bool => if c then 3 else -1)

end Transformer.Grokking.GradientEvidence

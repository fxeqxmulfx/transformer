module

public import Transformer.Modes.Section2_Morse

/-!
# The original Gaussian KDE mode and upcrossing sets agree almost surely

ArXiv:2412.09080v3, §2.1, the identity asserted after `eq:Fn`.
The original KDE is smooth, and a local maximum has zero first
derivative and nonpositive second derivative. Almost-sure
nondegeneracy of its critical points turns this into strict
negativity. The original field's upcrossing criterion then
identifies the entire two point sets, simultaneously for every
subset of the query line, on one full-probability set of samples.

The counterexample to the unqualified pathwise identity remains
valid. The probability-one statement uses the actual Gaussian
law and its proved critical-chart argument, not an assumed
genericity condition. It gives the equality of the original
extended counts, so no measurability or finiteness of the query
set needs to be added to the source expectation identity.
-/

@[expose] public section

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Modes

/-- A continuous real function has nonpositive second derivative at a local maximum.
Source: arXiv:2412.09080v3, §2.1, the necessary derivative condition after `eq:Fn`.
If the second derivative were positive, the second derivative test would also give
a local minimum; the resulting local constancy forces that derivative to vanish. -/
theorem deriv_deriv_nonpos_of_isLocalMax (f : ℝ → ℝ) (t : ℝ)
    (hf : IsLocalMax f t) (hc : ContinuousAt f t) : deriv (deriv f) t ≤ 0 := by
  by_contra h
  have hpos : 0 < deriv (deriv f) t := by linarith
  have hmin := isLocalMin_of_deriv_deriv_pos hpos hf.deriv_eq_zero hc
  have he : f =ᶠ[𝓝 t] fun _ => f t := by
    filter_upwards [hf, hmin] with s hs ht
    exact le_antisymm hs ht
  have hz := he.deriv.deriv_eq
  have hconst : deriv (fun _ : ℝ => f t) = fun _ => 0 := funext fun x => deriv_const x (f t)
  rw [hconst, deriv_const] at hz
  linarith

/-- The genuine original mode set equals the upcrossing set whenever its original critical points are nondegenerate.
Source: arXiv:2412.09080v3, §2.1, the corrected pathwise qualification of the identity after `eq:Fn`. -/
theorem modeSet_eq_upcrossingSet_of_nondegenerate {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (X : Idx n → ℝ)
    (hX : ∀ t : ℝ, deriv (kde β X) t = 0 → deriv (deriv (kde β X)) t ≠ 0) (T : Set ℝ) :
    modeSet (kde β X) T = upcrossingSet (fieldF β X) 0 T := by
  apply Set.Subset.antisymm ?_ (upcrossingSet_subset_modeSet hβ hn X T)
  rintro t ⟨ht, hm⟩
  have hd := hm.deriv_eq_zero
  have hdd := deriv_deriv_nonpos_of_isLocalMax (kde β X) t hm
    (contDiff_kde (k := 1) β X).continuous.continuousAt
  exact ⟨ht, (isUpcrossing_fieldF_iff hβ hn X t).mpr ⟨hd, lt_of_le_of_ne hdd (hX t hd)⟩⟩

/-- The actual extended mode and upcrossing counts agree for any query set under the proved pathwise qualification.
Source: arXiv:2412.09080v3, §2.1, counting the equal sets following `eq:Fn`. -/
theorem modeCount_eq_upcrossingCount_of_nondegenerate {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (X : Idx n → ℝ)
    (hX : ∀ t : ℝ, deriv (kde β X) t = 0 → deriv (deriv (kde β X)) t ≠ 0) (T : Set ℝ) :
    modeCount (kde β X) T = upcrossingCount (fieldF β X) 0 T := by
  rw [modeCount, upcrossingCount, modeSet_eq_upcrossingSet_of_nondegenerate hβ hn X hX T]

/-- One full-probability set of original Gaussian samples has equal mode and upcrossing sets for every query subset.
Source: arXiv:2412.09080v3, §2.1, the almost-sure correction of the set identity after `eq:Fn`. -/
theorem ae_gaussianSample_modeSet_eq_upcrossingSet {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    ∀ᵐ X ∂gaussianSample n, ∀ T : Set ℝ,
      modeSet (kde β X) T = upcrossingSet (fieldF β X) 0 T := by
  filter_upwards [ae_gaussianSample_kde_nondegenerate hβ hn] with X hX
  exact fun T => modeSet_eq_upcrossingSet_of_nondegenerate hβ hn X hX T

/-- One full-probability set of original Gaussian samples has equal extended counts for every query subset.
Source: arXiv:2412.09080v3, §2.1, the almost-sure count identity needed after `eq:Fn`. -/
theorem ae_gaussianSample_modeCount_eq_upcrossingCount {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    ∀ᵐ X ∂gaussianSample n, ∀ T : Set ℝ,
      modeCount (kde β X) T = upcrossingCount (fieldF β X) 0 T := by
  filter_upwards [ae_gaussianSample_kde_nondegenerate hβ hn] with X hX
  exact fun T => modeCount_eq_upcrossingCount_of_nondegenerate hβ hn X hX T

/-- At every original query, nondegenerate original critical points give exactly the source mode/upcrossing criterion.
Source: arXiv:2412.09080v3, §2.1, the qualified equivalence following `eq:Fn`. -/
theorem isLocalMax_kde_iff_isUpcrossing_of_nondegenerate {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (X : Idx n → ℝ)
    (hX : ∀ t : ℝ, deriv (kde β X) t = 0 → deriv (deriv (kde β X)) t ≠ 0) (t : ℝ) :
    IsLocalMax (kde β X) t ↔ IsUpcrossing (fieldF β X) 0 t := by
  have h := congrArg (fun S : Set ℝ => t ∈ S)
    (modeSet_eq_upcrossingSet_of_nondegenerate hβ hn X hX Set.univ)
  simpa only [modeSet, upcrossingSet, Set.mem_univ, true_and, Set.mem_ofPred_eq] using h.to_iff

/-- Almost surely the original pointwise mode/upcrossing equivalence holds at every real query.
Source: arXiv:2412.09080v3, §2.1, the almost-sure form of the assertion after `eq:Fn`. -/
theorem ae_gaussianSample_isLocalMax_iff_isUpcrossing {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) :
    ∀ᵐ X ∂gaussianSample n, ∀ t : ℝ,
      IsLocalMax (kde β X) t ↔ IsUpcrossing (fieldF β X) 0 t := by
  filter_upwards [ae_gaussianSample_kde_nondegenerate hβ hn] with X hX
  exact fun t => isLocalMax_kde_iff_isUpcrossing_of_nondegenerate hβ hn X hX t

/-- For any original query set, the actual extended counts are equal almost everywhere under the source sample law.
Source: arXiv:2412.09080v3, §2.1, the count equality used inside the expectations after `eq:Fn`. -/
theorem modeCount_ae_eq_upcrossingCount {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n) (T : Set ℝ) :
    (fun X => modeCount (kde β X) T) =ᵐ[gaussianSample n]
      fun X => upcrossingCount (fieldF β X) 0 T := by
  filter_upwards [ae_gaussianSample_modeCount_eq_upcrossingCount hβ hn] with X hX
  exact hX T

/-- The original singleton field has exactly one upcrossing, at its original sample, restricted to the query set.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, including its singleton normalization. -/
theorem upcrossingSet_fieldF_singleton (β : ℝ) (X : Idx 1 → ℝ) (T : Set ℝ) :
    upcrossingSet (fieldF β X) 0 T = {X 0} ∩ T := by
  ext t
  constructor
  · rintro ⟨ht, hf, hd⟩
    have hz : (t - X 0) * Real.exp (-(β / 2) * (t - X 0) ^ 2) = 0 := by
      simpa only [fieldF, Fin.sum_univ_one, Nat.cast_one, Real.sqrt_one, div_one, one_mul] using hf
    have he : t = X 0 := sub_eq_zero.mp ((mul_eq_zero.mp hz).resolve_right (Real.exp_ne_zero _))
    exact ⟨Set.mem_singleton_iff.mpr he, ht⟩
  · rintro ⟨he, ht⟩
    have hf : fieldF β X t = 0 := by
      rw [Set.mem_singleton_iff.mp he]
      simp [fieldF]
    refine ⟨ht, hf, ?_⟩
    rw [fieldF_singleton_deriv_at_zero β X t hf]
    norm_num

/-- A negative quadratic realizes the continuous local-maximum hypotheses of the second derivative condition. -/
example : IsLocalMax (fun t : ℝ => -t ^ 2) 0 ∧ ContinuousAt (fun t : ℝ => -t ^ 2) 0 := by
  constructor
  · exact Filter.Eventually.of_forall fun t => by simpa using neg_nonpos.mpr (sq_nonneg t)
  · fun_prop

/-- An actual singleton Gaussian KDE realizes positive bandwidth, positive size, and global critical nondegeneracy.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, for one original sample. -/
example : ∃ X : Idx 1 → ℝ, (0 : ℝ) < 1 ∧ (0 : ℕ) < 1 ∧
    (∀ t : ℝ, deriv (kde 1 X) t = 0 → deriv (deriv (kde 1 X)) t ≠ 0) ∧
    IsLocalMax (kde 1 X) 0 := by
  let X : Idx 1 → ℝ := fun _ => 0
  have hβ : (0 : ℝ) < 1 := by norm_num
  have hn : (0 : ℕ) < 1 := by norm_num
  have hX : ∀ t : ℝ, deriv (kde 1 X) t = 0 → deriv (deriv (kde 1 X)) t ≠ 0 := by
    intro t ht hdd
    have hf : fieldF 1 X t = 0 := by rw [fieldF_eq hβ hn, ht, mul_zero]
    have hd := fieldF_singleton_deriv_at_zero 1 X t hf
    rw [deriv_fieldF_eq_neg_mul_deriv_deriv hβ hn, hdd, mul_zero] at hd
    norm_num at hd
  refine ⟨X, hβ, hn, hX, ?_⟩
  apply isLocalMax_of_deriv_deriv_neg
  · rw [(hasDerivAt_deriv_kde 1 X 0).deriv]
    simp [X]
    positivity
  · rw [(hasDerivAt_kde 1 X 0).deriv]
    simp [X]
  · exact (contDiff_kde (k := 1) 1 X).continuous.continuousAt

end Transformer.Modes

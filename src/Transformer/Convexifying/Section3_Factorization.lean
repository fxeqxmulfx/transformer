/-
# Euclidean norms and balanced head factors

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2 and Appendix A.1/A.2.
These finite-dimensional norm identities make the rescaling and recovery
arguments precise, including zero rows that the paper divides by implicitly.
-/

import Transformer.Convexifying.Section3_Models

open scoped BigOperators

namespace Transformer.Convexifying

/-- The Mathlib Euclidean-space realization of a coordinate vector. -/
def toEuclidean {d : ℕ} (v : Vec d) : EuclideanSpace ℝ (Fin d) :=
  WithLp.toLp 2 v

/-- Our coordinate norm is the norm of Mathlib's Euclidean space. -/
theorem norm₂_eq_euclidean {d : ℕ} (v : Vec d) :
    norm₂ v = ‖toEuclidean v‖ := by
  rw [EuclideanSpace.norm_eq]
  simp [norm₂, normSq, toEuclidean, Real.norm_eq_abs, sq_abs]

/-- Euclidean homogeneity in coordinate notation. -/
theorem norm₂_smul {d : ℕ} (a : ℝ) (v : Vec d) :
    norm₂ (fun q => a * v q) = |a| * norm₂ v := by
  rw [norm₂_eq_euclidean]
  have heq : toEuclidean (fun q => a * v q) = a • toEuclidean v := by
    ext q
    simp [toEuclidean]
  rw [heq, norm_smul, Real.norm_eq_abs, ← norm₂_eq_euclidean]

/-- Triangle inequality for finite linear combinations. -/
theorem norm₂_linearCombination_le {h d : ℕ}
    (a : Fin h → ℝ) (v : Fin h → Vec d) :
    norm₂ (fun q => ∑ j, a j * v j q) ≤
      ∑ j, |a j| * norm₂ (v j) := by
  rw [norm₂_eq_euclidean]
  have heq : toEuclidean (fun q => ∑ j, a j * v j q) =
      ∑ j, a j • toEuclidean (v j) := by
    ext q
    simp [toEuclidean, Finset.sum_apply]
  rw [heq]
  calc
    ‖∑ j, a j • toEuclidean (v j)‖ ≤
        ∑ j, ‖a j • toEuclidean (v j)‖ := norm_sum_le _ _
    _ = ∑ j, |a j| * norm₂ (v j) := by
      simp [norm_smul, Real.norm_eq_abs, norm₂_eq_euclidean]

/-- A zero Euclidean norm means that every coordinate is zero. -/
theorem norm₂_eq_zero_iff {d : ℕ} (v : Vec d) :
    norm₂ v = 0 ↔ ∀ q, v q = 0 := by
  rw [norm₂_eq_euclidean]
  constructor
  · intro h q
    have he : toEuclidean v = 0 := norm_eq_zero.mp h
    have := congrArg (fun w : EuclideanSpace ℝ (Fin d) => w q) he
    simpa [toEuclidean] using this
  · intro h
    have he : toEuclidean v = 0 := by
      ext q
      simpa [toEuclidean] using h q
    rw [he]
    simp

/-- Balance a product vector into value and output weights with quadratic
penalty equal to its Euclidean norm.  The zero case is handled explicitly.
Source: arXiv:2211.11052v1, Appendix A.1/A.2, `eq:scaling`. -/
theorem exists_balanced_factor {d : ℕ} (v : Vec d) :
    ∃ u : Vec d, ∃ b : ℝ,
      (∀ q, u q * b = v q) ∧
      (normSq u + b ^ 2) / 2 = norm₂ v := by
  by_cases hv : norm₂ v = 0
  · have hzero := (norm₂_eq_zero_iff v).mp hv
    refine ⟨fun _ => 0, 0, ?_, ?_⟩
    · intro q; simp [hzero q]
    · simp [normSq, hv]
  · have hvpos : 0 < norm₂ v := lt_of_le_of_ne (norm₂_nonneg v) (Ne.symm hv)
    let t := Real.sqrt (norm₂ v)
    have htpos : 0 < t := Real.sqrt_pos.2 hvpos
    have ht2 : t ^ 2 = norm₂ v := Real.sq_sqrt hvpos.le
    let u : Vec d := fun q => v q / t
    refine ⟨u, t, ?_, ?_⟩
    · intro q
      dsimp [u]
      field_simp
    · have hs : normSq u = normSq v / t ^ 2 := by
        simp [normSq, u, div_pow, Finset.sum_div]
      rw [hs, ← norm₂_sq v, ← ht2]
      field_simp
      ring

/-- Put a product vector into the unit value ball, moving its norm into
the scalar output weight.  The zero case is handled explicitly.
Source: arXiv:2211.11052v1, §3.2, `lemma:scaling`. -/
theorem exists_unit_factor {d : ℕ} (v : Vec d) :
    ∃ u : Vec d, ∃ b : ℝ,
      (∀ q, u q * b = v q) ∧ norm₂ u ≤ 1 ∧ |b| = norm₂ v := by
  by_cases hv : norm₂ v = 0
  · have hzero := (norm₂_eq_zero_iff v).mp hv
    refine ⟨fun _ => 0, 0, ?_, ?_, ?_⟩
    · intro q; simp [hzero q]
    · simp [norm₂, normSq]
    · simpa using hv.symm
  · have hvpos : 0 < norm₂ v := lt_of_le_of_ne (norm₂_nonneg v) (Ne.symm hv)
    let u : Vec d := fun q => v q / norm₂ v
    refine ⟨u, norm₂ v, ?_, ?_, ?_⟩
    · intro q
      dsimp [u]
      field_simp
    · have hu : u = fun q => (1 / norm₂ v) * v q := by
        funext q
        dsimp [u]
        ring
      rw [hu, norm₂_smul, abs_of_nonneg (one_div_nonneg.mpr hvpos.le)]
      have heq : 1 / norm₂ v * norm₂ v = 1 := by field_simp
      rw [heq]
    · exact abs_of_pos hvpos

/-- A pair form of balanced factorization. -/
private theorem exists_balanced_pair {d : ℕ} (v : Vec d) :
    ∃ p : Vec d × ℝ,
    (∀ q, p.1 q * p.2 = v q) ∧
      (normSq p.1 + p.2 ^ 2) / 2 = norm₂ v := by
  obtain ⟨u, b, hprod, hreg⟩ := exists_balanced_factor v
  exact ⟨(u, b), hprod, hreg⟩

/-- A selected balanced factorization of a vector. -/
noncomputable def balancedPair {d : ℕ} (v : Vec d) : Vec d × ℝ :=
  Classical.choose (exists_balanced_pair v)

/-- The selected balanced pair has the required product and penalty. -/
theorem balancedPair_spec {d : ℕ} (v : Vec d) :
    (∀ q, (balancedPair v).1 q * (balancedPair v).2 = v q) ∧
      (normSq (balancedPair v).1 + (balancedPair v).2 ^ 2) / 2 = norm₂ v :=
  Classical.choose_spec (exists_balanced_pair v)

/-- A pair form of unit-ball factorization. -/
private theorem exists_unit_pair {d : ℕ} (v : Vec d) :
    ∃ p : Vec d × ℝ,
    (∀ q, p.1 q * p.2 = v q) ∧ norm₂ p.1 ≤ 1 ∧ |p.2| = norm₂ v := by
  obtain ⟨u, b, hprod, hunit, hscale⟩ := exists_unit_factor v
  exact ⟨(u, b), hprod, hunit, hscale⟩

/-- A selected unit-ball factorization of a vector. -/
noncomputable def unitPair {d : ℕ} (v : Vec d) : Vec d × ℝ :=
  Classical.choose (exists_unit_pair v)

/-- The selected unit-ball pair has the required product and norm. -/
theorem unitPair_spec {d : ℕ} (v : Vec d) :
    (∀ q, (unitPair v).1 q * (unitPair v).2 = v q) ∧
      norm₂ (unitPair v).1 ≤ 1 ∧ |(unitPair v).2| = norm₂ v :=
  Classical.choose_spec (exists_unit_pair v)

/-- A canonical unit-ball direction, with the zero vector handled without
division. -/
noncomputable def unitDirection {d : ℕ} (v : Vec d) : Vec d :=
  if norm₂ v = 0 then fun _ => 0 else fun q => v q / norm₂ v

/-- The canonical direction reconstructs `v` after multiplication by its
norm and lies in the unit ball.
Source: arXiv:2211.11052v1, Appendix A.1, normalization step. -/
theorem unitDirection_spec {d : ℕ} (v : Vec d) :
    (∀ q, unitDirection v q * norm₂ v = v q) ∧
      norm₂ (unitDirection v) ≤ 1 := by
  by_cases hv : norm₂ v = 0
  · have hzero := (norm₂_eq_zero_iff v).mp hv
    constructor
    · intro q
      simp [unitDirection, hv, hzero q]
    · have heq : unitDirection v = fun _ => 0 := by
        funext q
        simp [unitDirection, hv]
      rw [heq]
      simp [norm₂, normSq]
  · have hvpos : 0 < norm₂ v := lt_of_le_of_ne (norm₂_nonneg v) (Ne.symm hv)
    have heq : unitDirection v = fun q => v q / norm₂ v := by
      funext q
      simp [unitDirection, hv]
    constructor
    · intro q
      rw [heq]
      field_simp
    · rw [heq]
      have heq' : (fun q => v q / norm₂ v) =
          fun q => (1 / norm₂ v) * v q := by
        funext q
        ring
      rw [heq', norm₂_smul, abs_of_nonneg (one_div_nonneg.mpr hvpos.le)]
      have hunit : 1 / norm₂ v * norm₂ v = 1 := by field_simp
      rw [hunit]

end Transformer.Convexifying

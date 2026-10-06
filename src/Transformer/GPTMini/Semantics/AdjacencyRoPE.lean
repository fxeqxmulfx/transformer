import Transformer.GPTMini.Semantics.Routing
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# An actual RoPE pair can select the previous raw token

Source: apply_rope/rope_tables at f11b6e2 and MQAR's adjacent key/value
writes in synthetic/retrieval.py at cbafbe9. Use pair seven of the original
sixteen-dimensional small-model head, at the unchanged theta=10000.
Its frequency is positive and at most 1/100; positions through 128 remain
inside a monotone cosine interval. A one-position query phase shift makes
the immediately previous token the unique causal score maximizer.

The construction uses original RoPE and finite temperature. It is a
positional routing component, not yet a complete raw MQAR encoder or a
claim that learned query/key/value matrices satisfy this construction.
-/

namespace Transformer.GPTMini.Semantics

/-- Pair seven's actual frequency in the original small GPTMini head.
Source: rope_tables' theta**(-arange(half)/half), with head_dim=16 and theta=10000. -/
noncomputable def adjacentFrequency : ℝ := invFreq 16 10000 ⟨7, by decide⟩

/-- A genuine first-half unit vector, paired with coordinate fifteen by original RoPE.
Source: apply_rope's two contiguous halves, rather than adjacent-coordinate pairing. -/
noncomputable def adjacentDirection : EucSpace 16 :=
  EuclideanSpace.single (ropeFst 16 ⟨7, by decide⟩) 1

/-- The ordinary query vector has one position of backwards phase; the key vector is adjacentDirection.
Source: the actual linear RoPE rotation, which can be encoded in constant ordinary projection columns. -/
noncomputable def adjacentQuery : EucSpace 16 := applyRope 16 10000 (-1) adjacentDirection

/-- The chosen concrete frequency is positive and small enough for every Basis context.
Source: the exact original power law; no assumed positional score gap enters the proof. -/
theorem adjacentFrequency_bounds : 0 < adjacentFrequency ∧ adjacentFrequency ≤ 1 / 100 := by
  change 0 < (10000 : ℝ) ^ (-(14 : ℝ) / 16) ∧
    (10000 : ℝ) ^ (-(14 : ℝ) / 16) ≤ 1 / 100
  constructor
  · exact Real.rpow_pos_of_pos (by norm_num) _
  · have h := Real.rpow_le_rpow_of_exponent_le (by norm_num : (1 : ℝ) ≤ 10000)
      (by norm_num : -(14 : ℝ) / 16 ≤ -(1 / 2))
    rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow] at h
    have hs : Real.sqrt (10000 : ℝ) = 100 := by
      rw [show (10000 : ℝ) = (100 : ℝ) ^ 2 from by norm_num, Real.sqrt_sq (by norm_num)]
    rw [hs] at h
    norm_num at h ⊢
    exact h

/-- Even the longest Basis context remains strictly within the first cosine half-period.
Source: the proved frequency bound and the largest raw context cap 128. -/
theorem adjacentFrequency_context : 128 * adjacentFrequency < Real.pi := by
  nlinarith [adjacentFrequency_bounds.2, Real.pi_gt_three]

/-- Both ordinary phase-shifted vectors have exactly unit norm.
Source: the concrete unit direction and original RoPE's proved isometry. -/
theorem adjacent_vectors_norm : ‖adjacentDirection‖ = 1 ∧ ‖adjacentQuery‖ = 1 := by
  have hd : ‖adjacentDirection‖ = 1 := by simp [adjacentDirection, PiLp.norm_single]
  exact ⟨hd, by rw [adjacentQuery, applyRope_isometry, hd]⟩

/-- The actual rotated query/key inner product peaks at t=s-1.
Source: two applications of original RoPE's relative-position theorem and the exact pair coordinate. -/
theorem adjacent_rotated_inner (s t : ℝ) :
    inner (𝕜 := ℝ) (applyRope 16 10000 s adjacentQuery)
      (applyRope 16 10000 t adjacentDirection) =
        Real.cos ((t - s + 1) * adjacentFrequency) := by
  rw [applyRope_relative, adjacentQuery, applyRope_relative]
  simp only [adjacentDirection, EuclideanSpace.inner_single_left, map_one, one_mul]
  have hp : ropeSplit 16 (ropeFst 16 ⟨7, by decide⟩) =
      Sum.inl (Sum.inl (⟨7, by decide⟩ : Fin 8)) := by simp [ropeFst]
  rw [applyRope_apply, hp]
  have hne : ropeSnd 16 ⟨7, by decide⟩ ≠ ropeFst 16 ⟨7, by decide⟩ := by decide
  simp only [ropeCoord, Sum.elim_inl, PiLp.single_apply, ite_true,
    ite_eq_right hne, one_mul, zero_mul, sub_zero]
  apply congrArg Real.cos
  unfold ropeAngle adjacentFrequency
  ring

/-- A positive explicit margin separates the previous-token peak from every one-step-away score.
Source: strict cosine monotonicity on the proved actual frequency interval. -/
theorem adjacent_cosine_gap_pos : 0 < 1 - Real.cos adjacentFrequency := by
  have h := Real.cos_lt_cos_of_nonneg_of_le_pi (by norm_num : (0 : ℝ) ≤ 0)
    (by nlinarith [adjacentFrequency_bounds.2, Real.pi_gt_three] : adjacentFrequency ≤ Real.pi)
    adjacentFrequency_bounds.1
  rw [Real.cos_zero] at h
  linarith

/-- Every nonzero discrete displacement in a Basis context has a score at most cos(frequency).
Source: the actual frequency bound; d is an integer-token displacement, not a learned score assumption. -/
theorem adjacent_cosine_displacement (d : ℤ) (hne : d ≠ 0) (hbound : |(d : ℝ)| ≤ 128) :
    Real.cos ((d : ℝ) * adjacentFrequency) ≤ Real.cos adjacentFrequency := by
  have hd : 1 ≤ |(d : ℝ)| := by
    by_cases h : 0 ≤ d
    · rw [abs_of_nonneg (by exact_mod_cast h)]
      have hi : 1 ≤ d := by omega
      exact_mod_cast hi
    · rw [abs_of_nonpos (by exact_mod_cast (by omega : d ≤ 0))]
      have hi : (1 : ℤ) ≤ -d := by omega
      exact_mod_cast hi
  have hfreq := adjacentFrequency_bounds.1
  have hphase : |(d : ℝ) * adjacentFrequency| = |(d : ℝ)| * adjacentFrequency := by
    rw [abs_mul, abs_of_pos hfreq]
  rw [← Real.cos_abs, hphase]
  apply Real.cos_le_cos_of_nonneg_of_le_pi hfreq.le
  · nlinarith [adjacentFrequency_context]
  · nlinarith

example : (-1 : ℤ) ≠ 0 ∧ |((-1 : ℤ) : ℝ)| ≤ 128 := by norm_num

/-- True QKNorm retains the exact positional cosine at any epsilon no greater than the unit-vector norm.
Source: score/normL2 and the proved unit norms after original RoPE, not a hard-attention replacement. -/
theorem adjacent_score (alpha eps s t : ℝ) (heps : eps ≤ 1) :
    score alpha eps (applyRope 16 10000 s adjacentQuery)
      (applyRope 16 10000 t adjacentDirection) =
        Real.exp alpha * Real.cos ((t - s + 1) * adjacentFrequency) := by
  have hq : ‖applyRope 16 10000 s adjacentQuery‖ = 1 :=
    (applyRope_isometry _ _ _ _).trans adjacent_vectors_norm.2
  have hk : ‖applyRope 16 10000 t adjacentDirection‖ = 1 :=
    (applyRope_isometry _ _ _ _).trans adjacent_vectors_norm.1
  rw [score, normL2, normL2, hq, hk, max_eq_left heps]
  simp only [div_one, one_smul, adjacent_rotated_inner]

example : (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- The previous raw position has exactly the maximal finite-temperature score exp(alpha).
Source: the actual shifted RoPE/QKNorm formula, with the discrete predecessor equation made explicit. -/
theorem adjacent_score_peak (alpha eps : ℝ) (heps : eps ≤ 1)
    (i j : ℕ) (hprev : j + 1 = i) :
    score alpha eps (applyRope 16 10000 (i : ℝ) adjacentQuery)
      (applyRope 16 10000 (j : ℝ) adjacentDirection) = Real.exp alpha := by
  rw [adjacent_score _ _ _ _ heps]
  have hi : (j : ℝ) + 1 = i := by exact_mod_cast hprev
  have hd : (j : ℝ) - i + 1 = 0 := by linarith
  rw [hd, zero_mul, Real.cos_zero, mul_one]

example : (1 / 100000 : ℝ) ≤ 1 ∧ (1 : ℕ) + 1 = 2 := by norm_num

/-- The current position remains a strict lower-scoring competitor of its predecessor.
Source: the same shifted original RoPE pair; causal softmax's unmasked self entry cannot tie the peak. -/
theorem adjacent_self_score (alpha eps i : ℝ) (heps : eps ≤ 1) :
    score alpha eps (applyRope 16 10000 i adjacentQuery)
      (applyRope 16 10000 i adjacentDirection) = Real.exp alpha * Real.cos adjacentFrequency := by
  rw [adjacent_score _ _ _ _ heps]
  have hd : (i - i + 1) * adjacentFrequency = adjacentFrequency := by ring
  rw [hd]

example : (1 / 100000 : ℝ) ≤ 1 := by norm_num

end Transformer.GPTMini.Semantics

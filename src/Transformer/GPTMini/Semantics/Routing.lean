import Transformer.GPTMini.Semantics.Basic
import Transformer.GPTMini.AttentionLipschitz

/-!
# A retrieval head preserves semantic codes despite softmax leakage

Source: CausalMHA.forward at f11b6e2 and its actual QKNorm/RoPE/XSA
operators. The selected value may approximate a semantic code. XSA must
be orthogonal to that code; without this condition a self-aligned code
can be erased. The error estimate includes both finite-temperature
leakage and imperfect value encoding.

The second certificate propagates an approximate key copy into a new
query-key match. Thus the two stages needed for raw adjacent key/value
tokens need not copy or route exactly. These are universal conditional
operator laws, not claims that arbitrary learned parameters obey the
routing hypotheses on every Basis input.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- XSA preserves an orthogonal semantic target up to twice the input error.
Source: the actual clipped normalization and rank-one subtraction at f11b6e2. -/
theorem projection_error_le {d : ℕ} (eps : ℝ) (heps : 0 ≤ eps)
    (self y target : EucSpace d)
    (horthogonal : inner (𝕜 := ℝ) target (normL2 eps self) = 0) :
    ‖(y - (inner (𝕜 := ℝ) y (normL2 eps self)) • normL2 eps self) - target‖ ≤
      2 * ‖y - target‖ := by
  have hu := normL2_norm_le eps heps self
  have h := norm_proj_sub_proj_le y target (normL2 eps self) (normL2 eps self) hu hu
  simpa only [horthogonal, zero_smul, sub_zero, sub_self, norm_zero,
    mul_zero, add_zero] using h

example (target : EucSpace 16) : (0 : ℝ) ≤ 1 ∧
    inner (𝕜 := ℝ) target (normL2 1 (0 : EucSpace 16)) = 0 := by
  constructor
  · norm_num
  · simp [normL2]

/-- An actual RoPE/QKNorm/softmax/XSA head approximately retrieves the selected semantic code.
Source: CausalMHA.forward at f11b6e2; the score gap is measured after its actual RoPE. -/
theorem head_error_le (cfg : Config) {T : ℕ} (alpha eps gap B eta : ℝ)
    (heps : 0 ≤ eps) (hB : 0 ≤ B)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ)
    (i selected : Fin T) (target : EucSpace cfg.head_dim)
    (hselected : selected.val ≤ i.val)
    (hgap : ∀ j, j ≠ selected → j.val ≤ i.val →
      preScore cfg alpha eps
          (fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (q r))
          (fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (k r)) i j ≤
        preScore cfg alpha eps
          (fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (q r))
          (fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (k r)) i selected - gap)
    (hvalues : ∀ j, j ≠ selected → ‖v j - v selected‖ ≤ B)
    (hcode : ‖v selected - target‖ ≤ eta)
    (horthogonal : inner (𝕜 := ℝ) target (normL2 eps (v i)) = 0) :
    ‖attentionHead cfg alpha eps q k v positions i - target‖ ≤
      2 * (((T - 1 : ℕ) : ℝ) * Real.exp (-gap) * B + eta) := by
  let qr := fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (q r)
  let kr := fun r => applyRope cfg.head_dim cfg.rope_theta (positions r) (k r)
  let y := attnOutput cfg alpha eps qr kr v i
  have herr : ‖y - v selected‖ ≤ ((T - 1 : ℕ) : ℝ) * Real.exp (-gap) * B :=
    (output_error_le cfg alpha eps qr kr v i selected B hvalues).trans
      (mul_le_mul_of_nonneg_right
        (tail_mass_le cfg alpha eps gap qr kr i selected hselected hgap) hB)
  have hy : ‖y - target‖ ≤ ((T - 1 : ℕ) : ℝ) * Real.exp (-gap) * B + eta :=
    (norm_sub_le_norm_sub_add_norm_sub y (v selected) target).trans (add_le_add herr hcode)
  have hp := projection_error_le eps heps (v i) y target horthogonal
  unfold attentionHead xsaProjection
  simp only
  exact hp.trans (mul_le_mul_of_nonneg_left hy (by norm_num))

example (cfg : Config) : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : Fin 2).val ≤ (1 : Fin 2).val ∧
    (∀ j : Fin 2, j ≠ 0 → j.val ≤ (1 : Fin 2).val →
      preScore cfg 0 1
          (fun r : Fin 2 => applyRope cfg.head_dim cfg.rope_theta (r.val : ℝ) 0)
          (fun r : Fin 2 => applyRope cfg.head_dim cfg.rope_theta (r.val : ℝ) 0) 1 j ≤
        preScore cfg 0 1
          (fun r : Fin 2 => applyRope cfg.head_dim cfg.rope_theta (r.val : ℝ) 0)
          (fun r : Fin 2 => applyRope cfg.head_dim cfg.rope_theta (r.val : ℝ) 0) 1 0 - 0) ∧
    (∀ j : Fin 2, j ≠ 0 → ‖(0 : EucSpace cfg.head_dim) - 0‖ ≤ (1 : ℝ)) ∧
    ‖(0 : EucSpace cfg.head_dim) - 0‖ ≤ (0 : ℝ) ∧
    inner (𝕜 := ℝ) (0 : EucSpace cfg.head_dim) (normL2 1 0) = 0 := by
  refine ⟨by norm_num, by norm_num, by decide, ?_, ?_, ?_, ?_⟩
  · intro j hj hji
    simp [rope_zero, preScore, score, normL2]
  · intro j hj
    simp
  · simp
  · simp

/-- Approximate copied keys retain a match if their error is below the QKNorm score-gap budget.
Source: QKNorm's actual epsilon clipping, RoPE isometry and score_lipschitz at f11b6e2.
The ideal keys are semantic reference codes; no ideal final logits are assumed. -/
theorem copied_keys_gap (cfg : Config) {T : ℕ} (alpha eps gap eta : ℝ)
    (heps : 0 < eps) (q k reference : Fin T → EucSpace cfg.head_dim)
    (positions : Fin T → ℝ) (i selected : Fin T)
    (hkeys : ∀ j, ‖k j - reference j‖ ≤ eta)
    (hgap : ∀ j, j ≠ selected → j.val ≤ i.val →
      score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions j) (reference j)) ≤
        score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions selected) (reference selected)) - gap) :
    ∀ j, j ≠ selected → j.val ≤ i.val →
      score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) ≤
        score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions selected) (k selected)) -
          (gap - 4 * Real.exp alpha / eps * eta) := by
  have hshift (j : Fin T) :
      |score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) -
        score alpha eps (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
          (applyRope cfg.head_dim cfg.rope_theta (positions j) (reference j))| ≤
        2 * Real.exp alpha / eps * eta := by
    have hn := normL2_lipschitz eps heps
      (applyRope cfg.head_dim cfg.rope_theta (positions j) (k j))
      (applyRope cfg.head_dim cfg.rope_theta (positions j) (reference j))
    rw [applyRope_dist] at hn
    have hs := score_lipschitz alpha eps heps.le
      (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
      (applyRope cfg.head_dim cfg.rope_theta (positions j) (k j))
      (applyRope cfg.head_dim cfg.rope_theta (positions i) (q i))
      (applyRope cfg.head_dim cfg.rope_theta (positions j) (reference j))
    simp only [sub_self, norm_zero, zero_add] at hs
    calc _ ≤ Real.exp alpha * (2 / eps * ‖k j - reference j‖) :=
          hs.trans (mul_le_mul_of_nonneg_left hn (Real.exp_pos _).le)
      _ ≤ Real.exp alpha * (2 / eps * eta) := by gcongr; exact hkeys j
      _ = _ := by ring
  intro j hj hji
  have hjbounds := abs_le.mp (hshift j)
  have hsbounds := abs_le.mp (hshift selected)
  have hbudget : 4 * Real.exp alpha / eps * eta =
      2 * (2 * Real.exp alpha / eps * eta) := by ring
  rw [hbudget]
  linarith [hgap j hj hji, hjbounds.2, hsbounds.1]

example (cfg : Config) : (0 : ℝ) < 1 ∧
    (∀ (_ : Fin 2), ‖(0 : EucSpace cfg.head_dim) - 0‖ ≤ (0 : ℝ)) ∧
    ∀ j : Fin 2, j ≠ 0 → j.val ≤ (1 : Fin 2).val →
      score 0 1 (applyRope cfg.head_dim cfg.rope_theta 1 0)
          (applyRope cfg.head_dim cfg.rope_theta (j.val : ℝ) 0) ≤
        score 0 1 (applyRope cfg.head_dim cfg.rope_theta 1 0)
          (applyRope cfg.head_dim cfg.rope_theta 0 0) - 0 := by
  refine ⟨by norm_num, fun _ => by simp, ?_⟩
  intro j hj hji
  simp [rope_zero, score, normL2]

end Transformer.GPTMini.Semantics

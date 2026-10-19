module

public import Transformer.Landscape.Neural.SiLU

/-!
# An operand-interaction obstruction for the actual Rotate90 readout

Source: `grokking_abmod/web/own-channel/README.md`, Architecture and
initialization, at f49969393a7482490d45483d8a4fd3046b1adede. The four-row
argument is also stated in the separately inspected working note
`grokking_abmod/docs/rotate90-mathematics.md`, section 2.1 (2026-10-19;
that note was untracked and is not attributed to the recorded commit).

Deleting the residual branch leaves exactly E[a] + J E[b], positive-epsilon
RMSNorm and the tied readout. Its normalization is positive and common to
all classes at each input, so decisions reduce to additive operand scores.
A checkerboard with two labels cannot be separated, at any embedding width.
Two comparisons are weak and two strict: the result covers ties when the
first label has priority, as in the browser's smallest-index argmax.
The obstruction uses finite channel sums, with no additional assumptions
on the values or rank of the embedding rows.

This proves a necessary interaction requirement, not a training theorem or
an explanation of why a scalar MLP fails to find its representable solution.
It makes no assumption about the learned embedding's character spectrum.
-/

@[expose] public section

namespace Transformer.Landscape.Neural

variable {V K : Type*} [Fintype K]

/-- The actual classifier when the nonlinear residual branch is zero.
Source: the recorded architecture's x0 and final readout formulas. -/
noncomputable def noBranchLogit (eps : ℝ) (E : V → Channels K → ℝ)
    (a b q : V) : ℝ := tiedReadout E (rmsNorm eps (tiedInput E a b)) q

/-- Final RMSNorm scales every class score by the same input-dependent
denominator. Source: the recorded RMSNorm and tied readout formulas. -/
theorem tiedReadout_rmsNorm (E : V → Channels K → ℝ)
    (eps : ℝ) (v : Channels K → ℝ) (q : V) :
    tiedReadout E (rmsNorm eps v) q =
      tiedReadout E v q / Real.sqrt (eps + meanSquare v) := by
  unfold tiedReadout rmsNorm
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-- With no residual branch, the unnormalized readout is a sum of one
score for each operand. Source: the recorded x0 = E[a] + J E[b]. -/
theorem tiedReadout_tiedInput (E : V → Channels K → ℝ) (a b q : V) :
    tiedReadout E (tiedInput E a b) q =
      tiedReadout E (E a) q + tiedReadout E (rotate90 (E b)) q := by
  simp only [tiedReadout, tiedInput, add_mul, Finset.sum_add_distrib]

/-- Positive-epsilon normalization preserves weak and strict comparisons,
including zero hidden states. Source: the recorded epsilon = 1e-5 formula. -/
theorem tiedReadout_rmsNorm_comparison (E : V → Channels K → ℝ)
    {eps : ℝ} (heps : 0 < eps) (v : Channels K → ℝ) (q r : V) :
    (tiedReadout E (rmsNorm eps v) q ≤ tiedReadout E (rmsNorm eps v) r ↔
      tiedReadout E v q ≤ tiedReadout E v r) ∧
    (tiedReadout E (rmsNorm eps v) q < tiedReadout E (rmsNorm eps v) r ↔
      tiedReadout E v q < tiedReadout E v r) := by
  rw [tiedReadout_rmsNorm, tiedReadout_rmsNorm]
  have hp := Real.sqrt_pos_of_pos (rmsNorm_radicand_pos heps v)
  exact ⟨div_le_div_iff_of_pos_right hp, div_lt_div_iff_of_pos_right hp⟩

example (E : Fin 97 → Channels (Fin 64) → ℝ) (v : Channels (Fin 64) → ℝ) :
    (0 : ℝ) < 1 / 100000 ∧
    (tiedReadout E (rmsNorm (1 / 100000) v) 2 <
        tiedReadout E (rmsNorm (1 / 100000) v) 95 ↔
      tiedReadout E v 2 < tiedReadout E v 95) := by
  exact ⟨by norm_num, (tiedReadout_rmsNorm_comparison E (by norm_num) v 2 95).2⟩

/-- Two additive operand scores cannot separate a two-label checkerboard.
Source: the four-row argument of the inspected working note, section 2.1.
The diagonal label may win ties; the off-diagonal label must beat it. -/
theorem additive_not_checkerboard {A B Q : Type*} (u : A → Q → ℝ) (v : B → Q → ℝ)
    (a a' : A) (b b' : B) (q r : Q) :
    ¬ (u a r + v b r ≤ u a q + v b q ∧
      u a' r + v b' r ≤ u a' q + v b' q ∧
      u a q + v b' q < u a r + v b' r ∧
      u a' q + v b q < u a' r + v b r) := by
  rintro ⟨h1, h2, h3, h4⟩
  linarith

/-- The actual normalized tied readout cannot realize that checkerboard,
for any shared E. Source: the recorded architecture and the working note's
section 2.1. Normalization and class sharing are retained in the statement. -/
theorem noBranch_not_checkerboard (E : V → Channels K → ℝ)
    {eps : ℝ} (heps : 0 < eps) (a a' b b' q r : V) :
    ¬ (noBranchLogit eps E a b r ≤ noBranchLogit eps E a b q ∧
      noBranchLogit eps E a' b' r ≤ noBranchLogit eps E a' b' q ∧
      noBranchLogit eps E a b' q < noBranchLogit eps E a b' r ∧
      noBranchLogit eps E a' b q < noBranchLogit eps E a' b r) := by
  rintro ⟨h1, h2, h3, h4⟩
  have h1' := (tiedReadout_rmsNorm_comparison E heps (tiedInput E a b) r q).1.mp h1
  have h2' := (tiedReadout_rmsNorm_comparison E heps (tiedInput E a' b') r q).1.mp h2
  have h3' := (tiedReadout_rmsNorm_comparison E heps (tiedInput E a b') q r).2.mp h3
  have h4' := (tiedReadout_rmsNorm_comparison E heps (tiedInput E a' b) q r).2.mp h4
  rw [tiedReadout_tiedInput, tiedReadout_tiedInput] at h1' h2' h3' h4'
  exact additive_not_checkerboard (fun a q => tiedReadout E (E a) q)
    (fun b q => tiedReadout E (rotate90 (E b)) q) a a' b b' q r ⟨h1', h2', h3', h4'⟩

-- The epsilon hypothesis is satisfiable at the actual vocabulary and width.
example : (0 : ℝ) < 1 / 100000 ∧ Fintype.card (Fin 97) = 97 ∧
    Fintype.card (Channels (Fin 64)) = 128 := by
  norm_num [Channels]

/-- A concrete division rectangle from the browser's modulus, with its
original width and epsilon. Its diagonal answer is 2, its off-diagonal
answer 95, and 2 has priority in an argmax tie. Source: section 2.1's
(qb,b), (-qb,-b), (qb,-b), (-qb,b), with q = 2 and b = 1. -/
theorem noBranch_mod97_rectangle (E : Fin 97 → Channels (Fin 64) → ℝ) :
    ¬ (noBranchLogit (1 / 100000) E 2 1 95 ≤ noBranchLogit (1 / 100000) E 2 1 2 ∧
      noBranchLogit (1 / 100000) E 95 96 95 ≤ noBranchLogit (1 / 100000) E 95 96 2 ∧
      noBranchLogit (1 / 100000) E 2 96 2 < noBranchLogit (1 / 100000) E 2 96 95 ∧
      noBranchLogit (1 / 100000) E 95 1 2 < noBranchLogit (1 / 100000) E 95 1 95) := by
  exact noBranch_not_checkerboard E (by norm_num) 2 95 1 96 2 95

-- Multiplication checks the four division labels without computing inverses.
example : (2 * 1) % 97 = (2 : ℕ) ∧ (2 * 96) % 97 = (95 : ℕ) ∧
    (95 * 96) % 97 = (2 : ℕ) ∧ (95 * 1) % 97 = (95 : ℕ) := by
  norm_num

/-- A zero dense matrix deletes the SiLU branch of the complete model at
that parameter. Source: the recorded architecture with the analytic scalar
replacement of M9.2. This does not say the matrix remains zero under training. -/
theorem siluState_of_matrix_zero (eps : ℝ) (weights : SiluWeights V K)
    (hA : ∀ i j, siluMatrix weights i j = 0) (a b : V) :
    siluState eps weights a b = tiedInput (siluEmbedding weights) a b := by
  funext i
  simp [siluState, dense, hA, silu]

/-- The interaction obstruction applies to actual SiLU weights whose dense
matrix vanishes; E keeps all its shared roles. Source: M9.2's full classifier
and section 2.1's no-branch argument. This is a checkpoint identity, not an
invariant manifold or a claim about typical random initialization. -/
theorem siluLogit_of_matrix_zero (eps : ℝ) (weights : SiluWeights V K)
    (hA : ∀ i j, siluMatrix weights i j = 0) (a b q : V) :
    siluLogit eps weights a b q = noBranchLogit eps (siluEmbedding weights) a b q := by
  unfold siluLogit noBranchLogit
  rw [siluState_of_matrix_zero eps weights hA]

-- Zero weights at the source vocabulary and width satisfy both matrix hypotheses.
example : ∀ i j : Channels (Fin 64),
    siluMatrix (0 : SiluWeights (Fin 97) (Fin 64)) i j = 0 := by
  intro i j
  rfl

end Transformer.Landscape.Neural

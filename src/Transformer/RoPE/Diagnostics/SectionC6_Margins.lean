module

public import Transformer.RoPE.Diagnostics.SectionC6_Stability

/-!
# Conditional access guarantees for arbitrary fixed Q/K

Derived consequences of Wu, Du and Peng, arXiv:2609.39929v1, Appendix C.6.
The score-difference coefficients are arbitrary. A positive reference margin
that exceeds its spectral variation budget suffices to preserve a key's
ranking throughout a context window. No learning theorem or unverified
Gaussian calibration is substituted for that explicit hypothesis.

The common position coordinate may represent a query shift: each key's
baseline phase is absorbed using `score_phase_shift`. All statements still
hold activations fixed. They do not establish an end-to-end guarantee after
the input or earlier-layer activations change.
-/

@[expose] public section

open Set

namespace Transformer.RoPE.Diagnostics

/-- A target-minus-competitor score is another rotary score, with coefficient
differences. This retains cancellation that separate absolute bounds lose. -/
theorem head_score_difference {n : ℕ} (a b c d omega : Fin n → ℝ) (distance : ℝ) :
    headScore (a - c) (b - d) omega distance =
      headScore a b omega distance - headScore c d omega distance := by
  unfold headScore
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [score, Pi.sub_apply]
  ring

/-- Exact content-dependent stability budget for a pairwise score margin. -/
theorem pairwise_margin_bound {n : ℕ} (a b c d omega : Fin n → ℝ) (r s : ℝ) :
    |(headScore a b omega r - headScore c d omega r) -
      (headScore a b omega s - headScore c d omega s)| ≤
        |r - s| * spectralSlope (a - c) (b - d) omega := by
  rw [← head_score_difference a b c d omega r, ← head_score_difference a b c d omega s]
  exact head_score_lipschitz_bound _ _ _ _ _

/-- If the actual reference margin exceeds its variation budget, the target
still outranks the competitor at the new position. -/
theorem margin_preserved {n : ℕ} (a b c d omega : Fin n → ℝ) {r s : ℝ}
    (hmargin : |r - s| * spectralSlope (a - c) (b - d) omega <
      headScore a b omega s - headScore c d omega s) :
    headScore c d omega r < headScore a b omega r := by
  have h := (abs_le.mp (pairwise_margin_bound a b c d omega r s)).1
  linarith

example : headScore (fun _ : Fin 1 => 0) (fun _ => 0) (fun _ => 1) (1 / 2) <
    headScore (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 1) (1 / 2) := by
  apply margin_preserved
    (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 1) (s := 0)
  norm_num [headScore, spectralSlope, score, Pi.sub_apply]
  rw [show (16 : ℝ) = (4 : ℝ) ^ 2 by norm_num, Real.sqrt_sq_eq_abs]
  norm_num

/-- A single set of reference-margin checks protects access against every
competing key at every position up to the maximum distance. This is a
sufficient conditional guarantee, with no monotonicity of the scores assumed. -/
theorem window_argmax_preserved {bands keys : ℕ}
    (a b : Fin keys → Fin bands → ℝ) (omega : Fin bands → ℝ)
    (target : Fin keys) (width : ℝ)
    (hmargin : ∀ other, other ≠ target →
      width * spectralSlope (a target - a other) (b target - b other) omega <
        headScore (a target) (b target) omega 0 - headScore (a other) (b other) omega 0) :
    ∀ distance ∈ Icc 0 width, ∀ other, other ≠ target →
      headScore (a other) (b other) omega distance <
        headScore (a target) (b target) omega distance := by
  intro distance hr other hother
  have h := head_window_bound (a target - a other) (b target - b other) omega hr
  rw [head_score_difference, head_score_difference] at h
  have hlow := (abs_le.mp h).1
  have hgap := hmargin other hother
  linarith

example : ∀ distance ∈ Icc 0 (1 / 2 : ℝ), ∀ other : Fin 2, other ≠ 0 →
    headScore ((fun key _ => if key = (0 : Fin 2) then 4 else 0) other)
      (fun _ : Fin 1 => 0) (fun _ => 1) distance <
    headScore (fun _ : Fin 1 => 4) (fun _ => 0) (fun _ => 1) distance := by
  apply window_argmax_preserved
    (fun (key : Fin 2) (_ : Fin 1) => if key = 0 then 4 else 0)
    (fun _ _ => 0) (fun _ => 1) 0 (1 / 2)
  intro other hother
  fin_cases other
  · exact False.elim (hother rfl)
  · norm_num [headScore, spectralSlope, score, Pi.sub_apply]
    rw [show (16 : ℝ) = (4 : ℝ) ^ 2 by norm_num, Real.sqrt_sq_eq_abs]
    norm_num

/-- Without any amplitude restriction, even a fixed nonzero in-field phase
change permits arbitrarily large changes for realizable Q/K vectors. This
explains why an unconditional uniform guarantee over all weights is too strong. -/
theorem unbounded_score_change {omega displacement : ℝ}
    (hphase : omega * displacement ∈ Ioc 0 Real.pi) (bound : ℝ) :
    ∃ query key : ℝ × ℝ,
      bound < |pairScore query key omega displacement - pairScore query key omega 0| := by
  have hcos : Real.cos (omega * displacement) < 1 := by
    have h := Real.strictAntiOn_cos ⟨le_rfl, Real.pi_pos.le⟩
      ⟨hphase.1.le, hphase.2⟩ hphase.1
    simpa using h
  have hcontrast : 0 < 1 - Real.cos (omega * displacement) := by linarith
  let amplitude : ℝ := (|bound| + 1) / (1 - Real.cos (omega * displacement))
  refine ⟨(1, 0), (amplitude, 0), ?_⟩
  have hgap : pairScore (1, 0) (amplitude, 0) omega displacement -
      pairScore (1, 0) (amplitude, 0) omega 0 = -(|bound| + 1) := by
    simp only [pairScore, mul_zero, mul_one, zero_mul, add_zero, sub_zero,
      Real.cos_zero, Real.sin_zero]
    unfold amplitude
    field_simp
    ring
  rw [hgap, abs_neg, abs_of_pos (by positivity : 0 < |bound| + 1)]
  linarith [le_abs_self bound]

example : ∃ query key : ℝ × ℝ,
    (100 : ℝ) < |pairScore query key (Real.pi / 4) 2 - pairScore query key (Real.pi / 4) 0| :=
  unbounded_score_change ⟨by positivity, by nlinarith [Real.pi_pos]⟩ 100

end Transformer.RoPE.Diagnostics

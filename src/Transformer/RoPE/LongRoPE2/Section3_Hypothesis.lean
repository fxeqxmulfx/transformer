module

public import Transformer.RoPE.LongRoPE2.Section2_Prior

/-!
# LongRoPE2 §3.1: the RoPE OOD hypothesis

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, §3.1: "Hypothesis. Insufficient training in
higher RoPE dimensions extends empirical rotation periods beyond the theoretical `2π/θ_i`. This
discrepancy necessitates larger scale factors to mitigate RoPE OOD and lowering the critical
dimension index `d_rcd` below its theoretical `d_tcd`." The text before it explains the longer
periods by "reduced effective ranges (Fig. 3(b))": "the max relative token distance determines
the practical RoPE range". The hypothesis itself is empirical; its two consequences are proved
here.

* `effective_range_factor`: when training covers only the phases of the relative distances
  below `ρ·L_train`, `0 < ρ < 1`, keeping the phases of `L` positions inside them is equivalent to
  `λ_i ≥ L/(ρ·L_train)`, strictly above Eq. (10)'s `L/L_train`;
* `empirical_critical_le`: when every empirical period is at least the theoretical one and the
  empirical periods reach `L_train` exactly from the pair `c` on, `c ≤ d_tcd`, with `d_tcd`
  Eq. (7)'s pair index (`criticalPair`; `toNat` covers `L_train < 2π`, where the ceiling is
  negative). "Below" needs more: the empirical period of the pair `d_tcd − 1` has to reach
  `L_train` as well.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

open ScalingLaws

/-- §3.1: with phases trained only over the distances below `ρ·L_train`, `0 < ρ < 1`, the phase
bound at `L` asks for `λ ≥ L/(ρ·L_train)`, strictly above `L/L_train`. -/
theorem effective_range_factor {omega factor train target rho : ℝ} (homega : 0 < omega)
    (hfactor : 0 < factor) (htrain : 0 < train) (htarget : 0 < target) (hrho0 : 0 < rho)
    (hrho : rho < 1) :
    (target * scaledFrequency omega factor ≤ rho * train * omega ↔
        target / (rho * train) ≤ factor) ∧ target / train < target / (rho * train) :=
  ⟨phase_range_iff homega hfactor (mul_pos hrho0 htrain),
    div_lt_div_of_pos_left htarget (mul_pos hrho0 htrain) (by nlinarith)⟩

example : ((131072 : ℝ) * scaledFrequency 1 128 ≤ 1 / 2 * 2048 * 1 ↔
      (131072 : ℝ) / (1 / 2 * 2048) ≤ 128) ∧
    (131072 : ℝ) / 2048 < 131072 / (1 / 2 * 2048) :=
  effective_range_factor one_pos (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- §3.1: empirical periods at least the theoretical ones reach `L_train` from a pair `c` at most
`d_tcd`. -/
theorem empirical_critical_le {base train : ℝ} {d c : ℕ} {empirical : ℕ → ℝ} (hbase : 1 < base)
    (htrain : 0 < train) (hd : 0 < d)
    (hlonger : ∀ i, period (frequency base d i) ≤ empirical i)
    (hsplit : ∀ i, c ≤ i ↔ train ≤ empirical i) :
    c ≤ (criticalPair base d train).toNat := by
  by_contra h
  have hk := (hsplit (criticalPair base d train).toNat).not.mp h
  have hT := (period_lt_window_iff_critical hbase htrain hd _).mp
    ((hlonger _).trans_lt (not_le.mp hk))
  omega

example : 31 ≤ (criticalPair 10000 96 2048).toNat :=
  empirical_critical_le (empirical := fun i => period (frequency 10000 96 i)) (by norm_num)
    (by norm_num) (by norm_num) (fun _ => le_rfl)
    (fun i => by rw [← not_lt, ← not_lt, phi3_critical.2.2 i])

end Transformer.RoPE.LongRoPE2

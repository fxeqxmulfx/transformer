module

public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Grokking.FlatSpectrum.TablePart.ReadoutShare

/-!
# The cells every code of `P(a) + a b` needs

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT8. Let M return the
table of a code `ρ` from the cells `Ω`. Then the seen cells carry the centered energy of the code,
`(p − 1)² ‖ρ_c‖² ≤ (1 + √κ) p Σ_Ω |ρ(f)|²` (`centered_energy_le_of_isMinOn`). A code of phases with
mean `μ` needs `(p − 1)² (1 − |μ|²) ≤ (1 + √κ) |Ω|` (`phase_code_le_card`), and a balanced one
`(p − 1)² ≤ (1 + √κ) |Ω|` (`card_ge_of_balanced_phase`). For `a³ + a b` at `p = 97` that is at
least 447 of the 9409 cells (`card_ge_cubic`). The effective rank of a balanced code of phases is
at least `(p − 1)² / (p (1 + √κ))` (`effectiveRank_ge_of_balanced_phase`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- The table of `P(a) + a b` through the code `ρ`. -/
noncomputable abbrev codeTable (P : (ZMod p)[X]) (ρ : ZMod p → ℂ) :
    Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => P.eval a + a * b) ρ

/-- **If M returns the table of a code, its seen cells carry the centered energy of the code.** -/
theorem centered_energy_le_of_isMinOn {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) (ρ : ZMod p → ℂ)
    {Ω : Set (ZMod p × ZMod p)} [DecidablePred (· ∈ Ω)]
    (hK : IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = codeTable P ρ ij.1 ij.2}
      (codeTable P ρ)) :
    ((p : ℝ) - 1) ^ 2 * ∑ c, ‖centerCode ρ c‖ ^ 2 ≤
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) *
        (p * ∑ a, ∑ b, if (a, b) ∈ Ω then ‖codeTable P ρ a b‖ ^ 2 else 0) := by
  have h := nuclearNorm_sq_le_of_isMinOn hK
  rw [ZMod.card, min_self] at h
  exact (centered_energy_le_nuclearNorm_sq hm ρ).trans
    (mul_le_mul_of_nonneg_left h (by positivity))

example := centered_energy_le_of_isMinOn (P := (X ^ 2 : (ZMod 3)[X])) (Ω := Set.univ) (by simp)
  (fun c => (c.val : ℂ)) (isMinOn_univ _)

/-- **A code of phases with mean `μ`** needs `(p − 1)² (1 − |μ|²) ≤ (1 + √κ) |Ω|`. -/
theorem phase_code_le_card {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) {ρ : ZMod p → ℂ}
    (hρ : ∀ c, ‖ρ c‖ = 1) {Ω : Finset (ZMod p × ZMod p)}
    (hK : IsMinOn nuclearNorm {Y | ∀ ij ∈ (Ω : Set (ZMod p × ZMod p)),
      Y ij.1 ij.2 = codeTable P ρ ij.1 ij.2} (codeTable P ρ)) :
    ((p : ℝ) - 1) ^ 2 * (1 - (‖∑ c, ρ c‖ / p) ^ 2) ≤
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) * Ω.card := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hE : ∑ c, ‖centerCode ρ c‖ ^ 2 = p * (1 - (‖∑ c, ρ c‖ / p) ^ 2) := by
    have h := sum_norm_sq_centerCode ρ
    simp only [hρ, one_pow, Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul,
      mul_one] at h
    have e : (p : ℝ) * (1 - (‖∑ c, ρ c‖ / p) ^ 2) = p - ‖∑ c, ρ c‖ ^ 2 / p := by
      rw [div_pow]
      field_simp
    linarith
  have h1 := centered_energy_le_nuclearNorm_sq hm ρ
  have h2 := nuclearNorm_sq_le_card_of_isMinOn (K := codeTable P ρ)
    (fun a b => hρ (P.eval a + a * b)) hK
  rw [ZMod.card, min_self] at h2
  rw [hE] at h1
  have hA : 0 ≤ 1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)) := by positivity
  have h3 := h1.trans (mul_le_mul_of_nonneg_left h2 hA)
  refine le_of_mul_le_mul_left ?_ hp
  calc (p : ℝ) * (((p : ℝ) - 1) ^ 2 * (1 - (‖∑ c, ρ c‖ / p) ^ 2))
      = ((p : ℝ) - 1) ^ 2 * (p * (1 - (‖∑ c, ρ c‖ / p) ^ 2)) := by ring
    _ ≤ (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) * (p * Ω.card) := h3
    _ = p * ((1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) * Ω.card) := by ring

example := phase_code_le_card (P := (X ^ 2 : (ZMod 3)[X])) (ρ := fun c => (ZMod.stdAddChar c : ℂ))
  (Ω := univ) (by simp) (fun c => by simp) (by rw [Finset.coe_univ]; exact isMinOn_univ _)

/-- **A balanced code of phases** needs `(p − 1)² ≤ (1 + √κ) |Ω|`. -/
theorem card_ge_of_balanced_phase {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) {ρ : ZMod p → ℂ}
    (hρ : ∀ c, ‖ρ c‖ = 1) (hbal : ∑ c, ρ c = 0) {Ω : Finset (ZMod p × ZMod p)}
    (hK : IsMinOn nuclearNorm {Y | ∀ ij ∈ (Ω : Set (ZMod p × ZMod p)),
      Y ij.1 ij.2 = codeTable P ρ ij.1 ij.2} (codeTable P ρ)) :
    ((p : ℝ) - 1) ^ 2 ≤ (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) * Ω.card := by
  have h := phase_code_le_card hm hρ hK
  rwa [hbal, norm_zero, zero_div, zero_pow two_ne_zero, sub_zero, mul_one] at h

example := card_ge_of_balanced_phase (P := (X ^ 2 : (ZMod 3)[X]))
  (ρ := fun c => (ZMod.stdAddChar c : ℂ)) (Ω := univ) (by simp) (fun c => by simp)
  (by simpa using sum_stdAddChar_mul (p := 3) 1) (by rw [Finset.coe_univ]; exact isMinOn_univ _)

/-- **`a³ + a b` at `p = 97`: a balanced code of phases needs at least 447 of the 9409 cells.** -/
theorem card_ge_cubic {ρ : ZMod 97 → ℂ} (hρ : ∀ c, ‖ρ c‖ = 1) (hbal : ∑ c, ρ c = 0)
    {Ω : Finset (ZMod 97 × ZMod 97)}
    (hK : IsMinOn nuclearNorm {Y | ∀ ij ∈ (Ω : Set (ZMod 97 × ZMod 97)),
      Y ij.1 ij.2 = taskKernel (fun a b => a ^ 3 + a * b) ρ ij.1 ij.2}
      (taskKernel (fun a b => a ^ 3 + a * b) ρ)) : 447 ≤ Ω.card := by
  have : Fact (Nat.Prime 97) := ⟨by norm_num⟩
  have hk : taskKernel (fun a b : ZMod 97 => a ^ 3 + a * b) ρ = codeTable (X ^ 3) ρ := by
    ext a b
    simp [codeTable, taskKernel]
  rw [hk] at hK
  have h := card_ge_of_balanced_phase (P := X ^ 3) (by simp) hρ hbal hK
  rw [natDegree_X_pow] at h
  norm_num at h
  by_contra hlt
  have hΩ : (Ω.card : ℝ) ≤ 446 := by
    have h446 : Ω.card ≤ 446 := by omega
    exact_mod_cast h446
  have hs : √386 ≤ 19.65 := Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have h5 : (1 + √386) * (Ω.card : ℝ) ≤ 20.65 * 446 :=
    mul_le_mul (by linarith) hΩ (Nat.cast_nonneg _) (by norm_num)
  norm_num at h5
  linarith

example : 447 ≤ (univ : Finset (ZMod 97 × ZMod 97)).card :=
  have : Fact (Nat.Prime 97) := ⟨by norm_num⟩
  card_ge_cubic (ρ := fun c => (ZMod.stdAddChar c : ℂ)) (fun c => by simp)
    (by simpa using sum_stdAddChar_mul (p := 97) 1)
    (by rw [Finset.coe_univ]; exact isMinOn_univ _)

/-- **The effective rank of a balanced code of phases** is at least `(p − 1)² / (p (1 + √κ))`. -/
theorem effectiveRank_ge_of_balanced_phase {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    {ρ : ZMod p → ℂ} (hρ : ∀ c, ‖ρ c‖ = 1) (hbal : ∑ c, ρ c = 0) :
    ((p : ℝ) - 1) ^ 2 / (p * (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)))) ≤
      effectiveRank (codeTable P ρ) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hF : frobeniusSq (codeTable P ρ) = p * p := by
    simp [frobeniusSq, taskKernel, hρ, ZMod.card]
  have hE : ∑ c, ‖centerCode ρ c‖ ^ 2 = p := by
    have h := sum_norm_sq_centerCode ρ
    rw [hbal, norm_zero, zero_pow two_ne_zero, zero_div, add_zero] at h
    rw [← h]
    simp [hρ, ZMod.card]
  have h := centered_energy_le_nuclearNorm_sq hm ρ
  rw [hE] at h
  rw [effectiveRank, div_pow, Real.sq_sqrt (by rw [hF]; positivity), hF,
    div_le_div_iff₀ (by positivity) (by positivity)]
  calc ((p : ℝ) - 1) ^ 2 * (p * p) = p * (((p : ℝ) - 1) ^ 2 * p) := by ring
    _ ≤ p * ((1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) *
        nuclearNorm (codeTable P ρ) ^ 2) := mul_le_mul_of_nonneg_left h hp.le
    _ = _ := by ring

example := effectiveRank_ge_of_balanced_phase (P := (X ^ 2 : (ZMod 3)[X]))
  (ρ := fun c => (ZMod.stdAddChar c : ℂ)) (by simp) (fun c => by simp)
  (by simpa using sum_stdAddChar_mul (p := 3) 1)

end Transformer.Grokking.FlatSpectrum.TablePart

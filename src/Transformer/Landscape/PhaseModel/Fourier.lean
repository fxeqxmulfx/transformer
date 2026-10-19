module

public import Transformer.Landscape.PhaseModel.Orthogonality

/-!
# Phase model — a Fourier bound on the full division table

For phases `ψ`, let `û_j = Σ_x e^{i(ψ_x - 2πjx/n)}` be the Fourier
coefficients of `x ↦ e^{iψ_x}`, so that `Re û_j = Σ_x cos (ψ_x - 2πjx/n)`.
The orthogonality of the characters gives Parseval's identity
`Σ_j |û_j|² = n²` (`sum_normSq_fourier`) and the cubic identity
`Σ_j |û_j|² Re û_j = n Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`
(`sum_normSq_mul_re_fourier`), where the double sum is `n² - V_full(ψ)`
(`divisionEnergy_univ_eq`). Under the weights `|û_j|² / n²` the `Re û_j`
have the mean `n - V_full(ψ)/n`, so some `j` has
`n Σ_x (1 - cos (ψ_x - 2πjx/n)) ≤ V_full(ψ)` (`exists_mul_sum_one_sub_cos_le`).

Each theorem is that of `Landscape.PhaseModel.Abelian.Fourier` at `G = Fin n`: the
character `cyclicChar n j` has the values `e^{2πijx/n}`, and the sums over
the characters are sums over `j < n` (`bijective_cyclicChar`).

Source: item M6.8 of `landscape_plan.md`.
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel

variable {n : ℕ} [NeZero n]

/-- **Parseval's identity**: `Σ_j |û_j|² = n²` for
`û_j = Σ_x e^{i(ψ_x - 2πjx/n)}`. -/
theorem sum_normSq_fourier (ψ : EucSpace n) :
    ∑ j : Fin n, Complex.normSq (∑ x, Complex.exp
      (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) = n ^ 2 := by
  have h := Abelian.sum_normSq_fourier ψ
  rw [Fintype.card_fin] at h
  refine (Finset.sum_congr rfl fun j _ => ?_).trans ((bijective_cyclicChar.sum_comp _).trans h)
  simp only [Complex.ofReal_sub, sub_mul, Complex.exp_sub, Abelian.exp_characterPhase,
    ← cyclicChar_apply]

/-- **The cubic identity**:
`Σ_j |û_j|² Re û_j = n Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, the real part of
`Σ_j conj(û_j)² û_j = n Σ_{y,z} e^{i(ψ_{y+z} - ψ_y - ψ_z)}`. -/
theorem sum_normSq_mul_re_fourier (ψ : EucSpace n) :
    ∑ j : Fin n, Complex.normSq (∑ x, Complex.exp
        (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)) *
      (∑ x, Complex.exp (↑(ψ.ofLp x - (characterPhase n (j : ℕ)).ofLp x) * Complex.I)).re =
      n * ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  have h := Abelian.sum_normSq_mul_re_fourier ψ
  rw [Fintype.card_fin] at h
  refine (Finset.sum_congr rfl fun j _ => ?_).trans ((bijective_cyclicChar.sum_comp _).trans h)
  simp only [Complex.ofReal_sub, sub_mul, Complex.exp_sub, Abelian.exp_characterPhase,
    ← cyclicChar_apply]

/-- **The energy of the full table as a double sum**:
`V_full(ψ) = n² - Σ_{y,z} cos (ψ_{y+z} - ψ_y - ψ_z)`, with the pair
`(y + z, y)` for `x / y`. -/
theorem divisionEnergy_univ_eq (ψ : EucSpace n) :
    divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ =
      n ^ 2 - ∑ y, ∑ z, cos (ψ.ofLp (y + z) - ψ.ofLp y - ψ.ofLp z) := by
  have h := Abelian.divisionEnergy_univ_eq ψ
  rw [Fintype.card_fin] at h
  exact h

/-- **Some character is within `V_full(ψ)/n` of the phases**: there is `j`
with `n Σ_x (1 - cos (ψ_x - 2πjx/n)) ≤ V_full(ψ)`. Otherwise every
`n Re û_j < n² - V_full(ψ)`, and the weights `|û_j|²`, of sum `n²`, would
give `n Σ_j |û_j|² Re û_j < n² (n² - V_full(ψ))`, against the cubic
identity. -/
theorem exists_mul_sum_one_sub_cos_le (ψ : EucSpace n) : ∃ j : ℤ,
    n * ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) ≤
      divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ := by
  obtain ⟨χ, hχ⟩ := Abelian.exists_mul_sum_one_sub_cos_le ψ
  obtain ⟨j, rfl⟩ := bijective_cyclicChar.2 χ
  refine ⟨(j : ℕ), ?_⟩
  simp only [Fintype.card_fin] at hχ
  simp only [cos_sub_characterPhase]
  exact hχ

/-- The hypothesis `NeZero n` of the theorems above holds for `n = 4`; at the
phases `0`, a zero of `V_full`, Parseval's identity reads
`Σ_j |Σ_x e^{-2πijx/4}|² = 16`. -/
example : ∑ j : Fin 4, Complex.normSq (∑ x, Complex.exp
    (↑((0 : EucSpace 4).ofLp x - (characterPhase 4 (j : ℕ)).ofLp x) * Complex.I)) = 16 := by
  rw [sum_normSq_fourier]
  norm_num

end Transformer.Landscape.PhaseModel

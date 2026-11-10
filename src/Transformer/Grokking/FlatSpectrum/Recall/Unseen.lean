module

public import Transformer.Grokking.FlatSpectrum.OneCellMin
public import Transformer.Grokking.FlatSpectrum.Recall.Basic

/-!
# The minimum of M answers one unseen pair with the opposite phase

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC6, for the recall kernel of
*Associative recall has a flat spectrum* (added), items AR2 and AR3.

AR3 says that recall with a readout without zeros is a minimum of M from no proper set of
pairs, and not what the minimum is. With one unseen pair `(q, π)` it is known. The recall
kernel of a centered readout has `K Kᴴ = c 1` with `c = |ν|^{m−1} ∑_v |φ(v)|²`. Its cell is
`φ(π q)`, and the rest of its column is `g = ∑_{p ≠ q} |φ(π p)|²`, positive when there are two
keys (`restSq_recallKernel_pos`). So by OC5

- a minimum of M from the other pairs exists (`exists_isMinOn_recallKernel`), and
- every minimum answers the unseen pair with `−θ φ(π q)`, `g / (2c − g) ≤ θ ≤ g / (c − g)`
  (`exists_neg_mul_of_isMinOn_recallKernel`): the opposite phase, and for a real readout the
  wrong sign.

For a readout of modulus one, `g = m − 1` and `c = |ν|^m`
(`restSq_recallKernel_of_norm_eq_one`, `scale_of_norm_eq_one`), so
`(m − 1) / (2 |ν|^m − (m − 1)) ≤ θ ≤ (m − 1) / (|ν|^m − (m − 1))`
(`exists_neg_mul_of_isMinOn_recallKernel_of_norm_eq_one`): the answer is nearly zero, on the
wrong side.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

omit [Fintype ν] in
/-- **The other keys of a dictionary carry weight** (`review/flat_spectrum.md`, OC6): with two
keys and a readout without zeros, `g = ∑_{p ≠ q} |φ(π p)|² > 0`. -/
lemma restSq_recallKernel_pos [Nontrivial κ] {φ : ν → ℂ} (h0 : ∀ v, φ v ≠ 0) (q : κ)
    (π : κ → ν) : 0 < restSq (recallKernel φ) q π := by
  obtain ⟨p, hp⟩ := exists_ne q
  refine Finset.sum_pos' (fun p _ => by split_ifs <;> positivity) ⟨p, Finset.mem_univ p, ?_⟩
  split_ifs with h
  · exact absurd h hp.symm
  · exact pow_pos (norm_pos_iff.2 (h0 (π p))) 2

omit [Fintype ν] in
/-- For a readout of modulus one, `g = m − 1`. -/
lemma restSq_recallKernel_of_norm_eq_one {φ : ν → ℂ} (h1 : ∀ v, ‖φ v‖ = 1) (q : κ)
    (π : κ → ν) : restSq (recallKernel φ) q π = Fintype.card κ - 1 := by
  have h := norm_sq_add_restSq (recallKernel φ) q π
  have h2 : ∀ p, ‖recallKernel φ p π‖ = 1 := fun p => h1 (π p)
  simp only [h2, one_pow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at h
  linarith

omit [DecidableEq κ] in
/-- For a readout of modulus one, `c = |ν|^m`. -/
lemma scale_of_norm_eq_one [Nonempty κ] {φ : ν → ℂ} (h1 : ∀ v, ‖φ v‖ = 1) :
    (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2 =
      (Fintype.card ν : ℝ) ^ Fintype.card κ := by
  simp only [h1, one_pow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
  rw [← pow_succ, Nat.sub_add_cancel Fintype.card_pos]

variable [DecidableEq ν]

/-- **From all pairs but one, a minimum of M exists** (`review/flat_spectrum.md`, OC6). -/
theorem exists_isMinOn_recallKernel [Nontrivial κ] {φ : ν → ℂ} (hφ : ∑ v, φ v = 0)
    (h0 : ∀ v, φ v ≠ 0) (q : κ) (π : κ → ν) :
    ∃ X : Matrix κ (κ → ν) ℂ,
      (∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))), X kl.1 kl.2 = recallKernel φ kl.1 kl.2) ∧
      IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))),
        Y kl.1 kl.2 = recallKernel φ kl.1 kl.2} X :=
  exists_isMinOn_compl_singleton (scale_pos (h0 (π q))) (recallKernel_mul_conjTranspose hφ)
    (restSq_recallKernel_pos h0 q π)

/-- **The minimum of M answers one unseen pair with the opposite phase**
(`review/flat_spectrum.md`, OC6): for two keys or more and a centered readout without zeros,
every minimum of the nuclear norm among the matrices equal to the recall kernel at every pair
but `(q, π)` has `X(q, π) = −θ φ(π q)` with `0 < θ`, `g ≤ θ (2c − g)` and `θ (c − g) ≤ g`, where
`c = |ν|^{m−1} ∑_v |φ(v)|²` and `g = ∑_{p ≠ q} |φ(π p)|²`. -/
theorem exists_neg_mul_of_isMinOn_recallKernel [Nontrivial κ] {φ : ν → ℂ} (hφ : ∑ v, φ v = 0)
    (h0 : ∀ v, φ v ≠ 0) {q : κ} {π : κ → ν} {X : Matrix κ (κ → ν) ℂ}
    (hX : ∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))), X kl.1 kl.2 = recallKernel φ kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))),
      Y kl.1 kl.2 = recallKernel φ kl.1 kl.2} X) :
    ∃ θ : ℝ, 0 < θ ∧
      restSq (recallKernel φ) q π ≤
        θ * (2 * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2) -
          restSq (recallKernel φ) q π) ∧
      θ * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2 -
        restSq (recallKernel φ) q π) ≤ restSq (recallKernel φ) q π ∧
      X q π = -(θ : ℂ) * φ (π q) :=
  exists_neg_mul_of_isMinOn_compl_singleton (scale_pos (h0 (π q)))
    (recallKernel_mul_conjTranspose hφ) (h0 (π q)) (restSq_recallKernel_pos h0 q π) hX hmin

/-- **A readout of modulus one** (`review/flat_spectrum.md`, OC6): the unseen pair is answered
with `−θ φ(π q)`, `0 < θ`, `m − 1 ≤ θ (2 |ν|^m − (m − 1))` and
`θ (|ν|^m − (m − 1)) ≤ m − 1`. -/
theorem exists_neg_mul_of_isMinOn_recallKernel_of_norm_eq_one [Nontrivial κ] {φ : ν → ℂ}
    (hφ : ∑ v, φ v = 0) (h1 : ∀ v, ‖φ v‖ = 1) {q : κ} {π : κ → ν} {X : Matrix κ (κ → ν) ℂ}
    (hX : ∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))), X kl.1 kl.2 = recallKernel φ kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(q, π)}ᶜ : Set (κ × (κ → ν))),
      Y kl.1 kl.2 = recallKernel φ kl.1 kl.2} X) :
    ∃ θ : ℝ, 0 < θ ∧
      (Fintype.card κ : ℝ) - 1 ≤
        θ * (2 * (Fintype.card ν : ℝ) ^ Fintype.card κ - (Fintype.card κ - 1)) ∧
      θ * ((Fintype.card ν : ℝ) ^ Fintype.card κ - (Fintype.card κ - 1)) ≤ Fintype.card κ - 1 ∧
      X q π = -(θ : ℂ) * φ (π q) := by
  have h0 : ∀ v, φ v ≠ 0 := fun v => norm_pos_iff.1 (by rw [h1 v]; exact one_pos)
  obtain ⟨θ, hθ, hge, hle, hx⟩ := exists_neg_mul_of_isMinOn_recallKernel hφ h0 hX hmin
  rw [scale_of_norm_eq_one h1, restSq_recallKernel_of_norm_eq_one h1] at hge hle
  exact ⟨θ, hθ, hge, hle, hx⟩

/-- The readout `±1` of two values is centered. -/
lemma sum_sign_eq_zero : ∑ v : Fin 2, (![1, -1] : Fin 2 → ℂ) v = 0 := by
  simp [Fin.sum_univ_two]

/-- The readout `±1` of two values has modulus one. -/
lemma norm_sign_eq_one (v : Fin 2) : ‖(![1, -1] : Fin 2 → ℂ) v‖ = 1 := by
  fin_cases v <;> simp

/-- The readout `±1` of two values has no zero. -/
lemma sign_ne_zero (v : Fin 2) : (![1, -1] : Fin 2 → ℂ) v ≠ 0 :=
  norm_pos_iff.1 (by rw [norm_sign_eq_one v]; exact one_pos)

/-- Two keys with values `±1`, and the pair of the key `0` and the dictionary that binds both
keys to the value `0` unseen: a minimum of M exists, and it answers the pair with `−θ` in the
place of `1`, with `1 / 7 ≤ θ ≤ 1 / 3`. -/
example : ∃ X : Matrix (Fin 2) (Fin 2 → Fin 2) ℂ, ∃ θ : ℝ, 1 ≤ θ * 7 ∧ θ * 3 ≤ 1 ∧
    X 0 (fun _ => 0) = -(θ : ℂ) := by
  obtain ⟨X, hX, hmin⟩ :=
    exists_isMinOn_recallKernel (κ := Fin 2) sum_sign_eq_zero sign_ne_zero 0 fun _ => 0
  obtain ⟨θ, -, h1, h2, h3⟩ :=
    exists_neg_mul_of_isMinOn_recallKernel_of_norm_eq_one sum_sign_eq_zero norm_sign_eq_one hX hmin
  norm_num at h1 h2
  exact ⟨X, θ, by linarith, by linarith, by simpa using h3⟩

example := restSq_recallKernel_pos (κ := Fin 2) sign_ne_zero 0 fun _ => 0

example := restSq_recallKernel_of_norm_eq_one (κ := Fin 2) norm_sign_eq_one 0 fun _ => 0

example := scale_of_norm_eq_one (κ := Fin 2) norm_sign_eq_one

/-- The same pair, by the statement for every readout without zeros. -/
example : ∃ X : Matrix (Fin 2) (Fin 2 → Fin 2) ℂ, ∃ θ : ℝ, 0 < θ ∧
    X 0 (fun _ => 0) = -(θ : ℂ) * (![1, -1] : Fin 2 → ℂ) 0 := by
  obtain ⟨X, hX, hmin⟩ :=
    exists_isMinOn_recallKernel (κ := Fin 2) sum_sign_eq_zero sign_ne_zero 0 fun _ => 0
  obtain ⟨θ, h0, -, -, h3⟩ :=
    exists_neg_mul_of_isMinOn_recallKernel sum_sign_eq_zero sign_ne_zero hX hmin
  exact ⟨X, θ, h0, h3⟩

end Transformer.Grokking.FlatSpectrum.Recall

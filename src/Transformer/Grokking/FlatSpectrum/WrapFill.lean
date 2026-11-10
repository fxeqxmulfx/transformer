module

public import Transformer.Grokking.FlatSpectrum.CompareFill
public import Transformer.Grokking.FlatSpectrum.StepWrap

/-!
# The wrap of a step is not recovered at its boundary

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW6.
The sign of the step, `wrapKernel p`, is the comparison kernel with its columns reversed, and
reindexing rows and columns carries a minimum of the nuclear norm to a minimum
(`isMinOn_submatrix`). The boundary of the wrap is the pairs with `b + x = p − 1`, the last
sums that do not wrap, and with `b + x = p`, the first that do.

* The sign of the step is not a minimum among the matrices equal to it on `Ω` once `Ω` misses
  a cell of the boundary (`not_isMinOn_wrapKernel`).
* With one cell of the boundary unseen, no minimum holds the true sign there
  (`ne_of_isMinOn_wrapKernel`), and the kernel with that cell flipped is a minimum
  (`isMinOn_wrapKernel_flip`, `isMinOn_wrapKernel_flip_succ`).

So a learner that minimizes the nuclear norm of the sign of the step decides an unseen pair
at the wrap boundary against the truth: it answers `b − p` for `b`, or `b` for `b − p`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

/-- Reindexing the rows and the columns carries a minimum of the nuclear norm to a minimum. -/
theorem isMinOn_submatrix {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
    {K X : Matrix m n ℂ} {Ω : Set (m × n)} (σ : m ≃ m) (τ : n ≃ n)
    (h : IsMinOn nuclearNorm {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = K kl.1 kl.2} X) :
    IsMinOn nuclearNorm
      {Y | ∀ kl ∈ {kl : m × n | (σ kl.1, τ kl.2) ∈ Ω}, Y kl.1 kl.2 = K.submatrix σ τ kl.1 kl.2}
      (X.submatrix σ τ) := by
  rw [isMinOn_iff] at h ⊢
  intro Y hY
  have h1 := h (Y.submatrix σ.symm τ.symm) fun kl hkl => by
    have h2 := hY (σ.symm kl.1, τ.symm kl.2) (by simpa using hkl)
    simpa using h2
  rwa [nuclearNorm_submatrix, ← nuclearNorm_submatrix X σ τ] at h1

example : IsMinOn nuclearNorm
    {Y | ∀ kl ∈ {kl : Fin 2 × Fin 2 | (kl.1, Fin.revPerm kl.2) ∈ ({(1, 0)}ᶜ : Set _)},
      Y kl.1 kl.2 = (compareKernel 2).submatrix (Equiv.refl _) Fin.revPerm kl.1 kl.2}
    ((compareKernel 2 + (2 : ℂ) • single 1 0 1).submatrix (Equiv.refl _) Fin.revPerm) :=
  isMinOn_submatrix _ _ (isMinOn_compareKernel_flip_succ (i := 0) rfl)

variable {p : ℕ}

/-- **The wrap is not recovered at its boundary**, `review/flat_spectrum.md`, item CW6: the
sign of the step is not a minimum of the nuclear norm among the matrices equal to it on `Ω`
when `Ω` misses a pair with `b + x = p − 1` or `b + x = p`. -/
theorem not_isMinOn_wrapKernel {Ω : Set (Fin p × Fin p)} {b x : Fin p}
    (h : (b : ℕ) + x + 1 = p ∨ (b : ℕ) + x = p) (hbx : (b, x) ∉ Ω) :
    ¬IsMinOn nuclearNorm {X | ∀ kl ∈ Ω, X kl.1 kl.2 = wrapKernel p kl.1 kl.2} (wrapKernel p) := by
  intro hmin
  have hC := isMinOn_submatrix (Equiv.refl _) Fin.revPerm hmin
  rw [wrapKernel_submatrix] at hC
  rcases h with h | h
  · have hrev : Fin.rev b = x := Fin.ext (by rw [Fin.val_rev]; omega)
    exact not_isMinOn_compareKernel_diag (i := b) (by simpa [hrev] using hbx) hC
  · exact not_isMinOn_compareKernel_succ (i := Fin.rev x) (i' := b) (by rw [Fin.val_rev]; omega)
      (by simpa using hbx) hC

/-- Modulo `3`, with the pair `(1, 1)` unseen, `1 + 1 = 3 − 1`: the sign is not recovered. -/
example : ¬IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 1)}ᶜ : Set (Fin 3 × Fin 3)), X kl.1 kl.2 = wrapKernel 3 kl.1 kl.2}
    (wrapKernel 3) :=
  not_isMinOn_wrapKernel (b := 1) (x := 1) (Or.inl rfl) (by simp)

/-- **No minimum keeps the true sign**, `review/flat_spectrum.md`, item CW6: with one pair
of the wrap boundary unseen, every minimum of the nuclear norm among the matrices equal to the
sign of the step at the other pairs differs from it at that pair. -/
theorem ne_of_isMinOn_wrapKernel {b x : Fin p} (h : (b : ℕ) + x + 1 = p ∨ (b : ℕ) + x = p)
    {X : Matrix (Fin p) (Fin p) ℂ}
    (hX : ∀ kl ∈ ({(b, x)}ᶜ : Set (Fin p × Fin p)), X kl.1 kl.2 = wrapKernel p kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(b, x)}ᶜ : Set (Fin p × Fin p)), X kl.1 kl.2 = wrapKernel p kl.1 kl.2} X) :
    X b x ≠ wrapKernel p b x := by
  intro hbx
  have hXW : X = wrapKernel p := by
    rw [eq_add_single_of_mem hX, hbx, sub_self, zero_smul, add_zero]
  rw [hXW] at hmin
  exact not_isMinOn_wrapKernel h (by simp) hmin

/-- **The flipped pair is a minimum**, `review/flat_spectrum.md`, item CW6: with the pair
`(b, x)`, `b + x = p − 1`, `0 < b < p − 1`, unseen, the sign of the step with `−1` at that
pair, a wrap that does not happen, has the least nuclear norm among the matrices equal to the
sign of the step at the other pairs. -/
theorem isMinOn_wrapKernel_flip {b₀ b b₁ x : Fin p} (h₀ : (b₀ : ℕ) + 1 = b)
    (h₁ : (b : ℕ) + 1 = b₁) (hx : (b : ℕ) + x + 1 = p) :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(b, x)}ᶜ : Set (Fin p × Fin p)), X kl.1 kl.2 = wrapKernel p kl.1 kl.2}
      (wrapKernel p + (-2 : ℂ) • single b x 1) := by
  have hrev : Fin.rev x = b := Fin.ext (by rw [Fin.val_rev]; omega)
  have hC := isMinOn_submatrix (Equiv.refl _) Fin.revPerm (isMinOn_compareKernel_flip_diag h₀ h₁)
  have hK : (compareKernel p + (-2 : ℂ) • single b b 1).submatrix (Equiv.refl _) Fin.revPerm =
      wrapKernel p + (-2 : ℂ) • single b x 1 := by
    ext r c
    have hc : c.rev = b ↔ c = x := by rw [Fin.rev_eq_iff, ← hrev, Fin.rev_rev]
    rw [submatrix_apply, add_smul_single_one_apply, add_smul_single_one_apply,
      wrapKernel_eq_submatrix, submatrix_apply, Fin.revPerm_apply]
    simp only [Equiv.refl_apply, hc]
  rw [hK, ← wrapKernel_eq_submatrix, isMinOn_iff] at hC
  rw [isMinOn_iff]
  refine fun X hX => hC X fun kl hkl => hX kl fun hkl' => hkl ?_
  rw [Set.mem_singleton_iff] at hkl' ⊢
  simp [hkl', hrev]

/-- **The flipped pair is a minimum**, `review/flat_spectrum.md`, item CW6: with the pair
`(b, x)`, `b + x = p`, unseen, the sign of the step with `1` at that pair, a wrap left out,
has the least nuclear norm among the matrices equal to the sign of the step at the other
pairs. -/
theorem isMinOn_wrapKernel_flip_succ {b x : Fin p} (hx : (b : ℕ) + x = p) :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(b, x)}ᶜ : Set (Fin p × Fin p)), X kl.1 kl.2 = wrapKernel p kl.1 kl.2}
      (wrapKernel p + (2 : ℂ) • single b x 1) := by
  have h₁ : ((Fin.rev x : Fin p) : ℕ) + 1 = b := by rw [Fin.val_rev]; omega
  have hC := isMinOn_submatrix (Equiv.refl _) Fin.revPerm (isMinOn_compareKernel_flip_succ h₁)
  have hK : (compareKernel p + (2 : ℂ) • single b (Fin.rev x) 1).submatrix (Equiv.refl _)
      Fin.revPerm = wrapKernel p + (2 : ℂ) • single b x 1 := by
    ext r c
    rw [submatrix_apply, add_smul_single_one_apply, add_smul_single_one_apply,
      wrapKernel_eq_submatrix, submatrix_apply, Fin.revPerm_apply]
    simp only [Equiv.refl_apply, Fin.rev_inj]
  rw [hK, ← wrapKernel_eq_submatrix, isMinOn_iff] at hC
  rw [isMinOn_iff]
  refine fun X hX => hC X fun kl hkl => hX kl fun hkl' => hkl ?_
  rw [Set.mem_singleton_iff] at hkl' ⊢
  simp [hkl']

/-- Modulo `3`: with `(1, 1)` unseen the minimum says that `1 + 1` wraps, and with `(1, 2)`
unseen that `1 + 2` does not. -/
example : IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 1)}ᶜ : Set (Fin 3 × Fin 3)), X kl.1 kl.2 = wrapKernel 3 kl.1 kl.2}
    (wrapKernel 3 + (-2 : ℂ) • single 1 1 1) :=
  isMinOn_wrapKernel_flip (b₀ := 0) (b₁ := 2) rfl rfl rfl

example : IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 2)}ᶜ : Set (Fin 3 × Fin 3)), X kl.1 kl.2 = wrapKernel 3 kl.1 kl.2}
    (wrapKernel 3 + (2 : ℂ) • single 1 2 1) :=
  isMinOn_wrapKernel_flip_succ rfl

example : (wrapKernel 3 + (2 : ℂ) • single 1 2 1 : Matrix (Fin 3) (Fin 3) ℂ) 1 2 ≠
    wrapKernel 3 1 2 :=
  ne_of_isMinOn_wrapKernel (Or.inr rfl) (add_smul_single_apply_of_mem _)
    (isMinOn_wrapKernel_flip_succ rfl)

end Transformer.Grokking.FlatSpectrum

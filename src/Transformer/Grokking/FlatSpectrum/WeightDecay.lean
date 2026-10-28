module

public import Transformer.Grokking.FlatSpectrum.Factorization

/-!
# Weight decay at its optimum is M

Source: `review/flat_spectrum.md`, Lemma 1: "Weight decay on a bilinear layer therefore acts as
a nuclear-norm penalty. Training with weight decay then behaves like minimum-nuclear-norm
interpolation from Ω. This is the modeling step. It is exact for a pure factorization".

The pure factorization is `X = U Vᴴ`, with `U : m × r` and `V : n × r` for any width `r`, and
its weight decay is `½ (‖U‖² + ‖V‖²)`. A fit to a set `S` of matrices is a factorization with
`U Vᴴ ∈ S`; an optimal fit has the least weight decay among the fits of every width
(`IsOptimalFit`). Lemma 1 (`nuclearNorm_isLeast`) makes this exactly M:

* a fit is optimal exactly when its product has the least nuclear norm in `S` and its weight
  decay equals that norm (`isOptimalFit_iff`);
* the products of the optimal fits are exactly the matrices of least nuclear norm in `S`
  (`setOf_isOptimalFit_eq`);
* for `K ∈ S`, some fit is optimal and every optimal fit multiplies to `K` exactly when `K` is
  the only matrix of least nuclear norm in `S`, that is, when M recovers `K`
  (`forall_isMinOn_iff_iff_isOptimalFit`).

That training reaches this optimum is the modeling step of the source, and is not proved.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- A fit `U Vᴴ ∈ S` of least weight decay `½ (‖U‖² + ‖V‖²)` among the fits `U' V'ᴴ ∈ S` of
every width. -/
def IsOptimalFit (S : Set (Matrix m n ℂ)) {r : ℕ} (U : Matrix m (Fin r) ℂ)
    (V : Matrix n (Fin r) ℂ) : Prop :=
  U * Vᴴ ∈ S ∧ ∀ (r' : ℕ) (U' : Matrix m (Fin r') ℂ) (V' : Matrix n (Fin r') ℂ), U' * V'ᴴ ∈ S →
    (frobeniusSq U + frobeniusSq V) / 2 ≤ (frobeniusSq U' + frobeniusSq V') / 2

/-- **A fit is optimal exactly when its product has the least nuclear norm** in `S` and its
weight decay equals that norm, by Lemma 1. -/
theorem isOptimalFit_iff {S : Set (Matrix m n ℂ)} {r : ℕ} {U : Matrix m (Fin r) ℂ}
    {V : Matrix n (Fin r) ℂ} :
    IsOptimalFit S U V ↔ U * Vᴴ ∈ S ∧ IsMinOn nuclearNorm S (U * Vᴴ) ∧
      (frobeniusSq U + frobeniusSq V) / 2 = nuclearNorm (U * Vᴴ) := by
  have hlow := nuclearNorm_le_of_eq_mul U V rfl
  constructor
  · rintro ⟨hS, hopt⟩
    have hle (Y : Matrix m n ℂ) (hY : Y ∈ S) :
        (frobeniusSq U + frobeniusSq V) / 2 ≤ nuclearNorm Y := by
      obtain ⟨r', U', V', hY', hc⟩ := (nuclearNorm_isLeast Y).1
      rw [hc]
      exact hopt r' U' V' (hY' ▸ hY)
    exact ⟨hS, isMinOn_iff.mpr fun Y hY => hlow.trans (hle Y hY), le_antisymm (hle _ hS) hlow⟩
  · rintro ⟨hS, hmin, hc⟩
    refine ⟨hS, fun r' U' V' h' => ?_⟩
    rw [hc]
    exact (isMinOn_iff.mp hmin _ h').trans (nuclearNorm_le_of_eq_mul U' V' rfl)

example := isOptimalFit_iff (S := {X : Matrix (Fin 1) (Fin 1) ℂ | X = 1}) (r := 1) (U := 1)
  (V := 1)

/-- **The products of the optimal fits are exactly the matrices of least nuclear norm** in
`S`: weight decay at its optimum returns what M returns. -/
theorem setOf_isOptimalFit_eq (S : Set (Matrix m n ℂ)) :
    {X | ∃ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
      IsOptimalFit S U V ∧ X = U * Vᴴ} = {X | X ∈ S ∧ IsMinOn nuclearNorm S X} := by
  ext X
  constructor
  · rintro ⟨r, U, V, hopt, rfl⟩
    exact ⟨(isOptimalFit_iff.mp hopt).1, (isOptimalFit_iff.mp hopt).2.1⟩
  · rintro ⟨hS, hmin⟩
    obtain ⟨r, U, V, hX, hc⟩ := (nuclearNorm_isLeast X).1
    subst hX
    exact ⟨r, U, V, isOptimalFit_iff.mpr ⟨hS, hmin, hc.symm⟩, rfl⟩

/-- **Weight decay at its optimum recovers `K` exactly when M does** (`review/flat_spectrum.md`,
Lemma 1, "exact for a pure factorization"): for `K ∈ S`, some fit is optimal and every optimal
fit multiplies to `K` exactly when `K` is the only matrix of least nuclear norm in `S`. -/
theorem forall_isMinOn_iff_iff_isOptimalFit {S : Set (Matrix m n ℂ)} {K : Matrix m n ℂ}
    (hK : K ∈ S) :
    (∀ X ∈ S, (IsMinOn nuclearNorm S X ↔ X = K)) ↔
      (∃ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ), IsOptimalFit S U V) ∧
        ∀ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
          IsOptimalFit S U V → U * Vᴴ = K := by
  have hset := setOf_isOptimalFit_eq S
  have hprod (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ) (h : IsOptimalFit S U V) :
      U * Vᴴ ∈ S ∧ IsMinOn nuclearNorm S (U * Vᴴ) := by
    have : U * Vᴴ ∈ {X | ∃ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
        IsOptimalFit S U V ∧ X = U * Vᴴ} := ⟨r, U, V, h, rfl⟩
    rwa [hset] at this
  constructor
  · intro hrec
    have hKmin : K ∈ {X | X ∈ S ∧ IsMinOn nuclearNorm S X} := ⟨hK, (hrec K hK).mpr rfl⟩
    rw [← hset] at hKmin
    obtain ⟨r, U, V, hopt, -⟩ := hKmin
    exact ⟨⟨r, U, V, hopt⟩, fun r' U' V' h' => (hrec _ (hprod r' U' V' h').1).mp
      (hprod r' U' V' h').2⟩
  · rintro ⟨⟨r, U, V, hopt⟩, hall⟩ X hX
    constructor
    · intro hmin
      have hmem : X ∈ {X | X ∈ S ∧ IsMinOn nuclearNorm S X} := ⟨hX, hmin⟩
      rw [← hset] at hmem
      obtain ⟨r', U', V', hopt', rfl⟩ := hmem
      exact hall r' U' V' hopt'
    · rintro rfl
      rw [← hall r U V hopt]
      exact (hprod r U V hopt).2

example := forall_isMinOn_iff_iff_isOptimalFit (S := {X : Matrix (Fin 1) (Fin 1) ℂ | X = 1})
  (K := 1) rfl

end Transformer.Grokking.FlatSpectrum

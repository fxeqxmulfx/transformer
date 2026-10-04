/-
# PARITY is not definable in `TL[◁#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §3: a remark the authors left commented out in the source after
`thm:transformer_equivalence`,

> A relevant note is that PARITY is in `TL[◁#, ▷#]` but not `TL[◁#]`,

with no proof.  This file proves its second half; the first is
`Transformer.CRASP.ParityTwoSided`.  PARITY is the set of strings over
`{0, 1}` with an odd number of `1`s, `true` standing for `1` (`parity`).

On the strings `a^n`, a past-only formula without PNPs sees at a position `i`
the string `a^i` (`Form.sat_append`), so at the last position its value is a
function of `n` alone, and that function is eventually constant
(`Form.const_replicate`).  The terms are eventually affine in `n` with a
natural slope (`Term.affine_replicate`): once `φ` is constant, `◁#[φ]` grows
by `1` or by `0` at each step.  Of two such terms the steeper eventually wins
a comparison, and two of one slope compare by their offsets, so a comparison
is eventually constant, and the Boolean connectives keep that.  PARITY
alternates on `1^n`, so no formula of `TL[◁#]` defines it, at any depth
(`not_definableL_parity`); by `thm:transformer_equivalence`, no future-masked
rounded transformer recognizes it (`not_recognizes_parity`).

A Parikh numerical predicate is excluded, as it is from `TL[◁#]`: one reading
the number of `1`s defines PARITY at depth `0`.
-/

import Transformer.CRASP.BoundedExists
import Transformer.CRASP.Locality
import Transformer.CRASP.Transformers

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- PARITY: the strings over `{0, 1}`, with `true` for `1`, that hold an odd
number of `1`s. -/
def parity : Set (List Bool) := {w | Odd (w.count true)}

/-- On `a^(n+1)`, `◁#[φ]` at the last position is its value on `a^n` at the
last position there plus the indicator of `φ` at the new one: below the new
position, a past-only formula without PNPs reads `a^n` (Definition
`def:TLC_semantics`). -/
theorem val_countL_replicate_succ (a : σ) {φ : Form σ} (hp : φ.past = true)
    (hf : φ.pnpFree = true) (n : ℕ) :
    (Term.countL φ).val (List.replicate (n + 1) a) (n + 1) =
      (Term.countL φ).val (List.replicate n a) n +
        (if φ.sat (List.replicate (n + 1) a) (n + 1) = true then 1 else 0) := by
  rw [val_countL_succ, List.replicate_succ',
    Term.val_append _ _ (.countL φ) (by rwa [Term.past]) (by rwa [Term.pnpFree]) n (by simp)]

/-- The hypotheses of `val_countL_replicate_succ` are satisfiable: `Q_a` is a
past-only formula without PNPs. -/
example (a : σ) : (Form.sym a : Form σ).past = true ∧ (Form.sym a : Form σ).pnpFree = true :=
  ⟨rfl, rfl⟩

mutual

/-- **On the strings `a^n`, a past-only term without PNPs is eventually affine
in `n`**, with a natural slope: past some `N`, its value at the last position
of `a^n` is `c·n + d`. -/
theorem Term.affine_replicate (a : σ) : ∀ t : Term σ, t.past = true → t.pnpFree = true →
    ∃ (c : ℕ) (d : ℤ) (N : ℕ), ∀ n, N ≤ n → (t.val (List.replicate n a) n : ℤ) = c * n + d
  | .countL φ, hp, hf => by
      rw [Term.past] at hp
      rw [Term.pnpFree] at hf
      obtain ⟨b, N, hN⟩ := Form.const_replicate a φ hp hf
      refine ⟨b.toNat, (Term.countL φ).val (List.replicate N a) N - b.toNat * N, N,
        fun n hn => ?_⟩
      induction n, hn using Nat.le_induction with
      | base => ring
      | succ n hn ih =>
          rw [val_countL_replicate_succ a hp hf n, hN (n + 1) (by omega)]
          push_cast
          rw [ih]
          cases b
          · simp
          · simp
            ring
  | .countR _, hp, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf => by
      rw [Term.past, Bool.and_eq_true] at hp
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      obtain ⟨c₁, d₁, N₁, h₁⟩ := Term.affine_replicate a t₁ hp.1 hf.1
      obtain ⟨c₂, d₂, N₂, h₂⟩ := Term.affine_replicate a t₂ hp.2 hf.2
      refine ⟨c₁ + c₂, d₁ + d₂, max N₁ N₂, fun n hn => ?_⟩
      rw [Term.val, Nat.cast_add, h₁ n (le_of_max_le_left hn), h₂ n (le_of_max_le_right hn)]
      push_cast
      ring
  | .one, _, _ => ⟨0, 1, 0, fun n _ => by simp [Term.val]⟩

/-- **On the strings `a^n`, a past-only formula without PNPs is eventually
constant**: past some `N`, it has one value at the last position of `a^n`. -/
theorem Form.const_replicate (a : σ) : ∀ φ : Form σ, φ.past = true → φ.pnpFree = true →
    ∃ (b : Bool) (N : ℕ), ∀ n, N ≤ n → φ.sat (List.replicate n a) n = b
  | .sym c, _, _ => ⟨decide (a = c), 1, fun n hn => by
      simp [Form.sat, show n - 1 < n by omega]⟩
  | .lt t₁ t₂, hp, hf => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      obtain ⟨c₁, d₁, N₁, h₁⟩ := Term.affine_replicate a t₁ hp.1 hf.1
      obtain ⟨c₂, d₂, N₂, h₂⟩ := Term.affine_replicate a t₂ hp.2 hf.2
      have key : ∀ n, max N₁ N₂ ≤ n → (Form.lt t₁ t₂).sat (List.replicate n a) n =
          decide ((c₁ : ℤ) * n + d₁ < c₂ * n + d₂) := fun n hn => by
        rw [Form.sat, ← h₁ n (le_of_max_le_left hn), ← h₂ n (le_of_max_le_right hn)]
        simp only [Nat.cast_lt]
      rcases lt_trichotomy c₁ c₂ with hc | rfl | hc
      · refine ⟨true, max (max N₁ N₂) ((d₁ - d₂).toNat + 1), fun n hn => ?_⟩
        rw [key n (le_of_max_le_left hn), decide_eq_true_eq]
        have hd : d₁ - d₂ < n := by
          have := Int.self_le_toNat (d₁ - d₂)
          have := le_of_max_le_right hn
          omega
        have hc' : ((c₁ : ℤ) + 1) * n ≤ c₂ * n :=
          mul_le_mul_of_nonneg_right (by exact_mod_cast hc) (Nat.cast_nonneg n)
        linarith
      · exact ⟨decide (d₁ < d₂), max N₁ N₂, fun n hn => by rw [key n hn]; simp⟩
      · refine ⟨false, max (max N₁ N₂) (d₂ - d₁).toNat, fun n hn => ?_⟩
        rw [key n (le_of_max_le_left hn), decide_eq_false_iff_not, not_lt]
        have hd : d₂ - d₁ ≤ n := by
          have := Int.self_le_toNat (d₂ - d₁)
          have := le_of_max_le_right hn
          omega
        have hc' : ((c₂ : ℤ) + 1) * n ≤ c₁ * n :=
          mul_le_mul_of_nonneg_right (by exact_mod_cast hc) (Nat.cast_nonneg n)
        linarith
  | .neg φ, hp, hf => by
      rw [Form.past] at hp
      rw [Form.pnpFree] at hf
      obtain ⟨b, N, h⟩ := Form.const_replicate a φ hp hf
      exact ⟨!b, N, fun n hn => by rw [Form.sat, h n hn]⟩
  | .and φ₁ φ₂, hp, hf => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      obtain ⟨b₁, N₁, h₁⟩ := Form.const_replicate a φ₁ hp.1 hf.1
      obtain ⟨b₂, N₂, h₂⟩ := Form.const_replicate a φ₂ hp.2 hf.2
      exact ⟨b₁ && b₂, max N₁ N₂, fun n hn => by
        rw [Form.sat, h₁ n (le_of_max_le_left hn), h₂ n (le_of_max_le_right hn)]⟩
  | .pnp _, _, hf => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

end

/-- The hypotheses of `Term.affine_replicate` and `Form.const_replicate` are
satisfiable: `◁#[Q_a] < 1 + 1` is a past-only formula without PNPs, and so
are its terms. -/
example (a : σ) :
    (Form.lt (.countL (.sym a)) (.add .one .one) : Form σ).past = true ∧
      (Form.lt (.countL (.sym a)) (.add .one .one) : Form σ).pnpFree = true :=
  ⟨rfl, rfl⟩

/-- **PARITY is not definable in `TL[◁#]`, at any depth** (the second half of
the remark after `thm:transformer_equivalence`, §3).  A defining formula would
be eventually constant on `1^n` (`Form.const_replicate`), while `1^n ∈ PARITY`
exactly when `n` is odd. -/
theorem not_definableL_parity (k : ℕ) : ¬ DefinableL parity k := by
  rintro ⟨φ, ⟨hp, hf, -⟩, hlang⟩
  obtain ⟨b, N, h⟩ := Form.const_replicate true φ hp hf
  have hmem : ∀ n, N ≤ n → (b = true ↔ Odd n) := fun n hn => by
    have hw : List.replicate n true ∈ φ.lang ↔ List.replicate n true ∈ parity := by rw [hlang]
    change φ.sat (List.replicate n true) (List.replicate n true).length = true ↔
      Odd ((List.replicate n true).count true) at hw
    rwa [List.length_replicate, h n hn, List.count_replicate_self] at hw
  have h₀ := hmem N le_rfl
  have h₁ := hmem (N + 1) (Nat.le_succ N)
  rw [Nat.odd_add_one] at h₁
  exact iff_not_self (h₀.symm.trans h₁)

/-- **No future-masked rounded transformer recognizes PARITY**, whatever its
depth, width and precision: by `thm:transformer_equivalence` it would make
PARITY definable in `TL[◁#]`, against `not_definableL_parity`. -/
theorem not_recognizes_parity {p s d k : ℕ} (T : RTfr (Option Bool) p s d k) :
    ¬ T.Recognizes parity :=
  fun hT => not_definableL_parity k ((definableL_iff_recognizes parity k).mpr ⟨p, s, d, T, hT⟩)

/-- The plain logic is what fails: the Parikh numerical predicate "the number
of `1`s is odd" defines PARITY at depth `0` (Definition `def:PNP`). -/
example : (Form.pnp fun v _ => decide (Odd (v true)) : Form Bool).lang = parity := by
  ext w
  simp [Form.lang, Form.models, Form.sat, parikh, parity, List.count_eq_length_filter]

end CRASP
end Transformer

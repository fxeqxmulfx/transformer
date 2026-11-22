module

public import Transformer.Phasor.TripleTransport.ReadBasis

/-!
# A shared gate for exclusive dense reads

Source: “One gate for the two structural reads” in the triple-transport
experiment. The proposals select learned vectors and may abstain.
The gating state is a causal backbone output.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.SharedStructuralGate

variable {X : Type*} {d : ℕ}

def value (dictionary : X → Fin d → ℝ) (read : Option X) (i : Fin d) : ℝ :=
  read.elim 0 (fun x => dictionary x i)

/-- The two structural proposals have disjoint support. Keeping the first
defined proposal and attaching its coefficient preserves every coordinate of
the read at this prefix, independently of the learned dictionary geometry. -/
theorem select_read (D : X → Fin d → ℝ) (c t : Option X) (a b : ℝ)
    (disjoint : c = none ∨ t = none) (i : Fin d) :
    (if c.isSome then a else b) * value D (c.or t) i =
      a * value D c i + b * value D t i := by
  rcases disjoint with h | h
  · subst c
    cases t <;> simp [value]
  · subst t
    cases c <;> simp [value]

/-- Replacing both coefficients by one learned gate preserves the full
vector whenever the shared coefficient matches the one used at supported
positions. Abstentions and zero dictionary vectors need no agreement. -/
theorem shared_read (D : X → Fin d → ℝ) (c t : Option X) (a b s : ℝ)
    (disjoint : c = none ∨ t = none)
    (left : ∀ x, c = some x → D x ≠ 0 → s = a)
    (right : ∀ x, t = some x → D x ≠ 0 → s = b) (i : Fin d) :
    s * value D (c.or t) i = a * value D c i + b * value D t i := by
  rw [← select_read D c t a b disjoint i]
  cases c with
  | none =>
    cases t with
    | none => simp [value]
    | some y =>
      by_cases h : D y = 0
      · simp [value,h]
      · rw [right y rfl h]
        rfl
  | some y =>
    by_cases h : D y = 0
    · simp [value,h]
    · rw [left y rfl h]
      rfl

private def dictionary (x : Bool) : Fin 2 → ℝ :=
  if x then fun i => if i = 0 then 2 else -1 else 0

example : (if (none : Option Bool).isSome then (7 : ℝ) else 4) *
    value dictionary (none.or (some true)) 0 =
    7 * value dictionary none 0 + 4 * value dictionary (some true) 0 :=
  select_read dictionary _ _ 7 4 (.inl rfl) 0

example : (3 : ℝ) * value dictionary ((some true).or none) 1 =
    3 * value dictionary (some true) 1 + 7 * value dictionary none 1 := by
  exact shared_read dictionary _ _ 3 7 3 (.inr rfl) (by simp) (by simp) 1

/-- Exclusive roles alone need not be replaceable by one coefficient. Using
the same dense vector and gating state at two positions exposes the issue
without referring to any task or labels. -/
example : ¬ ∃ s : ℝ, s * value dictionary (some true) 0 = 2 ∧
    s * value dictionary (some true) 0 = 4 := by
  rintro ⟨s, a, b⟩
  linarith

end Transformer.Phasor.TripleTransport.SharedStructuralGate

namespace Transformer.Phasor.TripleTransport.CurrentGate
open scoped BigOperators
variable {d e : ℕ}

def affine (b : ℝ) (g h : Fin d → ℝ) : ℝ := b + ∑ i, g i * h i

/-- When the previous state is an affine read of the current causal state,
one affine gate on the current state represents every affine gate on its
predecessor. The parameter substitution is independent of the prefix. -/
theorem absorb (b : ℝ) (g off : Fin d → ℝ) (a : Fin d → Fin e → ℝ) (h : Fin e → ℝ) :
    affine b g (fun i => off i + ∑ j, a i j * h j) =
      affine (b + ∑ i, g i * off i) (fun j => ∑ i, g i * a i j) h := by
  unfold affine
  simp only [mul_add, Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ) (f := fun i j => g i * (a i j * h j))]
  simp only [mul_assoc, add_assoc]

/-- Any gate on an earlier state has the same value when that state
coincides. A current-state gate with different values on two such prefixes
cannot be expressed by changing the earlier gate parameters. -/
theorem separates {X Y : Type*} (earlier : X → Y) (score : X → ℝ) (p q : X)
    (same : earlier p = earlier q) (different : score p ≠ score q) :
    ¬ ∃ gate : Y → ℝ, ∀ x, gate (earlier x) = score x := by
  rintro ⟨gate, fits⟩
  exact different ((fits p).symm.trans ((congrArg gate same).trans (fits q)))

private def state (x : Bool × Bool) (i : Fin 2) : ℝ :=
  if i = 0 then (if x.1 then 1 else -1) else (if x.2 then 1 else -1)

example (h : Fin 2 → ℝ) :
    affine 1 (fun _ : Fin 1 => 2) (fun _ => 3 + ∑ j, (j.val + 1 : ℝ) * h j) =
      affine (1 + ∑ _ : Fin 1, (2:ℝ)*3) (fun j => ∑ _ : Fin 1, 2*(j.val+1 : ℝ)) h :=
  absorb 1 (fun _ => 2) (fun _ => 3) (fun _ j => j.val+1) h

example : ¬ ∃ gate : Bool → ℝ, ∀ x : Bool × Bool,
    gate x.1 = affine 0 (fun i => if i = 1 then 1 else 0) (state x) := by
  apply separates Prod.fst _ (false, false) (false, true) rfl
  norm_num [affine, state, Fin.sum_univ_two]

end Transformer.Phasor.TripleTransport.CurrentGate

module

public import Transformer.Rotate90.Training.Defs

/-!
# The observable obstruction from an aliased embedding

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, readout theorem.
All classes remain in the softmax, including labels outside the power map's image.
Equal embedding rows force equal logits for every hidden state. For a nonzero
answer, the subgroup orbit gives a log-cardinality lower bound on cross-entropy.
This is a statement about the actual tied output, independent of how E was found.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {K H : Type*} [Fintype K] [Fintype H]

/-- The tied output cannot distinguish two classes with equal embedding rows,
regardless of the hidden computation, width or number of residual blocks. -/
theorem stackLogit_mul {S : Subgroup (ZMod p)ˣ}
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (a b q : ZMod p) {h : (ZMod p)ˣ} (hh : h ∈ S) :
    Power.stackLogit eps (entry E) blocks a b ((h : ZMod p) * q) =
      Power.stackLogit eps (entry E) blocks a b q := by
  unfold Power.stackLogit tiedReadout
  apply Finset.sum_congr rfl
  intro i hi
  rw [hE h hh q i]

section Entropy

variable {V : Type*} [Fintype V]

/-- Equal-scored competing classes impose a lower bound on actual cross-entropy.
The finite set can contain the target or any other classes with its score. -/
theorem crossEntropy_ge_log_card (L : V → ℝ) (q : V) (A : Finset V)
    (hA : A.Nonempty) (hL : ∀ r ∈ A, L r = L q) :
    Real.log (A.card : ℝ) ≤ crossEntropy L q := by
  classical
  have hcard : (0 : ℝ) < A.card := by exact_mod_cast hA.card_pos
  have hsum : (A.card : ℝ) * Real.exp (L q) ≤ ∑ r, Real.exp (L r) := by
    calc
      (A.card : ℝ) * Real.exp (L q) = ∑ r ∈ A, Real.exp (L r) := by
        rw [Finset.sum_congr rfl (fun r hr => congrArg Real.exp (hL r hr))]
        simp
      _ ≤ ∑ r, Real.exp (L r) :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ A)
          (fun r hr hnot => (Real.exp_pos (L r)).le)
  have hlog := Real.log_le_log (mul_pos hcard (Real.exp_pos (L q))) hsum
  rw [Real.log_mul (ne_of_gt hcard) (ne_of_gt (Real.exp_pos (L q))), Real.log_exp] at hlog
  unfold crossEntropy
  linarith

example := crossEntropy_ge_log_card (V := Fin 3) (fun _ => (2 : ℝ)) 0
  {0, 1} (by simp) (by intro r hr; rfl)

end Entropy

/-- Multiplication by a nonzero answer embeds the subgroup into the class set. -/
theorem orbit_injective (S : Subgroup (ZMod p)ˣ) {q : ZMod p} (hq : q ≠ 0) :
    Function.Injective (fun h : S => ((h : (ZMod p)ˣ) : ZMod p) * q) := by
  intro h j heq
  apply Subtype.ext
  exact Units.val_injective (mul_right_cancel₀ hq heq)

/-- On every nonzero target, an S-aliased actual model has loss at least log |S|.
The zero target has a singleton orbit and is deliberately excluded. -/
theorem crossEntropy_ge_log_subgroup {S : Subgroup (ZMod p)ˣ}
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (a b : ZMod p) {q : ZMod p} (hq : q ≠ 0) :
    Real.log (Nat.card S : ℝ) ≤
      crossEntropy (Power.stackLogit eps (entry E) blocks a b) q := by
  classical
  let := Fintype.ofFinite S
  let orbit := Finset.univ.image (fun h : S => ((h : (ZMod p)ˣ) : ZMod p) * q)
  have hcard : orbit.card = Nat.card S := by
    rw [Finset.card_image_of_injective _ (orbit_injective S hq), Finset.card_univ,
      Nat.card_eq_fintype_card]
  have hnonempty : orbit.Nonempty := by
    exact Finset.Nonempty.image (by exact ⟨1, Finset.mem_univ _⟩) _
  rw [← hcard]
  apply crossEntropy_ge_log_card _ q orbit hnonempty
  intro r hr
  obtain ⟨h, hh, rfl⟩ := Finset.mem_image.mp hr
  exact stackLogit_mul hE eps blocks a b q h.property

/-- Applied to powered division, every pair with a nonzero numerator has this
loss floor for every natural exponent. This includes k=0 and b an arbitrary
nonzero denominator, rather than requiring the target itself as a hypothesis. -/
theorem poweredLoss_floor {S : Subgroup (ZMod p)ˣ}
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {a : ZMod p} (ha : a ≠ 0) (b : (ZMod p)ˣ) :
    Real.log (Nat.card S : ℝ) ≤
      crossEntropy (Power.stackLogit eps (entry E) blocks a b) ((a / (b : ZMod p)) ^ k) := by
  apply crossEntropy_ge_log_subgroup hE eps blocks a b
  exact pow_ne_zero k (div_ne_zero ha b.ne_zero)

example := poweredLoss_floor (p := 3) (K := Fin 8) (H := Fin 43)
  (aliased_zero_nonzero ⊤ (fun _ => 2) (fun _ => 1))
  (1 / 8388608) [Power.zeroBlock] 0 (a := 2) (by decide) 1
example := poweredLoss_floor (p := 3) (K := Fin 8) (H := Fin 43)
  (aliased_zero_nonzero ⊤ (fun _ => 2) (fun _ => 1))
  (1 / 8388608) [Power.zeroBlock] 5 (a := 1) one_ne_zero
  (Units.mk0 (2 : ZMod 3) (by decide))

/-- A nonidentity alias prevents a strict correct winner already at a=b=1.
This works for every natural exponent, including zero. -/
theorem not_solves_of_aliased {S : Subgroup (ZMod p)ˣ}
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {h : (ZMod p)ˣ} (hh : h ∈ S) (hne : h ≠ 1)
    (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ) :
    ¬ (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
      Power.stackLogit eps (entry E) blocks a b q <
        Power.stackLogit eps (entry E) blocks a b ((a / b) ^ k)) := by
  intro hsolve
  have hval : (h : ZMod p) ≠ 1 := by
    intro heq
    exact hne (Units.val_injective heq)
  have hlt := hsolve 1 1 one_ne_zero h (by simpa using hval)
  have heq := stackLogit_mul hE eps blocks 1 1 1 hh
  simp only [mul_one] at heq
  simp only [div_self one_ne_zero, one_pow] at hlt
  exact (ne_of_lt hlt) heq

example := stackLogit_mul (p := 3) (K := Fin 8) (H := Fin 43) (S := ⊤)
  (E := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (1 / 8388608) [Power.zeroBlock] 1 2 1
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp)
example := orbit_injective (p := 3) ⊤ (q := 1) one_ne_zero
example := crossEntropy_ge_log_subgroup (p := 3) (K := Fin 8) (H := Fin 43) (S := ⊤)
  (E := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (1 / 8388608) [Power.zeroBlock] 1 2 (q := 2) (by decide)
example := not_solves_of_aliased (p := 3) (K := Fin 8) (H := Fin 43) (S := ⊤)
  (E := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) (by decide)
  (1 / 8388608) [Power.zeroBlock] 1

end Transformer.Rotate90.Training

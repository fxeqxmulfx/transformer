module

public import Transformer.Rotate90.Training.Symmetry
public import Mathlib.GroupTheory.OrderOfElement

/-!
# The actual tied embedding gradient preserves quotient symmetry

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, gradient theorem.
The first lookup, second lookup and output each contribute to the same tensor's
gradient. The three contributions are proved invariant separately, using the
full-table arithmetic symmetry. Coprimality of k and the subgroup order makes
the output action cover the whole subgroup; omitting this condition is false.
All finite widths, hidden sizes and depths, and arbitrary branch matrices, are
covered. A partial training sample is not silently replaced by the full table.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural

variable {p : ℕ} [Fact p.Prime] {K H : Type*} [Fintype K] [Fintype H]

/-- The tied derivative differentiates all three occurrences of E together. -/
theorem fderiv_sharedLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (E v : Table (ZMod p) (Channels K)) :
    fderiv ℝ (sharedLoss eps blocks k) E v =
      fderiv ℝ (fullLoss eps blocks k) (diagonal E) (diagonal v) := by
  change fderiv ℝ (fullLoss eps blocks k ∘ diagonal) E v = _
  rw [fderiv_comp E (differentiableAt_fullLoss heps blocks k _) diagonal.differentiableAt,
    diagonal.fderiv]
  rfl

/-- The first input lookup has identical partial derivatives on each alias orbit. -/
theorem differential_first_mul {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {h : (ZMod p)ˣ} (hh : h ∈ S) (q : ZMod p) (i : Channels K) :
    fderiv ℝ (fullLoss eps blocks k) (diagonal E) (basis ((h : ZMod p) * q) i, 0, 0) =
      fderiv ℝ (fullLoss eps blocks k) (diagonal E) (basis q i, 0, 0) := by
  have hsym := differential_fullLoss_fixed heps blocks k hE hh S.one_mem
    (basis ((h : ZMod p) * q) i, 0, 0)
  change fderiv ℝ (fullLoss eps blocks k) (diagonal E)
    (permute (Units.mulLeft h) (basis ((h : ZMod p) * q) i),
      permute (Units.mulLeft 1) 0, permute (Units.mulLeft ((h / 1) ^ k)) 0) = _ at hsym
  have hb := permute_basis (Units.mulLeft h) q i
  simp only [Units.mulLeft_apply] at hb
  rw [hb] at hsym
  simpa only [map_zero] using hsym.symm

/-- The second lookup obeys the same orbit equality, with its inverse output action. -/
theorem differential_second_mul {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {h : (ZMod p)ˣ} (hh : h ∈ S) (q : ZMod p) (i : Channels K) :
    fderiv ℝ (fullLoss eps blocks k) (diagonal E) (0, basis ((h : ZMod p) * q) i, 0) =
      fderiv ℝ (fullLoss eps blocks k) (diagonal E) (0, basis q i, 0) := by
  have hsym := differential_fullLoss_fixed heps blocks k hE S.one_mem hh
    (0, basis ((h : ZMod p) * q) i, 0)
  change fderiv ℝ (fullLoss eps blocks k) (diagonal E)
    (permute (Units.mulLeft 1) 0, permute (Units.mulLeft h) (basis ((h : ZMod p) * q) i),
      permute (Units.mulLeft ((1 / h) ^ k)) 0) = _ at hsym
  have hb := permute_basis (Units.mulLeft h) q i
  simp only [Units.mulLeft_apply] at hb
  rw [hb] at hsym
  simpa only [map_zero] using hsym.symm

/-- The output lookup is invariant under kth powers of subgroup elements. -/
theorem differential_output_pow {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {h : (ZMod p)ˣ} (hh : h ∈ S) (q : ZMod p) (i : Channels K) :
    fderiv ℝ (fullLoss eps blocks k) (diagonal E)
        (0, 0, basis (((h ^ k : (ZMod p)ˣ) : ZMod p) * q) i) =
      fderiv ℝ (fullLoss eps blocks k) (diagonal E) (0, 0, basis q i) := by
  have hsym := differential_fullLoss_fixed heps blocks k hE hh S.one_mem
    (0, 0, basis (((h ^ k : (ZMod p)ˣ) : ZMod p) * q) i)
  change fderiv ℝ (fullLoss eps blocks k) (diagonal E)
    (permute (Units.mulLeft h) 0, permute (Units.mulLeft 1) 0,
      permute (Units.mulLeft ((h / 1) ^ k))
        (basis (((h ^ k : (ZMod p)ˣ) : ZMod p) * q) i)) = _ at hsym
  have hb := permute_basis (Units.mulLeft (h ^ k)) q i
  simp only [Units.mulLeft_apply] at hb
  rw [div_one, hb] at hsym
  simpa only [map_zero] using hsym.symm

/-- **Gradient invariance for the actual tied model.** Full-table learning has
no first-order symmetry-breaking signal at a subgroup-invariant embedding when
the task exponent is coprime to that subgroup's order. This is not stationarity:
the common rows and every branch matrix can continue moving. -/
theorem gradient_sharedLoss_aliased {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E) :
    Aliased S (gradient (sharedLoss eps blocks k) E) := by
  intro h hh q i
  rw [gradient_entry, gradient_entry, fderiv_sharedLoss heps, fderiv_sharedLoss heps]
  let D := fderiv ℝ (fullLoss eps blocks k) (diagonal E)
  have split (v : Table (ZMod p) (Channels K)) :
      D (diagonal v) = D (v, 0, 0) + D (0, v, 0) + D (0, 0, v) := by
    have hv : diagonal v = (v, 0, 0) + (0, v, 0) + (0, 0, v) := by
      simp [diagonal]
    rw [hv, map_add, map_add]
  obtain ⟨j, hj⟩ := (Nat.Coprime.pow_left_bijective (G := S) hk).surjective ⟨h, hh⟩
  have hj' : (j : (ZMod p)ˣ) ^ k = h := congrArg Subtype.val hj
  have hout := differential_output_pow heps blocks k hE j.property q i
  rw [hj'] at hout
  change D (diagonal (basis ((h : ZMod p) * q) i)) = D (diagonal (basis q i))
  rw [split, split, differential_first_mul heps blocks k hE hh,
    differential_second_mul heps blocks k hE hh, hout]

/-- An infinitesimal separation of two aliased rows has zero directional
derivative. This is a first-order statement, not a positive-curvature or basin
claim: higher-order changes and nonsymmetric data can still split the rows. -/
theorem alias_difference_derivative_zero {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {h : (ZMod p)ˣ} (hh : h ∈ S) (q : ZMod p) (i : Channels K) :
    fderiv ℝ (sharedLoss eps blocks k) E (basis ((h : ZMod p) * q) i - basis q i) = 0 := by
  rw [map_sub, ← gradient_entry, ← gradient_entry,
    gradient_sharedLoss_aliased heps blocks k hk hE h hh q i, sub_self]

example := alias_difference_derivative_zero (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤) (by simp)
  (aliased_zero_nonzero ⊤ (fun _ => 2) (fun _ => 1))
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) 1 (0, 0)

example := fderiv_sharedLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1
  (WithLp.toLp 2 fun _ => 1) (basis 1 (0, 0))
example := differential_first_mul (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) 1 (0, 0)
example := differential_second_mul (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) 1 (0, 0)
example := differential_output_pow (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) 1 (0, 0)
example := gradient_sharedLoss_aliased (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤) (by simp)
  (E := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])

end Transformer.Rotate90.Training

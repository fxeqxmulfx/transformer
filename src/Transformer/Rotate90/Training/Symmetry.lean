module

public import Transformer.Rotate90.Training.Basic

/-!
# The exact data symmetry behind quotient-code training

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, full-table
reindexing. Independently multiply the first operand by r and the second by s.
Their powered quotient is multiplied by (r/s)^k. Permuting the three auxiliary
tables accordingly leaves full-table cross-entropy unchanged. This identity
holds for all weights and every finite stack, not just a reduced feature model.
It does not assert invariance of an arbitrary partial-table sample.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural
open scoped BigOperators

section Coordinates

variable {V C : Type*} [Fintype V] [Fintype C]

/-- Row reindexing moves a unit coordinate perturbation to the corresponding row. -/
theorem permute_basis (σ : Equiv.Perm V) (q : V) (i : C) :
    permute σ (basis (σ q) i) = basis q i := by
  classical
  ext ⟨r, j⟩
  change entry (permute σ (basis (σ q) i)) r j = entry (basis q i) r j
  rw [entry_permute]
  simp [entry, basis, PiLp.single_apply, σ.injective.eq_iff]

/-- A scalar tensor gradient entry is its true Euclidean directional derivative. -/
theorem gradient_entry (f : Table V C → ℝ) (E : Table V C) (q : V) (i : C) :
    entry (gradient f E) q i = fderiv ℝ f E (basis q i) := by
  classical
  rw [← inner_gradient_left, basis, EuclideanSpace.inner_single_right]
  simp [entry]

example := permute_basis (V := Fin 3) (C := Fin 2) (Equiv.refl _) 1 0
example := gradient_entry (V := Fin 3) (C := Fin 2)
  (fun E => (entry E 1 0) ^ 2) (WithLp.toLp 2 fun _ => 2) 1 0

end Coordinates

variable {p : ℕ} [Fact p.Prime] {K H C : Type*} [Fintype K] [Fintype H] [Fintype C]

/-- Multiplying the operands induces the stated output permutation, including
zero numerators and exponent zero. Both denominator factors are units. -/
theorem target_mul (r s : (ZMod p)ˣ) (k : ℕ) (a : ZMod p) (b : (ZMod p)ˣ) :
    (((r : ZMod p) * a) / ((s * b : (ZMod p)ˣ) : ZMod p)) ^ k =
      (((r / s) ^ k : (ZMod p)ˣ) : ZMod p) * (a / (b : ZMod p)) ^ k := by
  push_cast
  rw [mul_div_mul_comm, mul_pow]

/-- **Exact full-table symmetry of the actual normalized network loss.** The
three table permutations are linked by the arithmetic task, not chosen freely. -/
theorem fullLoss_permute (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (k : ℕ) (tables : Tables (ZMod p) (Channels K)) (r s : (ZMod p)ˣ) :
    fullLoss eps blocks k
      (permuteThree (Units.mulLeft r) (Units.mulLeft s) (Units.mulLeft ((r / s) ^ k)) tables) =
        fullLoss eps blocks k tables := by
  let f : ZMod p → (ZMod p)ˣ → ℝ := fun a b =>
    crossEntropy (untiedLogit eps blocks tables a b) ((a / (b : ZMod p)) ^ k)
  have hrow (a : ZMod p) (b : (ZMod p)ˣ) :
      crossEntropy (untiedLogit eps blocks
        (permuteThree (Units.mulLeft r) (Units.mulLeft s) (Units.mulLeft ((r / s) ^ k)) tables)
          a b) ((a / (b : ZMod p)) ^ k) = f ((r : ZMod p) * a) (s * b) := by
    change crossEntropy
      (fun q => untiedLogit eps blocks tables ((r : ZMod p) * a) ((s * b : (ZMod p)ˣ) : ZMod p)
        ((Units.mulLeft ((r / s) ^ k)) q)) ((a / (b : ZMod p)) ^ k) = _
    rw [crossEntropy_permute]
    change crossEntropy
      (untiedLogit eps blocks tables ((r : ZMod p) * a) ((s * b : (ZMod p)ˣ) : ZMod p))
        ((((r / s) ^ k : (ZMod p)ˣ) : ZMod p) * (a / (b : ZMod p)) ^ k) = _
    rw [← target_mul]
  unfold fullLoss
  apply congrArg (fun x : ℝ => x / (Fintype.card (ZMod p) * Fintype.card (ZMod p)ˣ))
  change (∑ a : ZMod p, ∑ b : (ZMod p)ˣ, _) = ∑ a : ZMod p, ∑ b : (ZMod p)ˣ, f a b
  simp_rw [hrow]
  have hb (a : ZMod p) : (∑ b : (ZMod p)ˣ, f a (s * b)) = ∑ b : (ZMod p)ˣ, f a b :=
    Equiv.sum_comp (Equiv.mulLeft s) (f a)
  simp_rw [hb]
  exact Equiv.sum_comp (Units.mulLeft r) (fun a => ∑ b : (ZMod p)ˣ, f a b)

/-- A subgroup-invariant embedding is fixed by each operand permutation. -/
theorem permute_fixed {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) C}
    (hE : Aliased S E) {h : (ZMod p)ˣ} (hh : h ∈ S) :
    permute (Units.mulLeft h) E = E :=
  (aliased_iff S E).mp hE h hh

/-- The tied point is fixed by every arithmetic-compatible subgroup action on
the three auxiliary occurrences. Closure under quotients and powers is essential. -/
theorem permuteThree_diagonal_fixed {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) C}
    (hE : Aliased S E) {r s : (ZMod p)ˣ} (hr : r ∈ S) (hs : s ∈ S) (k : ℕ) :
    permuteThree (Units.mulLeft r) (Units.mulLeft s) (Units.mulLeft ((r / s) ^ k))
      (diagonal E) = diagonal E := by
  change (permute (Units.mulLeft r) E,
    permute (Units.mulLeft s) E, permute (Units.mulLeft ((r / s) ^ k)) E) = (E, E, E)
  rw [permute_fixed hE hr, permute_fixed hE hs,
    permute_fixed hE (S.pow_mem (S.div_mem hr hs) k)]

/-- A fixed linear symmetry of a differentiable loss also fixes its differential
at each symmetric point. This follows by differentiating the global identity. -/
theorem differential_fixed {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : F → ℝ) (T : F →L[ℝ] F) {x : F} (hf : DifferentiableAt ℝ f x)
    (hT : ∀ y, f (T y) = f y) (hx : T x = x) (v : F) :
    fderiv ℝ f x (T v) = fderiv ℝ f x v := by
  have hfun : f ∘ T = f := funext hT
  have hdiff : DifferentiableAt ℝ f (T x) := by rw [hx]; exact hf
  have h := fderiv_comp x hdiff T.differentiableAt
  rw [hfun, hx, T.fderiv] at h
  exact (congrArg (fun A : F →L[ℝ] ℝ => A v) h).symm

/-- **The actual loss differential respects the subgroup action at the tied
aliased point.** Analyticity proved in Basic supplies its derivative hypothesis. -/
theorem differential_fullLoss_fixed {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} {E : Table (ZMod p) (Channels K)} (hE : Aliased S E)
    {r s : (ZMod p)ˣ} (hr : r ∈ S) (hs : s ∈ S)
    (v : Tables (ZMod p) (Channels K)) :
    fderiv ℝ (fullLoss eps blocks k) (diagonal E)
      (permuteThree (Units.mulLeft r) (Units.mulLeft s) (Units.mulLeft ((r / s) ^ k)) v) =
        fderiv ℝ (fullLoss eps blocks k) (diagonal E) v := by
  exact differential_fixed _ _ (differentiableAt_fullLoss heps blocks k _)
    (fun tables => fullLoss_permute eps blocks k tables r s)
    (permuteThree_diagonal_fixed hE hr hs k) v

example := target_mul (p := 3) (1 : (ZMod 3)ˣ)
  (Units.mk0 (2 : ZMod 3) (by decide)) 2 0 1
example := fullLoss_permute (p := 3) (K := Fin 1) (H := Fin 2)
  (1 / 8388608) [Power.zeroBlock] 2
  (WithLp.toLp 2 (fun _ => 1), WithLp.toLp 2 (fun _ => 2), WithLp.toLp 2 (fun _ => 3))
  1 (Units.mk0 (2 : ZMod 3) (by decide))
example := permute_fixed (p := 3) (C := Fin 2) (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp)
example := permuteThree_diagonal_fixed (p := 3) (C := Fin 2) (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (r := 1) (s := 1) (by simp) (by simp) 2
example := differential_fixed (fun x : ℝ => x ^ 2)
  (-ContinuousLinearMap.id ℝ ℝ) (x := 0) (by fun_prop)
  (by intro y; simp) (by simp) 1
example := differential_fullLoss_fixed (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 (S := ⊤)
  (E := WithLp.toLp 2 fun _ => 1) (by intro h hh q i; rfl)
  (r := 1) (s := 1) (by simp) (by simp)
  (WithLp.toLp 2 (fun _ => 1), WithLp.toLp 2 (fun _ => 2), WithLp.toLp 2 (fun _ => 3))

end Transformer.Rotate90.Training

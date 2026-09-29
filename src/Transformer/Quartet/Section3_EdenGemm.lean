/-
# An inner-product counterexample with independent FP8 coins

Source: arXiv:2601.22813v2, §3.2–§3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenUnitOne

open MeasureTheory

namespace Transformer.Quartet

variable {k : ℕ}

/-- Independent FP8 coin vectors in §3.3, Algorithm 1 make the expected inner product the product of coordinate means. -/
theorem integral_dot_independent
    (f g : (Fin (2 ^ k) → ℝ) → Fin (2 ^ k) → Fin 16 → ℝ)
    (hf : ∀ a b, Integrable (fun u => f u a b) (volume.restrict (edenCoinCube k)))
    (hg : ∀ a b, Integrable (fun v => g v a b) (volume.restrict (edenCoinCube k))) :
    (∫ u in edenCoinCube k, ∫ v in edenCoinCube k,
      ∑ a : Fin (2 ^ k), ∑ b : Fin 16, f u a b * g v a b) =
      ∑ a : Fin (2 ^ k), ∑ b : Fin 16,
        (∫ u in edenCoinCube k, f u a b) * (∫ v in edenCoinCube k, g v a b) := by
  have hinner : ∀ u, (∫ v in edenCoinCube k,
      ∑ a : Fin (2 ^ k), ∑ b : Fin 16, f u a b * g v a b) =
      ∑ a : Fin (2 ^ k), ∑ b : Fin 16,
        f u a b * (∫ v in edenCoinCube k, g v a b) := by
    intro u
    rw [integral_finsetSum _ (fun a _ => integrable_finsetSum _
      (fun b _ => (hg a b).const_mul _))]
    apply Finset.sum_congr rfl
    intro a _
    rw [integral_finsetSum _ (fun b _ => (hg a b).const_mul _)]
    simp only [integral_const_mul]
  simp_rw [hinner]
  rw [integral_finsetSum _ (fun a _ => integrable_finsetSum _
    (fun b _ => (hf a b).mul_const _))]
  apply Finset.sum_congr rfl
  intro a _
  rw [integral_finsetSum _ (fun b _ => (hf a b).mul_const _)]
  simp only [integral_mul_const]

/-- The §3.2 RHT is also a right inverse of its inverse. -/
theorem rht_rhtInv (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    rht k ε (rhtInv k ε y) i j = y i j := by
  have hpoint : ∀ a b, (if ε a b then -rhtInv k ε y a b else rhtInv k ε y a b) =
      ∑ c : Fin (2 ^ k), ∑ d : Fin 16, hadamard k a c b d * y c d := by
    intro a b
    unfold rhtInv
    cases ε a b <;> simp
  unfold rht
  simp_rw [hpoint]
  exact sum_hadamard_sum_hadamard y i j

/-- The §3.2 rotation cancellation also applies to inverse images of any two quantized tensors. -/
theorem sum_rhtInv_mul_rhtInv (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (y z : Fin (2 ^ k) → Fin 16 → ℝ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16, rhtInv k ε y i j * rhtInv k ε z i j) =
      ∑ i : Fin (2 ^ k), ∑ j : Fin 16, y i j * z i j := by
  have h := sum_rht_mul_rht ε (rhtInv k ε y) (rhtInv k ε z)
  simpa only [rht_rhtInv] using h.symm

/-- The §3.3 conditional mean in rotated coordinates equals the RHT of the exact vector mean. -/
theorem integral_msEden_edenPair_rotated
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, msEden k s (edenPair k) ε u i j =
      rht k ε (edenMeanTarget k) i j := by
  let m : Fin (2 ^ k) → Fin 16 → ℝ :=
    fun a b => ∫ u in edenCoinCube k, msEden k s (edenPair k) ε u a b
  have hInv : rhtInv k ε m = edenMeanTarget k := by
    funext a b
    change rhtInv k ε (fun a b => ∫ u in edenCoinCube k, msEden k s (edenPair k) ε u a b)
      a b = edenMeanTarget k a b
    rw [← integral_rhtInv_of_integrable ε (fun u => msEden k s (edenPair k) ε u)
      (fun c d => integrable_msEden_coins s (edenPair k) ε c d)]
    exact integral_rhtInv_msEden_edenPair_vector hs₀ hs₁ ε a b
  have h := rht_rhtInv ε m i j
  rw [hInv] at h
  exact h.symm

/-- The §3.3 unit-coordinate operand also has exact mean in rotated coordinates. -/
theorem integral_msEden_edenUnitOne_rotated
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, msEden k s (edenUnitOne k) ε u i j =
      rht k ε (edenUnitOne k) i j := by
  let m : Fin (2 ^ k) → Fin 16 → ℝ :=
    fun a b => ∫ u in edenCoinCube k, msEden k s (edenUnitOne k) ε u a b
  have hInv : rhtInv k ε m = edenUnitOne k := by
    funext a b
    change rhtInv k ε (fun a b => ∫ u in edenCoinCube k, msEden k s (edenUnitOne k) ε u a b)
      a b = edenUnitOne k a b
    rw [← integral_rhtInv_of_integrable ε (fun u => msEden k s (edenUnitOne k) ε u)
      (fun c d => integrable_msEden_coins s (edenUnitOne k) ε c d)]
    exact integral_rhtInv_msEden_edenUnitOne hs₀ hs₁ ε a b
  have h := rht_rhtInv ε m i j
  rw [hInv] at h
  exact h.symm

/-- The exact scalar product of the §3.3 mean reconstructed vector with `e₁`. -/
theorem edenMeanTarget_dot_unitOne (k : ℕ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      edenMeanTarget k i j * edenUnitOne k i j) = 101 / 510 := by
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k))
    (fun i _ hi => Finset.sum_eq_zero (fun j _ => by simp [edenUnitOne, twoSupport, hi]))
    (by simp)]
  rw [Finset.sum_eq_single (1 : Fin 16)
    (fun j _ hj => by simp [edenUnitOne, twoSupport, hj]) (by simp)]
  norm_num [edenMeanTarget, edenUnitOne, twoSupport]

/-- A §3.3 inner-product counterexample with a common RHT and independent FP8 coins. -/
theorem integral_msEden_gemm_edenPair_unitOne
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) :
    (∫ u in edenCoinCube k, ∫ v in edenCoinCube k,
      ∑ i : Fin (2 ^ k), ∑ j : Fin 16,
        msEden k s (edenPair k) ε u i j * msEden k s (edenUnitOne k) ε v i j) =
      101 / 510 := by
  rw [integral_dot_independent
    (fun u => msEden k s (edenPair k) ε u)
    (fun v => msEden k s (edenUnitOne k) ε v)
    (fun a b => integrable_msEden_coins s (edenPair k) ε a b)
    (fun a b => integrable_msEden_coins s (edenUnitOne k) ε a b)]
  simp_rw [integral_msEden_edenPair_rotated hs₀ hs₁ ε,
    integral_msEden_edenUnitOne_rotated hs₀ hs₁ ε]
  rw [sum_rht_mul_rht]
  exact edenMeanTarget_dot_unitOne k

/-- The original unquantized inner product in the same §3.3 GEMM example. -/
theorem edenPair_dot_unitOne (k : ℕ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      edenPair k i j * edenUnitOne k i j) = 1 / 10 := by
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k))
    (fun i _ hi => Finset.sum_eq_zero (fun j _ => by simp [edenUnitOne, twoSupport, hi]))
    (by simp)]
  rw [Finset.sum_eq_single (1 : Fin 16)
    (fun j _ hj => by simp [edenUnitOne, twoSupport, hj]) (by simp)]
  norm_num [edenPair, edenUnitOne, twoSupport]

/-- The §3.3 exact product bias also survives averaging over the RHT sign seeds. -/
theorem mean_msEden_gemm_edenPair_unitOne
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93) :
    mean k (fun ε u => ∫ v in edenCoinCube k,
      ∑ i : Fin (2 ^ k), ∑ j : Fin 16,
        msEden k s (edenPair k) ε u i j * msEden k s (edenUnitOne k) ε v i j) =
      101 / 510 := by
  apply mean_eq_of_forall
  intro ε
  exact integral_msEden_gemm_edenPair_unitOne hs₀ hs₁ ε

/-- The independence and integrability hypotheses of the §3.3 product calculation are realized
by the two MS-EDEN operands themselves. -/
example (ε : Fin (2 ^ 3) → Fin 16 → Bool) :
    (∫ u in edenCoinCube 3, ∫ v in edenCoinCube 3,
      ∑ i : Fin (2 ^ 3), ∑ j : Fin 16,
        msEden 3 (3200 / 527) (edenPair 3) ε u i j *
          msEden 3 (3200 / 527) (edenUnitOne 3) ε v i j) =
    ∑ i : Fin (2 ^ 3), ∑ j : Fin 16,
      (∫ u in edenCoinCube 3, msEden 3 (3200 / 527) (edenPair 3) ε u i j) *
        (∫ v in edenCoinCube 3, msEden 3 (3200 / 527) (edenUnitOne 3) ε v i j) :=
  integral_dot_independent _ _
    (fun a b => integrable_msEden_coins (3200 / 527) (edenPair 3) ε a b)
    (fun a b => integrable_msEden_coins (3200 / 527) (edenUnitOne 3) ε a b)

/-- The §3.3 inner-product claim applies at the paper's rotation size and scale. -/
example :
    mean 3 (fun ε u => ∫ v in edenCoinCube 3,
      ∑ i : Fin (2 ^ 3), ∑ j : Fin 16,
        msEden 3 (3200 / 527) (edenPair 3) ε u i j *
          msEden 3 (3200 / 527) (edenUnitOne 3) ε v i j) = 101 / 510 :=
  mean_msEden_gemm_edenPair_unitOne (by norm_num) (by norm_num)

end Transformer.Quartet

/-
# The conditional vector mean of MS-EDEN

Source: arXiv:2601.22813v2, §3.2–§3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenBias

open MeasureTheory

namespace Transformer.Quartet

variable {k : ℕ}

/-- The cube of one independent FP8 stochastic-rounding coin per 16-entry group in §3.3, Algorithm 1. -/
def edenCoinCube (k : ℕ) : Set (Fin (2 ^ k) → ℝ) :=
  Set.univ.pi fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1

/-- The output of §3.3, Algorithm 1 is integrable in its FP8 rounding coins at every fixed input and seed. -/
theorem integrable_msEden_coins (s : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (i : Fin (2 ^ k)) (j : Fin 16) :
    Integrable (fun u => msEden k s x ε u i j) (volume.restrict (edenCoinCube k)) := by
  let y := rht k ε x
  let q := rtn fp4 (y i j / (groupScaleRTN s y i * tensorScaleRTN s y))
  let z := edenScale s y i * groupScaleRTN s y i
  let t := tensorScaleRTN s y
  have hform : (fun u => msEden k s x ε u i j) = fun u => (q * t) * sr fp8 z (u i) := by
    funext u
    unfold msEden
    dsimp [y, q, z, t]
    ring
  rw [hform]
  change Integrable (fun u => (q * t) * sr fp8 z (u i))
    (volume.restrict (Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1)))
  exact (integrable_pi_eval (integrable_sr fp8 z) i).const_mul _

/-- Integration commutes with the finite, linear inverse RHT of §3.2. -/
theorem integral_rhtInv_of_integrable
    (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (f : (Fin (2 ^ k) → ℝ) → Fin (2 ^ k) → Fin 16 → ℝ)
    (hf : ∀ a b, Integrable (fun u => f u a b) (volume.restrict (edenCoinCube k)))
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, rhtInv k ε (f u) i j =
      rhtInv k ε (fun a b => ∫ u in edenCoinCube k, f u a b) i j := by
  simp only [rhtInv]
  rw [integral_const_mul]
  rw [integral_finsetSum _ (fun a _ => ?_)]
  · congr 1
    apply Finset.sum_congr rfl
    intro a _
    rw [integral_finsetSum _ (fun b _ => (hf a b).const_mul _)]
    simp only [integral_const_mul]
  · exact integrable_finsetSum _ fun b _ => (hf a b).const_mul _

/-- The FP8 coin average of one entry of §3.3, Algorithm 1 on a two-valued rotation. -/
theorem integral_msEden_twoValued
    {x : Fin (2 ^ k) → Fin 16 → ℝ} {ε : Fin (2 ^ k) → Fin 16 → Bool}
    {P R M s qP qR : ℝ} (hy : rht k ε x = twoValued k P R)
    (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M)
    (h : |P| = M ∨ |R| = M)
    (hqP : rtn fp4 (P * s / M) = qP) (hqR : rtn fp4 (R * s / M) = qR)
    (hD : 0 < P * qP + R * qR)
    (hz : s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256 ≤ 448)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, msEden k s x ε u i j =
      (if (j : ℕ) % 2 = 0 then qP else qR) *
        ((P ^ 2 + R ^ 2) / (P * qP + R * qR)) := by
  let z : ℝ := s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256
  let q : ℝ := if (j : ℕ) % 2 = 0 then qP else qR
  let t : ℝ := M / (s * 256)
  have hform : ∀ u, msEden k s x ε u i j = (q * t) * sr fp8 z (u i) := by
    intro u
    rw [msEden_twoValued hy hs hM hP hR h hqP hqR hD.ne' u i j]
    dsimp [q, z, t]
    ring
  have hz0 : 0 ≤ z := by dsimp [z]; positivity
  have h0 : (0 : ℝ) ∈ fp8 := ⟨by norm_num, 0, 0, by norm_num⟩
  have h448 : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num⟩
  simp_rw [hform]
  rw [integral_const_mul]
  change (q * t) * (∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1),
    sr fp8 z (u i)) = _
  rw [integral_pi_eval (sr fp8 z) i, integral_sr fp8_finite ⟨0, h0, hz0⟩ ⟨448, h448, hz⟩]
  dsimp [q, t, z]
  field_simp

/-- The conditional mean after inverse RHT is the inverse RHT of the mean rotated tensor (§3.2–§3.3). -/
theorem integral_rhtInv_msEden_twoValued_vector
    {x : Fin (2 ^ k) → Fin 16 → ℝ} {ε : Fin (2 ^ k) → Fin 16 → Bool}
    {P R M s qP qR : ℝ} (hy : rht k ε x = twoValued k P R)
    (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M)
    (h : |P| = M ∨ |R| = M)
    (hqP : rtn fp4 (P * s / M) = qP) (hqR : rtn fp4 (R * s / M) = qR)
    (hD : 0 < P * qP + R * qR)
    (hz : s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256 ≤ 448)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, rhtInv k ε (msEden k s x ε u) i j =
      rhtInv k ε (twoValued k
        (qP * ((P ^ 2 + R ^ 2) / (P * qP + R * qR)))
        (qR * ((P ^ 2 + R ^ 2) / (P * qP + R * qR)))) i j := by
  rw [integral_rhtInv_of_integrable ε (fun u => msEden k s x ε u)
    (fun a b => integrable_msEden_coins s x ε a b)]
  have hfun : (fun a b => ∫ u in edenCoinCube k, msEden k s x ε u a b) =
      twoValued k (qP * ((P ^ 2 + R ^ 2) / (P * qP + R * qR)))
        (qR * ((P ^ 2 + R ^ 2) / (P * qP + R * qR))) := by
    funext a b
    rw [integral_msEden_twoValued hy hs hM hP hR h hqP hqR hD hz]
    unfold twoValued
    split_ifs <;> rfl
  rw [hfun]

/-- Two nonzero input coordinates, for a rotation block in §3.2. -/
def twoSupport (k : ℕ) (A B : ℝ) : Fin (2 ^ k) → Fin 16 → ℝ :=
  fun i j => if i = 0 ∧ j = 0 then A else if i = 0 ∧ j = 1 then B else 0

/-- The Sylvester RHT of a two-coordinate vector takes two values (§3.2). -/
theorem rht_twoSupport (A B : ℝ) (ε : Fin (2 ^ k) → Fin 16 → Bool) :
    rht k ε (twoSupport k A B) = fun (_ : Fin (2 ^ k)) (j : Fin 16) =>
      if (j : ℕ) % 2 = 0 then
        ((if ε 0 0 then (-1 : ℝ) else 1) * A +
          (if ε 0 1 then (-1 : ℝ) else 1) * B) / Real.sqrt (2 ^ (k + 4))
      else
        ((if ε 0 0 then (-1 : ℝ) else 1) * A -
          (if ε 0 1 then (-1 : ℝ) else 1) * B) / Real.sqrt (2 ^ (k + 4)) := by
  funext i j
  unfold rht
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k)) (fun b _ hb => ?_) (by simp)]
  · rw [Finset.sum_eq_add_of_mem (0 : Fin 16) 1 (Finset.mem_univ _) (Finset.mem_univ _)
      (by decide) (fun c _ hc => by simp [twoSupport, hc.1, hc.2])]
    simp only [hadamard_eq, twoSupport, Fin.val_zero, Fin.val_one, walsh_zero_right,
      walsh_four_one]
    rcases Nat.mod_two_eq_zero_or_one (j : ℕ) with h | h <;>
      cases ε 0 0 <;> cases ε 0 1 <;> simp [h] <;> ring
  · exact Finset.sum_eq_zero fun j' _ => by simp [twoSupport, hb]

/-- At one §3.3 group, the integrability and two-valued hypotheses above are realized by
`e₀ + e₁/10`, the all-positive seed, and `s = 6`. -/
example (ε : Fin (2 ^ 0) → Fin 16 → Bool) :
    ∫ u in edenCoinCube 0, rhtInv 0 ε (msEden 0 6 (edenPair 0) ε u) 0 0 =
      rhtInv 0 ε
        (fun a b => ∫ u in edenCoinCube 0, msEden 0 6 (edenPair 0) ε u a b) 0 0 :=
  integral_rhtInv_of_integrable ε _
    (fun a b => integrable_msEden_coins 6 (edenPair 0) ε a b) 0 0

/-- The §3.3 two-valued hypotheses, including the FP8 range, hold at `k = 0`, `s = 6`. -/
example :
    (∫ u in edenCoinCube 0,
      msEden 0 6 (edenPair 0) (fun _ _ => false) u 0 0 =
        6 * (((11 / 40 : ℝ) ^ 2 + (9 / 40) ^ 2) /
          ((11 / 40) * 6 + (9 / 40) * 4))) ∧
    (∫ u in edenCoinCube 0,
      rhtInv 0 (fun _ _ => false)
        (msEden 0 6 (edenPair 0) (fun _ _ => false) u) 0 0 =
      rhtInv 0 (fun _ _ => false)
        (twoValued 0
          (6 * (((11 / 40 : ℝ) ^ 2 + (9 / 40) ^ 2) / ((11 / 40) * 6 + (9 / 40) * 4)))
          (4 * (((11 / 40 : ℝ) ^ 2 + (9 / 40) ^ 2) / ((11 / 40) * 6 + (9 / 40) * 4)))) 0 0) := by
  have h4 : Real.sqrt (2 ^ (0 + 4) : ℝ) = 4 := by
    rw [show (2 : ℝ) ^ (0 + 4) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hy : rht 0 (fun _ _ => false) (edenPair 0) = twoValued 0 (11 / 40) (9 / 40) := by
    rw [rht_edenPair, h4]
    funext i j
    unfold twoValued
    norm_num
  have hqP : rtn fp4 ((11 / 40 : ℝ) * 6 / (11 / 40)) = 6 := by
    rw [show (11 / 40 : ℝ) * 6 / (11 / 40) = 6 by norm_num]
    exact rtn_fp4_six (by norm_num)
  have hqR : rtn fp4 ((9 / 40 : ℝ) * 6 / (11 / 40)) = 4 := by
    rw [show (9 / 40 : ℝ) * 6 / (11 / 40) = 54 / 11 by norm_num]
    exact rtn_fp4_four (by norm_num) (by norm_num)
  constructor
  · exact integral_msEden_twoValued hy (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (Or.inl (by norm_num)) hqP hqR
      (by norm_num) (by norm_num) 0 0
  · exact integral_rhtInv_msEden_twoValued_vector hy (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (Or.inl (by norm_num)) hqP hqR
      (by norm_num) (by norm_num) 0 0

end Transformer.Quartet

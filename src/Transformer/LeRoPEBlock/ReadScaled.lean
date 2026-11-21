module

public import Transformer.LeRoPEBlock.ReadDFT

/-!
# The DFT heads at any scale of the logits

Source: `review/lerope_block.md`, *Reading a slot* (added), RS5.

The model normalizes a head's query and key before the rotation, so a query read from a positive
coordinate is `λ · 1` with `λ = a / √(a² + ε)`, below `1` for `ε > 0`, and the key likewise. At
`β = 0` the logit is bilinear in the query and the key (`leropeLogit_smul_smul`): the DFT head's
logits become RS4's times `c = λ_Q λ_K`. The head keeps the shape RS4 reads with:
`P_c / (P_c + i)` on its key and `1 / (P_c + i)` on each other key, `P_c = (i + 1)^{c (1 + 1/(2R))}`
(`depthWeight_smul_dftScores`, from `depthWeight_ite`). So RS4's reading holds with `P_c` in place
of `P` for every `c > 0` (`dft_read_uniform_smul`).

Two heads of one shape read the difference of their keys at every position (`dft_read_sub`): the
weights off the two keys cancel, and the fixed combination `o − o'` is `(p − q)(v_s − v_{s'})`.
RS4's decoder instead subtracts `(i + 1) q` times a uniform average, a coefficient that changes with
the query position `i`: an output projection shared by the positions decodes with it at one
position only.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.Residual Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **At `β = 0` the logit is bilinear** in the query and the key: scaling them by `λ` and `μ`
scales the logit by `λ μ`. -/
theorem leropeLogit_smul_smul {X : Type*} {p : ℕ} (θ : Fin p → ℝ) (d l m : ℝ)
    (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (δ : ℕ) :
    leropeLogit θ 0 d (fun x => l • Q x) (fun y => m • K y) i x y δ =
      l * m * leropeLogit θ 0 d Q K i x y δ := by
  simp only [leropeLogit, phaseLogit, zero_mul, zero_div, add_zero, mulVec_smul, smul_dotProduct,
    dotProduct_smul, smul_eq_mul]
  ring

/-- Halving the query and the key quarters the logit. -/
example (Q K : Unit → Fin 2 × Fin 1 → ℝ) :
    leropeLogit (fun _ => 1) 0 2 (fun x => (1 / 2 : ℝ) • Q x) (fun y => (1 / 2 : ℝ) • K y) 3 ()
      () 1 = 1 / 2 * (1 / 2) * leropeLogit (fun _ => 1) 0 2 Q K 3 () () 1 :=
  leropeLogit_smul_smul _ _ _ _ _ _ _ _ _ _

/-- **Scores `a` on the key `s` and `b` on every other key** of `i + 1` give the weights
`e^a / (e^a + i e^b)` on `s` and `e^b / (e^a + i e^b)` elsewhere. -/
theorem depthWeight_ite {i : ℕ} (s : Fin (i + 1)) (a b : ℝ) (j : Fin (i + 1)) :
    depthWeight (fun j => if j = s then a else b) j =
      if j = s then Real.exp a / (Real.exp a + i * Real.exp b)
      else Real.exp b / (Real.exp a + i * Real.exp b) := by
  have h (j : Fin (i + 1)) : Real.exp (if j = s then a else b) =
      Real.exp b + if j = s then Real.exp a - Real.exp b else 0 := by
    split_ifs <;> ring
  have hZ : depthPartition (fun j : Fin (i + 1) => if j = s then a else b) =
      Real.exp a + i * Real.exp b := by
    simp only [depthPartition, h, sum_add_distrib, sum_const, card_univ, Fintype.card_fin,
      sum_ite_eq', mem_univ, ite_true, nsmul_eq_mul]
    push_cast
    ring
  rw [depthWeight, hZ]
  split_ifs <;> rfl

/-- Two keys, scores `0` and `0`: half each. -/
example : depthWeight (fun j : Fin 2 => if j = 0 then (0 : ℝ) else 0) 0 =
    Real.exp 0 / (Real.exp 0 + (1 : ℕ) * Real.exp 0) := by
  rw [depthWeight_ite]; simp

/-- `P_c = (i + 1)^{c (1 + 1/(2R))}`: the DFT head with logits scaled by `c` keeps `P_c / (P_c + i)`
on its key. -/
noncomputable def dftPowAt (c : ℝ) (R i : ℕ) : ℝ := ((i : ℝ) + 1) ^ (c * (1 + 1 / (2 * (R : ℝ))))

/-- `p − q = (P_c − 1)/(P_c + i)` for the DFT heads with logits scaled by `c`. -/
noncomputable def dftGapAt (c : ℝ) (R i : ℕ) : ℝ := (dftPowAt c R i - 1) / (dftPowAt c R i + i)

/-- At scale `1` these are RS4's `P` and `p − q`. -/
theorem dftPowAt_one (R i : ℕ) : dftPowAt 1 R i = dftPow R i := by
  rw [dftPowAt, dftPow, one_mul]

theorem one_lt_dftPowAt {c : ℝ} (hc : 0 < c) {R i : ℕ} (hi : 1 ≤ i) : 1 < dftPowAt c R i := by
  have : (1 : ℝ) ≤ i := by exact_mod_cast hi
  exact Real.one_lt_rpow (by linarith) (by positivity)

/-- **The DFT head with logits scaled by `c`**, turned by `i − s` at the query position `i ≤ 2R`:
`P_c / (P_c + i)` on the key `s` and `1 / (P_c + i)` on every other key. -/
theorem depthWeight_smul_dftScores {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi : i ≤ 2 * R)
    (x : ℕ → X) (c : ℝ) (s j : Fin (i + 1)) :
    depthWeight (c • dftScores R (i - s) i x) j =
      if j = s then dftPowAt c R i / (dftPowAt c R i + i) else 1 / (dftPowAt c R i + i) := by
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  set L := Real.log ((i : ℝ) + 1) with hL
  have hsc : c • dftScores R (i - s) i x =
      fun j : Fin (i + 1) => if j = s then c * L else c * -(L / (2 * R)) := by
    funext j
    simp only [Pi.smul_apply, smul_eq_mul, dftScores,
      leropeLogit_dft_shift_at hR (by omega : i - s ≤ 2 * R) i _ _ (by omega : i - j ≤ 2 * R),
      ← hL]
    by_cases hjs : j = s
    · subst hjs; simp
    · have hs := s.2
      have hj := j.2
      have h' : i - (j : ℕ) ≠ i - s := fun h => hjs (Fin.ext (by omega))
      simp [h', hjs]
  have hP : Real.exp (c * L) = dftPowAt c R i * Real.exp (c * -(L / (2 * R))) := by
    rw [dftPowAt, Real.rpow_def_of_pos (by positivity), ← Real.exp_add, ← hL]
    congr 1
    field_simp
    ring
  have he := Real.exp_pos (c * -(L / (2 * R)))
  have hpos : 0 < dftPowAt c R i + i := by unfold dftPowAt; positivity
  rw [hsc, depthWeight_ite, hP, ← add_mul]
  split_ifs
  · rw [mul_div_mul_right _ _ he.ne']
  · rw [div_mul_cancel_right₀ he.ne', one_div]

/-- The final position `3` of two copies, `16` bands, logits halved. -/
example (s j : Fin 4) := depthWeight_smul_dftScores (X := Unit) (R := 16) (i := 3) (by norm_num)
  (by norm_num) (fun _ => ()) (1 / 2) s j

theorem dftGapAt_eq (c : ℝ) (R i : ℕ) :
    dftGapAt c R i = dftPowAt c R i / (dftPowAt c R i + i) - 1 / (dftPowAt c R i + i) := by
  rw [dftGapAt, div_sub_div_same]

theorem dftGapAt_pos {c : ℝ} (hc : 0 < c) {R i : ℕ} (hi : 1 ≤ i) : 0 < dftGapAt c R i := by
  have hP := one_lt_dftPowAt hc (R := R) hi
  unfold dftGapAt
  exact div_pos (by linarith) (by positivity)

/-- **RS4's reading at the logit scale `c > 0`**: the head turned by `i − s` and a uniform head
give the value at `s` as `(o_s − ((i + 1)/(P_c + i)) o_0) / (p − q)`, for `1 ≤ i ≤ 2R`. -/
theorem dft_read_uniform_smul {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi1 : 1 ≤ i) (hi : i ≤ 2 * R)
    (x : ℕ → X) {c : ℝ} (hc : 0 < c) (v : Fin (i + 1) → E) (s : Fin (i + 1)) :
    (dftGapAt c R i)⁻¹ • (attend (c • dftScores R (i - s) i x) v -
      (((i : ℝ) + 1) / (dftPowAt c R i + i)) • attend (fun _ : Fin (i + 1) => (0 : ℝ)) v) =
      v s := by
  have hpos : 0 < dftPowAt c R i + i := by unfold dftPowAt; positivity
  have hP := one_lt_dftPowAt hc (R := R) hi1
  have hu : attend (fun _ : Fin (i + 1) => (0 : ℝ)) v = (1 / ((i : ℝ) + 1)) • ∑ j, v j := by
    simp [attend, weightedSum, depthWeight_zero, smul_sum]
  rw [hu, smul_smul, show ((i : ℝ) + 1) / (dftPowAt c R i + i) * (1 / ((i : ℝ) + 1)) =
    1 / (dftPowAt c R i + i) by field_simp, attend,
    funext (depthWeight_smul_dftScores hR hi x c s), dftGapAt_eq]
  exact decode_weightedSum_ite (fun h => by
    rw [div_eq_div_iff hpos.ne' hpos.ne'] at h
    nlinarith) v s

/-- Two copies end at the position `3`: the head turned by `3` and the uniform head read the key `0`
at the logit scale `1/2`. -/
example (v : Fin 4 → ℝ) := dft_read_uniform_smul (X := Unit) (R := 16) (i := 3) (by norm_num)
  (by norm_num) (by norm_num) (fun _ => ()) (by norm_num : (0 : ℝ) < 1 / 2) v 0

/-- **Two heads of one shape read the difference of their keys at every position**: the heads
turned by `i − s` and `i − s'` average to `(p − q)(v_s − v_{s'})` apart, at any logit scale `c`. -/
theorem dft_read_sub {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi : i ≤ 2 * R) (x : ℕ → X) (c : ℝ)
    (v : Fin (i + 1) → E) (s s' : Fin (i + 1)) :
    attend (c • dftScores R (i - s) i x) v - attend (c • dftScores R (i - s') i x) v =
      dftGapAt c R i • (v s - v s') := by
  rw [attend, attend, funext (depthWeight_smul_dftScores hR hi x c s),
    funext (depthWeight_smul_dftScores hR hi x c s'), weightedSum_ite_eq, weightedSum_ite_eq,
    dftGapAt_eq, smul_sub]
  abel

/-- At every position `i ≤ 32` of a window of `16` bands, any two keys `s, s'`. -/
example (i : ℕ) (hi : i ≤ 32) (v : Fin (i + 1) → ℝ) (s s' : Fin (i + 1)) :=
  dft_read_sub (X := Unit) (R := 16) (by norm_num) hi (fun _ => ()) 1 v s s'

end Transformer.LeRoPEBlock

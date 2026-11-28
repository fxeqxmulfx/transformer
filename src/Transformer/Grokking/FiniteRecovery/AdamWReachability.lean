module

public import Transformer.Grokking.FiniteRecovery.AdamWTrials
public import FixedPointTheorems.brouwer

/-! Finite-horizon reachability for the native AdamW update. Decoupled
decay contributes a positive scalar multiple of the initial weights.
The remaining displacement is uniformly bounded over all initializations.
Brouwer's theorem then gives simultaneous control of every coordinate
that was free at initialization, without shrinking the learning rates. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

open Metric

/-- The scalar by which decoupled decay would multiply the weights if
the adaptive directions were zero for the first `K` updates. -/
noncomputable def decayFactor (rate : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | t+1 => (1 - rate t / 10) * decayFactor rate t

theorem decayFactor_bounds (rate : ℕ → ℝ) (K : ℕ) (hr : ∀ t < K, 0 ≤ rate t ∧ rate t < 10) :
    0 < decayFactor rate K ∧ decayFactor rate K ≤ 1 := by
  induction K with
  | zero => exact ⟨by norm_num [decayFactor],le_rfl⟩
  | succ t ih =>
    obtain ⟨hp,h1⟩ := ih (fun j hj => hr j (Nat.lt_succ_of_lt hj))
    have h := hr t (Nat.lt_succ_self t)
    exact ⟨mul_pos (by linarith) hp,(mul_le_of_le_one_left hp.le (by linarith)).trans h1⟩

variable {I J : Type*} [Fintype I] [Fintype J]

/-- The center follows pure weight decay. Its error bound in every
coordinate is independent of the magnitude of the initial weights. -/
theorem decay_displacement (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ) (w : I → ℝ)
    (K : ℕ) (hr : ∀ t < K, 0 ≤ rate t ∧ rate t < 10) (i : I) :
    |(run rate g w K).weights i - decayFactor rate K * w i| ≤ 120 * (1 - decayFactor rate K) := by
  induction K with
  | zero => simp [decayFactor,run]
  | succ t ih =>
    let s := run rate g w
    let u := direction t ((s (t+1)).first i) ((s (t+1)).second i)
    have hu : |u| ≤ 12 := direction_bound t (moment_cone rate g w i (t+1))
    have h := hr t (Nat.lt_succ_self t)
    have hd : 0 ≤ 1 - rate t / 10 := by linarith
    have hind := ih (fun j hj => hr j (Nat.lt_succ_of_lt hj))
    have he : (s (t+1)).weights i - decayFactor rate (t+1) * w i =
        (1 - rate t/10) * ((s t).weights i - decayFactor rate t * w i) - rate t * u := by
      change (1 - rate t/10) * (s t).weights i - rate t * u - _ = _
      rw [decayFactor]; ring
    change |(s (t+1)).weights i - _| ≤ _
    rw [he]
    calc
      _ ≤ |(1 - rate t/10) * ((s t).weights i - decayFactor rate t * w i)| + |rate t * u| := abs_sub _ _
      _ ≤ (1 - rate t/10) * (120 * (1 - decayFactor rate t)) + rate t * 12 := by
        rw [abs_mul,abs_mul,abs_of_nonneg hd,abs_of_nonneg h.1]
        exact add_le_add (mul_le_mul_of_nonneg_left hind hd) (mul_le_mul_of_nonneg_left hu h.1)
      _ = _ := by rw [decayFactor]; ring

/-- A bounded continuous perturbation of a positive scalar multiple of
the identity on a finite-dimensional real space is onto. The fixed-point
equation for any requested output maps an explicit closed ball to itself. -/
theorem surjective_of_bounded_displacement {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] (F : E → E) (hF : Continuous F) {c R : ℝ} (hc : 0 < c) (hR : 0 ≤ R)
    (hb : ∀ x, ‖F x - c • x‖ ≤ R) : Function.Surjective F := by
  intro y
  let B := closedBall (0 : E) ((‖y‖ + R) / c)
  let P (x : E) := c⁻¹ • (y - (F x - c • x))
  have hP : Continuous P := (continuous_const.sub (hF.sub (continuous_id.const_smul c))).const_smul _
  have hB (x : E) : P x ∈ B := by
    change dist (P x) 0 ≤ _
    rw [dist_zero_right]
    calc
      ‖P x‖ = c⁻¹ * ‖y - (F x - c • x)‖ := by
        dsimp only [P]
        rw [norm_smul,Real.norm_eq_abs,abs_of_pos (inv_pos.mpr hc)]
      _ ≤ c⁻¹ * (‖y‖ + R) := mul_le_mul_of_nonneg_left
        ((norm_sub_le _ _).trans (add_le_add le_rfl (hb x))) (inv_nonneg.mpr hc.le)
      _ = _ := by ring
  let p : C(B,B) := ⟨fun x => ⟨P x,hB x⟩,(hP.comp continuous_subtype_val).subtype_mk _⟩
  obtain ⟨x,hx⟩ := brouwer_fixed_point B (convex_closedBall _ _) (isCompact_closedBall _ _)
    ⟨0,mem_closedBall_self (div_nonneg (add_nonneg (norm_nonneg _) hR) hc.le)⟩ p
  have he : P x = x := congrArg Subtype.val hx
  have hh := congrArg (fun a : E => c • a) he
  change c • (c⁻¹ • (y - (F x - c • (x : E)))) = _ at hh
  rw [smul_smul,mul_inv_cancel₀ hc.ne',one_smul] at hh
  refine ⟨x,?_⟩
  calc
    F x = c • (x : E) + (F x - c • (x : E)) := by abel
    _ = y := ((sub_eq_iff_eq_add).mp hh).symm

/-- Any chosen vector on the free coordinates can be reached exactly at
the declared update. Other coordinates may start deterministically and
still train jointly; all gradients may depend on all coordinates. -/
theorem reachable_coordinates (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (pick : J → I) (start : (J → ℝ) → I → ℝ)
    (hstart : Continuous start) (hfree : ∀ z j, start z (pick j) = z j)
    (K : ℕ) (hr : ∀ t < K, 0 ≤ rate t ∧ rate t < 10) :
    Function.Surjective (fun z j => (run rate g (start z) K).weights (pick j)) := by
  obtain ⟨hp,h1⟩ := decayFactor_bounds rate K hr
  have hR : 0 ≤ 120 * (1 - decayFactor rate K) := by linarith
  apply surjective_of_bounded_displacement _ _ hp hR
  · intro z
    apply (pi_norm_le_iff_of_nonneg hR).2
    intro j
    simpa only [Pi.sub_apply,Pi.smul_apply,smul_eq_mul,Real.norm_eq_abs,hfree] using
      decay_displacement rate g (start z) K hr (pick j)
  · exact continuous_pi fun j => (continuous_apply (pick j)).comp
      ((continuous_run rate g hg K).1.comp hstart)

/-- A coupled two-coordinate gradient and a determined second coordinate
already allow every terminal value for the freely initialized coordinate. -/
example (K : ℕ) : Function.Surjective (fun z : Unit → ℝ => fun _ : Unit =>
    (run (fun _ => 1/1000) (fun _ x i => x i + x (!i))
      (fun i : Bool => if i then (1 : ℝ) else z ()) K).weights false) :=
  reachable_coordinates (fun _ => 1/1000) _ (fun _ => continuous_pi fun i =>
    (continuous_apply i).add (continuous_apply (!i))) (fun _ => false) _
    (continuous_pi fun i => by cases i <;> simp [continuous_apply,continuous_const]) (fun _ _ => rfl)
    K (by intro; norm_num)

end Transformer.Grokking.FiniteRecovery.AdamW

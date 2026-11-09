module

public import Transformer.Grokking.GradientEvidence.FutureDirectionNorm
public import Transformer.Grokking.GradientEvidence.DampedPersistence

/-!
# A future AdamW tube without future forcing observations

Source: `review/flat_spectrum.md`, *Future forcing from checkpoint moments*,
FF4. The path uses the actual .9/.95 buffered direction. Its radius uses only
checkpoint cones and the known schedule. This theorem is exact-real; the
experiment's FP32 allowances are checked separately, not silently assumed
proved. A useful lower geometry bound can still expire at a finite horizon.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset
variable {K : Type*} [Fintype K]

/-- FF4: exact AdamW continuation from saved parameters and moment buffers. -/
def warmAdamPath (p : EuclideanSpace ℝ K) (m v : K → ℝ) (g : K → ℕ → ℝ)
    (ρ eps : ℝ) (η : ℕ → ℝ) (start : ℕ) : ℕ → EuclideanSpace ℝ K
  | 0 => p
  | h+1 => (1-η h*ρ) • warmAdamPath p m v g ρ eps η start h -
      η h • warmDirections m v g eps start h

omit [Fintype K] in
/-- FF4: every coordinate is the specified main-model buffered Adam update,
including absolute bias-correction time and decoupled weight decay. -/
theorem warm_adam_source (p : EuclideanSpace ℝ K) (m v : K → ℝ)
    (g : K → ℕ → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ) (start h : ℕ) (k : K) :
    warmAdamPath p m v g ρ eps η start (h+1) k =
      bufferedAdamUpdate (9/10) (19/20) eps (η h) ρ (start+h+1)
        (warmAdamPath p m v g ρ eps η start h k)
        (warmMoments (m k) (v k) (g k) (h+1)).1
        (warmMoments (m k) (v k) (g k) (h+1)).2 := by
  change (1-η h*ρ) * warmAdamPath p m v g ρ eps η start h k -
    η h * warmDirection (m k) (v k) (g k) eps start h = _
  unfold warmDirection bufferedAdamUpdate
  ring

/-- FF4: the actual forcing has a budget determined before future gradients.
Only initial moment validity, positive epsilon and a nonnegative rate are used. -/
theorem warm_adam_forcing_bound (p : EuclideanSpace ℝ K) (m v r : K → ℝ)
    (g : K → ℕ → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (start h : ℕ) :
    ‖warmAdamPath p m v g ρ eps η start (h+1) -
      (1-η h*ρ) • warmAdamPath p m v g ρ eps η start h‖ ≤
      η h * (Real.sqrt (∑ k, futureCone (r k) (h+1)) * futureBias start h) := by
  have he : warmAdamPath p m v g ρ eps η start (h+1) -
      (1-η h*ρ) • warmAdamPath p m v g ρ eps η start h =
      -(η h • warmDirections m v g eps start h) := by
    rw [warmAdamPath]
    abel
  rw [he, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_nonneg (hη h)]
  exact mul_le_mul_of_nonneg_left
    (warm_directions_norm m v r g eps hv hr hc heps start h) (hη h)

omit [Fintype K] in
/-- FF4: continuing the original AdamW path from a checkpoint produces the
same parameters at every later time. All earlier moment history is retained. -/
theorem main_adam_resume (p0 : K → ℝ) (g : K → ℕ → ℝ) (ρ eps : ℝ)
    (η : ℕ → ℝ) (start n : ℕ) :
    warmAdamPath (WithLp.toLp 2 (fun k => mainAdamPath (p0 k) ρ eps η (g k) start))
      (fun k => (mainMoments (g k) start).1) (fun k => (mainMoments (g k) start).2)
      (fun k h => g k (start+h)) ρ eps (fun h => η (start+h)) start n =
      WithLp.toLp 2 (fun k => mainAdamPath (p0 k) ρ eps η (g k) (start+n)) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [warmAdamPath, ih]
    ext k
    change (1-η (start+n)*ρ) * mainAdamPath (p0 k) ρ eps η (g k) (start+n) -
      η (start+n) * warmDirection (mainMoments (g k) start).1
        (mainMoments (g k) start).2 (fun h => g k (start+h)) eps start n =
      mainAdamPath (p0 k) ρ eps η (g k) (start+(n+1))
    rw [← main_direction_resume]
    rfl

/-- FF4: a checkpoint-only radius controls every future prefix for arbitrary
gradients. No future direction norm or forcing budget is a hypothesis. -/
theorem warm_adam_tube (p : EuclideanSpace ℝ K) (m v r : K → ℝ)
    (g : K → ℕ → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (ha : ∀ h, η h * ρ ≤ 1) (start n : ℕ) :
    ‖warmAdamPath p m v g ρ eps η start n -
      dampedScale (fun h => 1-η h*ρ) n • p‖ ≤
      forcingRadius (fun h => 1-η h*ρ)
        (fun h => η h * (Real.sqrt (∑ k, futureCone (r k) (h+1)) *
          futureBias start h)) n := by
  exact damped_forcing_tube (warmAdamPath p m v g ρ eps η start) _ _
    (fun h => sub_nonneg.mpr (ha h))
    (warm_adam_forcing_bound p m v r g ρ eps η hv hr hc heps hη start) n

/-- FF4: any fixed contractive linear projection inherits the future tube.
The projection premise concerns geometry of the operator, not future gradients. -/
theorem warm_adam_projected_tube {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (P : EuclideanSpace ℝ K →ₗ[ℝ] F) (hP : ∀ x, ‖P x‖ ≤ ‖x‖)
    (p : EuclideanSpace ℝ K) (m v r : K → ℝ) (g : K → ℕ → ℝ)
    (ρ eps : ℝ) (η : ℕ → ℝ) (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (ha : ∀ h, η h * ρ ≤ 1) (start n : ℕ) :
    ‖P (warmAdamPath p m v g ρ eps η start n) -
      dampedScale (fun h => 1-η h*ρ) n • P p‖ ≤
      forcingRadius (fun h => 1-η h*ρ)
        (fun h => η h * (Real.sqrt (∑ k, futureCone (r k) (h+1)) *
          futureBias start h)) n := by
  rw [← P.map_smul, ← P.map_sub]
  exact (hP _).trans (warm_adam_tube p m v r g ρ eps η hv hr hc heps hη ha start n)

/-- FF4: the same prospective radius supplies both norm bounds. If it exceeds
the damped initial norm, the lower bound is explicitly zero. -/
theorem warm_adam_norm_bounds (p : EuclideanSpace ℝ K) (m v r : K → ℝ)
    (g : K → ℕ → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (ha : ∀ h, η h * ρ ≤ 1) (start n : ℕ) :
    let q := dampedScale (fun h => 1-η h*ρ) n
    let R := forcingRadius (fun h => 1-η h*ρ)
      (fun h => η h * (Real.sqrt (∑ k, futureCone (r k) (h+1)) *
        futureBias start h)) n
    max 0 (q*‖p‖-R) ≤ ‖warmAdamPath p m v g ρ eps η start n‖ ∧
      ‖warmAdamPath p m v g ρ eps η start n‖ ≤ q*‖p‖+R := by
  exact damped_forcing_norm_bounds (warmAdamPath p m v g ρ eps η start) _ _
    (fun h => sub_nonneg.mpr (ha h))
    (warm_adam_forcing_bound p m v r g ρ eps η hv hr hc heps hη start) n

example (g : Fin 3 → ℕ → ℝ) (start h : ℕ) :=
  warm_adam_forcing_bound (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10)
    (fun _ => 1/100) (fun _ => 1) g (1/10) (1/100000000) (fun _ => 1/10000)
    (by intro k; norm_num) (by intro k; norm_num) (by intro k; norm_num)
    (by norm_num) (by intro h; norm_num) start h
example (g : Fin 3 → ℕ → ℝ) (start n : ℕ) :=
  warm_adam_tube (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10)
    (fun _ => 1/100) (fun _ => 1) g (1/10) (1/100000000) (fun _ => 1/10000)
    (by intro k; norm_num) (by intro k; norm_num) (by intro k; norm_num)
    (by norm_num) (by intro h; norm_num) (by intro h; norm_num) start n
example (g : Fin 3 → ℕ → ℝ) (start n : ℕ) :=
  warm_adam_norm_bounds (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10)
    (fun _ => 1/100) (fun _ => 1) g (1/10) (1/100000000) (fun _ => 1/10000)
    (by intro k; norm_num) (by intro k; norm_num) (by intro k; norm_num)
    (by norm_num) (by intro h; norm_num) (by intro h; norm_num) start n
example (g : Fin 3 → ℕ → ℝ) (start n : ℕ) :=
  warm_adam_projected_tube LinearMap.id (by intro x; exact le_rfl)
    (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10) (fun _ => 1/100) (fun _ => 1)
    g (1/10) (1/100000000) (fun _ => 1/10000) (by intro k; norm_num)
    (by intro k; norm_num) (by intro k; norm_num) (by norm_num)
    (by intro h; norm_num) (by intro h; norm_num) start n

end Transformer.Grokking.GradientEvidence

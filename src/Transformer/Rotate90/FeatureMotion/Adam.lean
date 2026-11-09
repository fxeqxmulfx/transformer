module

public import Transformer.Rotate90.FeatureMotion.Probability
public import Transformer.Grokking.GradientEvidence.FutureAdamTube

/-!
# Full-parameter Adam displacement about a saved checkpoint

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM5 and FF4.
The warm-moment tube is centered at the damped old parameters. Moving its
center back to the checkpoint supplies the displacement caps used by EM4.
The coordinate type may be any complete parameter block, not only a readout.
This is exact-real AdamW; floating-point update defects are not omitted
implicitly and require a separate bound before applying this to execution.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Grokking.GradientEvidence Finset
variable {I : Type*} [Fintype I]

/-- EM5: checkpoint-only parameter displacement; all future moments have
already been eliminated by the warm-Adam cone bound. -/
def adamRadius (p : EuclideanSpace ℝ I) (r : I → ℝ) (ρ : ℝ) (η : ℕ → ℝ)
    (start n : ℕ) : ℝ :=
  |dampedScale (fun h => 1-η h*ρ) n - 1| * ‖p‖ +
    forcingRadius (fun h => 1-η h*ρ)
      (fun h => η h * (Real.sqrt (∑ k, futureCone (r k) (h+1)) * futureBias start h)) n

/-- EM5: every future prefix of exact AdamW lies within a radius computed
from saved parameters/moments and the schedule. Gradients are arbitrary;
in particular no future direction norm is assumed or measured. -/
theorem warm_adam_checkpoint_motion (p : EuclideanSpace ℝ I) (m v r : I → ℝ)
    (g : I → ℕ → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (ha : ∀ h, η h * ρ ≤ 1) (start n : ℕ) :
    ‖warmAdamPath p m v g ρ eps η start n - p‖ ≤ adamRadius p r ρ η start n := by
  have ht := warm_adam_tube p m v r g ρ eps η hv hr hc heps hη ha start n
  let q := dampedScale (fun h => 1-η h*ρ) n
  have he : q • p - p = (q-1) • p := by rw [sub_smul, one_smul]
  have hb := (norm_sub_le_norm_sub_add_norm_sub
    (warmAdamPath p m v g ρ eps η start n) (q • p) p).trans (add_le_add ht le_rfl)
  rw [he, norm_smul, Real.norm_eq_abs] at hb
  simpa only [adamRadius, q, add_comm] using hb

/-- EM5: a fixed contractive parameter projection inherits the checkpoint
radius. For tighter block caps, apply the preceding theorem to that block's
own saved coordinates and moment ratios, as the native constructor does. -/
theorem warm_adam_projected_motion {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (P : EuclideanSpace ℝ I →ₗ[ℝ] F) (hP : ∀ x, ‖P x‖ ≤ ‖x‖)
    (p : EuclideanSpace ℝ I) (m v r : I → ℝ) (g : I → ℕ → ℝ)
    (ρ eps : ℝ) (η : ℕ → ℝ) (hv : ∀ k, 0 ≤ v k) (hr : ∀ k, 0 ≤ r k)
    (hc : ∀ k, (m k)^2 ≤ r k * v k) (heps : 0 < eps)
    (hη : ∀ h, 0 ≤ η h) (ha : ∀ h, η h * ρ ≤ 1) (start n : ℕ) :
    ‖P (warmAdamPath p m v g ρ eps η start n) - P p‖ ≤ adamRadius p r ρ η start n := by
  rw [← P.map_sub]
  exact (hP _).trans (warm_adam_checkpoint_motion p m v r g ρ eps η
    hv hr hc heps hη ha start n)

example (g : Fin 3 → ℕ → ℝ) (start n : ℕ) :=
  warm_adam_checkpoint_motion (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10)
    (fun _ => 1/100) (fun _ => 1) g (1/10) (1/100000000) (fun _ => 1/10000)
    (by intro k; norm_num) (by intro k; norm_num) (by intro k; norm_num)
    (by norm_num) (by intro h; norm_num) (by intro h; norm_num) start n
example (g : Fin 3 → ℕ → ℝ) (start n : ℕ) :=
  warm_adam_projected_motion LinearMap.id (by intro x; exact le_rfl)
    (WithLp.toLp 2 (fun _ => 2)) (fun _ => 1/10) (fun _ => 1/100) (fun _ => 1)
    g (1/10) (1/100000000) (fun _ => 1/10000) (by intro k; norm_num)
    (by intro k; norm_num) (by intro k; norm_num) (by norm_num)
    (by intro h; norm_num) (by intro h; norm_num) start n

end Transformer.Rotate90.FeatureMotion

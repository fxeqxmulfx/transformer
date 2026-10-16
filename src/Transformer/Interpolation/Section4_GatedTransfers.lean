/-
# Exact local transfers under one original gated neuron

ArXiv:2411.04551v3, §4, the point matching step in
`prop: interpolation.neural.ode`, and §1, `eq: neural.ode.sphere`.
If the original gate has a positive lower bound along a finite
projection segment, the actual integrated activation clock reaches
its terminal level in finite positive physical time. The proved exact
clock identity therefore gives an exact original gated transfer.

For distinct non-antipodal unit endpoints, the chord-axis projection
transfer admits a positive midpoint cap gate throughout its segment.
The resulting original ReLU flow sends the input exactly to the target
and fixes every unit point outside that cap. This local construction
retains the original neuron and its actual global flow.
-/

import Transformer.Interpolation.Section4_RidgeClocks
import Transformer.Interpolation.Section4_ProjectionSegments
import Transformer.Interpolation.Section3_RidgeTransport

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A uniformly active finite intrinsic segment is reached in finite
positive physical time by the actual original ReLU clock. Source
auxiliary: arXiv:2411.04551v3, §4, the local point matching clock. -/
theorem exists_ridgeProjectionClock_eq (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) (τ κ : ℝ) (hτ : 0 < τ) (hκ : 0 < κ)
    (hgate : ∀ s ∈ Icc (0 : ℝ) τ,
      κ ≤ max (b - inner ℝ (sphereProjectionFlow a s x : EucSpace d) u) 0) :
    ∃ T : ℝ, 0 < T ∧ ridgeProjectionClock a u b x T = τ ∧
      ridgeFlow (a : EucSpace d) u b x T = sphereProjectionFlow a τ x := by
  let c := ridgeProjectionClock a u b x
  let g (s : ℝ) := max (b - inner ℝ (ridgeFlow (a : EucSpace d) u b x s : EucSpace d) u) 0
  let R := 2 * τ / κ + 1
  have hR : 0 < R := by dsimp only [R]; positivity
  have hreach : τ ≤ c R := by
    by_contra hlt
    have hlt' : c R < τ := lt_of_not_ge hlt
    have hact (s : ℝ) (hs : s ∈ Ico (0 : ℝ) R) : κ ≤ g s := by
      dsimp only [g]
      rw [ridgeFlow_eq_projection_clock a u b x s hs.1]
      exact hgate (c s) ⟨ridgeProjectionClock_nonneg a u b x s hs.1,
        ((monotone_ridgeProjectionClock a u b x) hs.2.le).trans hlt'.le⟩
    have hf (s : ℝ) : HasDerivAt (fun r => (κ / 2) * r) (κ / 2) s := by
      simpa only [id_eq, mul_one] using (hasDerivAt_id s).const_mul (κ / 2)
    have hb := image_le_of_deriv_right_lt_deriv_boundary
      (f := fun s => (κ / 2) * s) (f' := fun _ => κ / 2)
      (B := c) (B' := g) (a := 0) (b := R)
      (continuous_const.mul continuous_id).continuousOn (fun s _ => (hf s).hasDerivWithinAt)
      (by
        rw [mul_zero]
        change 0 ≤ ridgeProjectionClock a u b x 0
        rw [ridgeProjectionClock_zero])
      (fun s => hasDerivAt_ridgeProjectionClock a u b x s)
      (fun s hs _ => by have hh := hact s hs; linarith)
    have hh := hb ⟨hR.le, le_rfl⟩
    have hlin : (κ / 2) * R = τ + κ / 2 := by dsimp only [R]; field_simp
    rw [hlin] at hh
    linarith
  obtain ⟨T, hT, hcT⟩ := intermediate_value_Icc hR.le
    (continuous_ridgeProjectionClock a u b x).continuousOn
    (show τ ∈ Icc (c 0) (c R) from ⟨by
      change ridgeProjectionClock a u b x 0 ≤ τ
      rw [ridgeProjectionClock_zero]
      exact hτ.le, hreach⟩)
  change ridgeProjectionClock a u b x T = τ at hcT
  have hTpos : 0 < T := by
    by_contra hh
    have he : T = 0 := le_antisymm (not_lt.mp hh) hT.1
    rw [he, ridgeProjectionClock_zero] at hcT
    linarith
  refine ⟨T, hTpos, hcT, ?_⟩
  rw [ridgeFlow_eq_projection_clock a u b x T hT.1, hcT]

/-- A single actual original neuron moves a distinct non-antipodal
unit pair exactly, in positive finite time, and fixes the whole
complement of its midpoint cap. Source auxiliary:
arXiv:2411.04551v3, §4, local point matching. -/
theorem exists_exact_gated_transfer (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    ∃ (a : SSphere d) (c T : ℝ), 0 < c ∧ 0 < T ∧
      ridgeFlow (a : EucSpace d) (-((p : EucSpace d) + (q : EucSpace d))) (-c) p T = q ∧
      ∀ z : SSphere d, inner ℝ (z : EucSpace d) ((p : EucSpace d) + (q : EucSpace d)) ≤ c →
        ridgeFlow (a : EucSpace d) (-((p : EucSpace d) + (q : EucSpace d))) (-c) z T = z := by
  obtain ⟨a, τ, hτ, hD, hmap, ham⟩ := exists_exact_projection_transfer p q hpq hanti
  let m : EucSpace d := (p : EucSpace d) + (q : EucSpace d)
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hq : ‖(q : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp q.2
  have hh := norm_sub_sq_real (q : EucSpace d) (p : EucSpace d)
  rw [hp, hq] at hh
  obtain ⟨hr, hr2⟩ := chord_norm_pos_lt_two p q hpq hanti
  have hm : inner ℝ (p : EucSpace d) m = 1 + inner ℝ (q : EucSpace d) (p : EucSpace d) := by
    dsimp only [m]
    rw [inner_add_right, real_inner_self_eq_norm_sq, hp, real_inner_comm]
    ring
  have hheight : 0 < inner ℝ (p : EucSpace d) m := by nlinarith
  let c := inner ℝ (p : EucSpace d) m / 2
  have hc : 0 < c := by dsimp only [c]; positivity
  have hcp : c < inner ℝ (p : EucSpace d) m := by dsimp only [c]; linarith
  obtain ⟨hk, hgate⟩ := projection_transfer_gate_floor a p m ham c hc.le hcp τ hD
  obtain ⟨T, hT, _, hend⟩ := exists_ridgeProjectionClock_eq a (-m) (-c) p τ
    (inner ℝ (p : EucSpace d) m - c) hτ hk hgate
  refine ⟨a, c, T, hc, hT, hend.trans hmap, ?_⟩
  intro z hz
  apply ridgeFlow_inactive
  rw [inner_neg_right]
  linarith

/-- The original weak endpoint is the same exact local transfer under
its actual single-neuron matrices. Source auxiliary:
arXiv:2411.04551v3, §4, `eq: neural.ode.sphere` inside `eq: cauchy.pb`. -/
theorem exists_exact_gated_dirac_controls (i : Fin d) (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    ∃ (a : SSphere d) (c T : ℝ), 0 < c ∧ 0 < T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d p ∧
        cauchyPB d (fun _ => reluRidgeParams i (a : EucSpace d)
          (-((p : EucSpace d) + (q : EucSpace d))) (-c)) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d p →
        cauchyPB d (fun _ => reluRidgeParams i (a : EucSpace d)
          (-((p : EucSpace d) + (q : EucSpace d))) (-c)) μ → μ T = Perspective.diracProb d q := by
  obtain ⟨a, c, T, hc, hT, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  let m : EucSpace d := (p : EucSpace d) + (q : EucSpace d)
  refine ⟨a, c, T, hc, hT, exists_cauchyPB_reluRidgeParams i (a : EucSpace d) (-m) (-c)
    (Perspective.diracProb d p), ?_⟩
  intro μ h0 hμ
  rw [cauchyPB_ridge_dirac_unique i (a : EucSpace d) (-m) (-c) p μ hμ h0]
  change Perspective.diracProb d (ridgeFlow (a : EucSpace d)
    (-((p : EucSpace d) + (q : EucSpace d))) (-c) p T) = _
  rw [hend]

/-- A complete original gated transfer fixes a quantitative
neighborhood of any unit pole orthogonal to its input vector.
Source auxiliary: arXiv:2411.04551v3, §4, protection of the other
points during a local point matching step. -/
theorem ridgeFlow_fixed_near_orthogonal_pole (a m : EucSpace d) (w z : SSphere d)
    (c T : ℝ) (hwm : inner ℝ (w : EucSpace d) m = 0)
    (hz : ‖(z : EucSpace d) - (w : EucSpace d)‖ * ‖m‖ ≤ c) :
    ridgeFlow a (-m) (-c) z T = z := by
  have hi := real_inner_le_norm ((z : EucSpace d) - (w : EucSpace d)) m
  rw [inner_sub_left, hwm, sub_zero] at hi
  apply ridgeFlow_inactive
  rw [inner_neg_right]
  linarith

/-- Distinct orthogonal unit atoms, a positive intrinsic horizon and
a positive constant activation floor meet all local transfer and clock
hypotheses jointly. -/
example : ∃ a p q : SSphere 2, p ≠ q ∧ q ≠ antipode 2 p ∧ (0 : ℝ) < 1 ∧
    ∀ s ∈ Icc (0 : ℝ) 1,
      (1 : ℝ) ≤ max (1 - inner ℝ (sphereProjectionFlow a s p : EucSpace 2) 0) 0 := by
  let p : SSphere 2 := ⟨PiLp.single 2 (0 : Fin 2) (1 : ℝ), by simp⟩
  let q : SSphere 2 := ⟨PiLp.single 2 (1 : Fin 2) (1 : ℝ), by simp⟩
  refine ⟨q, p, q, ?_, ?_, one_pos, fun _ _ => by norm_num⟩
  · intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (0 : Fin 2)) he
    norm_num [p, q] at hh
  · intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (1 : Fin 2)) he
    norm_num [p, q, antipode] at hh

/-- Nonzero orthogonal gate data and a positive neighborhood radius
also realize the quantitative protection hypotheses. -/
example : ∃ (w : SSphere 3) (m : EucSpace 3), ‖m‖ = 1 ∧
    inner ℝ (w : EucSpace 3) m = 0 ∧ ‖(w : EucSpace 3) - (w : EucSpace 3)‖ * ‖m‖ ≤ 1 := by
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  exact ⟨w, PiLp.single 2 (0 : Fin 3) 1, by simp,
    by simp [w, EuclideanSpace.inner_single_left], by simp⟩

end Transformer.Interpolation

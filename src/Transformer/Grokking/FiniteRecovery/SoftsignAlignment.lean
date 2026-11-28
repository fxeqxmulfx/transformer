module

public import Transformer.Grokking.FiniteRecovery.SoftsignMomentum
public import Mathlib.Analysis.SpecificLimits.Basic

/-! A finite-step estimate for ordinary softsign training. Expand the squared
distance of the actual update to one representative. One-point directional
alignment controls the linear term; minibatches can have different gradients.
The residual radius includes the quadratic term from the positive WSD floor. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.SoftsignMomentum

open Filter Transformer.AMSGradW.Kernite
open scoped Topology

variable {ι : Type*} [Fintype ι]

noncomputable def squaredError (u w : ι → ℝ) : ℝ := ∑ i, (w i - u i)^2

/-- Full negative update direction, including decoupled weight decay. -/
noncomputable def force (decay : ℝ) (s : State ι) (g : ι → ℝ) (i : ι) : ℝ :=
  decay*s.weights i + direction (9/10*s.moment i + 1/10*clip g i)

variable (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial u : ι → ℝ)

/-- Exact finite-dimensional energy balance for the native clipped update. -/
theorem squaredError_step (t : ℕ) :
    let s := run decay rate grad initial t
    let h := force decay s (grad t s.weights)
    squaredError u (run decay rate grad initial (t+1)).weights = squaredError u s.weights
      - 2*rate t * ∑ i, (s.weights i - u i)*h i + (rate t)^2 * ∑ i, (h i)^2 := by
  intro s h
  simp only [squaredError, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  change ((1-rate t*decay)*s.weights i - rate t * direction _ - u i)^2 =
    (s.weights i-u i)^2 - 2*rate t*((s.weights i-u i)*force decay s (grad t s.weights) i) +
      (rate t)^2*(force decay s (grad t s.weights) i)^2
  unfold force
  ring

/-- A bound for the quadratic term is available from the update itself for
every gradient stream. A tighter problem-specific bound can replace it. -/
theorem force_squared_bound (hdc : 0 < decay) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1) (t : ℕ) :
    let s := run decay rate grad initial t
    ∑ i, (force decay s (grad t s.weights) i)^2 ≤ (Fintype.card ι : ℝ)*(decay*max ‖initial‖ decay⁻¹ + 1)^2 := by
  intro s
  let K := decay*max ‖initial‖ decay⁻¹+1
  have hK : 0 ≤ K := by dsimp [K]; positivity
  calc
    _ ≤ ∑ _i : ι, K^2 := Finset.sum_le_sum (fun i _ => by
      have h := weight_bound decay rate grad initial hdc hr i t
      have hab : |force decay s (grad t s.weights) i| ≤ K := by
        apply (abs_add_le _ _).trans
        rw [abs_mul, abs_of_pos hdc]
        exact add_le_add (mul_le_mul_of_nonneg_left h hdc.le) (direction_bound _)
      simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hK).2 hab)
    _ = _ := by simp [K]

/-- On a constant-rate part of WSD, alignment with a representative gives an
explicit geometric term plus a residual. This involves only the direction
computed from the training batch, decay and the previous momentum. -/
theorem aligned_error_bound (B : ℕ) {η μ M : ℝ} (hη : 0 < η) (hμ : 0 < μ) (hM : 0 ≤ M)
    (hημ : 2*η*μ ≤ 1) (hr : ∀ k, rate (B+k) = η)
    (halign : ∀ k, let s := run decay rate grad initial (B+k)
      μ * squaredError u s.weights ≤ ∑ i, (s.weights i-u i)*force decay s (grad (B+k) s.weights) i)
    (hforce : ∀ k, let s := run decay rate grad initial (B+k)
      ∑ i, (force decay s (grad (B+k) s.weights) i)^2 ≤ M) (k : ℕ) :
    squaredError u (run decay rate grad initial (B+k)).weights ≤
      (1-2*η*μ)^k * squaredError u (run decay rate grad initial B).weights + η*M/(2*μ) := by
  let e j := squaredError u (run decay rate grad initial (B+j)).weights
  let r := 1-2*η*μ
  let b := η*M/(2*μ)
  have hb : 0 ≤ b := by dsimp [b]; positivity
  have hr0 : 0 ≤ r := sub_nonneg.mpr hημ
  have hfix : r*b+η^2*M=b := by dsimp [r,b]; field_simp; ring
  induction k with
  | zero => simpa using hb
  | succ k ih =>
    have hs := squaredError_step decay rate grad initial u (B+k)
    dsimp only at hs
    rw [hr k] at hs
    have ha := mul_le_mul_of_nonneg_left (halign k) (by positivity : 0 ≤ 2*η)
    have hm := mul_le_mul_of_nonneg_left (hforce k) (sq_nonneg η)
    have hd : e (k+1) ≤ r*e k+η^2*M := by
      change squaredError u (run decay rate grad initial (B+k+1)).weights ≤ _
      rw [hs]
      exact (add_le_add (sub_le_sub_left ha _) hm).trans_eq (by dsimp [e,r]; ring)
    change e (k+1) ≤ r^(k+1)*_+b
    change e k ≤ r^k*_+b at ih
    calc
      _ ≤ r*(r^k*squaredError u (run decay rate grad initial B).weights + b)+η^2*M :=
        hd.trans (add_le_add (mul_le_mul_of_nonneg_left ih hr0) le_rfl)
      _ = _ := by rw [mul_add, add_assoc, hfix, pow_succ']; ring

/-- Every open success set around the representative contains a squared-error
neighborhood, including in the empty-coordinate case. -/
theorem open_squaredError_radius {U : Set (ι → ℝ)} (hU : IsOpen U) {v : ι → ℝ} (hv : v ∈ U) :
    ∃ ρ > (0:ℝ), ∀ w, squaredError v w < ρ → w ∈ U := by
  obtain ⟨δ,hδ,hin⟩ := Metric.isOpen_iff.mp hU v hv
  refine ⟨(δ/2)^2, by positivity, fun w hw => hin (Metric.mem_ball.mpr ?_)⟩
  apply lt_of_le_of_lt _ (show δ/2 < δ by linarith)
  apply (dist_pi_le_iff (by linarith : 0 ≤ δ/2)).2
  intro i
  rw [Real.dist_eq]
  have hi : (w i-v i)^2 ≤ squaredError v w :=
    Finset.single_le_sum (f := fun j => (w j-v j)^2) (fun j _ => sq_nonneg _) (Finset.mem_univ i)
  nlinarith [sq_abs (w i-v i), abs_nonneg (w i-v i)]

/-- If the residual is inside a certified radius, all sufficiently late ordinary
iterates are in the success set. This allows any increasing observation grid. -/
theorem aligned_eventually_mem {U : Set (ι → ℝ)} {ρ η μ M : ℝ} (B : ℕ)
    (hη : 0 < η) (hμ : 0 < μ) (hM : 0 ≤ M) (hημ : 2*η*μ ≤ 1) (hr : ∀ k, rate (B+k) = η)
    (halign : ∀ k, let s := run decay rate grad initial (B+k)
      μ * squaredError u s.weights ≤ ∑ i, (s.weights i-u i)*force decay s (grad (B+k) s.weights) i)
    (hforce : ∀ k, let s := run decay rate grad initial (B+k)
      ∑ i, (force decay s (grad (B+k) s.weights) i)^2 ≤ M)
    (hres : η*M/(2*μ) < ρ) (hball : ∀ w, squaredError u w < ρ → w ∈ U) :
    ∀ᶠ k in atTop, (run decay rate grad initial (B+k)).weights ∈ U := by
  have hlim := (tendsto_pow_atTop_nhds_zero_of_lt_one (sub_nonneg.mpr hημ)
    (by nlinarith [mul_pos hη hμ] : 1-2*η*μ < 1)).mul_const (squaredError u (run decay rate grad initial B).weights)
  have ht := hlim.add_const (η*M/(2*μ))
  simp only [zero_mul, zero_add] at ht
  filter_upwards [ht.eventually (gt_mem_nhds hres)] with k hk
  exact hball _ ((aligned_error_bound decay rate grad initial u B hη hμ hM hημ hr halign hforce k).trans_lt hk)

example (n : ℕ) := force_squared_bound (ι := Fin 2) (1/10) (fun _ => 1/1000) (fun _ w => w)
  (fun _ => 2) (by norm_num) (fun _ => by norm_num) n

example := open_squaredError_radius (ι := Fin 2) (Metric.isOpen_ball (x := 0) (ε := 1))
  (Metric.mem_ball_self (by norm_num))

/-- An explicit aligned trajectory starts outside the target interval.
Here the observed loss is constant, so the decoupled decay supplies all its drift. -/
example : ∀ᶠ k in atTop, ((run (1/10) (fun _ => 1/100) (fun _ _ => 0)
    (fun _ : Fin 1 => 2) k).weights 0)^2 < 1 := by
  let fit := run (1/10) (fun _ => 1/100) (fun _ _ => 0) (fun _ : Fin 1 => 2)
  have hm (k : ℕ) : (fit k).moment = 0 := by
    induction k with
    | zero => rfl
    | succ t ih =>
      funext i
      change 99/100*(fit t).moment i + 1/100*clip 0 i = 0
      simp [ih,clip]
  have hf (k : ℕ) (i : Fin 1) : force (1/10) (fit k) 0 i = (1/10)*(fit k).weights i := by
    simp [force,hm,clip,direction]
  have ha (k : ℕ) : (1/10)*squaredError 0 (fit k).weights ≤
      ∑ i, ((fit k).weights i-(0:Fin 1 → ℝ) i)*force (1/10) (fit k) 0 i := by
    simp only [squaredError, hf, Pi.zero_apply, sub_zero, Finset.mul_sum]
    exact le_of_eq (Finset.sum_congr rfl fun _ _ => by ring)
  have hsq (k : ℕ) : ∑ i, (force (1/10) (fit k) 0 i)^2 ≤ 4 := by
    have h := force_squared_bound (ι := Fin 1) (1/10) (fun _ => 1/100)
      (fun _ _ => 0) (fun _ => 2) (by norm_num) (fun _ => by norm_num) k
    norm_num [Pi.norm_def, Fin.forall_fin_one, fit, Real.norm_eq_abs] at h ⊢
    exact h
  have h := aligned_eventually_mem (ι := Fin 1) (1/10) (fun _ => 1/100) (fun _ _ => 0)
    (fun _ => 2) 0 (U := {w | (w 0)^2 < 1}) (ρ := 1) (η := 1/100) (μ := 1/10) (M := 4) 0
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (fun _ => rfl)
    (fun k => by simpa only [Nat.zero_add, fit] using ha k) (fun k => by simpa only [Nat.zero_add, fit] using hsq k) (by norm_num)
    (fun w hw => by simpa [squaredError] using hw)
  simpa using h

end Transformer.Grokking.FiniteRecovery.SoftsignMomentum

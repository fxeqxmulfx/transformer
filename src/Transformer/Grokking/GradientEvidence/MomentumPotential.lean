module

public import Transformer.Grokking.GradientEvidence.MainAdamMoments
public import Transformer.Grokking.GradientEvidence.CECocoercivity

/-!
# A momentum potential for a dissipative CE component

Source: `review/flat_spectrum.md`, *Loss-specific momentum control*, CL2–CL3.
The .9 EMA coefficient is the main model's. The scalar step is fixed in a
multi-step use. This reference dynamics omits adaptive diagonal differences,
decoupled weight decay, and the tied encoder; CL3 keeps those as perturbations.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset
variable {K : Type*} [Fintype K]

/-- CL2: a positive quadratic potential retaining the saved first moment. -/
def momentumPotential (s : ℝ) (w m : K → ℝ) : ℝ :=
  ∑ k, ((w k-9*s*m k)^2+90*s^2*(m k)^2)

/-- CL2: the potential is nonnegative even for nonzero initial memory. -/
theorem momentum_potential_nonneg (s : ℝ) (w m : K → ℝ) :
    0 ≤ momentumPotential s w m := by
  unfold momentumPotential
  apply sum_nonneg
  intro k hk
  positivity

/-- CL2: exact relation showing the potential controls ordinary parameter energy. -/
theorem momentum_potential_coercive_identity (s : ℝ) (w m : K → ℝ) :
    19*momentumPotential s w m-10*(∑ k, (w k)^2) =
      9*∑ k, (w k-19*s*m k)^2 := by
  unfold momentumPotential
  simp only [mul_sum, ← sum_sub_distrib]
  apply sum_congr rfl
  intro k hk
  ring

/-- CL2: a bounded potential bounds unused contrasts, not only momentum. -/
theorem parameter_energy_le_potential (s : ℝ) (w m : K → ℝ) :
    (∑ k, (w k)^2) ≤ (19/10)*momentumPotential s w m := by
  have h := momentum_potential_coercive_identity s w m
  have hp : 0 ≤ ∑ k, (w k-19*s*m k)^2 := sum_nonneg (fun k hk => sq_nonneg _)
  linarith

/-- CL2: exact one-step identity; the saved-memory discrepancy is dissipative
in this potential, rather than an additional unknown positive error. -/
theorem momentum_potential_step_identity (s : ℝ) (w m g : K → ℝ) :
    momentumPotential s (fun k => w k-s*mainFirst (m k) (g k))
      (fun k => mainFirst (m k) (g k)) - momentumPotential s w m =
      -2*s*(∑ k, w k*g k)+19*s^2*(∑ k, (g k)^2)-
        (171/10)*s^2*∑ k, (m k-g k)^2 := by
  unfold momentumPotential
  simp only [mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  apply sum_congr rfl
  intro k hk
  unfold mainFirst
  ring

/-- CL2: quantitative gradient damping absorbs arbitrary memory at the
explicit step threshold. CL1 proves its gradient premises for actual partial CE. -/
theorem momentum_potential_step_nonincrease (s L : ℝ) (w m g : K → ℝ)
    (hs : 0 ≤ s) (hd : 0 ≤ ∑ k, w k*g k)
    (hc : (∑ k, (g k)^2) ≤ L*(∑ k, w k*g k)) (hstep : 19*s*L ≤ 2) :
    momentumPotential s (fun k => w k-s*mainFirst (m k) (g k))
      (fun k => mainFirst (m k) (g k)) ≤ momentumPotential s w m := by
  have he := momentum_potential_step_identity s w m g
  have h1 := mul_le_mul_of_nonneg_left hc (by positivity : (0 : ℝ) ≤ 19*s^2)
  have h2 := mul_le_mul_of_nonneg_right hstep (mul_nonneg hs hd)
  have h3 : 0 ≤ (171/10 : ℝ)*s^2*∑ k, (m k-g k)^2 := by
    exact mul_nonneg (by positivity) (sum_nonneg (fun k hk => sq_nonneg _))
  nlinarith

/-- CL2: common clipping rescales the cocoercivity constant by the same scalar.
The equality is valid for any scalar; nonnegative clipping supplies damping. -/
theorem scaled_cocoercivity (w g : K → ℝ) (L a : ℝ)
    (hc : (∑ k, (g k)^2) ≤ L*(∑ k, w k*g k)) :
    (∑ k, (a*g k)^2) ≤ (a*L)*(∑ k, w k*(a*g k)) := by
  have h := mul_le_mul_of_nonneg_left hc (sq_nonneg a)
  calc
    _ = a^2*(∑ k, (g k)^2) := by simp only [mul_pow, mul_sum]
    _ ≤ a^2*(L*(∑ k, w k*g k)) := h
    _ = _ := by
      simp only [mul_sum]
      apply sum_congr rfl
      intro k hk
      ring

/-- CL2: fixed-step momentum remains bounded across any sequence of dissipative,
cocoercive gradients. The feature table and minibatch may change at every step. -/
theorem momentum_potential_sequence (w m g : ℕ → K → ℝ) (s L : ℝ)
    (hs : 0 ≤ s) (hstep : 19*s*L ≤ 2)
    (hd : ∀ t, 0 ≤ ∑ k, w t k*g t k)
    (hc : ∀ t, (∑ k, (g t k)^2) ≤ L*(∑ k, w t k*g t k))
    (hm : ∀ t k, m (t+1) k = mainFirst (m t k) (g t k))
    (hw : ∀ t k, w (t+1) k = w t k-s*m (t+1) k) (n : ℕ) :
    momentumPotential s (w n) (m n) ≤ momentumPotential s (w 0) (m 0) := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    have hn := momentum_potential_step_nonincrease s L (w n) (m n) (g n)
      hs (hd n) (hc n) hstep
    have hm' : m (n+1) = fun k => mainFirst (m n k) (g n k) := funext (hm n)
    have hw' : w (n+1) = fun k => w n k-s*mainFirst (m n k) (g n k) := by
      funext k
      rw [hw, hm]
    rw [hm', hw']
    exact hn.trans ih

variable {J : Type*} [Fintype J]

omit [Fintype J] in
/-- CL3: potential perturbations retain the signed linear terms and one
nonnegative quadratic energy, with no loss of cross terms. -/
theorem momentum_potential_perturbation (s : ℝ) (w m dw dm : K → ℝ) :
    momentumPotential s (fun k => w k+dw k) (fun k => m k+dm k) -
      momentumPotential s w m =
      (∑ k, (2*(w k-9*s*m k)*(dw k-9*s*dm k)+180*s^2*m k*dm k)) +
        momentumPotential s dw dm := by
  unfold momentumPotential
  rw [← sum_sub_distrib, ← sum_add_distrib]
  apply sum_congr rfl
  intro k hk
  ring

/-- CL3: each source receives its signed linear credit. The total quadratic
term cannot be replaced by the sum of separate squared source terms. -/
theorem momentum_potential_sources (s : ℝ) (w m : K → ℝ) (dw dm : J → K → ℝ) :
    momentumPotential s (fun k => w k+∑ j, dw j k) (fun k => m k+∑ j, dm j k) -
      momentumPotential s w m =
      (∑ j, ∑ k, (2*(w k-9*s*m k)*(dw j k-9*s*dm j k)+180*s^2*m k*dm j k)) +
        momentumPotential s (fun k => ∑ j, dw j k) (fun k => ∑ j, dm j k) := by
  rw [momentum_potential_perturbation]
  congr 1
  rw [sum_comm]
  apply sum_congr rfl
  intro k hk
  simp only [← mul_sum, sum_sub_distrib, sum_add_distrib]

omit [Fintype J] in
/-- CL3: combine the exact reference and full-update changes. Any attempt
to extend the reference certificate must control the remaining signed term. -/
theorem momentum_full_step_identity (s : ℝ) (w m g dw dm : K → ℝ) :
    let mr := fun k => mainFirst (m k) (g k)
    let wr := fun k => w k-s*mr k
    momentumPotential s (fun k => wr k+dw k) (fun k => mr k+dm k) -
      momentumPotential s w m =
      -2*s*(∑ k, w k*g k)+19*s^2*(∑ k, (g k)^2)-
        (171/10)*s^2*(∑ k, (m k-g k)^2) +
      (∑ k, (2*(wr k-9*s*mr k)*(dw k-9*s*dm k)+180*s^2*mr k*dm k)) +
        momentumPotential s dw dm := by
  dsimp only
  have hp := momentum_potential_perturbation s
    (fun k => w k-s*mainFirst (m k) (g k)) (fun k => mainFirst (m k) (g k)) dw dm
  have hr := momentum_potential_step_identity s w m g
  linarith

example := momentum_potential_step_nonincrease (K := Bool) (1/100) 1
  (fun k => if k then 2 else -1) (fun k => if k then -3 else 4)
  (fun k => if k then 2 else -1)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := scaled_cocoercivity (K := Bool)
  (fun k => if k then 2 else -1) (fun k => if k then 2 else -1) 1 (1/2) (by norm_num)
example (n : ℕ) := momentum_potential_sequence (K := Fin 1)
  (fun _ _ => 2) (fun _ _ => 0) (fun _ _ => 0) (1/100) 1
  (by norm_num) (by norm_num) (by intro t; norm_num) (by intro t; norm_num)
  (by intro t k; norm_num [mainFirst]) (by intro t k; norm_num) n

end Transformer.Grokking.GradientEvidence

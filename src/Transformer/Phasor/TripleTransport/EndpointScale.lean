module

public import Transformer.Phasor.TripleTransport.PathCoefficients
public import Mathlib.Analysis.SpecialFunctions.Sqrt

/-! Scale a complete dense endpoint vote v by sqrt(d)/sqrt(1+sum_i v_i²).
This candidate for the Phasor lookup residual uses only hidden geometry.
The same residual scale acts on every tied vocabulary comparison. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.EndpointScale
open scoped BigOperators

def energy {d : ℕ} (v : Fin d → ℝ) := ∑ i, (v i)^2
noncomputable def scale (d : ℕ) (q : ℝ) := Real.sqrt d / Real.sqrt (1+q)
noncomputable def unitVote {d : ℕ} (v : Fin d → ℝ) (i : Fin d) := scale d (energy v) * v i

/-- The regularized vote has squared norm d*q/(1+q), strictly below d
for nonempty hidden width. Both a zero vote and very large votes are defined. -/
theorem bounded {d : ℕ} (v : Fin d → ℝ) (width : 0<d) :
    energy (unitVote v) = (d:ℝ)*energy v/(1+energy v) ∧ energy (unitVote v)<d := by
  have nonneg : 0≤energy v := Finset.sum_nonneg (fun i _ => sq_nonneg (v i))
  have den : 0<1+energy v := by positivity
  have dim : 0<(d:ℝ) := by exact_mod_cast width
  have h : energy (unitVote v) = scale d (energy v)^2 * energy v := by
    unfold unitVote energy
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    rw [mul_pow]
  rw [h]
  have eqn : scale d (energy v)^2 * energy v = (d:ℝ)*energy v/(1+energy v) := by
    simp only [scale,div_pow, Real.sq_sqrt dim.le, Real.sq_sqrt den.le]
    ring
  rw [eqn]
  exact ⟨rfl, (div_lt_iff₀ den).2 (by nlinarith)⟩

/-- If the isolated lookup favors a class by a positive margin, the complete
model favors that class precisely above the competing backbone threshold. -/
theorem complete_margin (base vote gain : ℝ) (wins : 0 < vote) :
    0 < base+gain*vote ↔ -base/vote<gain := by
  rw [div_lt_iff₀ wins]
  constructor <;> intro h <;> linarith

/-- The regularized normalization retains a nonzero learning signal in every
nonzero coordinate direction at the all-zero vote. The denominator is even
along the direction with squared norm q. -/
theorem initial_derivative (d : ℕ) (q w : ℝ) :
    HasDerivAt (fun t : ℝ => scale d (t^2*q)*(t*w)) (Real.sqrt d*w) 0 := by
  have tt := (((hasDerivAt_id (0:ℝ)).fun_pow 2).mul_const q).const_add 1
  have deriv : HasDerivAt (fun t : ℝ => Real.sqrt (1+t^2*q)) 0 0 := by
    simpa using tt.sqrt (by norm_num)
  have factor := (hasDerivAt_const (0:ℝ) (Real.sqrt d)).fun_div deriv (by norm_num)
  simpa [scale] using factor.fun_mul ((hasDerivAt_id (0:ℝ)).mul_const w)

example : energy (unitVote (fun _ : Fin 3 => (2:ℝ))) = (3:ℝ)*energy (fun _ : Fin 3 => (2:ℝ))/
    (1+energy (fun _ : Fin 3 => (2:ℝ))) ∧ energy (unitVote (fun _ : Fin 3 => (2:ℝ))) < 3 :=
  bounded _ (by decide)
example : 0 < (-3:ℝ)+2*2 ↔ -(-3:ℝ)/2<2 := complete_margin _ _ _ (by norm_num)
end Transformer.Phasor.TripleTransport.EndpointScale

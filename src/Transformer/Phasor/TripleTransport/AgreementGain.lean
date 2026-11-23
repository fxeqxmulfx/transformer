module
public import Transformer.Phasor.TripleTransport.PowerGain

/-! A shared coefficient read from two coordinates of the learned path
table. Each coordinate uses a different part of the hash of the same
dense address tuple. Their product discounts a single accidental match. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.AgreementGain

noncomputable def gain (n : ℕ) (m b a c : ℝ) : ℝ :=
  ((a+c)/2) * ((a*c+b)/(m^2+b))^n

/-- Turning off concentration retains the signed mean of the two
coordinates in the frozen-weight ablation. -/
theorem unscaled (m b a c : ℝ) : gain 0 m b a c = (a+c)/2 := by simp [gain]

/-- Equal signed coefficients recover the existing power gain, including
for a negative coefficient and at a tied table maximum. -/
theorem agrees (n : ℕ) (m b a : ℝ) :
    gain n m b a a = PowerGain.gain n m b a := by
  simp only [gain, PowerGain.gain, sq]
  ring

/-- A different tuple sharing only the first live coordinate contributes
exactly this residual, independent of the sign of that coordinate. -/
theorem single_match (n : ℕ) (m b a : ℝ) :
    gain n m b a 0 = (a/2)*(b/(m^2+b))^n := by simp [gain]

/-- If the learned maximum reaches the square-root floor, an unmatched
coordinate is discounted by at least a factor two per power. -/
theorem single_bound (n : ℕ) (m b a : ℝ) (floor : 0<b) (size : b≤m^2) :
    |gain n m b a 0| ≤ (|a|/2)*(1/2 : ℝ)^n := by
  have d : 0<m^2+b := by positivity
  have lo : 0≤b/(m^2+b) := by positivity
  have q : b/(m^2+b)≤1/2 := (div_le_iff₀ d).2 (by linarith)
  rw [single_match, abs_mul, abs_div, abs_pow, abs_of_nonneg lo, abs_of_pos (by norm_num : 0<(2:ℝ))]
  exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ lo q n) (by positivity)

/-- The ordinary loss can train either coordinate from an all-zero table.
Along every common weight direction the first derivative is their mean. -/
theorem initial_derivative (n : ℕ) (m b a c : ℝ) (floor : 0<b) :
    HasDerivAt (fun t : ℝ => gain n (t*m) b (t*a) (t*c)) ((a+c)/2) 0 := by
  have linear (v : ℝ) : HasDerivAt (fun t : ℝ => t*v) v 0 := by
    simpa using (hasDerivAt_id (0:ℝ)).mul_const v
  have sum : HasDerivAt (fun t : ℝ => (t*a+t*c)/2) ((a+c)/2) 0 := by
    simpa using ((linear a).fun_add (linear c)).div_const 2
  have numerator : HasDerivAt (fun t : ℝ => (t*a)*(t*c)+b) 0 0 := by
    simpa using ((linear a).fun_mul (linear c)).add_const b
  have denominator : HasDerivAt (fun t : ℝ => (t*m)^2+b) 0 0 := by
    simpa using ((linear m).fun_pow 2).add_const b
  have nz : (0*m)^2+b≠0 := by simpa using floor.ne'
  simpa [gain,floor.ne'] using sum.fun_mul ((numerator.fun_div denominator nz).fun_pow n)

example : |gain 4 2 1 (-1) 0| ≤ (|(-1:ℝ)|/2)*(1/2:ℝ)^4 :=
  single_bound _ _ _ _ (by norm_num) (by norm_num)
example : HasDerivAt (fun t : ℝ => gain 4 (t*2) 1 (t*(-2)) (t*4)) ((-2+4)/2) 0 :=
  initial_derivative _ _ _ _ _ (by norm_num)
end Transformer.Phasor.TripleTransport.AgreementGain

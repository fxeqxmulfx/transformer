module

public import Transformer.Phasor.TripleTransport.EndpointScale
public import Mathlib.Analysis.SpecificLimits.Basic

/-! General concentration of a learned coefficient table, suggested by the
frozen ablation in the Phasor depth study. A common exponent acts on all
signed coefficients on every task, including at the all-zero initialization.
-/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.PowerGain

noncomputable def gain (n : ℕ) (m b w : ℝ) : ℝ := w * ((w^2+b)/(m^2+b))^n

theorem keeps_maximum (n : ℕ) (m b w : ℝ) (floor : 0<b) (leader : |w|=m) :
    gain n m b w = w := by
  have hw : w^2=m^2 := by rw [←leader,sq_abs]
  have pos : 0<m^2+b := by positivity
  simp [gain,hw,pos.ne']

/-- Stronger concentration strictly shrinks every live submaximal
coefficient, even if that coefficient has a negative sign. -/
theorem shrinks (n : ℕ) (m b w : ℝ) (floor : 0<b) (live : w≠0) (bound : |w|<m) :
    |gain (n+1) m b w| < |gain n m b w| := by
  have den : 0<m^2+b := by positivity
  have num : 0<w^2+b := by positivity
  have square : w^2<m^2 := by
    nlinarith [sq_abs w, mul_pos (sub_pos.mpr bound) (show 0<m+|w| by linarith [abs_nonneg w])]
  have ratio : (w^2+b)/(m^2+b)<1 := (div_lt_one den).2 (by linarith)
  have pos := div_pos num den
  rw [gain,gain,pow_succ,←mul_assoc,abs_mul,abs_of_pos pos]
  exact mul_lt_of_lt_one_right (abs_pos.mpr (mul_ne_zero live (pow_ne_zero _ pos.ne'))) ratio

/-- The response to any joint table direction at initialization is the
original direction for every finite exponent. -/
theorem initial_derivative (n : ℕ) (m b w : ℝ) (floor : 0<b) :
    HasDerivAt (fun t : ℝ => gain n (t*m) b (t*w)) w 0 := by
  have linear (v : ℝ) : HasDerivAt (fun t : ℝ => t*v) v 0 := by
    simpa using (hasDerivAt_id (0:ℝ)).mul_const v
  have quad (v : ℝ) : HasDerivAt (fun t : ℝ => (t*v)^2+b) 0 0 := by
    simpa using ((linear v).fun_pow 2).add_const b
  have den : (0*m)^2+b≠0 := by simpa using floor.ne'
  simpa [gain,floor.ne'] using (linear w).fun_mul (((quad w).fun_div (quad m) den).fun_pow n)

example : gain 8 3 1 (-3) = -3 := keeps_maximum _ _ _ _ (by norm_num) (by norm_num)
example : |gain 8 3 1 (-2)| < |gain 7 3 1 (-2)| :=
  shrinks 7 3 1 (-2) (by norm_num) (by norm_num) (by norm_num)
example : HasDerivAt (fun t : ℝ => gain 7 (t*3) 2 (t*(-1))) (-1) 0 :=
  initial_derivative _ _ _ _ (by norm_num)

end Transformer.Phasor.TripleTransport.PowerGain

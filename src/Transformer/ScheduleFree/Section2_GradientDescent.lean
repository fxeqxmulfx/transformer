module

public import Transformer.ScheduleFree.Section2_Averaging
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The base optimizer's regret

arXiv:2405.15682v4, §2.1, the OGD argument following Theorem 2.
These identities also supply Appendix F's large-step analysis. The
gradients can be arbitrary vectors: they need not be evaluated at `z`.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- §2: linearized regret of the base sequence against comparator `u`. -/
def regret (z g : ℕ → E) (u : E) (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, inner ℝ (g t) (z t - u)

/-- §2: unprojected constant-step base updates. -/
def baseRun (η : ℝ) (g : ℕ → E) (initial : E) : ℕ → E
  | 0 => initial
  | t + 1 => baseRun η g initial t - η • g t

/-- §2.1 / Appendix F, eq. (15): exact OGD energy accounting for one step. -/
theorem descent_energy (η : ℝ) (z g u : E) :
    ‖z - η • g - u‖ ^ 2 =
      ‖z - u‖ ^ 2 - 2 * η * inner ℝ g (z - u) + η ^ 2 * ‖g‖ ^ 2 := by
  rw [show z - η • g - u = (z - u) - η • g by module,
    norm_sub_sq_real, real_inner_smul_right, real_inner_comm, norm_smul,
    Real.norm_eq_abs, mul_pow, sq_abs]
  ring

/-- §2.1: telescope the actual base updates, retaining the final distance. -/
theorem regret_energy (η : ℝ) (z g : ℕ → E) (u : E) (T : ℕ)
    (hstep : ∀ t, z (t + 1) = z t - η • g t) :
    2 * η * regret z g u T = ‖z 0 - u‖ ^ 2 - ‖z T - u‖ ^ 2 +
      η ^ 2 * ∑ t ∈ Finset.range T, ‖g t‖ ^ 2 := by
  induction T with
  | zero => simp [regret]
  | succ T ih =>
    have h := descent_energy η (z T) (g T) u
    rw [← hstep T] at h
    simp only [regret, Finset.sum_range_succ] at ih ⊢
    nlinarith

example (η : ℝ) (g : ℕ → E) (initial u : E) (T : ℕ) :
    2 * η * regret (baseRun η g initial) g u T =
      ‖baseRun η g initial 0 - u‖ ^ 2 - ‖baseRun η g initial T - u‖ ^ 2 +
        η ^ 2 * ∑ t ∈ Finset.range T, ‖g t‖ ^ 2 :=
  regret_energy η _ g u T (fun _ => rfl)

/-- The OGD bound used after Theorem 2; no constraint on the gradient locations. -/
theorem regret_bound {η : ℝ} (hη : 0 < η) (z g : ℕ → E) (u : E) (T : ℕ)
    (hstep : ∀ t, z (t + 1) = z t - η • g t) :
    regret z g u T ≤
      (‖z 0 - u‖ ^ 2 + η ^ 2 * ∑ t ∈ Finset.range T, ‖g t‖ ^ 2) / (2 * η) := by
  apply (le_div_iff₀ (by positivity : 0 < 2 * η)).mpr
  have h := regret_energy η z g u T hstep
  nlinarith [sq_nonneg ‖z T - u‖]

example (g : ℕ → E) (initial u : E) (T : ℕ) :
    regret (baseRun 1 g initial) g u T ≤
      (‖baseRun 1 g initial 0 - u‖ ^ 2 + 1 ^ 2 *
        ∑ t ∈ Finset.range T, ‖g t‖ ^ 2) / (2 * 1) :=
  regret_bound zero_lt_one _ g u T (fun _ => rfl)

/-- §2.1: a uniform gradient bound gives the usual `D²/(2η)+ηTG²/2` regret. -/
theorem bounded_regret {η G : ℝ} (hη : 0 < η) (hG : 0 ≤ G)
    (z g : ℕ → E) (u : E) (T : ℕ)
    (hstep : ∀ t, z (t + 1) = z t - η • g t)
    (hbound : ∀ t < T, ‖g t‖ ≤ G) :
    regret z g u T ≤ (‖z 0 - u‖ ^ 2 + η ^ 2 * (T : ℝ) * G ^ 2) / (2 * η) := by
  have hs : (∑ t ∈ Finset.range T, ‖g t‖ ^ 2) ≤ (T : ℝ) * G ^ 2 := by
    calc
      _ ≤ ∑ _t ∈ Finset.range T, G ^ 2 := by
        apply Finset.sum_le_sum
        intro t ht
        exact (sq_le_sq₀ (norm_nonneg _) hG).mpr (hbound t (Finset.mem_range.mp ht))
      _ = _ := by simp
  apply (regret_bound hη z g u T hstep).trans
  apply (div_le_div_iff_of_pos_right (by positivity : 0 < 2 * η)).mpr
  nlinarith [mul_le_mul_of_nonneg_left hs (sq_nonneg η)]

example (initial u : E) (T : ℕ) :
    regret (baseRun 1 (fun _ => 0) initial) (fun _ => 0) u T ≤
      (‖baseRun 1 (fun _ => 0) initial 0 - u‖ ^ 2 + 1 ^ 2 * (T : ℝ) * 0 ^ 2) /
        (2 * 1) :=
  bounded_regret zero_lt_one le_rfl _ _ u T (fun _ => rfl) (by intro t ht; simp)

/-- Theorem 1's optimal step, with explicit nonzero-distance/oracle cases.
The `D=0` case has a stationary zero-step run and is handled separately. -/
theorem optimal_regret {D G : ℝ} (hD : 0 < D) (hG : 0 < G) {T : ℕ} (hT : 0 < T)
    (z g : ℕ → E) (u : E) (hinit : ‖z 0 - u‖ = D)
    (hstep : ∀ t, z (t + 1) = z t - (D / (G * Real.sqrt T)) • g t)
    (hbound : ∀ t < T, ‖g t‖ ≤ G) :
    regret z g u T ≤ D * G * Real.sqrt T := by
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  have hs : 0 < Real.sqrt T := Real.sqrt_pos.mpr ht
  have h := bounded_regret (div_pos hD (mul_pos hG hs)) hG.le z g u T hstep hbound
  rw [hinit] at h
  convert h using 1
  field_simp
  nlinarith [Real.sq_sqrt ht.le]

example : regret (baseRun 1 (fun _ => (0 : ℝ)) 1) (fun _ => 0) 0 1 ≤
    1 * 1 * Real.sqrt 1 := by
  simpa using optimal_regret (E := ℝ) zero_lt_one zero_lt_one (T := 1) (by decide)
    (baseRun 1 (fun _ => 0) 1) (fun _ => 0) 0 (by simp [baseRun])
    (by intro t; simp [baseRun]) (by intro t ht; simp)

/-- Theorem 1's degenerate cases: zero distance or zero observed gradients.
Real division gives a zero step; both cases have zero linearized regret. -/
theorem optimal_regret_nonneg {D G : ℝ} (hD : 0 ≤ D) (hG : 0 ≤ G) {T : ℕ} (hT : 0 < T)
    (z g : ℕ → E) (u : E) (hinit : ‖z 0 - u‖ = D)
    (hstep : ∀ t, z (t + 1) = z t - (D / (G * Real.sqrt T)) • g t)
    (hbound : ∀ t < T, ‖g t‖ ≤ G) : regret z g u T ≤ D * G * Real.sqrt T := by
  by_cases hd : D = 0
  · have hz0 : z 0 = u := sub_eq_zero.mp (norm_eq_zero.mp (hinit.trans hd))
    have hz (t : ℕ) : z t = u := by
      induction t with
      | zero => exact hz0
      | succ t ih => simpa [hd, ih] using hstep t
    simp [regret, hz, hd]
  by_cases hg : G = 0
  · have hz (t : ℕ) (ht : t ∈ Finset.range T) : g t = 0 :=
      norm_eq_zero.mp (le_antisymm (by simpa [hg] using hbound t (Finset.mem_range.mp ht))
        (norm_nonneg _))
    have hr : regret z g u T = 0 :=
      Finset.sum_eq_zero (fun t ht => by rw [hz t ht, inner_zero_left])
    simp [hr, hg]
  exact optimal_regret (lt_of_le_of_ne hD (Ne.symm hd)) (lt_of_le_of_ne hG (Ne.symm hg))
    hT z g u hinit hstep hbound

example (T : ℕ) (hT : 0 < T) : regret (fun _ => (0 : ℝ)) (fun _ => 0) 0 T ≤
    (0 : ℝ) * 0 * Real.sqrt T :=
  optimal_regret_nonneg le_rfl le_rfl hT _ _ 0 (by simp) (by intro t; simp) (by intro t ht; simp)

end Transformer.ScheduleFree

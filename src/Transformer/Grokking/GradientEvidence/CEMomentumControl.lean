module

public import Transformer.Grokking.GradientEvidence.MomentumPotential

/-!
# Apply the momentum potential to the actual unused-class CE derivative

Source: `review/flat_spectrum.md`, *Loss-specific momentum control*, CL1–CL3.
The derivative and class means are computed. There is no hypothesis that
saved momentum agrees with the present gradient. The partial dynamics can
use changing features and minibatches; full AdamW perturbations remain CL3.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes
variable {I C K U : Type*} [Fintype I] [Fintype C] [Fintype K]
  [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K]

/-- CL1: flatten the computed centered class block, without an assumed mean. -/
def unusedCoordinates (E : C → K → ℝ) (classes : U → C) (j : U × K) : ℝ :=
  E (classes j.1) j.2-classMean (fun u => E (classes u) j.2)

/-- CL1: the actual partial gradient is dissipative in flattened coordinates. -/
theorem unused_gradient_radial_nonneg (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (hu : ∀ u i, classes u ≠ y i) :
    0 ≤ ∑ j, unusedCoordinates E classes j *
      unusedCoordinates (readoutGradient T y E) classes j := by
  simpa only [Fintype.sum_prod_type, unusedCoordinates] using
    unused_group_gradient_damping T y E classes hu

/-- CL1: the computed block satisfies the quantitative coefficient used by CL2. -/
theorem unused_gradient_energy_bound (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (q : ℝ)
    (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q) :
    (∑ j, (unusedCoordinates (readoutGradient T y E) classes j)^2) ≤
      (q*(matrixEnergy T/Fintype.card I)) *
        ∑ j, unusedCoordinates E classes j * unusedCoordinates (readoutGradient T y E) classes j := by
  simpa only [Fintype.sum_prod_type, unusedCoordinates] using
    unused_group_cocoercive T y E classes q hu hq

/-- CL2: actual partial CE, common clipping, and arbitrary saved memory obey
the potential bound under the explicit step condition. It is not a full tied update. -/
theorem unused_ce_momentum_nonincrease (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (m : U × K → ℝ) (s q clip : ℝ)
    (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q)
    (hs : 0 ≤ s) (hclip : 0 ≤ clip)
    (hstep : 19*s*(clip*(q*(matrixEnergy T/Fintype.card I))) ≤ 2) :
    let w := unusedCoordinates E classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T y E) classes j
    momentumPotential s (fun j => w j-s*mainFirst (m j) (g j))
      (fun j => mainFirst (m j) (g j)) ≤ momentumPotential s w m := by
  dsimp only
  apply momentum_potential_step_nonincrease s
    (clip*(q*(matrixEnergy T/Fintype.card I))) _ m _ hs _ _ hstep
  · have hd := mul_nonneg hclip (unused_gradient_radial_nonneg T y E classes hu)
    convert hd using 1
    rw [mul_sum]
    apply sum_congr rfl
    intro j hj
    ring
  · exact scaled_cocoercivity _ _ _ clip
      (unused_gradient_energy_bound T y E classes q hu hq)

/-- CL2: a whole fixed-step partial-CE trajectory has a bounded potential.
Only the declared recurrence and a per-step probability/feature coefficient
are assumed; neither favorable saved memory nor a fixed feature table is needed. -/
theorem unused_ce_momentum_sequence (T : ℕ → I → K → ℝ) (y : ℕ → I → C)
    (E : ℕ → C → K → ℝ) (classes : U → C) (m : ℕ → U × K → ℝ)
    (s q : ℝ) (clip : ℕ → ℝ)
    (hu : ∀ u t i, classes u ≠ y t i)
    (hq : ∀ u t i, probability (fun v => linearScore (E t v) (T t i)) (classes u) ≤ q)
    (hs : 0 ≤ s) (hclip : ∀ t, 0 ≤ clip t)
    (hstep : ∀ t, 19*s*(clip t*(q*(matrixEnergy (T t)/Fintype.card I))) ≤ 2)
    (hm : ∀ t j, m (t+1) j = mainFirst (m t j)
      (clip t*unusedCoordinates (readoutGradient (T t) (y t) (E t)) classes j))
    (hw : ∀ t j, unusedCoordinates (E (t+1)) classes j =
      unusedCoordinates (E t) classes j-s*m (t+1) j) (n : ℕ) :
    momentumPotential s (unusedCoordinates (E n) classes) (m n) ≤
      momentumPotential s (unusedCoordinates (E 0) classes) (m 0) := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    have hn := unused_ce_momentum_nonincrease (T n) (y n) (E n) classes (m n) s q (clip n)
      (fun u => hu u n) (fun u => hq u n) hs (hclip n) (hstep n)
    have hm' : m (n+1) = fun j => mainFirst (m n j)
        (clip n*unusedCoordinates (readoutGradient (T n) (y n) (E n)) classes j) := funext (hm n)
    have hw' : unusedCoordinates (E (n+1)) classes = fun j =>
        unusedCoordinates (E n) classes j-s*mainFirst (m n j)
          (clip n*unusedCoordinates (readoutGradient (T n) (y n) (E n)) classes j) := by
      funext j
      rw [hw, hm]
    rw [hw', hm']
    exact hn.trans ih

/-- CL3: the moment discrepancy of the partial-CE reference is exactly the
current encoder input plus execution residual; earlier memory is retained. -/
theorem main_moment_reference_error (m gR gE em : ℝ) :
    mainFirst m (gR+gE)+em-mainFirst m gR = (1/10)*gE+em := by
  unfold mainFirst
  ring

/-- CL3: source AdamW position error around the scalar partial-CE reference.
The second moment and both absolute-time bias corrections remain actual inputs. -/
theorem main_adam_reference_error (w m v gR gE em ep eta s rho eps : ℝ) (t : ℕ) :
    let mn := mainFirst m (gR+gE)+em
    bufferedAdamUpdate (9/10) (19/20) eps eta rho (t+1) w mn v+ep-
      (w-s*mainFirst m gR) =
      -eta*rho*w +
      (s-eta*adamDiagonal (19/20) eps (t+1) v/(1-(9/10 : ℝ)^(t+1)))*mn -
      s*(1/10)*gE + (ep-s*em) := by
  dsimp only
  rw [bufferedAdamUpdate_factor]
  unfold correctedMoment mainFirst
  ring

example := unused_gradient_radial_nonneg (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (by intro u i; exact Fin.succ_ne_zero u)
example := unused_gradient_energy_bound (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ) 1
  (by intro u i; exact Fin.succ_ne_zero u) (by intro u i; exact probability_le_one _ _)
example := unused_ce_momentum_nonincrease (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun j => if j.1.val = 0 then -1 else 2) (1/100) 1 (1/2)
  (by intro u i; exact Fin.succ_ne_zero u) (by intro u i; exact probability_le_one _ _)
  (by norm_num) (by norm_num) (by norm_num [matrixEnergy])
example (n : ℕ) := unused_ce_momentum_sequence (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ _ => 0) (fun _ _ => 0) (fun _ c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun _ _ => 0) (1/100) 1 (fun _ => 1/2)
  (by intro u t i; exact Fin.succ_ne_zero u) (by intro u t i; exact probability_le_one _ _)
  (by norm_num) (by intro t; norm_num) (by intro t; norm_num [matrixEnergy])
  (by intro t j; simp [unusedCoordinates, readoutGradient_eq, classMean, mainFirst])
  (by intro t j; simp) n

/-- CL3: positive perturbation energy alone has no sign for the full change;
this example has a strictly decreasing reference and increasing full potential. -/
theorem momentum_perturbation_can_reverse :
    let w : Fin 1 → ℝ := fun _ => 1
    let m : Fin 1 → ℝ := fun _ => 0
    let g : Fin 1 → ℝ := fun _ => 1
    let mr := fun k => mainFirst (m k) (g k)
    let wr := fun k => w k-(1/100)*mr k
    momentumPotential (1/100) wr mr < momentumPotential (1/100) w m ∧
      momentumPotential (1/100) (fun k => wr k+1/10) mr >
        momentumPotential (1/100) w m := by
  norm_num [momentumPotential, mainFirst]

example := main_adam_reference_error 2 (1/10) (1/100) (1/5) (-1/20)
  (1/1000000) (-1/1000000) (1/10000) (1/100) (1/10) (1/100000000) 4000
example := momentum_potential_sources (K := Fin 1) (J := Bool) (1/100)
  (fun _ => 2) (fun _ => -1)
  (fun j _ => if j then 1/100 else -1/200)
  (fun j _ => if j then -1/20 else 1/50)

end Transformer.Grokking.GradientEvidence

module

public import Transformer.Grokking.GradientEvidence.RecipeDecay

/-!
# The main model's AdamW depends on rate and decay through their product

Source: `review/flat_spectrum.md`, *The rate of a recipe* (added), items RR1 and RR3.

`mainAdamRun G ρ eps η p0` is the main model's AdamW on all coordinates at once, with the
gradient field `G`: betas `0.9` and `0.95` and the coordinate update `bufferedAdamUpdate`.
Every coordinate follows `mainAdamPath` with the gradients it receives (`mainAdamRun_fst`), so
this is the recurrence of AD3 in closed loop.

* A coordinate whose gradient is zero at every point decays by the product of the schedule
  (`mainAdamRun_of_zero`).
* For `c > 0` and a field homogeneous of degree `-1` at the scale `c`, the run with the rates
  `c * η`, the decay `ρ / c` and the epsilon `eps / c` from `c • p0` is the run with `η`, `ρ`,
  `eps` from `p0` at the scale `c`: weights times `c`, first moments over `c`, second moments
  over `c ^ 2` (`mainAdamRun_atScale`).

The clip of the gradient norm of the recipe is not in this model: it has a scale of its own.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset

variable {ι : Type*}

/-- One update of every coordinate: weights, first moments, second moments. -/
def mainAdamStep (G : (ι → ℝ) → ι → ℝ) (ρ eps η : ℝ) (t : ℕ)
    (s : (ι → ℝ) × (ι → ℝ) × (ι → ℝ)) : (ι → ℝ) × (ι → ℝ) × (ι → ℝ) :=
  (fun i => bufferedAdamUpdate (9 / 10) (19 / 20) eps η ρ (t + 1) (s.1 i)
      (mainFirst (s.2.1 i) (G s.1 i)) (mainSecond (s.2.2 i) (G s.1 i)),
    fun i => mainFirst (s.2.1 i) (G s.1 i), fun i => mainSecond (s.2.2 i) (G s.1 i))

/-- The main model's AdamW from `p0` with zero moments, the gradient taken at the weights. -/
def mainAdamRun (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ) :
    ℕ → (ι → ℝ) × (ι → ℝ) × (ι → ℝ)
  | 0 => (p0, fun _ => 0, fun _ => 0)
  | t + 1 => mainAdamStep G ρ eps (η t) t (mainAdamRun G ρ eps η p0 t)

/-- The gradients that a coordinate receives along the run. -/
def mainAdamGradients (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ) (i : ι) :
    ℕ → ℝ :=
  fun t => G (mainAdamRun G ρ eps η p0 t).1 i

/-- The moments of a coordinate of the run are the moments of AD1 of its gradients. -/
theorem mainAdamRun_moments (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ)
    (i : ι) (t : ℕ) :
    ((mainAdamRun G ρ eps η p0 t).2.1 i, (mainAdamRun G ρ eps η p0 t).2.2 i) =
      mainMoments (mainAdamGradients G ρ eps η p0 i) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [mainMoments, ← ih]
    rfl

/-- **The run is the path of AD3 in closed loop**: a coordinate follows `mainAdamPath` with
the gradients it receives. -/
theorem mainAdamRun_fst (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ)
    (i : ι) (t : ℕ) :
    (mainAdamRun G ρ eps η p0 t).1 i =
      mainAdamPath (p0 i) ρ eps η (mainAdamGradients G ρ eps η p0 i) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [mainAdamPath_source, ← ih, ← mainAdamRun_moments]
    rfl

/-- **RR1 in closed loop**, `review/flat_spectrum.md`: a coordinate whose gradient is zero at
every point is its initial value times the product of the decay factors. -/
theorem mainAdamRun_of_zero (G : (ι → ℝ) → ι → ℝ) (i : ι) (hi : ∀ w, G w i = 0) (ρ eps : ℝ)
    (η : ℕ → ℝ) (p0 : ι → ℝ) (t : ℕ) :
    (mainAdamRun G ρ eps η p0 t).1 i = (∏ s ∈ range t, (1 - η s * ρ)) * p0 i := by
  have h : mainAdamGradients G ρ eps η p0 i = fun _ => 0 := funext fun _ => hi _
  rw [mainAdamRun_fst, h, mainAdamPath_zero]

/-- The second of two coordinates has no gradient; the first has the gradient `1 / w`. -/
example (t : ℕ) :=
  mainAdamRun_of_zero (fun w (i : Fin 2) => if i = 0 then 1 / w 0 else 0) 1 (fun _ => by simp)
    (1 / 10) (1 / 100000000) (fun _ => 1 / 1000) (fun _ => 1) t

/-- A state at the scale `c`: weights times `c`, first moments over `c`, second moments over
`c ^ 2`. -/
def atScale (c : ℝ) (s : (ι → ℝ) × (ι → ℝ) × (ι → ℝ)) : (ι → ℝ) × (ι → ℝ) × (ι → ℝ) :=
  (fun i => c * s.1 i, fun i => s.2.1 i / c, fun i => s.2.2 i / c ^ 2)

/-- **One update at the scale `c`**, `review/flat_spectrum.md`, item RR3: for a gradient field
homogeneous of degree `-1` at the scale `c`, the update with the rate `c * η`, the decay
`ρ / c` and the epsilon `eps / c` of a state at the scale `c` is the update at that scale. -/
theorem mainAdamStep_atScale {c : ℝ} (hc : 0 < c) (G : (ι → ℝ) → ι → ℝ)
    (hG : ∀ w i, G (fun j => c * w j) i = G w i / c) (ρ eps η : ℝ) (t : ℕ)
    (s : (ι → ℝ) × (ι → ℝ) × (ι → ℝ)) :
    mainAdamStep G (ρ / c) (eps / c) (c * η) t (atScale c s) =
      atScale c (mainAdamStep G ρ eps η t s) := by
  have hc0 : c ≠ 0 := hc.ne'
  have hm : ∀ i, mainFirst (s.2.1 i / c) (G s.1 i / c) = mainFirst (s.2.1 i) (G s.1 i) / c :=
    fun i => by simp only [mainFirst]; ring
  have hv : ∀ i, mainSecond (s.2.2 i / c ^ 2) (G s.1 i / c) =
      mainSecond (s.2.2 i) (G s.1 i) / c ^ 2 :=
    fun i => by simp only [mainSecond]; field_simp
  simp only [mainAdamStep, atScale, hG, hm, hv]
  refine Prod.ext (funext fun i => ?_) rfl
  simp only [bufferedAdamUpdate]
  rw [Real.sqrt_div' _ (sq_nonneg c), Real.sqrt_sq hc.le]
  have e1 : c * η * (ρ / c) = η * ρ := by field_simp
  have e2 : ∀ A M V S : ℝ, c * η / A * (M / c) / (V / c / S + eps / c) =
      c * (η / A * M / (V / S + eps)) := by
    intro A M V S
    have h : V / c / S + eps / c = (V / S + eps) / c := by ring
    have h' : c * η / A * (M / c) = η / A * M := by field_simp
    rw [h, h', div_div_eq_mul_div]
    ring
  rw [e1, e2]
  ring

example (s : (Fin 1 → ℝ) × (Fin 1 → ℝ) × (Fin 1 → ℝ)) :
    mainAdamStep (fun w i => 1 / w i) ((1 / 10) / 2) ((1 / 100000000) / 2) (2 * (1 / 1000)) 0
        (atScale 2 s) =
      atScale 2 (mainAdamStep (fun w i => 1 / w i) (1 / 10) (1 / 100000000) (1 / 1000) 0 s) :=
  mainAdamStep_atScale two_pos _ (fun w i => by rw [div_div, mul_comm]) _ _ _ 0 s

/-- **The runs are the same at the scale `c`**, `review/flat_spectrum.md`, item RR3. -/
theorem mainAdamRun_atScale {c : ℝ} (hc : 0 < c) (G : (ι → ℝ) → ι → ℝ)
    (hG : ∀ w i, G (fun j => c * w j) i = G w i / c) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ)
    (t : ℕ) :
    mainAdamRun G (ρ / c) (eps / c) (fun s => c * η s) (fun i => c * p0 i) t =
      atScale c (mainAdamRun G ρ eps η p0 t) := by
  induction t with
  | zero => simp [mainAdamRun, atScale]
  | succ t ih => rw [mainAdamRun, ih, mainAdamStep_atScale hc G hG, mainAdamRun]

/-- A constant rate `0.03` with the decay `1`, and the rate `0.001` with the decay `30` from
a point thirty times smaller, on the gradient `1 / w` of `log |w|`. -/
example (t : ℕ) :
    mainAdamRun (fun w (i : Fin 1) => 1 / w i) (1 / (1 / 30)) (1 / 100000000 / (1 / 30))
        (fun _ => 1 / 30 * (3 / 100)) (fun _ => 1 / 30 * 1) t =
      atScale (1 / 30) (mainAdamRun (fun w (i : Fin 1) => 1 / w i) 1 (1 / 100000000)
        (fun _ => 3 / 100) (fun _ => 1) t) :=
  mainAdamRun_atScale (by norm_num) _ (fun w i => by rw [div_div, mul_comm]) _ _ _ _ t

end Transformer.Grokking.GradientEvidence

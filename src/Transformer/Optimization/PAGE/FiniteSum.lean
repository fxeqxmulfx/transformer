module

public import Transformer.Optimization.Basic
public import Mathlib.Analysis.Calculus.FDeriv.Add

/-! The finite-sum PAGE optimizer of Li, Bao, Zhang and Richtárik (ICML 2021),
Algorithm 1 with exact refreshes and a uniform finite correction batch. An
index can denote a single example or one of equally weighted minibatches.
The two gradients in a correction always use the same index. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Finset
open scoped InnerProductSpace

variable {I E V : Type*} [Fintype I]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Uniform average in real coordinates, for either scalar losses or gradients. -/
noncomputable def mean [AddCommGroup V] [Module ℝ V] (v : I → V) : V :=
  (Fintype.card I : ℝ)⁻¹ • ∑ i, v i

theorem mean_const [Nonempty I] [AddCommGroup V] [Module ℝ V] (v : V) : mean (fun _ : I => v) = v := by
  have h : (Fintype.card I : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [mean, sum_const, card_univ, ← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ h,
    one_smul]

theorem mean_add [AddCommGroup V] [Module ℝ V] (u v : I → V) :
    mean (fun i => u i + v i) = mean u + mean v := by simp [mean, sum_add_distrib, smul_add]

theorem mean_sub [AddCommGroup V] [Module ℝ V] (u v : I → V) :
    mean (fun i => u i - v i) = mean u - mean v := by simp [mean, sum_sub_distrib, smul_sub]

theorem mean_mul (c : ℝ) (u : I → ℝ) : mean (fun i => c * u i) = c * mean u := by
  simp only [mean, smul_eq_mul, ← mul_sum]
  ring

theorem mean_mono {u v : I → ℝ} (h : ∀ i, u i ≤ v i) : mean u ≤ mean v := by
  exact mul_le_mul_of_nonneg_left (sum_le_sum fun i _ => h i) (by positivity)

theorem mean_inner (z : E) (u : I → E) : mean (fun i => ⟪z, u i⟫_ℝ) = ⟪z, mean u⟫_ℝ := by
  simp [mean, real_inner_smul_right, inner_sum]

variable [CompleteSpace E]

noncomputable def objective (loss : I → E → ℝ) (x : E) : ℝ := mean (fun i => loss i x)

noncomputable def full (loss : I → E → ℝ) (x : E) : E := mean (fun i => gradient (loss i) x)

/-- A refresh computes the genuine gradient of the finite-sum objective. -/
theorem gradient_objective (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i)) (x : E) :
    gradient (objective loss) x = full loss x := by
  apply (InnerProductSpace.toDual ℝ E).injective
  rw [toDual_gradient]
  have h := (HasFDerivAt.fun_sum (u := (univ : Finset I))
    (fun i _ => (hd i x).hasFDerivAt)).const_smul (Fintype.card I : ℝ)⁻¹
  change fderiv ℝ ((Fintype.card I : ℝ)⁻¹ • (fun y => ∑ i, loss i y)) x = _
  rw [h.fderiv]
  simp [full, mean, map_sum, toDual_gradient]

structure State (E : Type*) where
  weights : E
  estimate : E

noncomputable def start (loss : I → E → ℝ) (x : E) : State E := ⟨x, full loss x⟩

/-- A sampled training update. None refreshes, some i evaluates two component
gradients and carries the estimate. Only the refresh reads all components. -/
noncomputable def step (loss : I → E → ℝ) (η : ℝ) (s : State E) (draw : Option I) : State E :=
  let y := s.weights - η • s.estimate
  ⟨y, match draw with
    | none => full loss y
    | some i => s.estimate + (gradient (loss i) y - gradient (loss i) s.weights)⟩

/-- An actual trajectory given the refresh decisions and sampled components. -/
noncomputable def run (loss : I → E → ℝ) (η : ℝ) (x : E) (draw : ℕ → Option I) : ℕ → State E
  | 0 => start loss x
  | t+1 => step loss η (run loss η x draw t) (draw t)

/-- Every sampler fixes a common stationary point of the component losses. -/
theorem run_fixed (loss : I → E → ℝ) (η : ℝ) (x : E)
    (hx : ∀ i, gradient (loss i) x = 0) (draw : ℕ → Option I) (t : ℕ) :
    run loss η x draw t = ⟨x, 0⟩ := by
  have hf : full loss x = 0 := by simp [full, mean, hx]
  induction t with
  | zero => simp [run, start, hf]
  | succ t ih => cases h : draw t <;> simp [run, ih, step, h, hf, hx]

/-- One conditional expectation, with refresh probability p and a fresh
uniform index on the other branch. This operator is used in the proof. -/
noncomputable def next (loss : I → E → ℝ) (η p : ℝ) (h : State E → ℝ) (s : State E) : ℝ :=
  p * h (step loss η s none) + (1-p) * mean (fun i => h (step loss η s (some i)))

/-- Expected observable after t independent sampled updates, by finite conditioning. -/
noncomputable def expected (loss : I → E → ℝ) (η p : ℝ) :
    ℕ → (State E → ℝ) → State E → ℝ
  | 0, h => h
  | t+1, h => expected loss η p t (next loss η p h)

/-- Product law of the first t refresh decisions, obtained by independently
extending each finite history. Entries past t use an arbitrary default. -/
noncomputable def historyMean (p : ℝ) : ℕ → ((ℕ → Option I) → ℝ) → ℝ
  | 0, h => h (fun _ => none)
  | t+1, h => historyMean p t fun path =>
    p * h (Function.update path t none) +
      (1-p) * mean (fun i => h (Function.update path t (some i)))

private theorem run_prefix (loss : I → E → ℝ) (η : ℝ) (x : E) (t : ℕ)
    {a b : ℕ → Option I} (h : ∀ j < t, a j = b j) :
    run loss η x a t = run loss η x b t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [run, run, ih (fun j hj => h j (by omega)), h t (by omega)]

/-- The expectation used in the rate theorems equals averaging the actual
training trajectory over all independent, correctly weighted finite paths. -/
theorem expected_eq_run (loss : I → E → ℝ) (η p : ℝ) (x : E) (t : ℕ) (h : State E → ℝ) :
    expected loss η p t h (start loss x) =
      historyMean p t (fun path => h (run loss η x path t)) := by
  induction t generalizing h with
  | zero => rfl
  | succ t ih =>
    rw [expected, ih, historyMean]
    apply congrArg (historyMean p t)
    funext path
    have fixPrefix (a : Option I) : run loss η x (Function.update path t a) t = run loss η x path t :=
      run_prefix _ _ _ _ (fun j hj => Function.update_of_ne (Nat.ne_of_lt hj) _ _)
    simp only [next, run, fixPrefix, Function.update_self]

theorem expected_mono (loss : I → E → ℝ) (η : ℝ) {p : ℝ} (hp : 0 ≤ p) (hp' : p ≤ 1)
    {u v : State E → ℝ} (h : ∀ s, u s ≤ v s) (t : ℕ) (s : State E) :
    expected loss η p t u s ≤ expected loss η p t v s := by
  induction t generalizing u v with
  | zero => exact h s
  | succ t ih =>
    apply ih
    intro a
    exact add_le_add (mul_le_mul_of_nonneg_left (h _) hp)
      (mul_le_mul_of_nonneg_left (mean_mono fun _ => h _) (sub_nonneg.mpr hp'))

theorem expected_const [Nonempty I] (loss : I → E → ℝ) (η p c : ℝ) (t : ℕ) (s : State E) :
    expected loss η p t (fun _ => c) s = c := by
  induction t with
  | zero => rfl
  | succ t ih =>
    have h : next loss η p (fun _ => c) = fun _ => c := by
      funext a
      simp only [next, mean_const]
      ring
    simpa only [expected, h] using ih

theorem expected_sub_mul (loss : I → E → ℝ) (η p c : ℝ) (t : ℕ) (u v : State E → ℝ) (s : State E) :
    expected loss η p t (fun a => u a - c * v a) s =
      expected loss η p t u s - c * expected loss η p t v s := by
  induction t generalizing u v with
  | zero => rfl
  | succ t ih =>
    have h : next loss η p (fun a => u a - c*v a) =
        fun a => next loss η p u a - c * next loss η p v a := by
      funext a
      simp only [next, mean_sub, mean_mul]
      ring
    simpa only [expected, h] using ih (next loss η p u) (next loss η p v)

end Transformer.Optimization.PAGE

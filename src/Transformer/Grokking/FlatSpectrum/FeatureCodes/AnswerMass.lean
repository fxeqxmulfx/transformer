module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.FiberCodes

/-!
# Pool signed feature mass by answer

Source: `review/flat_spectrum.md`, *Combined answer contrasts*, AC1–AC2.
The observable coordinates are sums over answer fibers. Pooling commutes with
every linear prediction operator and contracts the sum of absolute errors.
Signed coordinates are allowed; no probabilistic interpretation is assumed.
A decoder constant on each fiber reads only these pooled coordinates. A latent
score offset common to every output class cancels from every class contrast.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset

variable {S A V I : Type*} [Fintype S] [Fintype A] [DecidableEq A]

/-- The total signed mass in each answer fiber, AC1. -/
def answerMass (f : S → A) (x : S → ℝ) (a : A) : ℝ :=
  ∑ s ∈ univ.filter (fun s => f s = a), x s

/-- An arbitrary learned real decoder on answer coordinates, AC2. -/
def answerScore (C : V → A → ℝ) (p : A → ℝ) (v : V) : ℝ := ∑ a, C v a * p a

/-- Pooling preserves total mass, AC1. -/
theorem sum_answerMass (f : S → A) (x : S → ℝ) :
    ∑ a, answerMass f x a = ∑ s, x s :=
  sum_fiberwise univ f x

/-- **Pooling contracts absolute error**, AC1. Errors of opposite signs in
one answer fiber cancel before the absolute value is taken. -/
theorem answerMass_abs_sum_le (f : S → A) (e : S → ℝ) :
    ∑ a, |answerMass f e a| ≤ ∑ s, |e s| := by
  calc
    _ ≤ ∑ a, ∑ s ∈ univ.filter (fun s => f s = a), |e s| :=
      sum_le_sum fun _ _ => abs_sum_le_sum_abs _ _
    _ = _ := sum_fiberwise univ f (fun s => |e s|)

omit [Fintype A] in
/-- **Linear prediction and answer pooling commute**, AC1. In particular,
fitting fixed-design ridge probes to pooled targets equals pooling their fitted
unpooled predictions; pooling alone does not discover a new representation. -/
theorem answerMass_linear_prediction [Fintype I] (f : S → A)
    (w : I → ℝ) (x : I → S → ℝ) (a : A) :
    answerMass f (fun s => ∑ i, w i * x i s) a =
      ∑ i, w i * answerMass f (x i) a := by
  simp only [answerMass, mul_sum]
  exact sum_comm

/-- **Any fiber-constant decoder factors through the pooled mass**, AC2. -/
theorem answerScore_pool (C : V → A → ℝ) (f : S → A) (x : S → ℝ) (v : V) :
    (∑ s, C v (f s) * x s) = answerScore C (answerMass f x) v := by
  unfold answerScore answerMass
  simp_rw [mul_sum]
  rw [← sum_fiberwise univ f (fun s => C v (f s) * x s)]
  apply sum_congr rfl
  intro a ha
  apply sum_congr rfl
  intro s hs
  rw [(mem_filter.mp hs).2]

/-- A common latent score offset disappears from all contrasts, AC2.
Its magnitude is unrestricted, as is the signed mass of the representation. -/
theorem answer_contrast_common_offset (C : V → A → ℝ) (f : S → A)
    (b x : S → ℝ) (t v : V) :
    (∑ s, (C t (f s) + b s) * x s) - (∑ s, (C v (f s) + b s) * x s) =
      answerScore C (answerMass f x) t - answerScore C (answerMass f x) v := by
  simp_rw [add_mul, sum_add_distrib, answerScore_pool]
  ring

/-- An error with zero mass in each fiber is invisible to every pooled
decoder, AC2; individual latent errors can be arbitrarily large. -/
theorem answerScore_ignore_balanced (C : V → A → ℝ) (f : S → A)
    (x e : S → ℝ) (he : ∀ a, answerMass f e a = 0) (v : V) :
    answerScore C (answerMass f (x + e)) v = answerScore C (answerMass f x) v := by
  have hp : answerMass f (x + e) = answerMass f x := by
    funext a
    simp only [answerMass, Pi.add_apply, sum_add_distrib] at *
    rw [he a, add_zero]
  rw [hp]

example := sum_answerMass (S := Bool × Bool) Prod.fst (fun s => if s.2 then 3 else -2)
example := answerMass_abs_sum_le (S := Bool × Bool) Prod.fst
  (fun s => if s.2 then 100 else -100)
example := answerMass_linear_prediction (S := Bool × Bool) (I := Bool) Prod.fst
  (fun i => if i then 2 else -1) (fun i s => if i = s.2 then 1 else 0) true
example := answerScore_pool (S := Bool × Bool) (V := Bool)
  (fun v a => if v = a then 1 else 0) Prod.fst (fun s => if s.2 then 2 else -1) true
example := answer_contrast_common_offset (S := Bool × Bool) (V := Bool)
  (fun v a => if v = a then 1 else 0) Prod.fst (fun _ => 10000)
  (fun s => if s.2 then 2 else -1) true false

example (M : ℝ) := answerScore_ignore_balanced (S := Bool × Bool) (V := Bool)
  (fun v a => if v = a then 1 else 0) Prod.fst (Pi.single (true, false) 1)
  (fun s => if s.2 then M else -M)
  (by intro a; cases a <;> simp [answerMass, sum_filter,
    Fintype.sum_prod_type]) true

end Transformer.Grokking.FlatSpectrum.FeatureCodes

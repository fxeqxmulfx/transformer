module

public import Transformer.Optimization.PAGE.FiniteSum
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-! PAGE on an empirical loss with a coupled quadratic penalty about c.
At constant step η and penalty λ=1/η the linear decay cancels the previous
weight in a refresh, and cancels the old displacement in a correction. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Finset
open scoped InnerProductSpace

variable {I E : Type*} [Fintype I] [Nonempty I]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Standard L2 weight decay on the trainable displacement w-c. -/
noncomputable def anchoredLoss (loss : I → E → ℝ) (c : E) (lam : ℝ) (i : I) (w : E) : ℝ :=
  loss i w + (lam/2) * ‖w-c‖^2

omit [Fintype I] [Nonempty I] in
theorem gradient_anchoredLoss (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    (c : E) (lam : ℝ) (i : I) (x : E) :
    gradient (anchoredLoss loss c lam i) x = gradient (loss i) x + lam • (x-c) := by
  have hs := (hasFDerivAt_id (𝕜 := ℝ) x).sub_const c
  have hpen : HasFDerivAt (fun y : E => (lam/2) * ‖y-c‖^2) (lam • innerSL ℝ (x-c)) x := by
    have h := (((hasStrictFDerivAt_norm_sq (x-c)).hasFDerivAt).comp x hs).const_mul (lam/2)
    convert h using 1 <;> try rfl
    ext z
    simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.id_apply, two_smul,
      smul_apply, add_apply, smul_eq_mul]
    ring
  apply (InnerProductSpace.toDual ℝ E).injective
  rw [toDual_gradient, map_add, map_smul, toDual_gradient]
  exact ((hd i x).hasFDerivAt.add hpen).fderiv

theorem full_anchoredLoss (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    (c : E) (lam : ℝ) (x : E) : full (anchoredLoss loss c lam) x = full loss x + lam • (x-c) := by
  simp only [full, gradient_anchoredLoss loss hd, mean_add, mean_const]

/-- Weight equations of the ordinary PAGE implementation, including every
sampled correction. The estimator has not been replaced by the full gradient. -/
theorem anchored_weight_updates (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    (c y : E) {η : ℝ} (hη : 0 < η) (path : ℕ → Option I) :
    let w := fun t => (run (anchoredLoss loss c η⁻¹) η y path t).weights
    let update := fun x => c - η • full loss x
    w 0 = y ∧ w 1 = update y ∧ ∀ t,
      w (t+2) = match path t with
        | none => update (w (t+1))
        | some i => w (t+1) - η • (gradient (loss i) (w (t+1)) - gradient (loss i) (w t)) := by
  let obj := anchoredLoss loss c η⁻¹
  let fit := run obj η y path
  let w (t : ℕ) := (fit t).weights
  let g (t : ℕ) := (fit t).estimate
  have move (t : ℕ) : w (t+1) = w t - η • g t := rfl
  have mulinv (x : E) : η • (η⁻¹ • x) = x := by rw [smul_smul, mul_inv_cancel₀ hη.ne', one_smul]
  have refresh (x : E) : x - η • full obj x = c - η • full loss x := by
    rw [full_anchoredLoss loss hd, smul_add, mulinv]
    abel
  refine ⟨rfl,refresh y,?_⟩
  intro t
  change w (t+2) = _
  rw [move (t+1)]
  cases hp : path t with
  | none =>
    have he : g (t+1) = full obj (w (t+1)) := by simp [g,w,fit,run,step,hp]
    rw [he]
    exact refresh _
  | some i =>
    have he : g (t+1) = g t + (gradient (obj i) (w (t+1)) - gradient (obj i) (w t)) := by
      simp [g,w,fit,run,step,hp]
    rw [he]
    change w (t+1) - η • (g t + (gradient (obj i) (w (t+1)) - gradient (obj i) (w t))) =
      w (t+1) - η • (gradient (loss i) (w (t+1)) - gradient (loss i) (w t))
    simp only [obj, gradient_anchoredLoss loss hd, smul_add, smul_sub, mulinv]
    have hx : η • g t = w t - w (t+1) := by rw [move,sub_sub_cancel]
    rw [hx]
    abel

end Transformer.Optimization.PAGE

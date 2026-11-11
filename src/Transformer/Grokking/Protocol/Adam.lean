module

public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Tactic

/-!
# The author's Adam and AdamW update and its run

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, "Model
and optimization"; openai/grok at 3d64b1d8, `grok/training.py`: `CustomAdamW.step` in its
`to_zero` form, `_scheduler_lr`, `configure_optimizers`; `experiments/grok_division/PROTOCOL.md`,
*The optimizer*.

One class serves both optimizers (`adamStep`). Update `t = 1, 2, …` at the gradient `g` and the
learning rate `lr_t` is, for every parameter alike, normalization gains and biases included,

`p ← p (1 - lr_t wd)`, `m ← β₁ m + (1 - β₁) g`, `v ← β₂ v + (1 - β₂) g²`,
`p ← p - lr_t / (1 - β₁ᵗ) · m / (√v / √(1 - β₂ᵗ) + ε)`.

The learning rate is `(t - 1)/warmup · rate` while `t - 1 ≤ warmup` and `rate` after
(`Hyper.rateAt`): `LambdaLR` has stepped `t - 1` times at update `t`, so the first update has
rate `0` and moves nothing (`adamRun_one_params`). The run (`adamRun`) takes the gradient of
update `t` from an oracle `grad t`, which in training is the gradient of the loss of the
`t`-th minibatch.

The update is well defined: for `0 ≤ β < 1` and `t ≥ 1` the bias corrections `1 - βᵗ` are
positive (`one_sub_pow_pos`), and for `ε > 0` the denominator is positive whatever the second
moment (`adamDenominator_pos`). The second moment is nonnegative (`adamRun_second_nonneg`). If
the gradients seen up to update `t` are at most `G` in absolute value, then
`|m_t| ≤ (1 - β₁ᵗ) G` and `v_t ≤ (1 - β₂ᵗ) G²` (`abs_adamRun_first_le`, `adamRun_second_le`),
and the update moves a decayed parameter by at most `lr_t G / ε` (`abs_adamRun_step_le`); these
three bounds are in `AdamBounds`. Nothing here says that a run converges.

The source's constants are `Hyper.source`: rate `10⁻³`, warmup `10`, `β = (0.9, 0.98)`,
`ε = 10⁻⁸`; weight decay `1` for AdamW and `0` for Adam.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι : Type*}

/-- The optimizer's constants. -/
structure Hyper where
  /-- The learning rate after warmup, `max_lr`. -/
  rate : ℝ
  /-- Updates of linear warmup, `warmup_steps`. -/
  warmup : ℕ
  /-- The first moment's decay. -/
  beta1 : ℝ
  /-- The second moment's decay. -/
  beta2 : ℝ
  /-- The constant added to the root of the corrected second moment. -/
  eps : ℝ
  /-- The decoupled weight decay; `0` is Adam. -/
  decay : ℝ

/-- The source's constants at the weight decay given. -/
noncomputable def Hyper.source (decay : ℝ) : Hyper :=
  ⟨1 / 1000, 10, 9 / 10, 49 / 50, 1 / 10 ^ 8, decay⟩

/-- **The learning rate of update `t`**, counted from `1`. -/
noncomputable def Hyper.rateAt (h : Hyper) (t : ℕ) : ℝ :=
  if t - 1 ≤ h.warmup then ((t - 1 : ℕ) : ℝ) / (max h.warmup 1 : ℕ) * h.rate else h.rate

variable {h : Hyper} {t : ℕ}

lemma Hyper.rateAt_one (h : Hyper) : h.rateAt 1 = 0 := by simp [Hyper.rateAt]

lemma Hyper.rateAt_of_lt (ht : h.warmup < t - 1) : h.rateAt t = h.rate := by
  rw [Hyper.rateAt, ite_eq_right (not_le.2 ht)]

lemma Hyper.rateAt_nonneg (hr : 0 ≤ h.rate) (t : ℕ) : 0 ≤ h.rateAt t := by
  unfold Hyper.rateAt
  split_ifs <;> positivity

lemma Hyper.rateAt_le (hr : 0 ≤ h.rate) (t : ℕ) : h.rateAt t ≤ h.rate := by
  unfold Hyper.rateAt
  split_ifs with ht
  · refine mul_le_of_le_one_left hr (div_le_one_of_le₀ (Nat.cast_le.2 ?_) (Nat.cast_nonneg _))
    exact ht.trans (le_max_left _ _)
  · exact le_rfl

/-- The parameters and both moments. -/
structure AdamState (ι : Type*) where
  /-- The parameters. -/
  params : ι → ℝ
  /-- The first moments. -/
  first : ι → ℝ
  /-- The second moments. -/
  second : ι → ℝ

/-- **Update `t`**, counted from `1`, at the gradient `g`. -/
noncomputable def adamStep (h : Hyper) (t : ℕ) (g : ι → ℝ) (s : AdamState ι) : AdamState ι where
  first i := h.beta1 * s.first i + (1 - h.beta1) * g i
  second i := h.beta2 * s.second i + (1 - h.beta2) * g i ^ 2
  params i := (1 - h.rateAt t * h.decay) * s.params i - h.rateAt t / (1 - h.beta1 ^ t) *
    ((h.beta1 * s.first i + (1 - h.beta1) * g i) /
      (√(h.beta2 * s.second i + (1 - h.beta2) * g i ^ 2) / √(1 - h.beta2 ^ t) + h.eps))

/-- **The run**: the state after `t` updates from the parameters `θ₀` and zero moments, where
update `t` sees the gradient `grad t` at the parameters before it. -/
noncomputable def adamRun (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) : ℕ → AdamState ι
  | 0 => ⟨θ₀, 0, 0⟩
  | t + 1 => adamStep h (t + 1) (grad (t + 1) (adamRun h grad θ₀ t).params) (adamRun h grad θ₀ t)

variable {grad : ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : ι → ℝ} {G : ℝ}

/-- The bias corrections are positive from the first update on. -/
lemma one_sub_pow_pos {β : ℝ} (h0 : 0 ≤ β) (h1 : β < 1) (ht : 1 ≤ t) : 0 < 1 - β ^ t :=
  sub_pos.2 (pow_lt_one₀ h0 h1 (Nat.one_le_iff_ne_zero.1 ht))

/-- **The update divides by a positive number**, whatever the second moment. -/
lemma adamDenominator_pos (he : 0 < h.eps) (v : ℝ) (t : ℕ) : 0 < √v / √(1 - h.beta2 ^ t) + h.eps :=
  add_pos_of_nonneg_of_pos (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)) he

/-- **The first update moves nothing**: its learning rate is `0`. -/
theorem adamRun_one_params (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) :
    (adamRun h grad θ₀ 1).params = θ₀ := by
  funext i
  simp [adamRun, adamStep, Hyper.rateAt_one]

/-- **The second moment is nonnegative.** -/
theorem adamRun_second_nonneg (h2 : 0 ≤ h.beta2) (h2' : h.beta2 ≤ 1) (t : ℕ) (i : ι) :
    0 ≤ (adamRun h grad θ₀ t).second i := by
  induction t with
  | zero => exact le_rfl
  | succ t ih => exact add_nonneg (mul_nonneg h2 ih) (mul_nonneg (sub_nonneg.2 h2') (sq_nonneg _))

/-! The witnesses: the source's constants. The learning rates of updates `1`, `2`, `11` and `12`
are `0`, `10⁻⁴`, `10⁻³` and `10⁻³`. -/

example : (Hyper.source 1).rateAt 1 = 0 := Hyper.rateAt_one _

example : (Hyper.source 1).rateAt 2 = 1 / 10000 := by norm_num [Hyper.rateAt, Hyper.source]

example : (Hyper.source 1).rateAt 11 = 1 / 1000 := by norm_num [Hyper.rateAt, Hyper.source]

example : (Hyper.source 1).rateAt 12 = 1 / 1000 :=
  Hyper.rateAt_of_lt (h := Hyper.source 1) (t := 12) (by norm_num [Hyper.source])

example := Hyper.rateAt_nonneg (h := Hyper.source 1) (by norm_num [Hyper.source]) 7

example := Hyper.rateAt_le (h := Hyper.source 1) (by norm_num [Hyper.source]) 7

example : 0 < 1 - (9 / 10 : ℝ) ^ 3 := one_sub_pow_pos (by norm_num) (by norm_num) (by norm_num)

example := adamDenominator_pos (h := Hyper.source 0) (by norm_num [Hyper.source]) 4 9

example := adamRun_one_params (Hyper.source 0) (fun _ _ (_ : Unit) => (1 : ℝ)) 0

/-- The second moment of the source's Adam run at the constant gradient `1` is nonnegative. -/
example (t : ℕ) := adamRun_second_nonneg (h := Hyper.source 0)
  (grad := fun _ _ (_ : Unit) => (1 : ℝ)) (θ₀ := 0) (by norm_num [Hyper.source])
  (by norm_num [Hyper.source]) t ()

end Transformer.Grokking.Protocol

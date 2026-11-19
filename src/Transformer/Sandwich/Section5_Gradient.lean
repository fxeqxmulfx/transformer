module

public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.FDeriv.Pi
public import Transformer.Sandwich.Section5_WindowLayer

/-!
# §5.1 The cumulative normalized gradient and the empirical receptive field

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §5.1: "We first calculate the normalized gradient
of each input token w.r.t the prediction of the next token: `s_m = ‖g_m‖₂ / Σ_{n=1}^{L_ex} ‖g_n‖₂`,
where `g_m` is the gradient vector of the input embedding `e_m`. We then calculate the cumulative
sum as: `c_m = Σ_{n=m}^{L_ex} s_n, 0 ≤ c_m ≤ 1` … We define the ERF of a model as:
`ERF = min {m | c_m > 0.99}`."

* The objective `f` is any function of the `L` input embeddings, such as the log-probability of the
  next token; `grad f e n` is `g_n`, `normGrad` is `s_n` and `cumGrad` is `c_m`, with
  `0 ≤ c_m ≤ 1` (`cumGrad_mem_Icc`). Positions are `0, …, L − 1` for the paper's `1, …, L_ex`.
* **The printed ERF is always the first position**: `c_1 = Σ_n s_n = 1 > 0.99` as soon as some
  gradient is nonzero (`erfPrinted_eq_zero`). The ERFs the paper reports (768, 1024 and 1536
  tokens when predicting the 2048-th token) count the most recent tokens: `erf` is the least `t`
  with `c_{L − t} > 0.99`, the number of most recent tokens that hold more than 99% of the
  normalized gradient.
* A token on which `f` does not depend has the gradient `0` (`grad_eq_zero`); so the ERF is at
  most the number of most recent tokens with a gradient (`erf_le_of_grad_eq_zero`), and a windowed
  model's ERF is at most its TRF `R(w − 1) + 1` (`Model.erf_le`): "a neural model often uses a
  subset of TRF".
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset

variable {L d k H v w : ℕ}

/-- The gradient `g_n` of the objective `f` with respect to the input embedding `e_n`, at the
coordinate `i` (§5.1). -/
noncomputable def grad (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) (n : Fin L)
    (i : Fin d) : ℝ :=
  fderiv ℝ f e (Pi.single n (Pi.single i 1))

/-- The norm `‖g_n‖₂` of the gradient with respect to `e_n`. -/
noncomputable def gradNorm (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) (n : Fin L) :
    ℝ :=
  Real.sqrt (∑ i, grad f e n i ^ 2)

/-- The normalized gradient `s_n = ‖g_n‖₂ / Σ_j ‖g_j‖₂` (§5.1). -/
noncomputable def normGrad (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) (n : Fin L) :
    ℝ :=
  gradNorm f e n / ∑ j, gradNorm f e j

/-- The cumulative normalized gradient `c_m = Σ_{n ≥ m} s_n` (§5.1). -/
noncomputable def cumGrad (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) (m : ℕ) : ℝ :=
  ∑ n ∈ univ.filter fun n : Fin L => m ≤ (n : ℕ), normGrad f e n

/-- The ERF as printed in §5.1: `min {m | c_m > 0.99}`. -/
noncomputable def erfPrinted (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) : ℕ :=
  sInf {m | 0.99 < cumGrad f e m}

/-- The ERF as §5.1 uses it: the least number `t` of most recent tokens that hold more than 99% of
the normalized gradient, `min {t | c_{L − t} > 0.99}`. -/
noncomputable def erf (f : (Fin L → Fin d → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) : ℕ :=
  sInf {t | 0.99 < cumGrad f e (L - t)}

variable {f : (Fin L → Fin d → ℝ) → ℝ} {e : Fin L → Fin d → ℝ}

/-- The normalized gradients are nonnegative. -/
theorem normGrad_nonneg (n : Fin L) : 0 ≤ normGrad f e n :=
  div_nonneg (Real.sqrt_nonneg _) (sum_nonneg fun _ _ => Real.sqrt_nonneg _)

/-- **`0 ≤ c_m ≤ 1`** (§5.1). -/
theorem cumGrad_mem_Icc (m : ℕ) : 0 ≤ cumGrad f e m ∧ cumGrad f e m ≤ 1 := by
  refine ⟨sum_nonneg fun n _ => normGrad_nonneg n, ?_⟩
  calc cumGrad f e m ≤ ∑ n, normGrad f e n :=
        sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun n _ _ => normGrad_nonneg n
    _ = (∑ n, gradNorm f e n) / ∑ j, gradNorm f e j := by rw [sum_div]; rfl
    _ ≤ 1 := div_self_le_one _

/-- The normalized gradients sum to `1` when some gradient is nonzero. -/
theorem sum_normGrad (h : ∃ n i, grad f e n i ≠ 0) : ∑ n, normGrad f e n = 1 := by
  obtain ⟨n, i, hni⟩ := h
  have hpos : 0 < ∑ j, gradNorm f e j := by
    refine lt_of_lt_of_le ?_ (single_le_sum (fun j _ => Real.sqrt_nonneg _) (mem_univ n))
    exact Real.sqrt_pos.2 (lt_of_lt_of_le (pow_pos (abs_pos.2 hni) 2 |>.trans_eq (sq_abs _))
      (single_le_sum (fun j _ => sq_nonneg (grad f e n j)) (mem_univ i)))
  simp only [normGrad, ← sum_div]
  exact div_self hpos.ne'

/-- The objective `e ↦ e₀₀`, the first coordinate of the first embedding, has the gradient `1`
there: `grad` sees the input. -/
theorem grad_apply_zero (e : Fin 1 → Fin 1 → ℝ) :
    grad (fun e : Fin 1 → Fin 1 → ℝ => e 0 0) e 0 0 = 1 := by
  have h := (hasFDerivAt_apply (𝕜 := ℝ) (0 : Fin 1) (e 0)).comp e
    (hasFDerivAt_apply (𝕜 := ℝ) (0 : Fin 1) e)
  have h' : fderiv ℝ (fun e : Fin 1 → Fin 1 → ℝ => e 0 0) e = _ := h.fderiv
  simp [grad, h']

/-- The hypothesis of `sum_normGrad` holds for the objective `e ↦ e₀₀`. -/
example (e : Fin 1 → Fin 1 → ℝ) :
    ∑ n, normGrad (fun e : Fin 1 → Fin 1 → ℝ => e 0 0) e n = 1 :=
  sum_normGrad ⟨0, 0, by rw [grad_apply_zero]; exact one_ne_zero⟩

/-- **The printed ERF is the first position** (§5.1): `c_1 = Σ_n s_n = 1 > 0.99` as soon as some
gradient is nonzero, so `min {m | c_m > 0.99}` is the first position, `1` in the paper and `0`
here, whatever the model. -/
theorem erfPrinted_eq_zero (h : ∃ n i, grad f e n i ≠ 0) : erfPrinted f e = 0 := by
  refine Nat.sInf_eq_zero.2 (Or.inl ?_)
  show 0.99 < cumGrad f e 0
  simp only [cumGrad, zero_le, Finset.filter_true]
  rw [sum_normGrad h]
  norm_num

/-- The hypothesis of `erfPrinted_eq_zero` holds for the objective `e ↦ e₀₀`. -/
example (e : Fin 1 → Fin 1 → ℝ) : erfPrinted (fun e : Fin 1 → Fin 1 → ℝ => e 0 0) e = 0 :=
  erfPrinted_eq_zero ⟨0, 0, by rw [grad_apply_zero]; exact one_ne_zero⟩

/-- **A token on which the objective does not depend has the gradient `0`.** -/
theorem grad_eq_zero {n : Fin L} (h : ∀ e', (∀ j, j ≠ n → e' j = e j) → f e' = f e)
    (i : Fin d) : grad f e n i = 0 := by
  by_cases hf : DifferentiableAt ℝ f e
  · have hline : HasDerivAt (fun t : ℝ => e + t • (Pi.single n (Pi.single i 1) : Fin L → Fin d → ℝ))
        (Pi.single n (Pi.single i 1)) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const
        (Pi.single n (Pi.single i 1) : Fin L → Fin d → ℝ)).const_add e
    have hf' : HasFDerivAt f (fderiv ℝ f e)
        (e + (0 : ℝ) • (Pi.single n (Pi.single i 1) : Fin L → Fin d → ℝ)) := by
      simpa using hf.hasFDerivAt
    have h1 := hf'.comp_hasDerivAt (0 : ℝ) hline
    have hconst : (f ∘ fun t : ℝ => e + t • (Pi.single n (Pi.single i 1) : Fin L → Fin d → ℝ)) =
        fun _ => f e := funext fun t => h _ fun j hj => by simp [Pi.single_eq_of_ne hj]
    rw [hconst] at h1
    exact h1.unique (hasDerivAt_const (x := (0 : ℝ)) (c := f e))
  · simp [grad, fderiv_zero_of_not_differentiableAt hf]

/-- The hypothesis of `grad_eq_zero` holds for a constant objective. -/
example (e : Fin L → Fin d → ℝ) (n : Fin L) (i : Fin d) :
    grad (fun _ : Fin L → Fin d → ℝ => (0 : ℝ)) e n i = 0 :=
  grad_eq_zero (fun _ _ => rfl) i

/-- **The ERF is at most the number of most recent tokens with a nonzero gradient.** -/
theorem erf_le_of_grad_eq_zero {W : ℕ} (h : ∀ n : Fin L, (n : ℕ) + W < L → ∀ i, grad f e n i = 0) :
    erf f e ≤ W := by
  by_cases hZ : ∑ j, gradNorm f e j = 0
  · have hempty : {t | 0.99 < cumGrad f e (L - t)} = ∅ := by
      ext t
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_lt, cumGrad,
        normGrad, hZ, div_zero, sum_const_zero]
      norm_num
    simp [erf, hempty]
  · apply Nat.sInf_le
    show 0.99 < cumGrad f e (L - W)
    have hfilter : cumGrad f e (L - W) = ∑ n, normGrad f e n := by
      refine sum_filter_of_ne fun n _ hn => ?_
      by_contra hlt
      apply hn
      simp only [normGrad, gradNorm, h n (by omega), ne_eq, OfNat.ofNat_ne_zero,
        not_false_eq_true, zero_pow, sum_const_zero, Real.sqrt_zero, zero_div]
    have hsum : ∑ n, normGrad f e n = 1 := by
      simp only [normGrad, ← sum_div]
      exact div_self hZ
    rw [hfilter, hsum]
    norm_num

/-- The hypothesis of `erf_le_of_grad_eq_zero` holds for a constant objective, whose ERF is
`0`. -/
example (e : Fin L → Fin d → ℝ) : erf (fun _ : Fin L → Fin d → ℝ => (0 : ℝ)) e ≤ 0 :=
  erf_le_of_grad_eq_zero fun _ _ i => grad_eq_zero (fun _ _ => rfl) i

/-- The zero extension of `L` input embeddings to a sequence. -/
noncomputable def extend (e : Fin L → Fin d → ℝ) : Embs d :=
  fun n => if h : n < L then e ⟨n, h⟩ else 0

/-- **A windowed model's ERF is at most its TRF** (§5: "a neural model often uses a subset of
TRF, named empirical receptive field (ERF)"), for the prediction of the token after the last of
`L` tokens and any objective `ℓ` of it. -/
theorem Model.erf_le (M : Model d k H v) {b : Fin H → ℕ → EReal} (hb : ∀ h, IsWindowed w (b h))
    (hw : 1 ≤ w) (ℓ : (Fin v → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) :
    erf (fun e => ℓ (M.predict b (extend e) (L - 1))) e ≤ M.depth * (w - 1) + 1 := by
  apply erf_le_of_grad_eq_zero
  intro n hn i
  apply grad_eq_zero
  intro e' he'
  show ℓ (M.predict b (extend e') (L - 1)) = ℓ (M.predict b (extend e) (L - 1))
  rw [(M.hasWindow_predict hb hw).1 (extend e') (extend e) (L - 1) fun j _ hLj => ?_]
  simp only [extend]
  split_ifs with hjL
  · exact he' ⟨j, hjL⟩ fun hjn => by
      have := congrArg Fin.val hjn
      simp only at this
      omega
  · rfl

/-- The hypotheses of `Model.erf_le` hold for windowed attention of size `1`. -/
example (M : Model d k H v) (ℓ : (Fin v → ℝ) → ℝ) (e : Fin L → Fin d → ℝ) :
    erf (fun e => ℓ (M.predict (fun _ => windowBias 1) (extend e) (L - 1))) e ≤
      M.depth * (1 - 1) + 1 :=
  M.erf_le (fun _ => isWindowed_windowBias 1) le_rfl ℓ e

end Transformer.Sandwich

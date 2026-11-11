module

public import Mathlib.Analysis.SpecificLimits.Basic
public import Transformer.Grokking.Protocol.Adam

/-!
# The limit of the author's Adam and AdamW run on one gradient field

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step` in its `to_zero` form;
`experiments/grok_division/PROTOCOL.md`, *The optimizer*; `review/flat_spectrum.md`,
*Training reaches the optimal fit* (added), item WD13.

Let every update see the gradient of one continuous field `g`, the full batch, let the
constants be regular (`Hyper.Regular`: a nonzero learning rate, `0 ≤ β₁, β₂ < 1`, `ε > 0`), and
let the parameters of the run (`adamRun`) converge to `θ`. The gradients seen converge to `g θ`,
both moments converge, `m → g θ` and `v → (g θ)²` (`tendsto_adamRun_first`,
`tendsto_adamRun_second`: a moving average of a convergent sequence converges to its limit,
`tendsto_movingAverage`), the bias corrections tend to `1`, and the update itself has a limit.
So in every coordinate

    wd · θ = − g θ / (|g θ| + ε)

(`decay_mul_eq_of_tendsto_adamRun`). Consequences:

* `|wd · θ| < 1` in every coordinate (`abs_decay_mul_lt_one_of_tendsto_adamRun`): with weight
  decay a limit lies in the open box of half-width `1 / wd`;
* without weight decay a limit is a critical point of the field
  (`eq_zero_of_tendsto_adamRun`);
* a limit is stationary for the loss with the penalty `(wd/2) ‖θ‖²`, `g θ + wd · θ = 0`, in a
  coordinate only if the gradient there is `0` or has the absolute value `1 − ε`
  (`eq_zero_or_abs_eq_of_tendsto_adamRun`).

So the decoupled weight decay of AdamW is not the squared penalty of gradient descent with
weight decay: the minimizers of the penalized loss are in general not limits of the run. The
witnesses are in `AdamLimitWitness`. Nothing here says that a run converges.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

/-- A moving average of a convergent sequence converges to its limit. -/
lemma tendsto_movingAverage {β a : ℝ} (hβ0 : 0 ≤ β) (hβ1 : β < 1) {x g : ℕ → ℝ}
    (hx : ∀ t, x (t + 1) = β * x t + (1 - β) * g t) (hg : Tendsto g atTop (𝓝 a)) :
    Tendsto x atTop (𝓝 a) := by
  rw [Metric.tendsto_atTop] at hg ⊢
  intro ε hε
  obtain ⟨T, hT⟩ := hg (ε / 2) (half_pos hε)
  have h1 : (0 : ℝ) ≤ 1 - β := by linarith
  have hk : ∀ k, |x (T + k) - a| ≤ β ^ k * |x T - a| + ε / 2 := by
    intro k
    induction k with
    | zero => rw [pow_zero, one_mul, add_zero]; linarith
    | succ k ih =>
      have e : x (T + (k + 1)) - a = β * (x (T + k) - a) + (1 - β) * (g (T + k) - a) := by
        rw [← add_assoc, hx]; ring
      have hgk : |g (T + k) - a| ≤ ε / 2 := by
        have := (hT (T + k) (Nat.le_add_right T k)).le
        rwa [Real.dist_eq] at this
      calc |x (T + (k + 1)) - a| ≤ β * |x (T + k) - a| + (1 - β) * |g (T + k) - a| := by
            rw [e]
            refine (abs_add_le _ _).trans (le_of_eq ?_)
            rw [abs_mul, abs_mul, abs_of_nonneg hβ0, abs_of_nonneg h1]
        _ ≤ β * (β ^ k * |x T - a| + ε / 2) + (1 - β) * (ε / 2) :=
            add_le_add (mul_le_mul_of_nonneg_left ih hβ0) (mul_le_mul_of_nonneg_left hgk h1)
        _ = β ^ (k + 1) * |x T - a| + ε / 2 := by ring
  have hpow : Tendsto (fun k => β ^ k * |x T - a|) atTop (𝓝 0) := by
    simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one hβ0 hβ1).mul_const (|x T - a|)
  obtain ⟨K, hK⟩ := (Metric.tendsto_atTop.1 hpow) (ε / 2) (half_pos hε)
  refine ⟨T + K, fun t ht => ?_⟩
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le (show T ≤ t by omega)
  have h2 := hK k (by omega)
  rw [Real.dist_eq, sub_zero, abs_of_nonneg (mul_nonneg (pow_nonneg hβ0 k) (abs_nonneg _))] at h2
  rw [Real.dist_eq]
  linarith [hk k]

variable {ι : Type*} {h : Hyper} {grad : ℕ → (ι → ℝ) → ι → ℝ} {g : (ι → ℝ) → ι → ℝ}
  {θ₀ θ : ι → ℝ}

/-- The first moment after update `t + 1`. -/
lemma adamRun_succ_first (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (t : ℕ) (i : ι) :
    (adamRun h grad θ₀ (t + 1)).first i = h.beta1 * (adamRun h grad θ₀ t).first i +
      (1 - h.beta1) * grad (t + 1) (adamRun h grad θ₀ t).params i := rfl

/-- The second moment after update `t + 1`. -/
lemma adamRun_succ_second (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (t : ℕ)
    (i : ι) :
    (adamRun h grad θ₀ (t + 1)).second i = h.beta2 * (adamRun h grad θ₀ t).second i +
      (1 - h.beta2) * grad (t + 1) (adamRun h grad θ₀ t).params i ^ 2 := rfl

/-- The parameters after update `t + 1`, in the moments after it. -/
lemma adamRun_succ_params (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (t : ℕ)
    (i : ι) :
    (adamRun h grad θ₀ (t + 1)).params i =
      (1 - h.rateAt (t + 1) * h.decay) * (adamRun h grad θ₀ t).params i -
        h.rateAt (t + 1) / (1 - h.beta1 ^ (t + 1)) * ((adamRun h grad θ₀ (t + 1)).first i /
          (√((adamRun h grad θ₀ (t + 1)).second i) / √(1 - h.beta2 ^ (t + 1)) + h.eps)) := rfl

/-- **The first moment converges to the gradient at the limit**, for one continuous field. -/
theorem tendsto_adamRun_first (h1 : 0 ≤ h.beta1) (h1' : h.beta1 < 1) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι) :
    Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).first i) atTop (𝓝 (g θ i)) :=
  tendsto_movingAverage h1 h1' (fun t => adamRun_succ_first h _ θ₀ t i)
    ((((continuous_apply i).comp hg).tendsto θ).comp hlim)

/-- **The second moment converges to the squared gradient at the limit.** -/
theorem tendsto_adamRun_second (h2 : 0 ≤ h.beta2) (h2' : h.beta2 < 1) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι) :
    Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).second i) atTop (𝓝 (g θ i ^ 2)) :=
  tendsto_movingAverage h2 h2' (fun t => adamRun_succ_second h _ θ₀ t i)
    (((((continuous_apply i).comp hg).tendsto θ).comp hlim).pow 2)

/-- The constants for which the limit of a run is computed: a nonzero learning rate, moment
decays in `[0, 1)` and a positive `ε`. -/
def Hyper.Regular (h : Hyper) : Prop :=
  h.rate ≠ 0 ∧ 0 ≤ h.beta1 ∧ h.beta1 < 1 ∧ 0 ≤ h.beta2 ∧ h.beta2 < 1 ∧ 0 < h.eps

/-- The source's constants are regular, for every weight decay. -/
lemma Hyper.source_regular (wd : ℝ) : (Hyper.source wd).Regular := by
  norm_num [Hyper.Regular, Hyper.source]

/-- **The limit of an Adam or AdamW run** (`review/flat_spectrum.md`, item WD13): if every
update sees the gradient of one continuous field `g` and the parameters converge to `θ`, then
`wd · θ = − g θ / (|g θ| + ε)` in every coordinate. -/
theorem decay_mul_eq_of_tendsto_adamRun (hh : h.Regular) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι) :
    h.decay * θ i = -(g θ i / (|g θ i| + h.eps)) := by
  obtain ⟨hr, h1, h1', h2, h2', he⟩ := hh
  have hp : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params i) atTop (𝓝 (θ i)) :=
    (tendsto_pi_nhds.1 hlim) i
  have hm := (tendsto_add_atTop_iff_nat 1).2 (tendsto_adamRun_first h1 h1' hg hlim i)
  have hv := (tendsto_add_atTop_iff_nat 1).2 (tendsto_adamRun_second h2 h2' hg hlim i)
  have hrate : Tendsto (fun t => h.rateAt (t + 1)) atTop (𝓝 h.rate) :=
    tendsto_const_nhds.congr' ((eventually_gt_atTop h.warmup).mono fun t ht =>
      (Hyper.rateAt_of_lt (by simpa using ht)).symm)
  have hb1 : Tendsto (fun t : ℕ => 1 - h.beta1 ^ (t + 1)) atTop (𝓝 (1 - 0)) :=
    tendsto_const_nhds.sub
      ((tendsto_add_atTop_iff_nat 1).2 (tendsto_pow_atTop_nhds_zero_of_lt_one h1 h1'))
  have hb2 : Tendsto (fun t : ℕ => √(1 - h.beta2 ^ (t + 1))) atTop (𝓝 √(1 - 0)) :=
    (Real.continuous_sqrt.tendsto _).comp (tendsto_const_nhds.sub
      ((tendsto_add_atTop_iff_nat 1).2 (tendsto_pow_atTop_nhds_zero_of_lt_one h2 h2')))
  rw [sub_zero] at hb1
  rw [sub_zero, Real.sqrt_one] at hb2
  have hpos : 0 < |g θ i| + h.eps := add_pos_of_nonneg_of_pos (abs_nonneg _) he
  have hden : Tendsto (fun t => √((adamRun h (fun _ => g) θ₀ (t + 1)).second i) /
      √(1 - h.beta2 ^ (t + 1)) + h.eps) atTop (𝓝 (|g θ i| + h.eps)) := by
    have h3 := (((Real.continuous_sqrt.tendsto _).comp hv).div hb2 one_ne_zero).add_const h.eps
    rwa [Real.sqrt_sq_eq_abs, div_one] at h3
  have hnext : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ (t + 1)).params i) atTop
      (𝓝 ((1 - h.rate * h.decay) * θ i - h.rate / 1 * (g θ i / (|g θ i| + h.eps)))) :=
    (((tendsto_const_nhds.sub (hrate.mul_const h.decay)).mul hp).sub
      ((hrate.div hb1 one_ne_zero).mul (hm.div hden hpos.ne'))).congr fun t =>
        (adamRun_succ_params h _ θ₀ t i).symm
  have huniq := tendsto_nhds_unique ((tendsto_add_atTop_iff_nat 1).2 hp) hnext
  rw [div_one] at huniq
  have h0 : h.rate * (h.decay * θ i + g θ i / (|g θ i| + h.eps)) = 0 := by
    linear_combination huniq
  rcases mul_eq_zero.1 h0 with h4 | h4
  · exact absurd h4 hr
  · linarith

/-- **With weight decay a limit lies in the open box**: `|wd · θ| < 1` in every coordinate. -/
theorem abs_decay_mul_lt_one_of_tendsto_adamRun (hh : h.Regular) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι) :
    |h.decay * θ i| < 1 := by
  have hpos : 0 < |g θ i| + h.eps := add_pos_of_nonneg_of_pos (abs_nonneg _) hh.2.2.2.2.2
  rw [decay_mul_eq_of_tendsto_adamRun hh hg hlim i, abs_neg, abs_div, abs_of_pos hpos]
  exact (div_lt_one hpos).2 (lt_add_of_pos_right _ hh.2.2.2.2.2)

/-- **Without weight decay a limit is a critical point of the field.** -/
theorem eq_zero_of_tendsto_adamRun (hh : h.Regular) (hd : h.decay = 0) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι) :
    g θ i = 0 := by
  have h0 := decay_mul_eq_of_tendsto_adamRun hh hg hlim i
  rw [hd, zero_mul, eq_comm, neg_eq_zero, div_eq_zero_iff] at h0
  exact h0.resolve_right (add_pos_of_nonneg_of_pos (abs_nonneg _) hh.2.2.2.2.2).ne'

/-- **A limit that is stationary for the squared penalty** (`review/flat_spectrum.md`, item
WD13): if a limit of the run satisfies `g θ + wd · θ = 0` in a coordinate, the gradient there
is `0` or has the absolute value `1 − ε`. -/
theorem eq_zero_or_abs_eq_of_tendsto_adamRun (hh : h.Regular) (hg : Continuous g)
    (hlim : Tendsto (fun t => (adamRun h (fun _ => g) θ₀ t).params) atTop (𝓝 θ)) (i : ι)
    (hstat : g θ i + h.decay * θ i = 0) : g θ i = 0 ∨ |g θ i| = 1 - h.eps := by
  have h0 := decay_mul_eq_of_tendsto_adamRun hh hg hlim i
  have hpos : 0 < |g θ i| + h.eps := add_pos_of_nonneg_of_pos (abs_nonneg _) hh.2.2.2.2.2
  have h4 : g θ i = g θ i / (|g θ i| + h.eps) := by linarith
  rw [eq_div_iff hpos.ne'] at h4
  have h3 : g θ i * (|g θ i| + h.eps - 1) = 0 := by linear_combination h4
  rcases mul_eq_zero.1 h3 with h5 | h5
  · exact Or.inl h5
  · exact Or.inr (by linarith)

end Transformer.Grokking.Protocol

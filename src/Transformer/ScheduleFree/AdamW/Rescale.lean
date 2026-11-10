module

public import Transformer.ScheduleFree.AdamW.Invariants

/-!
# Revised AdamW depends on rate and decay through their product

Source: `review/flat_spectrum.md`, *The rate of a recipe* (added), items RR3 and RR4. The
update is that of facebookresearch/schedule_free, commit
`59ed9b662d5ec5799aded7390b2482758423653c`, as `Defs.lean` has it.

For `c > 0` let `scaled c p` have the learning rate `c * lr`, the decay `decay / c` and the
epsilon `epsilon / c` of `p`, and nothing else changed. Let the oracle be homogeneous of degree
`-1` at the scale `c`: `g (c • y) = g y / c`, as the gradient of a loss that does not depend
on the scale of its argument is. Then the run of `scaled c p` from `c • point` is the run of
`p` from `point` multiplied by `c`: its evaluation, base and query points (`run_scaled`).

So on such a loss a recipe is the products `lr * decay` and `lr * epsilon` and the ratio of
its first point to `lr`. The historical recipe `lr = 0.03`, `decay = 1` is, at the rate
`0.001`, the decay `30` (`scaled_historical`).
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- The recipe with the rate multiplied by `c`, and the decay and the epsilon divided by it. -/
def scaled (c : ℝ) (p : Parameters) : Parameters :=
  { p with adam := { p.adam with
      learningRate := c * p.adam.learningRate
      decay := p.adam.decay / c
      epsilon := p.adam.epsilon / c } }

/-- The historical recipe of the DiffGPT recheck: rate `0.03`, decay `1`, ten warm-up steps,
betas `0.9` and `0.98`, epsilon `1e-8`, the default averaging powers. -/
def historical : Parameters := ⟨⟨3 / 100, 1, 10, 9 / 10, 49 / 50, 1 / 100000000⟩, 0, 2⟩

/-- **The historical recipe at the rate `0.001` is the decay `30`**,
`review/flat_spectrum.md`, item RR4. -/
theorem scaled_historical :
    scaled (1 / 30) historical = ⟨⟨1 / 1000, 30, 10, 9 / 10, 49 / 50, 3 / 10000000⟩, 0, 2⟩ := by
  simp only [scaled, historical]
  norm_num

/-- The rate of the scaled recipe. -/
theorem rate_scaled (c : ℝ) (p : Parameters) (t : ℕ) : rate (scaled c p) t = c * rate p t := by
  simp only [rate, scaled, mul_assoc]

/-- `s'` is `s` at the scale `c`: the points are multiplied by `c`, the second moments
divided by `c ^ 2`, the averaging total multiplied by `c ^ ratePower`, and the maximal rates
agree at the scale `c` once a nonnegative rate has entered them. -/
def IsScaled (c : ℝ) (p : Parameters) (s s' : State ι) : Prop :=
  (∀ i, s'.evaluation i = c * s.evaluation i) ∧ (∀ i, s'.base i = c * s.base i) ∧
    (∀ i, s'.history.second i = s.history.second i / c ^ 2) ∧
    s'.history.total = c ^ p.ratePower * s.history.total ∧
    ∀ r : ℝ, 0 ≤ r → max (c * r) s'.history.maximumRate = c * max r s.history.maximumRate

/-- The first states of the two runs: the sentinel `-1` of the maximal rate does not scale,
and no nonnegative rate sees it. -/
theorem isScaled_initial {c : ℝ} (hc : 0 < c) (p : Parameters) (point : ι → ℝ) :
    IsScaled c p (initial point) (initial fun i => c * point i) := by
  refine ⟨fun _ => rfl, fun _ => rfl, fun _ => by simp [initial], by simp [initial], ?_⟩
  intro r hr
  have h : (0 : ℝ) ≤ c * r := mul_nonneg hc.le hr
  simp only [initial]
  rw [max_eq_left (by linarith), max_eq_left (by linarith)]

/-- The query point of a scaled state. -/
theorem query_scaled {c : ℝ} {p : Parameters} {s s' : State ι} (h : IsScaled c p s s')
    (i : ι) : query (scaled c p) s' i = c * query p s i := by
  simp only [query, scaled, h.1 i, h.2.1 i]
  ring

example (i : Fin 1) : query (scaled 2 historical) (initial fun _ : Fin 1 => 2 * 3) i =
    2 * query historical (initial fun _ : Fin 1 => 3) i :=
  query_scaled (isScaled_initial two_pos historical fun _ : Fin 1 => 3) i

/-- **One step at the scale `c`**, `review/flat_spectrum.md`, item RR3: with the gradient
divided by `c`, the step of the scaled recipe from the scaled state is the scaled step. -/
theorem step_isScaled {c : ℝ} (hc : 0 < c) {p : Parameters} (hlr : 0 ≤ p.adam.learningRate)
    (t : ℕ) {s s' : State ι} (h : IsScaled c p s s') (g : ι → ℝ) :
    IsScaled c p (step p t s g) (step (scaled c p) t s' fun i => g i / c) := by
  obtain ⟨hx, hz, hv, hT, hM⟩ := h
  have hc0 : c ≠ 0 := hc.ne'
  have hk : 0 < c ^ p.ratePower := Real.rpow_pos_of_pos hc _
  have hN : nextMaximum (scaled c p) t s'.history = c * nextMaximum p t s.history := by
    rw [nextMaximum, rate_scaled, hM _ (rate_nonneg p t hlr), nextMaximum]
  have hW : weight (scaled c p) t s'.history = c ^ p.ratePower * weight p t s.history := by
    rw [weight, hN, Real.mul_rpow hc.le (maximum_nonneg p t s.history hlr), weight]
    simp only [scaled]
    ring
  have hC : coefficient (scaled c p) t s'.history = coefficient p t s.history := by
    rw [coefficient, hW, hT, ← mul_add, mul_div_mul_left _ _ hk.ne', coefficient]
  have hS : ∀ i, nextSecond (scaled c p) s'.history (fun i => g i / c) i =
      nextSecond p s.history g i / c ^ 2 := by
    intro i
    simp only [nextSecond, scaled, hv i]
    field_simp
  have hD : ∀ i, denominator (scaled c p) t s'.history (fun i => g i / c) i =
      denominator p t s.history g i / c := by
    intro i
    have hb : correction (scaled c p).adam.betaTwo t = correction p.adam.betaTwo t := rfl
    rw [denominator, hS i, hb, div_right_comm, Real.sqrt_div' _ (sq_nonneg c),
      Real.sqrt_sq hc.le, denominator]
    simp only [scaled]
    ring
  have hU : ∀ i, direction (scaled c p) t s'.history (query (scaled c p) s')
      (fun i => g i / c) i = direction p t s.history (query p s) g i := by
    intro i
    rw [direction, hD i, query_scaled ⟨hx, hz, hv, hT, hM⟩ i,
      div_div_div_cancel_right₀ hc0, direction]
    simp only [scaled]
    field_simp
  have hB : ∀ i, (step (scaled c p) t s' fun i => g i / c).base i =
      c * (step p t s g).base i := by
    intro i
    simp only [step, hz i, hU i, rate_scaled]
    ring
  refine ⟨fun i => ?_, hB, hS, ?_, fun r hr => ?_⟩
  · have h1 := hB i
    simp only [step] at h1 ⊢
    rw [hC, h1, hx i]
    ring
  · simp only [step, nextHistory, hW, hT]
    ring
  · simp only [step, nextHistory, hN]
    exact (mul_max_of_nonneg _ _ hc.le).symm

example : IsScaled 2 historical
    (step historical 0 (initial fun _ : Fin 1 => 3) fun _ => 1)
    (step (scaled 2 historical) 0 (initial fun _ : Fin 1 => 2 * 3) fun _ => 1 / 2) :=
  step_isScaled two_pos (by norm_num [historical]) 0
    (isScaled_initial two_pos historical fun _ : Fin 1 => 3) fun _ => 1

/-- **The runs are the same at the scale `c`**, `review/flat_spectrum.md`, item RR3: for an
oracle homogeneous of degree `-1` at the scale `c`, the run of the scaled recipe from the
scaled point is the scaled run. -/
theorem run_isScaled {c : ℝ} (hc : 0 < c) {p : Parameters} (hlr : 0 ≤ p.adam.learningRate)
    (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (hg : ∀ t y i, oracle t (fun j => c * y j) i = oracle t y i / c) (point : ι → ℝ) (t : ℕ) :
    IsScaled c p (run p oracle point t) (run (scaled c p) oracle (fun i => c * point i) t) := by
  induction t with
  | zero => exact isScaled_initial hc p point
  | succ t ih =>
    have hq : query (scaled c p) (run (scaled c p) oracle (fun i => c * point i) t) =
        fun j => c * query p (run p oracle point t) j := funext fun j => query_scaled ih j
    have ho : oracle t (query (scaled c p) (run (scaled c p) oracle (fun i => c * point i) t)) =
        fun i => oracle t (query p (run p oracle point t)) i / c := by
      rw [hq]
      exact funext fun i => hg t _ i
    simp only [run]
    rw [ho]
    exact step_isScaled hc hlr t ih _

/-- The evaluation, base and query points of the scaled run are those of the run, times `c`. -/
theorem run_scaled {c : ℝ} (hc : 0 < c) {p : Parameters} (hlr : 0 ≤ p.adam.learningRate)
    (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (hg : ∀ t y i, oracle t (fun j => c * y j) i = oracle t y i / c) (point : ι → ℝ) (t : ℕ)
    (i : ι) :
    (run (scaled c p) oracle (fun i => c * point i) t).evaluation i =
        c * (run p oracle point t).evaluation i ∧
      (run (scaled c p) oracle (fun i => c * point i) t).base i =
        c * (run p oracle point t).base i ∧
      query (scaled c p) (run (scaled c p) oracle (fun i => c * point i) t) i =
        c * query p (run p oracle point t) i :=
  have h := run_isScaled hc hlr oracle hg point t
  ⟨h.1 i, h.2.1 i, query_scaled h i⟩

/-- The oracle `y ↦ 1 / y`, the gradient of `log |y|`, is homogeneous of degree `-1`: the
historical recipe and the recipe with the rate `0.001` and the decay `30` run alike. -/
example (t : ℕ) (i : Fin 1) :
    (run (scaled (1 / 30) historical) (fun _ y j => 1 / y j) (fun _ => 1 / 30 * 3)
      t).evaluation i = 1 / 30 * (run historical (fun _ y j => 1 / y j) (fun _ => 3)
      t).evaluation i :=
  (run_scaled (by norm_num) (by norm_num [historical]) (fun _ y j => 1 / y j)
    (fun _ y j => by rw [div_div, mul_comm]) (fun _ : Fin 1 => 3)
    t i).1

end Transformer.ScheduleFree.AdamW

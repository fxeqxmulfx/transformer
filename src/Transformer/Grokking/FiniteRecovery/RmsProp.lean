module

public import Transformer.Grokking.FiniteRecovery.DecoupledDecay
public import Transformer.Grokking.FiniteRecovery.AdamWTrajectory

/-! Bias-corrected uncentered RMSPropW, in real arithmetic. The implementation
in `experiments/argand/src/engine/rmsprop.rs` stores one mean-square buffer.
All coordinate directions use the current clipped gradient. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.RmsProp

open Transformer.AMSGradW.Kernite

variable {ι : Type*} [Fintype ι]

structure State (ι : Type*) where
  weights : ι → ℝ
  second : ι → ℝ

noncomputable def meanSquare (v g : ℝ) : ℝ := (95/100)*v+(5/100)*g^2

/-- The native denominator uses the same positive epsilon and second-moment
bias correction as the two-moment control. -/
noncomputable def direction (t : ℕ) (v g : ℝ) : ℝ := g / AdamW.scale t (meanSquare v g)

noncomputable def run (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) :
    ℕ → State ι
  | 0 => ⟨initial,0⟩
  | t+1 =>
    let s := run decay rate grad initial t
    let g := clip (grad t s.weights)
    ⟨fun i => (1-rate t*decay)*s.weights i-rate t*direction t (s.second i) (g i),
     fun i => meanSquare (s.second i) (g i)⟩

private theorem bias_pos (t : ℕ) : 0 < Real.sqrt (1-(95/100 : ℝ)^(t+1)) :=
  Real.sqrt_pos.2 (sub_pos.2 (pow_lt_one₀ (by norm_num) (by norm_num) (Nat.succ_ne_zero t)))

/-- The current gradient contributes 5% of its square to the denominator.
Five bounds the direction uniformly, independently of dimension and epsilon. -/
theorem direction_bound (t : ℕ) {v : ℝ} (hv : 0 ≤ v) (g : ℝ) : |direction t v g| ≤ 5 := by
  let q := meanSquare v g
  have hq : 0 ≤ q := by dsimp [q,meanSquare]; positivity
  have hg : |g| ≤ 5*Real.sqrt q := by
    apply (sq_le_sq₀ (abs_nonneg _) (by positivity)).mp
    rw [sq_abs, mul_pow, Real.sq_sqrt hq]
    dsimp [q,meanSquare]
    nlinarith [sq_nonneg g]
  have hb := bias_pos t
  have hc : Real.sqrt (1-(95/100 : ℝ)^(t+1)) ≤ 1 := Real.sqrt_le_one.2 (sub_le_self _ (by positivity))
  have hs : Real.sqrt q ≤ AdamW.scale t q := (le_div_self (Real.sqrt_nonneg _) hb hc).trans
    (le_add_of_nonneg_right (by norm_num))
  have hp : 0 < AdamW.scale t q := by unfold AdamW.scale; positivity
  change |g / AdamW.scale t q| ≤ _
  rw [abs_div,abs_of_pos hp,div_le_iff₀ hp]
  exact hg.trans (mul_le_mul_of_nonneg_left hs (by norm_num))

variable (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)

/-- The corrected mean square is at most one because the common Euclidean clip
bounds each coordinate of each gradient. -/
theorem second_bounds (t : ℕ) (i : ι) :
    0 ≤ (run decay rate grad initial t).second i ∧ (run decay rate grad initial t).second i ≤ 1-(95/100 : ℝ)^t := by
  induction t with
  | zero => norm_num [run]
  | succ t ih =>
    let s := run decay rate grad initial t
    let g := clip (grad t s.weights) i
    have hg : g^2 ≤ 1 := by
      simpa only [sq_abs,one_pow] using (sq_le_sq₀ (abs_nonneg g) zero_le_one).2 (abs_clip_le _ i)
    change 0 ≤ meanSquare (s.second i) g ∧ meanSquare (s.second i) g ≤ _
    change 0 ≤ s.second i ∧ s.second i ≤ _ at ih
    rw [meanSquare,pow_succ (95/100 : ℝ) t]
    constructor <;> nlinarith [ih.1,ih.2,sq_nonneg g]

/-- Unlike a stale first moment, this direction makes an acute angle with each
nonzero observed gradient. Its inner product per coordinate has a lower bound. -/
theorem gradient_alignment (t : ℕ) (i : ι) :
    let s := run decay rate grad initial t
    let g := clip (grad t s.weights) i
    g^2/(1+1/100000000) ≤ g*direction t (s.second i) g := by
  intro s g
  have hv := (second_bounds decay rate grad initial (t+1) i).2
  change meanSquare (s.second i) g ≤ _ at hv
  have hb := bias_pos t
  have hs : AdamW.scale t (meanSquare (s.second i) g) ≤ 1+1/100000000 :=
    add_le_add ((div_le_one hb).2 (Real.sqrt_le_sqrt hv)) le_rfl
  have hp : 0 < AdamW.scale t (meanSquare (s.second i) g) := by unfold AdamW.scale; positivity
  rw [direction, ← mul_div_assoc, ← sq]
  exact div_le_div_of_nonneg_left (sq_nonneg _) hp hs

/-- Every complete weight vector stays in a common box from arbitrary starts. -/
theorem weight_bound (hdc : 0 < decay) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1) (i : ι) (n : ℕ) :
    |(run decay rate grad initial n).weights i| ≤ max ‖initial‖ (5/decay) := by
  let s := run decay rate grad initial
  let d t i := direction t ((s t).second i) (clip (grad t (s t).weights) i)
  exact DecoupledDecay.coordinate_bound (fun t => (s t).weights) d rate hdc hr (fun _ _ => rfl)
    (fun t j => direction_bound t (second_bounds decay rate grad initial t j).1 _) i n

example (i : Fin 3) (t : ℕ) := gradient_alignment (1/5) (fun _ => 1/1000)
  (fun _ w => w) (fun _ => 2) t i

example (i : Fin 3) (t : ℕ) := weight_bound (1/5) (fun _ => 1/1000)
  (fun _ w => w) (fun _ => 2) (by norm_num) (fun _ => by norm_num) i t

end Transformer.Grokking.FiniteRecovery.RmsProp

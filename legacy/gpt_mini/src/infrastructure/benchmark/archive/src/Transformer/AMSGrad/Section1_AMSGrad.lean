/-
# AMSGrad — the algorithm, the regret, and Theorem A

§1 and §2 of arXiv:1904.03590v4 (Tran, Le): Algorithm 1 (AMSGrad, Reddi et
al.), the regret it is measured by, and Theorem A, Reddi et al.'s convergence
theorem, which the paper calls problematic.

**What the source says and what is carried here.**

* Vectors are `Vec d = Fin d → ℝ`, and every operation of the algorithm on them
  is coordinatewise: `g²`, `m / √v̂`, `max`.  `grad f x` is the vector of
  partial derivatives, `∇f(x)`.

* `Π_{F, √V̂}(y) = argmin_{x ∈ F} ‖V̂^{1/4}(x - y)‖` is carried as a function
  `proj w y` of the weight vector `w = √v̂` and the point `y`, satisfying
  `IsWeightedProj F`: it lands in `F` and minimizes `Σᵢ wᵢ (xᵢ - yᵢ)²` there.
  The source takes `V̂` positive definite; a coordinate of `v̂` that is `0` has
  seen only zero gradients, and `IsWeightedProj` then allows any minimizer.
  Clamping, `boxProj`, is such a projection onto a box.

* The two algorithms of the paper differ only in the update of `v̂`; a run is
  `Setup.state` for a `Rule`, the map `(t, v̂_{t-1}, v_t) ↦ v̂_t`, and
  `amsgradRule` is Algorithm 1.  `x_t` is `S.x R t` for `t ≥ 1`.

* Theorem A contains `1/(1 - γ)` with `γ = β₁/√β₂`, under the hypothesis
  `γ ≤ 1`; at `γ = 1` the bound is undefined.  It is stated for `γ < 1` and
  `0 < β₂ < 1`, which `γ` and `√(1 - β₂)` presuppose.

* The regret is `R(T) = Σ_{t=1}^T [f_t(x_t) - f_t(x*)]` with `x*` a minimizer
  of `Σ f_t` over `F`.  The proofs use only `x* ∈ F`, and every bound is stated
  for every `x* ∈ F`, which contains the source's.

Source: arXiv:1904.03590v4, §1, Algorithm 1, Theorem A; §2.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Finset

namespace Transformer
namespace AMSGrad

/-- A point of `ℝ^d`, with coordinatewise operations.  arXiv:1904.03590v4, §2. -/
abbrev Vec (d : ℕ) := Fin d → ℝ

variable {d : ℕ}

/-- `∇f(x)`, the vector of partial derivatives.  arXiv:1904.03590v4, §2. -/
noncomputable def grad (f : Vec d → ℝ) (x : Vec d) : Vec d :=
  fun i => fderiv ℝ f x (Pi.single i 1)

/-- The coordinatewise square root `√v`.  arXiv:1904.03590v4, §2. -/
noncomputable def vsqrt (v : Vec d) : Vec d := fun i => Real.sqrt (v i)

/-- `proj w y` is a point of `F` minimizing `Σᵢ wᵢ (xᵢ - yᵢ)² = ‖W^{1/2}(x - y)‖²`
over `F`, for every weight `w ≥ 0`: the projection `Π_{F, W}` of
arXiv:1904.03590v4, §2, with `W = diag(w)`. -/
structure IsWeightedProj (F : Set (Vec d)) (proj : Vec d → Vec d → Vec d) : Prop where
  /-- The projection lands in `F`. -/
  mem : ∀ w y, (∀ i, 0 ≤ w i) → proj w y ∈ F
  /-- The projection minimizes the weighted distance over `F`. -/
  le : ∀ w y, (∀ i, 0 ≤ w i) → ∀ z ∈ F,
    ∑ i, w i * (proj w y i - y i) ^ 2 ≤ ∑ i, w i * (z i - y i) ^ 2

/-- Clamping each coordinate to `[a, b]`.  arXiv:1904.03590v4, §3, Example 3.2. -/
def boxProj (a b : ℝ) (y : Vec d) : Vec d := fun i => max a (min b (y i))

/-- Clamping is the weighted projection onto the box `[a, b]^d`, whatever the
weight. -/
theorem isWeightedProj_boxProj {a b : ℝ} (hab : a ≤ b) :
    IsWeightedProj (Set.Icc (fun _ : Fin d => a) (fun _ => b)) (fun _ => boxProj a b) where
  mem w y _ := ⟨fun i => le_max_left _ _, fun i => max_le hab (min_le_left _ _)⟩
  le w y hw z hz := by
    refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_ (hw i)
    have h1 := hz.1 i
    have h2 := hz.2 i
    simp only [boxProj]
    rcases le_total (y i) a with h | h
    · rw [min_eq_right (h.trans hab), max_eq_left h]; nlinarith
    · rcases le_total (y i) b with h' | h'
      · rw [min_eq_right h', max_eq_right h]; simp; positivity
      · rw [min_eq_left h', max_eq_right hab]; nlinarith

/-- A rule `(t, v̂_{t-1}, v_t) ↦ v̂_t`: what distinguishes AMSGrad from AdamX. -/
abbrev Rule (d : ℕ) := ℕ → Vec d → Vec d → Vec d

/-- AMSGrad's `v̂_t = max(v̂_{t-1}, v_t)`.  arXiv:1904.03590v4, Algorithm 1. -/
def amsgradRule : Rule d := fun _ a b => a ⊔ b

/-- The data of a run: projection, costs `f_t`, step sizes `α_t`, `β_{1,t}`,
`β₂`, and the starting point `x₁`.  arXiv:1904.03590v4, Algorithm 1. -/
structure Setup (d : ℕ) where
  /-- The projection `(w, y) ↦ Π_{F, diag w}(y)`. -/
  proj : Vec d → Vec d → Vec d
  /-- The cost functions `f_t`. -/
  f : ℕ → Vec d → ℝ
  /-- The step sizes `α_t`. -/
  α : ℕ → ℝ
  /-- The first-moment parameters `β_{1,t}`. -/
  β₁ : ℕ → ℝ
  /-- The second-moment parameter `β₂`. -/
  β₂ : ℝ
  /-- The starting point `x₁`. -/
  x₁ : Vec d

/-- The state after step `t`: `x_{t+1}`, `m_t`, `v_t`, `v̂_t`. -/
structure State (d : ℕ) where
  /-- The next iterate `x_{t+1}`. -/
  x : Vec d
  /-- The first moment `m_t`. -/
  m : Vec d
  /-- The second moment `v_t`. -/
  v : Vec d
  /-- The corrected second moment `v̂_t`. -/
  vhat : Vec d

namespace Setup

/-- Step `t` of the loop body of Algorithm 1, with `v̂_t` given by the rule `R`.
arXiv:1904.03590v4, Algorithm 1. -/
noncomputable def step (S : Setup d) (R : Rule d) (t : ℕ) (s : State d) : State d :=
  let g := grad (S.f t) s.x
  let m := S.β₁ t • s.m + (1 - S.β₁ t) • g
  let v := S.β₂ • s.v + (1 - S.β₂) • g ^ 2
  let vhat := R t s.vhat v
  ⟨S.proj (vsqrt vhat) (s.x - S.α t • (m / vsqrt vhat)), m, v, vhat⟩

/-- The run: `m₀ = v₀ = v̂₀ = 0`, then steps `1, 2, …`.
arXiv:1904.03590v4, Algorithm 1. -/
noncomputable def state (S : Setup d) (R : Rule d) : ℕ → State d
  | 0 => ⟨S.x₁, 0, 0, 0⟩
  | t + 1 => S.step R (t + 1) (S.state R t)

/-- The iterate `x_t`, `t ≥ 1`. -/
noncomputable def x (S : Setup d) (R : Rule d) (t : ℕ) : Vec d := (S.state R (t - 1)).x
/-- The first moment `m_t`. -/
noncomputable def m (S : Setup d) (R : Rule d) (t : ℕ) : Vec d := (S.state R t).m
/-- The second moment `v_t`. -/
noncomputable def v (S : Setup d) (R : Rule d) (t : ℕ) : Vec d := (S.state R t).v
/-- The corrected second moment `v̂_t`. -/
noncomputable def vhat (S : Setup d) (R : Rule d) (t : ℕ) : Vec d := (S.state R t).vhat
/-- The gradient `g_t = ∇f_t(x_t)`. -/
noncomputable def g (S : Setup d) (R : Rule d) (t : ℕ) : Vec d := grad (S.f t) (S.x R t)

/-- The regret `R(T) = Σ_{t=1}^T [f_t(x_t) - f_t(x*)]`.  arXiv:1904.03590v4, §2. -/
noncomputable def regret (S : Setup d) (R : Rule d) (xstar : Vec d) (T : ℕ) : ℝ :=
  ∑ t ∈ Icc 1 T, (S.f t (S.x R t) - S.f t xstar)

/-- `‖g_{1:T,i}‖₂`, the `ℓ²` norm of the `i`-th coordinates of `g_1, …, g_T`. -/
noncomputable def gnorm (S : Setup d) (R : Rule d) (T : ℕ) (i : Fin d) : ℝ :=
  Real.sqrt (∑ t ∈ Icc 1 T, S.g R t i ^ 2)

end Setup

end AMSGrad
end Transformer

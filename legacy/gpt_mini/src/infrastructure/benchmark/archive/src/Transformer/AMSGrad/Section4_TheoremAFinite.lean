import Transformer.AMSGrad.Section4_TheoremAProof

/-
# AMSGrad — Theorem A under the paper's finite-horizon assumptions

Theorem A of arXiv:1904.03590v4, §1, assumes properties of the losses,
steps, and first-moment coefficients for `1 ≤ t ≤ T`. We extend those data
after `T` without changing any iterate or statistic used in the bound and
apply the schedule-free proof in `Section4_TheoremAProof`.
-/

open Finset
namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- The online convex setup of arXiv:1904.03590v4, §1–2, with
loss assumptions only through the chosen horizon `T`. -/
structure IsOnlineConvexHorizon (S : Setup d) (F : Set (Vec d)) (D G : ℝ) (T : ℕ) : Prop where
  proj : IsWeightedProj F S.proj
  convex : Convex ℝ F
  x₁_mem : S.x₁ ∈ F
  convexOn : ∀ t, 1 ≤ t → t ≤ T → ConvexOn ℝ Set.univ (S.f t)
  differentiable : ∀ t, 1 ≤ t → t ≤ T → Differentiable ℝ (S.f t)
  diam : ∀ x ∈ F, ∀ y ∈ F, ∀ i, |x i - y i| ≤ D
  grad_le : ∀ t, 1 ≤ t → t ≤ T → ∀ x ∈ F, ∀ i, |grad (S.f t) x i| ≤ G

/-- Continue the loss sequence by zero losses and use the prescribed step
schedule after `T`. This is a proof device for arXiv:1904.03590v4, §1,
Theorem A. -/
private noncomputable def finiteSetup (S : Setup d) (α : ℝ) (T : ℕ) : Setup d :=
  { S with
    f := fun t => if 1 ≤ t ∧ t ≤ T then S.f t else fun _ => 0
    α := fun t => α / Real.sqrt t
    β₁ := fun t => if t ≤ T then S.β₁ t else S.β₁ 1 }

/-- Algorithm 1's state through `T` is unchanged by the continuation.
Source: arXiv:1904.03590v4, Algorithm 1. -/
private theorem finiteSetup_state_eq (S : Setup d) (R : Rule d) {α : ℝ} {T : ℕ}
    (hαt : ∀ t, 1 ≤ t → t ≤ T → S.α t = α / Real.sqrt t)
    (n : ℕ) (hn : n ≤ T) : (finiteSetup S α T).state R n = S.state R n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hn' : n ≤ T := by omega
    change (finiteSetup S α T).step R (n + 1) ((finiteSetup S α T).state R n) =
      S.step R (n + 1) (S.state R n)
    rw [ih hn']
    have hat := hαt (n + 1) (by omega) hn
    simp [Setup.step, finiteSetup, hn, hat]

/-- The continued setup satisfies the global convex-loss assumptions.
Source: arXiv:1904.03590v4, §1–2. -/
private theorem finiteSetup_online {S : Setup d} {F : Set (Vec d)} {D G α : ℝ} {T : ℕ}
    (hT : 1 ≤ T) (hS : IsOnlineConvexHorizon S F D G T) :
    IsOnlineConvex (finiteSetup S α T) F D G := by
  refine {
    proj := hS.proj
    convex := hS.convex
    x₁_mem := hS.x₁_mem
    convexOn := ?_
    differentiable := ?_
    diam := hS.diam
    grad_le := ?_ }
  · intro t
    by_cases ht : 1 ≤ t ∧ t ≤ T
    · simpa [finiteSetup, ht] using hS.convexOn t ht.1 ht.2
    · simp only [finiteSetup, ht, ↓reduceIte]
      exact convexOn_const 0 convex_univ
  · intro t
    by_cases ht : 1 ≤ t ∧ t ≤ T
    · simpa [finiteSetup, ht] using hS.differentiable t ht.1 ht.2
    · simp only [finiteSetup, ht, ↓reduceIte]
      exact differentiable_const 0
  · intro t x hx i
    by_cases ht : 1 ≤ t ∧ t ≤ T
    · simpa [finiteSetup, ht] using hS.grad_le t ht.1 ht.2 x hx i
    · have hG : 0 ≤ G := (abs_nonneg _).trans
        (hS.grad_le 1 le_rfl hT x hx i)
      simp [finiteSetup, ht, grad, hG]

/-- **Theorem A** (Reddi et al., Theorem 4) with the printed three-term
regret bound. It assumes `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`,
convex differentiable losses, and bounded gradients only for `1 ≤ t ≤ T`.
The conclusion holds for every feasible comparator, hence for the paper's
best fixed comparator.

The source writes `β₁/√β₂ ≤ 1` but divides by `1−γ`; we require `γ < 1`
for the finite expression. Nonnegative moment coefficients, `β₁ < 1`,
and `0 < β₂ < 1` are made explicit. The paper's §3 refutation of the old
proof step remains valid; this theorem uses a different projection estimate.

Source: arXiv:1904.03590v4, §1, Theorem A, and §3. -/
theorem theorem_A {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    {T : ℕ} (hT : 1 ≤ T) (hS : IsOnlineConvexHorizon S F D G T)
    {α : ℝ} (hα : 0 < α)
    (hαt : ∀ t, 1 ≤ t → t ≤ T → S.α t = α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → t ≤ T → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1)
    (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  let S' := finiteSetup S α T
  have hβ₁eq : S'.β₁ 1 = S.β₁ 1 := by simp [S', finiteSetup, hT]
  have hβ₁global : ∀ t, 1 ≤ t → 0 ≤ S'.β₁ t ∧ S'.β₁ t ≤ S'.β₁ 1 := by
    intro t ht
    by_cases htt : t ≤ T
    · simpa [S', finiteSetup, htt, hβ₁eq] using hβ₁ t ht htt
    · simpa [S', finiteSetup, htt, hβ₁eq] using hβ₁ 1 le_rfl hT
  have hglobal := theorem_A_global (S := S') (F := F) (D := D) (G := G) (α := α)
    (finiteSetup_online hT hS) hα (by rfl) hβ₁global
    (by simpa [hβ₁eq] using hβ₁') hβ₂ hβ₂'
    (by simpa [S', finiteSetup, hβ₁eq] using hγ) hT hxstar
  have hs : ∀ n ≤ T, S'.state amsgradRule n = S.state amsgradRule n :=
    fun n hn => finiteSetup_state_eq S amsgradRule hαt n hn
  have hx : ∀ t, t ≤ T + 1 → S'.x amsgradRule t = S.x amsgradRule t := by
    intro t ht
    unfold Setup.x
    exact congrArg State.x (hs (t - 1) (by omega))
  have hv : ∀ n ≤ T, S'.vhat amsgradRule n = S.vhat amsgradRule n := by
    intro n hn
    exact congrArg State.vhat (hs n hn)
  have hg : ∀ t, 1 ≤ t → t ≤ T → S'.g amsgradRule t = S.g amsgradRule t := by
    intro t ht ht'
    unfold Setup.g
    rw [hx t (by omega)]
    simp [S', finiteSetup, ht, ht']
  have hr : S'.regret amsgradRule xstar T = S.regret amsgradRule xstar T := by
    unfold Setup.regret
    apply sum_congr rfl
    intro t ht
    have htt := (mem_Icc.mp ht)
    rw [hx t (by omega)]
    simp [S', finiteSetup, htt.1, htt.2]
  have hn : ∀ i, S'.gnorm amsgradRule T i = S.gnorm amsgradRule T i := by
    intro i
    unfold Setup.gnorm
    congr 1
    apply sum_congr rfl
    intro t ht
    rw [hg t (mem_Icc.mp ht).1 (mem_Icc.mp ht).2]
  have hq : ∀ i, ∑ t ∈ Icc 1 T, S'.β₁ t * Real.sqrt (S'.vhat amsgradRule t i) / S'.α t =
      ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    intro i
    apply sum_congr rfl
    intro t ht
    have htt := (mem_Icc.mp ht)
    rw [hv t htt.2]
    simp [S', finiteSetup, htt.2, (hαt t htt.1 htt.2)]
  rw [hr, hβ₁eq] at hglobal
  simp only [S', hv T le_rfl, hq, hn] at hglobal
  exact hglobal

/-- The hypotheses of `theorem_A` and of the finite-horizon continuation
lemmas hold for zero losses on `[-1,1]`, `α = 1`, `β₁,t = 0`,
`β₂ = 1/2`, and `T = 1`. -/
example :=
  theorem_A (S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t)
      (fun _ => 0) (1 / 2))
    (F := Set.Icc (fun _ => -1) (fun _ => 1)) (D := 2) (G := 0)
    (α := 1) (T := 1) (xstar := (0 : Vec 1))
    le_rfl
    (by
      let h := isOnlineConvex_zero (d := 1)
        (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
      exact {
        proj := h.proj
        convex := h.convex
        x₁_mem := h.x₁_mem
        convexOn := fun t _ _ => h.convexOn t
        differentiable := fun t _ _ => h.differentiable t
        diam := h.diam
        grad_le := fun t _ _ => h.grad_le t })
    one_pos (fun _ _ _ => rfl)
    (fun _ _ _ => ⟨le_rfl, le_rfl⟩)
    (by simp [zeroSetup])
    (by norm_num [zeroSetup]) (by norm_num [zeroSetup])
    (by simp [zeroSetup])
    ⟨fun _ => by norm_num, fun _ => by norm_num⟩

end AMSGrad
end Transformer

module

public import Transformer.Grokking.FiniteRecovery.Basic
public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Tactic

/-! The exact probability threshold for simultaneous agreement on the hidden
inputs of a finite problem. A fiber consists of distinct admissible functions
giving identical observed labels. A randomized learner has just one output
law on that fiber. Both the upper bound and its attainment are finite sums. -/

@[expose] public section
open scoped Classical

namespace Transformer.Grokking.FiniteRecovery

open Finset Set

variable {X A : Type*} [Fintype X] [Fintype A]

/-- Real probability of getting every unobserved input right under a finite
output law p. The prediction on the observed inputs is immaterial here. -/
noncomputable def hiddenMass (S : Finset X) (p : (X → A) → ℝ) (f : X → A) : ℝ :=
  ∑ g, if EqOn g f (S : Set X)ᶜ then p g else 0

omit [Fintype X] [Fintype A] in
private theorem eq_of_same_hidden {S : Finset X} {f h g : X → A}
    (ho : EqOn f h (S : Set X)) (hf : EqOn g f (S : Set X)ᶜ)
    (hh : EqOn g h (S : Set X)ᶜ) : f = h := by
  funext x
  by_cases hx : x ∈ S
  · exact ho hx
  · exact (hf hx).symm.trans (hh hx)

/-- Among m targets with the same observations, some target has success
probability at most 1/m on the hidden inputs, for any randomized learner. -/
theorem hidden_fiber_upper (S : Finset X) (V : Finset (X → A)) (hV : V.Nonempty)
    (ho : ∀ f ∈ V, ∀ h ∈ V, EqOn f h (S : Set X)) (p : (X → A) → ℝ)
    (hp : ∀ g, 0 ≤ p g) (ht : ∑ g, p g = 1) :
    ∃ f ∈ V, hiddenMass S p f ≤ 1 / (V.card : ℝ) := by
  classical
  have hcol (g : X → A) : (∑ f ∈ V, if EqOn g f (S : Set X)ᶜ then p g else 0) ≤ p g := by
    by_cases h : ∃ f ∈ V, EqOn g f (S : Set X)ᶜ
    · obtain ⟨f,hf,hg⟩ := h
      rw [sum_eq_single f]
      · simp [hg]
      · intro j hj hne
        exact ite_eq_right (fun hw => hne (eq_of_same_hidden (ho _ hj _ hf) hw hg))
      · simp [hf]
    · have hn (f) (hf : f ∈ V) : ¬ EqOn g f (S : Set X)ᶜ := fun hw => h ⟨f,hf,hw⟩
      rw [sum_eq_zero fun f hf => ite_eq_right (hn f hf)]
      exact hp g
  have hs : (∑ f ∈ V, hiddenMass S p f) ≤ 1 := by
    rw [hiddenMass_sum, ← ht]
    exact sum_le_sum (fun g _ => hcol g)
  by_contra! hn
  have hlt := sum_lt_sum_of_nonempty hV (fun f hf => hn f hf)
  have hc : (V.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (card_pos.mpr hV).ne'
  simpa [nsmul_eq_mul, hc] using hlt.trans_le hs
where
  hiddenMass_sum : (∑ f ∈ V, hiddenMass S p f) =
      ∑ g, ∑ f ∈ V, if EqOn g f (S : Set X)ᶜ then p g else 0 := sum_comm

/-- Uniformly choosing a member of the observation fiber attains the upper
bound for every target in it. This choice uses only the observed labels
and the admissible class, even when more than one completion is possible. -/
theorem hidden_fiber_attained (S : Finset X) (V : Finset (X → A)) (hV : V.Nonempty)
    (ho : ∀ f ∈ V, ∀ h ∈ V, EqOn f h (S : Set X)) :
    ∃ p : (X → A) → ℝ, (∀ g, 0 ≤ p g) ∧ (∑ g, p g = 1) ∧
      ∀ f ∈ V, hiddenMass S p f = 1 / (V.card : ℝ) := by
  classical
  let p := fun g => if g ∈ V then 1 / (V.card : ℝ) else 0
  have hc : (V.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (card_pos.mpr hV).ne'
  refine ⟨p,fun g => by dsimp [p]; split <;> positivity,?_,?_⟩
  · simp [p, nsmul_eq_mul, hc]
  · intro f hf
    unfold hiddenMass
    rw [sum_eq_single f]
    · simp [p,hf,Set.EqOn]
    · intro g _ hne
      by_cases hg : g ∈ V
      · exact ite_eq_right (fun h => hne (eq_of_same_hidden (ho _ hg _ hf) (fun ⦃_⦄ _ => rfl) h))
      · simp [p,hg]
    · simp

/-- A single observation-only output law beats one half for every compatible
target exactly when this nonempty fiber has a single member. -/
theorem hidden_strict_majority_iff (S : Finset X) (V : Finset (X → A)) (hV : V.Nonempty)
    (ho : ∀ f ∈ V, ∀ h ∈ V, EqOn f h (S : Set X)) :
    (∃ p : (X → A) → ℝ, (∀ g, 0 ≤ p g) ∧ (∑ g, p g = 1) ∧
      ∀ f ∈ V, 1/2 < hiddenMass S p f) ↔ V.card = 1 := by
  constructor
  · rintro ⟨p,hp,ht,hs⟩
    obtain ⟨f,hf,hu⟩ := hidden_fiber_upper S V hV ho p hp ht
    by_contra hn
    have hc := card_pos.mpr hV
    have htwo : (2 : ℝ) ≤ V.card := by exact_mod_cast (show 2 ≤ V.card by omega)
    exact ((hs _ hf).trans_le (hu.trans (one_div_le_one_div_of_le (by norm_num) htwo))).false
  · intro hone
    obtain ⟨p,hp,ht,hs⟩ := hidden_fiber_attained S V hV ho
    exact ⟨p,hp,ht,fun f hf => by rw [hs f hf,hone]; norm_num⟩

example (S : Finset (Fin 8)) (f : Fin 8 → Bool) :=
  hidden_strict_majority_iff S {f} (singleton_nonempty _) (by simp [Set.EqOn])

/-- A single observed zero is consistent with copying and with the constant
zero. No randomized prediction can beat one half on the one hidden bit for both. -/
example : ¬ ∃ p : (Bool → Bool) → ℝ, (∀ g, 0 ≤ p g) ∧ (∑ g, p g = 1) ∧
    ∀ f ∈ ({id,fun _ => false} : Finset (Bool → Bool)), 1/2 < hiddenMass {false} p f := by
  let V : Finset (Bool → Bool) := {id,fun _ => false}
  have hne : (id : Bool → Bool) ≠ (fun _ => false) := fun h => Bool.false_ne_true (congrFun h true).symm
  have hagree : ∀ f ∈ V, ∀ g ∈ V, EqOn f g (({false} : Finset Bool) : Set Bool) := by
    intro f hf g hg x hx
    have hx' : x = false := Finset.mem_singleton.mp hx
    subst x
    simp only [V,Finset.mem_insert,Finset.mem_singleton] at hf hg
    rcases hf with rfl | rfl <;> rcases hg with rfl | rfl <;> rfl
  rintro ⟨p,hp,ht,hs⟩
  have hc := (hidden_strict_majority_iff {false} V (Finset.insert_nonempty _ _) hagree).mp
    ⟨p,hp,(Finset.sum_congr (by ext; simp) (fun _ _ => rfl)).trans ht,hs⟩
  simp [V,hne] at hc

end Transformer.Grokking.FiniteRecovery

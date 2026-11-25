module

public import Transformer.Grokking.FlatSpectrum.ProductDepth
public import Transformer.InklingLoop

/-!
# A loop of product layers: passes count as depth

Source: `review/flat_spectrum.md`, *Loops*, item LP1. Inkling 2 Mini runs its prelude once and its
middle blocks several times; every pass starts again from the saved prelude state, and its only
other input is a message read from the previous pass (`InklingLoop.passes`). In NV7's model the
shared core is a product net `L` on the prelude's stream extended by the message, so the message
enters the pass through linear readings, as a gate with fixed weights would put it into the
inputs of the blocks. The state of a pass is what `L` writes on the prelude's stream, the message
is a linear reading `R` of it, and the first pass gets the zero message (`loopState`). At every
position this is `InklingLoop.passes` with `L` as the core (`passes_loopCore`), so once two
consecutive passes agree, all later ones do (`loopState_add_of_succ_eq`, by
`InklingLoop.stable_prefix`).

A pass of a core of depth `d` multiplies the order of NV7 by at most `2^d`: the prelude state and
the message enter it only through linear readings (`loopState_mem_orderSpan`). So a linear readout
of a coda of depth `d₂` after `T` passes, behind a prelude of depth `d₀` on the one-hot features of
`k` pairs, is `ψ(∑ᵢ q(xᵢ, yᵢ))` only if `1 + ⌈log₂ k⌉ ≤ d₀ + T d + d₂`, at any width
(`le_passes_of_loop_copies`, through `one_add_clog_le_of_mem_orderSpan`). The bound does not use
that the passes share their weights.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι μ : Type*}

/-- The state after pass `n + 1` of a loop: the shared net `L` reads the prelude state `z` and the
message, and the state of the pass is what `L` writes on the prelude's stream. The message is the
reading `R` of the previous pass's state, and zero on the first pass. -/
def loopState (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ) (z : ι → ℂ) :
    ℕ → ι → ℂ
  | 0 => L (Sum.elim z 0) ∘ Sum.inl
  | n + 1 => L (Sum.elim z (R (loopState L R z n))) ∘ Sum.inl

/-- The core of the loop at every position, for `InklingLoop.passes`: there is no attention, and
the first pass, which has no message, reads the zero message. -/
def loopCore (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) :
    (ℕ → ι → ℂ) → (ℕ → Option (μ → ℂ)) → ℕ → ι → ℂ :=
  fun pre msg t => L (Sum.elim (pre t) ((msg t).getD 0)) ∘ Sum.inl

/-- **The loop is Inkling's**: at every position, `InklingLoop.passes` with the core
`loopCore L` and the message reader `R` is the loop of that position's prelude state. -/
theorem passes_loopCore (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    (pre : ℕ → ι → ℂ) (n t : ℕ) :
    InklingLoop.passes (loopCore L) (fun s t => R (s t)) pre n t = loopState L R (pre t) n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [InklingLoop.passes, loopCore, Option.getD_some, ih, loopState]

/-- **A fixed point stays**: once two consecutive passes agree, every later pass agrees with them.
This is `InklingLoop.stable_prefix` at one position. -/
theorem loopState_add_of_succ_eq {L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ} {R : (ι → ℂ) →ₗ[ℂ] μ → ℂ}
    {z : ι → ℂ} {n : ℕ} (h : loopState L R z (n + 1) = loopState L R z n) (j : ℕ) :
    loopState L R z (n + j) = loopState L R z n := by
  have key := InklingLoop.stable_prefix (loopCore L) (fun s t => R (s t)) (fun _ => z) 0 n
    (fun x y hxy i hi => by
      obtain rfl : i = 0 := Nat.le_zero.1 hi
      simp only [loopCore, Option.getD_some, hxy 0 le_rfl])
    (fun i hi => by
      obtain rfl : i = 0 := Nat.le_zero.1 hi
      rw [passes_loopCore, passes_loopCore]
      exact h) j 0 le_rfl
  rwa [passes_loopCore, passes_loopCore] at key

/-- A core that writes nothing keeps the prelude state. -/
lemma loopState_id (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ) (z : ι → ℂ) (n : ℕ) :
    loopState (id : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) R z n = z := by
  cases n <;> rfl

example (z : Unit → ℂ) (j : ℕ) :
    loopState (id : (Unit ⊕ Unit → ℂ) → Unit ⊕ Unit → ℂ) 0 z (1 + j) =
      loopState (id : (Unit ⊕ Unit → ℂ) → Unit ⊕ Unit → ℂ) 0 z 1 :=
  loopState_add_of_succ_eq (by rw [loopState_id, loopState_id]) j

variable {κ α β : Type}

/-- **A pass multiplies the order by at most `2^d`**: if the coordinates of the prelude state have
order at most `r`, those of the state after pass `n + 1` of a core of depth `d` have order at most
`2^{d(n+1)} r`. -/
theorem loopState_mem_orderSpan [Fintype ι] [Fintype μ] {m d r : ℕ}
    {L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ} (hL : IsProductNet m d L) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    {Z : (κ → α) → (κ → β) → ι → ℂ} (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (n : ℕ)
    (c : ι) : (fun x y => loopState L R (Z x y) n c) ∈ orderSpan (2 ^ (d * (n + 1)) * r) := by
  induction n generalizing c with
  | zero =>
    have hin : ∀ c', (fun x y => Sum.elim (Z x y) (0 : μ → ℂ) c') ∈ orderSpan r := by
      rintro (c' | j)
      · exact hZ c'
      · exact (orderSpan r).zero_mem
    rw [zero_add, mul_one]
    exact hL.order (Z := fun x y => Sum.elim (Z x y) (0 : μ → ℂ)) hin (Sum.inl c)
  | succ n ih =>
    have hin : ∀ c', (fun x y => Sum.elim (Z x y) (R (loopState L R (Z x y) n)) c') ∈
        orderSpan (2 ^ (d * (n + 1)) * r) := by
      rintro (c' | j)
      · exact orderSpan_mono (Nat.le_mul_of_pos_left r (by positivity)) (hZ c')
      · exact linear_mem_orderSpan (Z := fun x y => loopState L R (Z x y) n) ih
          ((LinearMap.proj j).comp R)
    rw [show 2 ^ (d * (n + 1 + 1)) * r = 2 ^ d * (2 ^ (d * (n + 1)) * r) by ring]
    exact hL.order (Z := fun x y => Sum.elim (Z x y) (R (loopState L R (Z x y) n))) hin
      (Sum.inl c)

example := loopState_mem_orderSpan (κ := Unit) (α := Bool) (β := Bool) (r := 0)
  (IsProductNet.id_net (ι := Unit ⊕ Unit) 0 1) 0 (Z := fun _ _ => 0)
  (fun _ => (orderSpan 0).zero_mem) 2 ()

/-- **From the order to the depth**, as in `le_depth_of_productNet_copies`: if `ψ(∑ᵢ q(xᵢ, yᵢ))`
has order at most `2^D` and `ψ ∘ q` has a nonzero mixed difference, then
`1 + ⌈log₂ k⌉ ≤ D`. -/
theorem one_add_clog_le_of_mem_orderSpan {p k : ℕ} [NeZero p] (hk : 0 < k) (q : α → β → ZMod p)
    (hq : ∃ a₀ a₁ b₀ b₁, ZMod.stdAddChar (q a₀ b₀) - ZMod.stdAddChar (q a₁ b₀) -
      ZMod.stdAddChar (q a₀ b₁) + ZMod.stdAddChar (q a₁ b₁) ≠ 0) {D : ℕ}
    (h : (fun (x : Fin k → α) (y : Fin k → β) => ZMod.stdAddChar (∑ i, q (x i) (y i))) ∈
      orderSpan (2 ^ D)) :
    1 + Nat.clog 2 k ≤ D := by
  obtain ⟨a₀, a₁, b₀, b₁, hne⟩ := hq
  have h2 : 2 * k ≤ 2 ^ D := by
    by_contra hlt
    have h0 := cornerSum_eq_zero (fun _ => a₀) (fun _ => a₁) (fun _ => b₀) (fun _ => b₁)
      (by rw [Fintype.card_fin]; omega) h
    have e : (fun (x : Fin k → α) (y : Fin k → β) => ZMod.stdAddChar (∑ i, q (x i) (y i))) =
        fun x y => ∏ i, (fun (_ : Fin k) a b => ZMod.stdAddChar (q a b)) i (x i) (y i) :=
      funext fun x => funext fun y => stdAddChar_sum _ _ _
    rw [e,cornerSum_prod (φ := fun (_ : Fin k) a b => ZMod.stdAddChar (q a b))] at h0
    exact prod_ne_zero_iff.2 (fun i _ => hne) h0
  obtain _ | D := D
  · rw [pow_zero] at h2
    omega
  · have hk' : k ≤ 2 ^ D := by rw [pow_succ] at h2; omega
    have := Nat.clog_le_of_le_pow hk'
    omega

example : 1 + Nat.clog 2 2 ≤ 1 + Nat.clog 2 2 := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 2) (by norm_num) fun a b => a ^ 3 + a * b
  have h := readout_mem_orderSpan hF (LinearMap.proj (Sum.inr root))
  simp only [LinearMap.proj_apply, hval] at h
  exact one_add_clog_le_of_mem_orderSpan (by norm_num) (fun a b => a ^ 3 + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩ h

variable [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]

/-- **Passes count as depth**: if a linear readout of a coda of depth `d₂` after `n + 1` passes
of a core of depth `d₁`, behind a prelude of depth `d₀` on the one-hot features of `k` pairs and a
zero scratch, is `ψ(∑ᵢ q(xᵢ, yᵢ))`, and `ψ ∘ q` has a nonzero mixed difference, then
`1 + ⌈log₂ k⌉ ≤ d₀ + (n + 1) d₁ + d₂`, at any width. -/
theorem le_passes_of_loop_copies {p k : ℕ} [NeZero p] (hk : 0 < k) (q : α → β → ZMod p)
    (hq : ∃ a₀ a₁ b₀ b₁, ZMod.stdAddChar (q a₀ b₀) - ZMod.stdAddChar (q a₁ b₀) -
      ZMod.stdAddChar (q a₀ b₁) + ZMod.stdAddChar (q a₁ b₁) ≠ 0)
    {T ν : Type} [Fintype T] [Fintype ν] {m₀ m₁ m₂ d₀ d₁ d₂ : ℕ}
    {P : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ}
    (hP : IsProductNet m₀ d₀ P)
    {L : ((((Fin k × α) ⊕ (Fin k × β)) ⊕ T) ⊕ ν → ℂ) → (((Fin k × α) ⊕ (Fin k × β)) ⊕ T) ⊕ ν → ℂ}
    (hL : IsProductNet m₁ d₁ L) (R : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) →ₗ[ℂ] ν → ℂ)
    {C : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ}
    (hC : IsProductNet m₂ d₂ C) (ℓ : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (C (loopState L R (P (Sum.elim (pairFeatures x y) 0)) n)) =
      ZMod.stdAddChar (∑ i, q (x i) (y i))) :
    1 + Nat.clog 2 k ≤ d₀ + (n + 1) * d₁ + d₂ := by
  refine one_add_clog_le_of_mem_orderSpan hk q hq ?_
  have h0 := hP.order (Z := fun x y => Sum.elim (pairFeatures x y) 0) features_mem_orderSpan
  have h1 := loopState_mem_orderSpan hL R
    (Z := fun x y => P (Sum.elim (pairFeatures x y) 0)) h0 n
  have h2 := hC.order (Z := fun x y => loopState L R (P (Sum.elim (pairFeatures x y) 0)) n) h1
  have h3 := linear_mem_orderSpan h2 ℓ
  rw [show (fun x y => ℓ (C (loopState L R (P (Sum.elim (pairFeatures x y) 0)) n))) =
    fun x y => ZMod.stdAddChar (∑ i, q (x i) (y i)) from funext fun x => funext (hval x)] at h3
  rwa [show 2 ^ d₂ * (2 ^ (d₁ * (n + 1)) * (2 ^ d₀ * 1)) = 2 ^ (d₀ + (n + 1) * d₁ + d₂) by
    ring] at h3

/-- NV7's network of depth `1 + ⌈log₂ 2⌉` as the prelude, and a core and coda that write nothing:
the bound holds with equality at every number of passes. -/
example (n : ℕ) : 1 + Nat.clog 2 2 ≤ (1 + Nat.clog 2 2) + (n + 1) * 0 + 0 := by
  obtain ⟨T, _, root, F, -, hF, hval⟩ := exists_productNet_copies (p := 7) (α := ZMod 7)
    (β := ZMod 7) (k := 2) (by norm_num) fun a b => a ^ 3 + a * b
  exact le_passes_of_loop_copies (by norm_num) (fun a b => a ^ 3 + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩ hF
    (IsProductNet.id_net (ι := (((Fin 2 × ZMod 7) ⊕ (Fin 2 × ZMod 7)) ⊕ T) ⊕ Unit) 0 0) 0
    (IsProductNet.id_net 0 0) (LinearMap.proj (Sum.inr root)) n
    fun x y => by rw [loopState_id]; exact hval x y

end Transformer.Grokking.FlatSpectrum

module

public import Transformer.Grokking.FlatSpectrum.ProductLoopWalk

/-!
# A scan pays linearly: the message on one side of every product

Source: `review/flat_spectrum.md`, *Loops*, item LP3. A scan follows one link per pass: every unit
of its core multiplies a reading of the tables, which the prelude state holds, by a reading of the
current key, which the message brings. Call a layer a scan layer if its gates read only the
prelude state, so that the message enters every product on at most one side (`IsScanLayer`). It
is a product layer (`IsScanLayer.isProductLayer`), so LP1's bound holds for it too.

In a scan layer the order grows by addition, not by doubling. On a prelude state of order `r` and
a message of order `s ≥ r` it writes order at most `r + s` (`IsScanLayer.order`), so pass `n + 1`
has order at most `(n + 2) r` (`loopState_mem_orderSpan_of_scan`). The answer of `2K` links has the
full order `2K` (`cornerSum_pairWalk_ne_zero`), so a scan answers them only if
`2K ≤ 2^{d₀ + d₂} (n + 2)`, at any width (`le_passes_of_scan_walk`). With no prelude and no coda,
`N` hops take at least `N − 1` passes of a scan layer, against `⌈log₂ N⌉` for a core that may
multiply the message by itself.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {κ α β : Type} {ι μ : Type*}

/-- A product layer on the prelude's stream and the message whose gates read only the prelude's
coordinates: the message enters every product of the layer on at most one side. -/
def IsScanLayer (m : ℕ) (F : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) : Prop :=
  ∃ (η : Type) (_ : Fintype η), Fintype.card η ≤ m ∧
    ∃ (gate : η → (ι → ℂ) →ₗ[ℂ] ℂ) (up : η → (ι ⊕ μ → ℂ) →ₗ[ℂ] ℂ) (out : η → ι ⊕ μ → ℂ),
      ∀ w, F w = w + ∑ u, (gate u (w ∘ Sum.inl) * up u w) • out u

/-- A scan layer is a product layer. -/
lemma IsScanLayer.isProductLayer {m : ℕ} {F : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ} (hF : IsScanLayer m F) :
    IsProductLayer m F := by
  obtain ⟨η, _, hm, g, w, o, hF⟩ := hF
  exact ⟨η, inferInstance, hm, fun u => (g u).comp (LinearMap.funLeft ℂ ℂ Sum.inl), w, o,
    fun z => hF z⟩

/-- **A scan pass adds orders**: on a prelude state of order `r` and a message of order `s ≥ r`,
a scan layer writes coordinates of order at most `r + s`. -/
theorem IsScanLayer.order [Fintype ι] [Fintype μ] {m r s : ℕ} {F : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ}
    (hF : IsScanLayer m F) {Z : (κ → α) → (κ → β) → ι → ℂ} {M : (κ → α) → (κ → β) → μ → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (hM : ∀ j, (fun x y => M x y j) ∈ orderSpan s)
    (hrs : r ≤ s) (c : ι ⊕ μ) :
    (fun x y => F (Sum.elim (Z x y) (M x y)) c) ∈ orderSpan (r + s) := by
  obtain ⟨η, _, -, g, w, o, hF⟩ := hF
  have hin : ∀ c', (fun x y => Sum.elim (Z x y) (M x y) c') ∈ orderSpan s := by
    rintro (c' | j)
    · exact orderSpan_mono hrs (hZ c')
    · exact hM j
  have h : (fun x y => F (Sum.elim (Z x y) (M x y)) c) =
      (fun x y => Sum.elim (Z x y) (M x y) c) +
        ∑ u, o u c • ((fun x y => g u (Z x y)) * fun x y => w u (Sum.elim (Z x y) (M x y))) := by
    funext x y
    simp [hF, Finset.sum_apply, mul_comm]
  rw [h]
  refine Submodule.add_mem _ (orderSpan_mono (Nat.le_add_left s r) (hin c))
    (Submodule.sum_mem _ fun u _ => Submodule.smul_mem _ _ ?_)
  exact mul_mem_orderSpan (linear_mem_orderSpan hZ (g u)) (linear_mem_orderSpan hin (w u))

/-- **Scan passes add up**: after pass `n + 1` of a scan layer on a prelude state of order `r`,
every coordinate has order at most `(n + 2) r`. -/
theorem loopState_mem_orderSpan_of_scan [Fintype ι] [Fintype μ] {m r : ℕ}
    {L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ} (hL : IsScanLayer m L) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    {Z : (κ → α) → (κ → β) → ι → ℂ} (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (n : ℕ)
    (c : ι) : (fun x y => loopState L R (Z x y) n c) ∈ orderSpan ((n + 2) * r) := by
  induction n generalizing c with
  | zero =>
    rw [show (0 + 2) * r = r + r by ring]
    exact hL.order (M := fun _ _ => 0) hZ (fun _ => (orderSpan r).zero_mem) le_rfl (Sum.inl c)
  | succ n ih =>
    rw [show (n + 1 + 2) * r = r + (n + 2) * r by ring]
    exact hL.order (M := fun x y => R (loopState L R (Z x y) n)) hZ
      (fun j => linear_mem_orderSpan (Z := fun x y => loopState L R (Z x y) n) ih
        ((LinearMap.proj j).comp R)) (Nat.le_mul_of_pos_left r (by omega)) (Sum.inl c)

/-- A function of order at most `r` with a nonzero corner sum on some box has `2 |κ| ≤ r`. -/
lemma two_mul_card_le_of_cornerSum_ne_zero [Fintype κ] [DecidableEq κ] {r : ℕ}
    {φ : (κ → α) → (κ → β) → ℂ} (hφ : φ ∈ orderSpan r) {x₀ x₁ : κ → α} {y₀ y₁ : κ → β}
    (h : cornerSum x₀ x₁ y₀ y₁ φ ≠ 0) : 2 * Fintype.card κ ≤ r :=
  not_lt.1 fun hlt => h (cornerSum_eq_zero x₀ x₁ y₀ y₁ hlt hφ)

variable {γ : Type} [DecidableEq γ]

/-- **A scan pays linearly**: if a linear readout of a coda of depth `d₂` after `n + 1` passes of
a scan layer, behind a prelude of depth `d₀` on features that read one table each, tells `a` from
`b` at the end of every walk through `2K` tables, then `2K ≤ 2^{d₀ + d₂} (n + 2)`, at any
width. -/
theorem le_passes_of_scan_walk {K : ℕ} (hK : 0 < K) {a b : γ} (g : γ → ℂ) (hg : g a ≠ g b)
    [Fintype ι] [Fintype μ] {m₀ m₁ m₂ d₀ d₂ : ℕ}
    (Z : (Fin K → γ → γ) → (Fin K → γ → γ) → ι → ℂ) (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan 1)
    {P : (ι → ℂ) → ι → ℂ} (hP : IsProductNet m₀ d₀ P)
    {L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ} (hL : IsScanLayer m₁ L) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    {C : (ι → ℂ) → ι → ℂ} (hC : IsProductNet m₂ d₂ C) (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (C (loopState L R (P (Z x y)) n)) = g (pairWalk K x y a)) :
    2 * K ≤ 2 ^ (d₀ + d₂) * (n + 2) := by
  have h1 := loopState_mem_orderSpan_of_scan hL R (Z := fun x y => P (Z x y)) (hP.order hZ) n
  have h3 := linear_mem_orderSpan (hC.order (Z := fun x y => loopState L R (P (Z x y)) n) h1) ℓ
  rw [show (fun x y => ℓ (C (loopState L R (P (Z x y)) n))) = fun x y => g (pairWalk K x y a) from
    funext fun x => funext (hval x), show 2 ^ d₂ * ((n + 2) * (2 ^ d₀ * 1)) =
      2 ^ (d₀ + d₂) * (n + 2) by ring] at h3
  simpa using two_mul_card_le_of_cornerSum_ne_zero h3 (cornerSum_pairWalk_ne_zero hK g hg)

/-- **Eight links take seven passes of a scan layer**, with no prelude and no coda, at any width;
LP1 asks a core of one layer for three. -/
example {T ν : Type} [Fintype T] [Fintype ν] {m : ℕ}
    {L : ((((Fin 4 × Bool × Bool) ⊕ (Fin 4 × Bool × Bool)) ⊕ T) ⊕ ν → ℂ) →
      (((Fin 4 × Bool × Bool) ⊕ (Fin 4 × Bool × Bool)) ⊕ T) ⊕ ν → ℂ}
    (hL : IsScanLayer m L) (R : (((Fin 4 × Bool × Bool) ⊕ (Fin 4 × Bool × Bool)) ⊕ T → ℂ) →ₗ[ℂ]
      ν → ℂ) (ℓ : (((Fin 4 × Bool × Bool) ⊕ (Fin 4 × Bool × Bool)) ⊕ T → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (loopState L R (Sum.elim (linkFeatures x y) 0) n) =
      if pairWalk 4 x y false = false then 1 else 0) : 7 ≤ n + 1 := by
  have h := le_passes_of_scan_walk (by norm_num) (a := false) (b := true)
    (fun c => if c = false then 1 else 0) (by simp) _ linkFeatures_mem_orderSpan
    (IsProductNet.id_net 0 0) hL R (IsProductNet.id_net 0 0) ℓ n hval
  norm_num at h
  omega

end Transformer.Grokking.FlatSpectrum

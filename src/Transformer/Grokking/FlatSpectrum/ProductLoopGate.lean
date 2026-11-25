module

public import Transformer.Grokking.FlatSpectrum.ProductLoopCopies
public import Transformer.Phasor.PlainRouter

/-!
# Phasor's write gate in a layer and on the loop's message

Source: `review/flat_spectrum.md`, *Loops*, item LP5. Phasor's router scales what a sublayer
writes into stream `i` by the write gate `2σ(wᵢ)` (`MHC.writeGate`, `Phasor.Router.step_apply`),
and `2σ(w) = 1 + tanh(w/2)` (`writeGate_eq_one_add_tanh`): the `tanh` of Inkling's message gate,
shifted by one. The logit `wᵢ` is a learned reading of the RMS-normalized state plus a bias
(`Phasor.Router.routing`). A *gated layer* idealizes this with a gate in front of every product
unit, `F z = z + ∑_u 2σ(ℓ_u z + b_u) (g_u z)(w_u z) o_u`, whose logit reads the state itself
through a real-linear `ℓ_u` (`gatedLayer`, `IsGatedLayer`). Phasor has one gate per stream
instead, and normalizes both the state its logits read and the branch output its gates scale.

A gate whose logit does not read the state is a number. A gated layer with zero readings is a
product layer (`isProductLayer_gatedLayer_zero`), so LP1 and LP2 hold for loops of such layers.
Every product layer is a gated layer whose gates are those of Phasor's plain router, `2σ(0) = 1`
(`IsProductLayer.isGatedLayer`, by `Phasor.writeGate_zero`). The same holds on the loop's message,
where Inkling puts its gate (`gatedLoopState`): a gate with a zero reading scales the message by
`2σ(b)` (`gatedLoopState_zero`), so LP1's count holds (`le_passes_of_gatedLoop_copies`), and the
plain router's gate leaves the loop as it was (`gatedLoopState_plain`).

A gate whose logit reads the state takes the values `2/(1 + 2^s)` on the logits `−s ln 2`
(`writeGate_neg_mul_log_two`). That is not a polynomial in `s`: with three of its values and a
quadratic, `(−1)^s = 361 − 136 s + 14 s² − 360 · 2σ(−s ln 2)` for `s ≤ 3`
(`neg_one_pow_eq_gate`). `ProductLoopGateParity` builds on it a core that LP1's count of the
order does not bound.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Transformer.MHC

/-- **Phasor's write gate is a shifted `tanh`**: `2σ(z) = 1 + tanh(z/2)`. -/
theorem writeGate_eq_one_add_tanh (z : ℝ) : writeGate z = 1 + Real.tanh (z / 2) := by
  have ha := Real.exp_pos (z / 2)
  have hb := Real.exp_pos (-(z / 2))
  have hab : Real.exp (z / 2) * Real.exp (-(z / 2)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hz : Real.exp (-z) = Real.exp (-(z / 2)) * Real.exp (-(z / 2)) := by
    rw [← Real.exp_add]
    ring_nf
  rw [writeGate, gate, Real.tanh_eq, hz]
  field_simp
  nlinarith [hab]

/-- At the logit `0` the shifted `tanh` gives the plain router's gate `1`. -/
example : writeGate 0 = 1 := by simpa using writeGate_eq_one_add_tanh 0

/-- On the logits `−s ln 2` the write gate is `2/(1 + 2^s)`. -/
theorem writeGate_neg_mul_log_two (s : ℕ) :
    writeGate (-((s : ℝ) * Real.log 2)) = 2 / (1 + 2 ^ s) := by
  rw [writeGate, gate, neg_neg, Real.exp_nat_mul, Real.exp_log two_pos]
  ring

/-- **Three values of the gate and a quadratic give the parity**:
`(−1)^s = 361 − 136 s + 14 s² − 360 · 2σ(−s ln 2)` for `s ≤ 3`. -/
theorem neg_one_pow_eq_gate {s : ℕ} (hs : s ≤ 3) :
    ((-1 : ℂ) ^ s) = 361 - 136 * (s : ℂ) + 14 * ((s : ℂ) * s) -
      360 * (writeGate (-((s : ℝ) * Real.log 2)) : ℂ) := by
  rw [writeGate_neg_mul_log_two]
  interval_cases s <;> norm_num

variable {ι : Type*}

/-- **A gated layer**: unit `u` multiplies two linear readings of the state, as a unit of a
product layer, behind Phasor's write gate `2σ(ℓ_u z + b_u)`, whose logit is a real-linear reading
`ℓ_u` of the state plus a bias `b_u`. -/
noncomputable def gatedLayer {η : Type*} [Fintype η] (logit : η → (ι → ℂ) →ₗ[ℝ] ℝ) (bias : η → ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) : (ι → ℂ) → ι → ℂ :=
  fun z => z + ∑ u, ((writeGate (logit u z + bias u) : ℂ) * (gate u z * up u z)) • out u

/-- `F` is one gated layer with at most `m` units. -/
def IsGatedLayer (m : ℕ) (F : (ι → ℂ) → ι → ℂ) : Prop :=
  ∃ (η : Type) (_ : Fintype η), Fintype.card η ≤ m ∧
    ∃ (logit : η → (ι → ℂ) →ₗ[ℝ] ℝ) (bias : η → ℝ) (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ)
      (out : η → ι → ℂ), F = gatedLayer logit bias gate up out

/-- **A gate that does not read the state is a number**: a gated layer whose logit readings are
zero is a product layer, each unit writing along `2σ(b_u) o_u`. -/
theorem isProductLayer_gatedLayer_zero {η : Type} [Fintype η] (bias : η → ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) :
    IsProductLayer (Fintype.card η) (gatedLayer 0 bias gate up out) := by
  refine ⟨η, inferInstance, le_rfl, gate, up, fun u => (writeGate (bias u) : ℂ) • out u,
    fun z => ?_⟩
  simp only [gatedLayer, Pi.zero_apply, LinearMap.zero_apply, zero_add, smul_smul]
  congr 1
  exact Finset.sum_congr rfl fun u _ => by rw [mul_comm]

/-- **Every product layer is a gated layer**, with the gates of Phasor's plain router:
zero readings and zero biases, so every gate is `2σ(0) = 1`. -/
theorem IsProductLayer.isGatedLayer {m : ℕ} {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F) :
    IsGatedLayer m F := by
  obtain ⟨η, _, hm, g, w, o, hF⟩ := hF
  refine ⟨η, inferInstance, hm, 0, 0, g, w, o, funext fun z => ?_⟩
  simp [gatedLayer, hF, Phasor.writeGate_zero]

/-- The identity is a gated layer without units. -/
example (m : ℕ) : IsGatedLayer m (id : (Unit → ℂ) → Unit → ℂ) :=
  (isProductLayer_id m).isGatedLayer

/-- One unit behind a gate of bias `ln 3` on a single coordinate is a product layer. -/
example : IsProductLayer 1 (gatedLayer (ι := Unit) (η := Unit) 0 (fun _ => Real.log 3)
    (fun _ => LinearMap.proj ()) (fun _ => LinearMap.proj ()) fun _ => 1) := by
  simpa using isProductLayer_gatedLayer_zero (ι := Unit) (η := Unit) (fun _ => Real.log 3)
    (fun _ => LinearMap.proj ()) (fun _ => LinearMap.proj ()) fun _ => 1

variable {μ : Type*}

/-- **A loop whose message passes Phasor's write gate**: as `loopState`, but pass `n + 2` gets
the message `2σ(ℓ s + b) · R s` of the last state `s`, where Inkling puts its message gate. -/
noncomputable def gatedLoopState (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    (logit : (ι → ℂ) →ₗ[ℝ] ℝ) (bias : ℝ) (z : ι → ℂ) : ℕ → ι → ℂ
  | 0 => L (Sum.elim z 0) ∘ Sum.inl
  | n + 1 =>
    let s := gatedLoopState L R logit bias z n
    L (Sum.elim z ((writeGate (logit s + bias) : ℂ) • R s)) ∘ Sum.inl

/-- **A message gate that does not read the state scales the message**: with a zero reading the
gated loop is the loop of `2σ(b) R`. -/
theorem gatedLoopState_zero (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ) (bias : ℝ)
    (z : ι → ℂ) (n : ℕ) :
    gatedLoopState L R 0 bias z n = loopState L ((writeGate bias : ℂ) • R) z n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [gatedLoopState, loopState, ih, LinearMap.zero_apply, zero_add,
      LinearMap.smul_apply]

/-- **The plain router passes the message unchanged**: its gate is `2σ(0) = 1`. -/
theorem gatedLoopState_plain (L : (ι ⊕ μ → ℂ) → ι ⊕ μ → ℂ) (R : (ι → ℂ) →ₗ[ℂ] μ → ℂ)
    (z : ι → ℂ) (n : ℕ) : gatedLoopState L R 0 0 z n = loopState L R z n := by
  rw [gatedLoopState_zero, Phasor.writeGate_zero, Complex.ofReal_one, one_smul]

/-- **LP1 behind a constant gate**: if a loop whose message passes the gate `2σ(b)` holds
`ψ(∑ᵢ q(xᵢ, yᵢ))` after pass `n + 1`, with a product prelude, core and coda of depths `d₀`, `d₁`
and `d₂`, then `1 + ⌈log₂ k⌉ ≤ d₀ + (n + 1) d₁ + d₂`, as in `le_passes_of_loop_copies`. -/
theorem le_passes_of_gatedLoop_copies {α β : Type} [Fintype α] [DecidableEq α] [Fintype β]
    [DecidableEq β] {p k : ℕ} [NeZero p] (hk : 0 < k) (q : α → β → ZMod p)
    (hq : ∃ a₀ a₁ b₀ b₁, ZMod.stdAddChar (q a₀ b₀) - ZMod.stdAddChar (q a₁ b₀) -
      ZMod.stdAddChar (q a₀ b₁) + ZMod.stdAddChar (q a₁ b₁) ≠ 0)
    {T ν : Type} [Fintype T] [Fintype ν] {m₀ m₁ m₂ d₀ d₁ d₂ : ℕ}
    {P : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ}
    (hP : IsProductNet m₀ d₀ P)
    {L : ((((Fin k × α) ⊕ (Fin k × β)) ⊕ T) ⊕ ν → ℂ) → (((Fin k × α) ⊕ (Fin k × β)) ⊕ T) ⊕ ν → ℂ}
    (hL : IsProductNet m₁ d₁ L) (R : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) →ₗ[ℂ] ν → ℂ)
    (bias : ℝ) {C : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) → ((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ}
    (hC : IsProductNet m₂ d₂ C) (ℓ : (((Fin k × α) ⊕ (Fin k × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ) (n : ℕ)
    (hval : ∀ x y, ℓ (C (gatedLoopState L R 0 bias (P (Sum.elim (pairFeatures x y) 0)) n)) =
      ZMod.stdAddChar (∑ i, q (x i) (y i))) :
    1 + Nat.clog 2 k ≤ d₀ + (n + 1) * d₁ + d₂ :=
  le_passes_of_loop_copies hk q hq hP hL ((writeGate bias : ℂ) • R) hC ℓ n fun x y => by
    rw [← gatedLoopState_zero]
    exact hval x y

/-- Behind the plain router's gate, LP2's two passes for three copies of `x³ + x y` meet the
bound. -/
example : 1 + Nat.clog 2 3 ≤ 1 + (1 + 1) * 1 + 0 :=
  le_passes_of_gatedLoop_copies (k := 3) (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b)
    ⟨0, 1, 0, 1, mixed_stdAddChar_ne_zero (by norm_num) fun a => a ^ 3⟩
    (IsProductNet.of_layer (isProductLayer_leafLayer (copiesPair (k := 3) (by norm_num))
      (copiesValue fun a b : ZMod 7 => a ^ 3 + a * b)))
    (IsProductNet.of_layer isProductLayer_treeCore) treeRead 0 (IsProductNet.id_net 0 0)
    (LinearMap.proj (prodSlot ⟨0, by norm_num⟩)) 1 fun x y => by
      rw [gatedLoopState_plain]
      exact loopCopies_root (k := 3) (by norm_num) (fun a b : ZMod 7 => a ^ 3 + a * b)
        (Nat.clog_le_of_le_pow (by norm_num)) x y

end Transformer.Grokking.FlatSpectrum

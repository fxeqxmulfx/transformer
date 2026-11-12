module

public import Transformer.Grokking.Protocol.Target

/-!
# The complete state of a run, and its continuation under later shuffles

Source: openai/grok at 3d64b1d8, `grok/training.py` (`CustomAdamW.step`: the optimizer keeps,
for every parameter, its value, `exp_avg`, `exp_avg_sq` and the count of steps) and
`grok/data.py` (`ArithmeticIterator`: a new permutation of the training set for each epoch);
`experiments/grok_division/PROTOCOL.md`, *The optimizer* and *Randomness*. The paper,
arXiv:2201.02177v1 (Power et al.), states nothing about the state of the optimizer.

The parameters alone do not fix what a run does next: the update divides the first moment by
the root of the second. The complete state after `t` updates is the parameters, both moments
and `t` (`stateAt`; its parameters are the `trajectory`). The run from a complete state
(`adamRunFrom`) continues a run (`adamRun_eq_adamRunFrom`, `adamRunFrom_add`) and sees the
gradients of the later updates only (`adamRunFrom_congr`).

An epoch has `E = batchesPerEpoch k b` updates, and the batches of the epochs from `e` on are
cut from the shuffles from `e` on (`batchSlots_mul_add`). So at the end of an epoch what
follows is a function of the split, the epoch, the complete state and the later shuffles
(`continuation`): a sample's state `n` updates after the end of epoch `e` is the continuation
of its state at that end (`stateAt_mul_add`); continuations compose (`continuation_mul_add`);
and the first `E a` updates of a continuation see the first `a` of its shuffles only
(`continuation_congr`), as does the state of a sample (`stateAt_congr`). Nothing here says
where a run goes.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι X : Type*} {N k : ℕ}

/-- **The run from a complete state**: the state `n` updates after update `t₀`, from the state
`s` after update `t₀`, where update `t` sees the gradient `grad t` at the parameters before
it. -/
noncomputable def adamRunFrom (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (t₀ : ℕ)
    (s : AdamState ι) : ℕ → AdamState ι
  | 0 => s
  | n + 1 => adamStep h (t₀ + n + 1) (grad (t₀ + n + 1) (adamRunFrom h grad t₀ s n).params)
      (adamRunFrom h grad t₀ s n)

/-- **A run is the run from its initial state**: the initial parameters and zero moments. -/
theorem adamRun_eq_adamRunFrom (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (θ₀ : ι → ℝ) (n : ℕ) :
    adamRun h grad θ₀ n = adamRunFrom h grad 0 ⟨θ₀, 0, 0⟩ n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [adamRun, adamRunFrom, ih, Nat.zero_add]

/-- **Runs from complete states compose**: `a + n` updates are `n` updates from the state after
`a`. -/
theorem adamRunFrom_add (h : Hyper) (grad : ℕ → (ι → ℝ) → ι → ℝ) (t₀ : ℕ) (s : AdamState ι)
    (a n : ℕ) :
    adamRunFrom h grad t₀ s (a + n) =
      adamRunFrom h grad (t₀ + a) (adamRunFrom h grad t₀ s a) n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [← Nat.add_assoc, adamRunFrom, adamRunFrom, ih, Nat.add_assoc t₀ a n]

/-- **The run from the state after update `t₀` sees the gradients of the later updates only.** -/
theorem adamRunFrom_congr (h : Hyper) {grad grad' : ℕ → (ι → ℝ) → ι → ℝ} (t₀ : ℕ)
    (s : AdamState ι) {n : ℕ} (hg : ∀ m, m < n → grad (t₀ + m + 1) = grad' (t₀ + m + 1)) :
    adamRunFrom h grad t₀ s n = adamRunFrom h grad' t₀ s n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [adamRunFrom, adamRunFrom, ih fun m hm => hg m (Nat.lt_succ_of_lt hm),
      hg n (Nat.lt_succ_self n)]

/-- A batch is cut from the shuffle of its epoch. -/
lemma batchSlots_congr (k b : ℕ) {σ σ' : ℕ → Equiv.Perm (Fin k)} {t : ℕ}
    (h : σ ((t - 1) / batchesPerEpoch k b) = σ' ((t - 1) / batchesPerEpoch k b)) :
    batchSlots k b σ t = batchSlots k b σ' t := by
  rw [batchSlots, batchSlots, h]

/-- **The batches of the epochs from `e` on are cut from the shuffles from `e` on.** -/
lemma batchSlots_mul_add (k b : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) (e : ℕ)
    (h : 0 < batchesPerEpoch k b) (m : ℕ) :
    batchSlots k b σ (batchesPerEpoch k b * e + m + 1) =
      batchSlots k b (fun j => σ (e + j)) (m + 1) := by
  simp only [batchSlots, Nat.add_sub_cancel, Nat.mul_add_div h, Nat.mul_add_mod]

/-- **The complete state of a sample after `t` updates**: the parameters and both moments. -/
noncomputable def stateAt (h : Hyper) (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X)
    (hk : k ≤ N) (b : ℕ) (ω : Sample N k ι) (t : ℕ) : AdamState ι :=
  adamRun h (fun s θ => lossGrad ((batchSlots k b ω.2.2 s).map (trainAt eqs hk ω.1)) θ) ω.2.1 t

/-- **The continuation of a complete state**: the state `n` updates after the end of epoch `e`,
from the state `s` at its end, under the split `π` and the shuffles `τ 0, τ 1, …` of the
epochs `e, e + 1, …`. -/
noncomputable def continuation (h : Hyper) (lossGrad : List X → (ι → ℝ) → ι → ℝ)
    (eqs : Fin N → X) (hk : k ≤ N) (b : ℕ) (π : Equiv.Perm (Fin N)) (e : ℕ) (s : AdamState ι)
    (τ : ℕ → Equiv.Perm (Fin k)) (n : ℕ) : AdamState ι :=
  adamRunFrom h (fun t θ => lossGrad
    ((batchSlots k b τ (t - batchesPerEpoch k b * e)).map (trainAt eqs hk π)) θ)
    (batchesPerEpoch k b * e) s n

variable {h : Hyper} {lossGrad : List X → (ι → ℝ) → ι → ℝ} {eqs : Fin N → X} {hk : k ≤ N}
  {b : ℕ}

/-- The parameters of the complete state are the trajectory. -/
lemma stateAt_params (ω : Sample N k ι) (t : ℕ) :
    (stateAt h lossGrad eqs hk b ω t).params = trajectory h lossGrad eqs hk b ω t := rfl

/-- The state of a sample is the continuation of its initial state under all its shuffles. -/
lemma stateAt_eq_continuation (ω : Sample N k ι) (n : ℕ) :
    stateAt h lossGrad eqs hk b ω n =
      continuation h lossGrad eqs hk b ω.1 0 ⟨ω.2.1, 0, 0⟩ ω.2.2 n := by
  simp only [stateAt, continuation, adamRun_eq_adamRunFrom, Nat.mul_zero, Nat.sub_zero]

/-- **Continuations compose**: `E a + n` updates after the end of epoch `e` are `n` updates after
the end of epoch `e + a`, from the state reached there, under the shuffles from `a` on. -/
theorem continuation_mul_add (hE : 0 < batchesPerEpoch k b) (π : Equiv.Perm (Fin N)) (e : ℕ)
    (s : AdamState ι) (τ : ℕ → Equiv.Perm (Fin k)) (a n : ℕ) :
    continuation h lossGrad eqs hk b π e s τ (batchesPerEpoch k b * a + n) =
      continuation h lossGrad eqs hk b π (e + a)
        (continuation h lossGrad eqs hk b π e s τ (batchesPerEpoch k b * a))
        (fun j => τ (a + j)) n := by
  simp only [continuation, adamRunFrom_add, ← Nat.mul_add]
  refine adamRunFrom_congr h _ _ fun m _ => ?_
  funext θ
  have h1 : batchesPerEpoch k b * (e + a) + m + 1 - batchesPerEpoch k b * e =
      batchesPerEpoch k b * a + m + 1 := by
    rw [Nat.mul_add]
    omega
  rw [h1, batchSlots_mul_add k b τ a hE m, Nat.add_assoc (batchesPerEpoch k b * (e + a)),
    Nat.add_sub_cancel_left]

/-- **A sample's state `n` updates after the end of epoch `e` is the continuation of its state
at the end of that epoch**, under its split and its later shuffles. -/
theorem stateAt_mul_add (hE : 0 < batchesPerEpoch k b) (ω : Sample N k ι) (e n : ℕ) :
    stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e + n) =
      continuation h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e)) (fun j => ω.2.2 (e + j)) n := by
  rw [stateAt_eq_continuation, stateAt_eq_continuation, continuation_mul_add hE, Nat.zero_add]

/-- **The first `E a` updates of a continuation see the first `a` of its shuffles only.** -/
theorem continuation_congr (π : Equiv.Perm (Fin N)) (e : ℕ) (s : AdamState ι)
    {τ τ' : ℕ → Equiv.Perm (Fin k)} {a n : ℕ}
    (hn : n ≤ batchesPerEpoch k b * a) (hτ : ∀ j, j < a → τ j = τ' j) :
    continuation h lossGrad eqs hk b π e s τ n = continuation h lossGrad eqs hk b π e s τ' n := by
  refine adamRunFrom_congr h _ _ fun m hm => ?_
  funext θ
  rw [batchSlots_congr k b (hτ _ ?_)]
  rw [Nat.add_assoc, Nat.add_sub_cancel_left, Nat.add_sub_cancel]
  exact Nat.div_lt_of_lt_mul (lt_of_lt_of_le hm hn)

/-- **The state of a sample at the end of epoch `a` and before is fixed by its split, its
initial parameters and its first `a` shuffles.** -/
theorem stateAt_congr {ω ω' : Sample N k ι} {a n : ℕ}
    (hn : n ≤ batchesPerEpoch k b * a) (h1 : ω.1 = ω'.1) (h2 : ω.2.1 = ω'.2.1)
    (hτ : ∀ j, j < a → ω.2.2 j = ω'.2.2 j) :
    stateAt h lossGrad eqs hk b ω n = stateAt h lossGrad eqs hk b ω' n := by
  rw [stateAt_eq_continuation, stateAt_eq_continuation, h1, h2]
  exact continuation_congr _ _ _ hn hτ

/-! The witnesses: a run on one coordinate whose gradient is the parameter; two oracles that
differ at the first update only; and four equations of which two are trained on in batches of
one, so that an epoch has two updates, with sequences of shuffles that share the first. -/

example (n : ℕ) := adamRun_eq_adamRunFrom (Hyper.source 1) (fun _ θ => θ) (fun _ : Unit => 1) n

example (s : AdamState Unit) (n : ℕ) := adamRunFrom_add (Hyper.source 1) (fun _ θ => θ) 4 s 3 n

example (s : AdamState Unit) :
    adamRunFrom (Hyper.source 1) (fun t θ => if t = 1 then 0 else θ) 1 s 5 =
      adamRunFrom (Hyper.source 1) (fun _ θ => θ) 1 s 5 :=
  adamRunFrom_congr (Hyper.source 1) 1 s fun m _ => by
    funext θ
    rw [ite_eq_right (by omega)]

example (σ : ℕ → Equiv.Perm (Fin 2)) := batchSlots_mul_add 2 1 σ 3 (by decide) 1

example (ω : Sample 4 2 Unit) (n : ℕ) :=
  stateAt_mul_add (h := Hyper.source 1) (lossGrad := fun (_ : List (Fin 4)) θ => θ) (eqs := id)
    (hk := by norm_num) (b := 1) (by decide) ω 3 n

example (π : Equiv.Perm (Fin 4)) (s : AdamState Unit) (τ : ℕ → Equiv.Perm (Fin 2)) (n : ℕ) :=
  continuation_mul_add (h := Hyper.source 1) (lossGrad := fun (_ : List (Fin 4)) θ => θ)
    (eqs := id) (hk := by norm_num) (b := 1) (by decide) π 5 s τ 1 n

example (π : Equiv.Perm (Fin 4)) (s : AdamState Unit) (τ : ℕ → Equiv.Perm (Fin 2)) :
    continuation (Hyper.source 1) (fun (_ : List (Fin 4)) θ => θ) id (by norm_num) 1 π 5 s τ 2 =
      continuation (Hyper.source 1) (fun (_ : List (Fin 4)) θ => θ) id (by norm_num) 1 π 5 s
        (fun j => if j = 0 then τ 0 else 1) 2 :=
  continuation_congr (a := 1) π 5 s (by decide) fun j hj => by
    rw [ite_eq_left (by omega), show j = 0 by omega]

example (π : Equiv.Perm (Fin 4)) (τ : ℕ → Equiv.Perm (Fin 2)) :
    stateAt (Hyper.source 1) (fun (_ : List (Fin 4)) θ => θ) id (by norm_num) 1
        ((π, 0, τ) : Sample 4 2 Unit) 2 =
      stateAt (Hyper.source 1) (fun (_ : List (Fin 4)) θ => θ) id (by norm_num) 1
        ((π, 0, fun j => if j = 0 then τ 0 else 1) : Sample 4 2 Unit) 2 :=
  stateAt_congr (a := 1) (by decide) rfl rfl fun j hj => by
    change τ j = if j = 0 then τ 0 else 1
    rw [ite_eq_left (by omega), show j = 0 by omega]

end Transformer.Grokking.Protocol

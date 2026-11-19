module

public import Mathlib.Data.Set.Card
public import Mathlib.Order.Interval.Finset.Nat

/-!
# §5 Receptive fields: windows of sequence maps

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §5 (Receptive Field Measurement): "an `R` layer
transformer has a theoretical receptive field (TRF) of `wR`, which is the maximum number of
tokens that contribute to the prediction of the next token."

The sequence maps of a windowed model are causal and read a window of the most recent tokens.
This file is the bookkeeping of such windows, for any map `F : (ℕ → α) → ℕ → β` from input
sequences to output sequences.

* `IsLocal w F`: the output at `m` depends only on the inputs at the positions `n` with
  `n + w > m ≥ n` of §3.3, the `w` most recent tokens.
* `IsShiftInvariant w F`: once `m + 1 ≥ w`, dropping the first `s` tokens does not change the
  output at `s + m`: the map does not see absolute positions. Windowed attention has both
  properties with its size `w`, LayerNorm and the readout with `1`.
* `HasWindow.comp`: a map with the window `w'` after one with the window `w` has the window
  `w + w' − 1`, not `w + w'`: the two windows share the token where they meet. Hence `R` layers
  have the window `R(w − 1) + 1` (`HasWindow.iterate`), not the paper's `wR`.
* `IsCausal F`: the output at `m` depends only on the inputs at the positions `n ≤ m`.
* `Contributes F m n`: changing the input at `n` alone can change the output at `m`. A local map
  has at most `w` contributing tokens at each position (`IsLocal.ncard_contributes_le`).
-/

@[expose] public section

namespace Transformer.Sandwich

variable {α β γ δ : Type*} {w w' : ℕ}

/-- The output at `m` depends only on the inputs at the positions `n` with `n + w > m ≥ n`, the
`w` most recent tokens (the window of §3.3). -/
def IsLocal (w : ℕ) (F : (ℕ → α) → ℕ → β) : Prop :=
  ∀ x y m, (∀ n, n ≤ m → m < n + w → x n = y n) → F x m = F y m

/-- Dropping the first `s` tokens does not change the output at `s + m` once the window of `w`
tokens before `m` lies in the sequence, `m + 1 ≥ w`. -/
def IsShiftInvariant (w : ℕ) (F : (ℕ → α) → ℕ → β) : Prop :=
  ∀ x s m, w ≤ m + 1 → F x (s + m) = F (fun j => x (s + j)) m

/-- `F` has the window `w`: it is local and shift-invariant with `w`. -/
def HasWindow (w : ℕ) (F : (ℕ → α) → ℕ → β) : Prop := IsLocal w F ∧ IsShiftInvariant w F

/-- The identity has the window `1`. -/
theorem hasWindow_id : HasWindow 1 fun x : ℕ → α => x :=
  ⟨fun _ _ m h => h m le_rfl (Nat.lt_succ_self m), fun _ _ _ _ => rfl⟩

/-- A position-wise map `φ` after `F` keeps the window of `F`. -/
theorem HasWindow.map {F : (ℕ → α) → ℕ → β} (hF : HasWindow w F) (φ : β → γ) :
    HasWindow w fun x m => φ (F x m) :=
  ⟨fun x y m h => congrArg φ (hF.1 x y m h), fun x s m h => congrArg φ (hF.2 x s m h)⟩

/-- The hypothesis of `HasWindow.map` holds for the identity. -/
example (φ : α → γ) : HasWindow 1 fun (x : ℕ → α) m => φ (x m) := hasWindow_id.map φ

/-- **Windows of a composition**: `G` with the window `w'` after `F` with the window `w` has the
window `w + w' − 1`. -/
theorem HasWindow.comp {G : (ℕ → β) → ℕ → γ} {F : (ℕ → α) → ℕ → β} (hG : HasWindow w' G)
    (hF : HasWindow w F) (hw : 1 ≤ w) (hw' : 1 ≤ w') :
    HasWindow (w + w' - 1) fun x => G (F x) := by
  refine ⟨fun x y m h => hG.1 _ _ m fun n hn hmn => hF.1 x y n fun j hj hnj => h j ?_ ?_,
    fun x s m h => ?_⟩
  · omega
  · omega
  show G (F x) (s + m) = G (F fun j => x (s + j)) m
  rw [hG.2 (F x) s m (by omega)]
  exact hG.1 _ _ m fun n _ hmn => hF.2 x s n (by omega)

/-- The hypotheses of `HasWindow.comp` hold for two identities. -/
example : HasWindow (1 + 1 - 1) fun x : ℕ → α => x :=
  hasWindow_id.comp hasWindow_id le_rfl le_rfl

/-- **`R` layers of the window `w`** have the window `R(w − 1) + 1`. -/
theorem HasWindow.iterate {F : (ℕ → α) → ℕ → α} (hF : HasWindow w F) (hw : 1 ≤ w) (R : ℕ) :
    HasWindow (R * (w - 1) + 1) F^[R] := by
  induction R with
  | zero =>
    rw [Function.iterate_zero, Nat.zero_mul, Nat.zero_add]
    exact hasWindow_id
  | succ R ih =>
    rw [Function.iterate_succ]
    have h := ih.comp hF hw (by omega)
    rwa [show w + (R * (w - 1) + 1) - 1 = (R + 1) * (w - 1) + 1 by
      rw [Nat.add_mul, one_mul]; omega] at h

/-- The hypotheses of `HasWindow.iterate` hold for the identity, whose iterates have the window
`1`. -/
example (R : ℕ) : HasWindow (R * (1 - 1) + 1) (fun x : ℕ → α => x)^[R] :=
  hasWindow_id.iterate le_rfl R

/-- The input at `n` contributes to the output of `F` at `m`: changing it alone can change that
output (§5, "tokens that contribute to the prediction"). -/
def Contributes (F : (ℕ → α) → ℕ → β) (m n : ℕ) : Prop :=
  ∃ x y : ℕ → α, (∀ j, j ≠ n → x j = y j) ∧ F x m ≠ F y m

/-- Only the tokens of the window contribute to a local map. -/
theorem IsLocal.contributes {F : (ℕ → α) → ℕ → β} (hF : IsLocal w F) {m n : ℕ}
    (h : Contributes F m n) : n ≤ m ∧ m < n + w := by
  by_contra hn
  obtain ⟨x, y, hxy, hne⟩ := h
  exact hne (hF x y m fun j hj hmj => hxy j fun hjn => hn (hjn ▸ ⟨hj, hmj⟩))

/-- The hypotheses of `IsLocal.contributes` hold for the identity. -/
example {m n : ℕ} (h : Contributes (fun x : ℕ → α => x) m n) : n ≤ m ∧ m < n + 1 :=
  hasWindow_id.1.contributes h

/-- **At most `w` tokens contribute** to each output of a local map. -/
theorem IsLocal.ncard_contributes_le {F : (ℕ → α) → ℕ → β} (hF : IsLocal w F) (m : ℕ) :
    {n | Contributes F m n}.ncard ≤ w := by
  calc {n | Contributes F m n}.ncard ≤ (↑(Finset.Ico (m + 1 - w) (m + 1)) : Set ℕ).ncard :=
        Set.ncard_le_ncard (fun n hn => by
          have := hF.contributes hn
          simp only [Finset.coe_Ico, Set.mem_Ico]
          omega) (Finset.finite_toSet _)
    _ = m + 1 - (m + 1 - w) := by rw [Set.ncard_coe_finset, Nat.card_Ico]
    _ ≤ w := by omega

/-- The hypotheses of `IsLocal.ncard_contributes_le` hold for the identity. -/
example (m : ℕ) : {n | Contributes (fun x : ℕ → α => x) m n}.ncard ≤ 1 :=
  hasWindow_id.1.ncard_contributes_le m


/-- `F` is causal: its output at `m` depends only on the inputs at the positions `n ≤ m`. -/
def IsCausal (F : (ℕ → α) → ℕ → β) : Prop :=
  ∀ x y m, (∀ n, n ≤ m → x n = y n) → F x m = F y m

/-- A causal map after a causal map is causal. -/
theorem IsCausal.comp {G : (ℕ → β) → ℕ → γ} {F : (ℕ → α) → ℕ → β} (hG : IsCausal G)
    (hF : IsCausal F) : IsCausal fun x => G (F x) :=
  fun x y m h => hG _ _ m fun n hn => hF x y n fun j hj => h j (hj.trans hn)

/-- The hypotheses of `IsCausal.comp` hold for two identities. -/
example : IsCausal fun x : ℕ → α => x :=
  IsCausal.comp (G := fun x : ℕ → α => x) (fun _ _ m h => h m le_rfl) fun _ _ m h => h m le_rfl

/-- A position-wise map before and after a causal map keeps it causal. -/
theorem IsCausal.map {F : (ℕ → α) → ℕ → β} (hF : IsCausal F) (T : ℕ → δ → α) (φ : β → γ) :
    IsCausal fun e m => φ (F (fun j => T j (e j)) m) :=
  fun x y m h => congrArg φ (hF _ _ m fun n hn => by rw [h n hn])

/-- The hypotheses of `IsCausal.map` hold for the identity. -/
example (φ : α → γ) : IsCausal fun (e : ℕ → α) m => φ ((fun x : ℕ → α => x) (fun j => e j) m) :=
  IsCausal.map (F := fun x : ℕ → α => x) (fun _ _ m h => h m le_rfl) (fun _ a => a) φ

/-- Only the tokens up to `m` contribute to a causal map's output at `m`. -/
theorem IsCausal.contributes {F : (ℕ → α) → ℕ → β} (hF : IsCausal F) {m n : ℕ}
    (h : Contributes F m n) : n ≤ m := by
  by_contra hn
  obtain ⟨x, y, hxy, hne⟩ := h
  exact hne (hF x y m fun j hj => hxy j fun hjn => hn (hjn ▸ hj))

/-- The hypotheses of `IsCausal.contributes` hold for the identity. -/
example {m n : ℕ} (h : Contributes (fun x : ℕ → α => x) m n) : n ≤ m :=
  IsCausal.contributes (fun _ _ m h => h m le_rfl) h

end Transformer.Sandwich

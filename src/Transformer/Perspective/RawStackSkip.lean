/-
# A skip is gauge-covariant, and the gauge damps it

**Not a statement of any paper.**  Modded-nanogpt
(github.com/KellerJordan/modded-nanogpt, `train_gpt.py`, `GPT.forward`) skips
attention at layer `6` and adds an earlier state of the stream instead,

  `skip_gate_out = torch.sigmoid(skip_lambda) * post_skip_gate`
  `x = x + skip_gate_out * cache[3]`,

where `cache[3]` is the snapshot of the residual stream at layer `3` and the
gate is one scalar per token (`gate[..., 28:29]` of `unpack_post_mudd_gate`).
That is the one update of the model `Perspective.RawStack` does not cover: it
adds not a bounded output of a block but the stream itself, at an earlier
depth.

In the gauge it is still exact.  Write `y_{k,i} = x_{k,i} / Λ_{k,i}` for the
stream with its gains divided out (`ungauged`, which is `gaugeStack` when there
are no skips).  Then a stack

  `x_{k+1,i} = λ_{k,i} x_{k,i} + s_{k,i} x_{m(k),i} + g_{k,i}`

is, in `y`, the same stack with every gain `1` and the skip weight rescaled:

  `y_{k+1,i} = y_{k,i} + (s_{k,i} Λ_{m(k),i} / Λ_{k+1,i}) y_{m(k),i} + g_{k,i}/Λ_{k+1,i}`

(`ungauged_rec_skip`), and the directions are those of `y` whatever the gains
are (`normalize_ungauged`).  So a skip is a gauge-covariant
operation, and what reaches the directions is not its gate but the gate divided
by the gain accumulated between the snapshot and the skip.

`Perspective.RawStackSkipDamp` reads that weight at gains above `1 + c`, and
reads it back against the record's own numbers.
-/

import Transformer.Perspective.RawStack

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The stream with its gauge divided out:** `y_{k,i} = x_{k,i} / Λ_{k,i}`. -/
noncomputable def ungauged (lam : ℕ → Idx n → ℝ) (x : ℕ → Idx n → EucSpace d) (k : ℕ)
    (i : Idx n) : EucSpace d :=
  (gainProd lam k i)⁻¹ • x k i

/-- **Dividing the gauge out does not move a direction.**

Source: none — posed here; `normalize_smul_of_pos` at `k = Λ_{k,i}⁻¹`. -/
theorem normalize_ungauged {lam : ℕ → Idx n → ℝ} (hlam : ∀ j i, 0 < lam j i)
    (x : ℕ → Idx n → EucSpace d) (k : ℕ) (i : Idx n) :
    ‖ungauged lam x k i‖⁻¹ • ungauged lam x k i = ‖x k i‖⁻¹ • x k i := by
  rw [ungauged]
  exact normalize_smul_of_pos (inv_pos.2 (gainProd_pos hlam k i)) (x k i)

/-- **Without skips the ungauged stream is the gauge stack of
`Perspective.RawStack`.**

Source: none — posed here; `rawStack_eq_gauge`, divided by `Λ_{k,i}`. -/
theorem ungauged_eq_gaugeStack {lam : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d}
    (hlam : ∀ j i, 0 < lam j i) (hx : ∀ k i, x (k + 1) i = lam k i • x k i + g k i)
    (k : ℕ) (i : Idx n) : ungauged lam x k i = gaugeStack lam (x 0) g k i := by
  rw [ungauged, rawStack_eq_gauge hlam hx k i, smul_smul,
    inv_mul_cancel₀ (gainProd_pos hlam k i).ne', one_smul]

/-- The hypotheses of `normalize_ungauged` and `ungauged_eq_gaugeStack` are
satisfiable: gains `1`, no block output, a stream at rest. -/
example : (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    ∀ (k : ℕ) (i : Idx 1), (fun _ (_ : Idx 1) => (basePoint 0 : EucSpace 1)) (k + 1) i
      = (1 : ℝ) • (fun _ (_ : Idx 1) => (basePoint 0 : EucSpace 1)) k i + 0 :=
  ⟨fun _ _ => one_pos, fun _ _ => by simp⟩

/-- **One step of the ungauged stream.**  Whatever the `k`-th block adds to the
stream, the ungauged stream moves by that much divided by the gauge it is added
behind — the gains themselves disappear from the recursion.

Source: none — posed here; the residual update `x ← λ x + (anything)` of
modded-nanogpt (`train_gpt.py`, `GPT.forward`), divided by `Λ_{k+1,i}`. -/
theorem ungauged_step {lam : ℕ → Idx n → ℝ} {x : ℕ → Idx n → EucSpace d} {v : EucSpace d}
    {k : ℕ} {i : Idx n} (hlam : ∀ j i, 0 < lam j i)
    (hx : x (k + 1) i = lam k i • x k i + v) :
    ungauged lam x (k + 1) i = ungauged lam x k i + (gainProd lam (k + 1) i)⁻¹ • v := by
  have hkne : gainProd lam k i ≠ 0 := (gainProd_pos hlam k i).ne'
  have hlne : lam k i ≠ 0 := (hlam k i).ne'
  have hstep : gainProd lam k i * lam k i = gainProd lam (k + 1) i :=
    (Finset.prod_range_succ (fun j => lam j i) k).symm
  have h1 : (gainProd lam (k + 1) i)⁻¹ * lam k i = (gainProd lam k i)⁻¹ := by
    rw [← hstep]; field_simp
  simp only [ungauged, hx, smul_add, smul_smul, h1]

/-- **A skip is gauge-covariant.**  If

  `x_{k+1,i} = λ_{k,i} x_{k,i} + s_{k,i} x_{m(k),i} + g_{k,i}`

with every gain positive, then the ungauged stream obeys the same recursion
with every gain `1`, the step `g_{k,i}/Λ_{k+1,i}` and the skip weight
`s_{k,i} Λ_{m(k),i} / Λ_{k+1,i}`.  The depth `m(k)` the skip reads is
arbitrary, and so are the gates and the outputs.

Source: none — posed here; `x = x + skip_gate_out * cache[3]` of
modded-nanogpt (`train_gpt.py`, `GPT.forward`), whose gate is one scalar per
token, against the residual update `rawStack_eq_gauge` is written for. -/
theorem ungauged_rec_skip {lam s : ℕ → Idx n → ℝ} {x g : ℕ → Idx n → EucSpace d} {m : ℕ → ℕ}
    (hlam : ∀ j i, 0 < lam j i)
    (hx : ∀ k i, x (k + 1) i = lam k i • x k i + s k i • x (m k) i + g k i)
    (k : ℕ) (i : Idx n) :
    ungauged lam x (k + 1) i
      = ungauged lam x k i
        + (s k i * gainProd lam (m k) i / gainProd lam (k + 1) i) • ungauged lam x (m k) i
        + (gainProd lam (k + 1) i)⁻¹ • g k i := by
  have hmne : gainProd lam (m k) i ≠ 0 := (gainProd_pos hlam (m k) i).ne'
  have hkne : gainProd lam (k + 1) i ≠ 0 := (gainProd_pos hlam (k + 1) i).ne'
  have h2 : s k i * gainProd lam (m k) i / gainProd lam (k + 1) i * (gainProd lam (m k) i)⁻¹
      = (gainProd lam (k + 1) i)⁻¹ * s k i := by
    field_simp
  rw [ungauged_step hlam (by rw [hx k i, add_assoc]), smul_add, ← add_assoc]
  simp only [ungauged, smul_smul, h2]

/-- The hypotheses of `ungauged_rec_skip` are satisfiable, and not only at
`s = 0`: gains `1`, a skip of weight `1` onto the state it is at, no output,
and the stack `x_k = 2^k e₀`. -/
example : (∀ (_ : ℕ) (_ : Idx 1), (0 : ℝ) < 1) ∧
    ∀ (k : ℕ) (i : Idx 1),
      (fun k (_ : Idx 1) => (2 : ℝ) ^ k • (basePoint 0 : EucSpace 1)) (k + 1) i
        = (1 : ℝ) • (fun k (_ : Idx 1) => (2 : ℝ) ^ k • (basePoint 0 : EucSpace 1)) k i
          + (1 : ℝ) • (fun k (_ : Idx 1) => (2 : ℝ) ^ k • (basePoint 0 : EucSpace 1)) k i + 0 :=
  ⟨fun _ _ => one_pos, fun _ _ => by
    simp only [pow_succ]
    module⟩

end Perspective
end Transformer

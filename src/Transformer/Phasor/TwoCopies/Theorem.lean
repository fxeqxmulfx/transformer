module

public import Transformer.Phasor.TwoCopies.Logits
public import Transformer.Phasor.TwoCopies.Margin

/-!
# Two copies of any task through main's layers

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

**T3c for two copies.** Let `q₀, q₁ : ℤ/p × ℤ/p → ℤ/p` be any two tasks, `p ≥ 2`, with the
operands `x₀, y₀, x₁, y₁` at the positions `0, …, 3`. A one-stream model of main's shape with the
width `D ≥ 4p + 7`, the head width `d ≥ max(p, 4)`, four heads, `F ≥ 2p` units and `L ≥ 2` layers
puts its largest logit at the position `3` on `q₀(x₀, y₀) + q₁(x₁, y₁)` and nowhere else, for
every `ε ≥ 0` and every depth scale `s > 0` (`exists_oneModel_twoCopies`). By R0 so does main's
model with mHC (`exists_model_twoCopies`). The projections are real-linear, as
`Transformer.Phasor` allows: at main's widths, `D = 128`, `d = 32` and `F = 171` complex units,
that is every `p ≤ 30`, and the example is `x³ + xy` over `ℤ/29`. Main's own projections are
complex-linear, and with them the design needs `D ≥ 8p + 8`, so `p ≤ 15` (NV8, by hand).

The logit of `v` is `κ (κ_emb (b² + [v = t₃] + cos(θ_v − θ_{t₃})) + A cos(θ_v − θ_u))`
(`logits_twoModel`), `u` the answer, and `Margin` makes the answer's term win
(`logits_twoModel_lt`).
-/

@[expose] public section

open Finset Real

namespace Transformer.Phasor.TwoCopies

variable {D d F p : ℕ}

/-- The bias of the construction, `b = 5/(s (1 − cos(2π/p)))`. -/
noncomputable def bTwo (p : ℕ) (s : ℝ) : ℝ := 5 / (s * (1 - cos (2 * π / p)))

/-- The gain of the second branch, which makes its product `√(D ε) + 1`. -/
noncomputable def gTwo (D p : ℕ) (eps s : ℝ) : ℝ :=
  (Real.sqrt (D * eps) + 1) / ampProd D eps (bTwo p s) s

variable {e : Slot p → EucSpace D} {f : Fin p → EucSpace d} {ι₀ : Fin 2 × Fin p ↪ Fin F}
  {ι₁ : Unit ↪ Fin F} {q : Fin 2 → Fin p → Fin p → Fin p}

/-- **The construction computes the answer**: at the position `3` of every depth `L ≥ 2` the logit
of `w₀ + w₁` beats every other logit, for any orthonormal slots and codes. -/
theorem logits_twoModel_lt (hp : 2 ≤ p) (he : Orthonormal ℝ e) (hf : Orthonormal ℝ f)
    (hd : 4 ≤ d) (hD : 0 < D) {eps s : ℝ} (heps : 0 ≤ eps) (hs : 0 < s) (tokens : ℕ → Fin p)
    {L : ℕ} (hL : 2 ≤ L) {v : Fin p} (hv : v ≠ copyVals q tokens 0 + copyVals q tokens 1) :
    (twoModel e f ι₀ ι₁ q eps (bTwo p s) (gTwo D p eps s)).logits eps s tokens L 3 v <
      (twoModel e f ι₀ ι₁ q eps (bTwo p s) (gTwo D p eps s)).logits eps s tokens L 3
        (copyVals q tokens 0 + copyVals q tokens 1) := by
  have hc := one_sub_cos_pos hp
  have hb : 0 < bTwo p s := div_pos (by norm_num) (mul_pos hs hc)
  obtain ⟨κ, hκ, hlog⟩ := logits_twoModel (ι₀ := ι₀) (ι₁ := ι₁) (q := q) he hf hd hD heps hb s
    (gTwo D p eps s) tokens hL
  rw [hlog, hlog, sub_self, cos_zero]
  refine mul_lt_mul_of_pos_left ?_ hκ
  have key : 3 * kEmb D eps (bTwo p s) <
      ampOut D eps (bTwo p s) s (gTwo D p eps s) * (1 - cos (2 * π / p)) :=
    three_kEmb_lt hD heps hs hc
  have hK := kEmb_pos (b := bTwo p s) hD heps
  have hA : 0 < ampOut D eps (bTwo p s) s (gTwo D p eps s) := by
    by_contra h
    have := mul_nonpos_of_nonpos_of_nonneg (not_lt.1 h) hc.le
    linarith
  have hi1 : (if v = toks tokens 3 then (1 : ℝ) else 0) ≤ 1 := by split_ifs <;> norm_num
  have hi0 : (0 : ℝ) ≤ if copyVals q tokens 0 + copyVals q tokens 1 = toks tokens 3 then 1 else 0 :=
    by split_ifs <;> norm_num
  have h1 := cos_le_one (angle p v - angle p (toks tokens 3))
  have h2 := neg_one_le_cos (angle p (copyVals q tokens 0 + copyVals q tokens 1) -
    angle p (toks tokens 3))
  have hK3 := mul_le_mul_of_nonneg_left (show (if v = toks tokens 3 then (1 : ℝ) else 0) +
    cos (angle p v - angle p (toks tokens 3)) ≤
      (if copyVals q tokens 0 + copyVals q tokens 1 = toks tokens 3 then 1 else 0) +
        cos (angle p (copyVals q tokens 0 + copyVals q tokens 1) - angle p (toks tokens 3)) + 3
    by linarith) hK.le
  have hAc := mul_le_mul_of_nonneg_left (cos_angle_sub_le hv) hA.le
  nlinarith

/-- Two copies of `x³ + xy` over `ℤ/29` at main's widths with real-linear projections, on the zero
tokens, against `v = 1`. -/
example (ι₀ : Fin 2 × Fin 29 ↪ Fin 171) (ι₁ : Unit ↪ Fin 171) :=
  logits_twoModel_lt (ι₀ := ι₀) (ι₁ := ι₁) (q := fun _ x y => x ^ 3 + x * y) (by norm_num)
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128))
    (orthonormal_coordCodes (by norm_num : 29 ≤ 32)) (by norm_num) (by norm_num)
    (by norm_num : (0 : ℝ) ≤ 1 / 8388608) (by positivity : (0 : ℝ) < 1 / Real.sqrt 8)
    (fun _ => 0) (L := 4) (by norm_num) (v := 1) (by decide)

/-- **T3c, two copies, the ordinary residual.** For any two tasks `q₀, q₁` over `ℤ/p`, `p ≥ 2`, a
one-stream model with `D ≥ 4p + 7`, `d ≥ max(p, 4)`, four heads and `F ≥ 2p` units puts, at every
depth `L ≥ 2`, its only largest logit at the position `3` on `q₀(x₀, y₀) + q₁(x₁, y₁)`. -/
theorem exists_oneModel_twoCopies (hp : 2 ≤ p) (hD : 4 * p + 7 ≤ D) (hd : p ≤ d) (hd4 : 4 ≤ d)
    (hF : 2 * p ≤ F) {eps : ℝ} (heps : 0 ≤ eps) {s : ℝ} (hs : 0 < s)
    (q : Fin 2 → Fin p → Fin p → Fin p) :
    ∃ M : OneModel D d 4 F p, ∀ (tokens : ℕ → Fin p) (L : ℕ), 2 ≤ L → ∀ v,
      v ≠ q 0 (tokens 0) (tokens 1) + q 1 (tokens 2) (tokens 3) →
        M.logits eps s tokens L 3 v <
          M.logits eps s tokens L 3 (q 0 (tokens 0) (tokens 1) + q 1 (tokens 2) (tokens 3)) :=
  ⟨twoModel (coordSlots hD) (coordCodes hd) (finProdFinEquiv.toEmbedding.trans (Fin.castLEEmb hF))
      ⟨fun _ => ⟨0, by omega⟩, fun _ _ _ => rfl⟩ q eps (bTwo p s) (gTwo D p eps s),
    fun tokens _ hL _ hv => logits_twoModel_lt hp (orthonormal_coordSlots hD)
      (orthonormal_coordCodes hd) hd4 (by omega) heps hs tokens hL hv⟩

/-- **T3c, two copies, main's model**: the same with mHC, its routers plain (R0). -/
theorem exists_model_twoCopies (hp : 2 ≤ p) (hD : 4 * p + 7 ≤ D) (hd : p ≤ d) (hd4 : 4 ≤ d)
    (hF : 2 * p ≤ F) {eps : ℝ} (heps : 0 ≤ eps) {s : ℝ} (hs : 0 < s)
    (q : Fin 2 → Fin p → Fin p → Fin p) :
    ∃ M : Model D d 4 F p, ∀ (tokens : ℕ → Fin p) (L : ℕ), 2 ≤ L → ∀ v,
      v ≠ q 0 (tokens 0) (tokens 1) + q 1 (tokens 2) (tokens 3) →
        M.logits eps s tokens L 3 v <
          M.logits eps s tokens L 3 (q 0 (tokens 0) (tokens 1) + q 1 (tokens 2) (tokens 3)) := by
  obtain ⟨M, hM⟩ := exists_oneModel_twoCopies hp hD hd hd4 hF heps hs q
  refine ⟨M.withRouters (fun _ => Router.plain D) (fun _ => Router.plain D),
    fun tokens L hL v hv => ?_⟩
  simp only [M.logits_withRouters (fun _ => Router.plain_isPlain D)
    (fun _ => Router.plain_isPlain D)]
  exact hM tokens L hL v hv

/-- **Main's widths, real-linear projections**: `D = 128`, `d = 32`, four heads, `F = 171`,
`ε = 2⁻²³`, four layers (`α = 1/√8`), two copies of `x³ + xy` over `ℤ/29`, the ordinary
residual. -/
example : ∃ M : OneModel 128 32 4 171 29, ∀ (tokens : ℕ → Fin 29) (L : ℕ), 2 ≤ L → ∀ v,
    v ≠ (tokens 0 ^ 3 + tokens 0 * tokens 1) + (tokens 2 ^ 3 + tokens 2 * tokens 3) →
      M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L 3 v <
        M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L 3
          ((tokens 0 ^ 3 + tokens 0 * tokens 1) + (tokens 2 ^ 3 + tokens 2 * tokens 3)) :=
  exists_oneModel_twoCopies (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by positivity) (fun _ x y => x ^ 3 + x * y)

/-- The same with main's mHC. -/
example : ∃ M : Model 128 32 4 171 29, ∀ (tokens : ℕ → Fin 29) (L : ℕ), 2 ≤ L → ∀ v,
    v ≠ (tokens 0 ^ 3 + tokens 0 * tokens 1) + (tokens 2 ^ 3 + tokens 2 * tokens 3) →
      M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L 3 v <
        M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L 3
          ((tokens 0 ^ 3 + tokens 0 * tokens 1) + (tokens 2 ^ 3 + tokens 2 * tokens 3)) :=
  exists_model_twoCopies (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by positivity) (fun _ x y => x ^ 3 + x * y)

end Transformer.Phasor.TwoCopies

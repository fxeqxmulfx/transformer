module

public import Transformer.SequenceModel.Attention.Expressivity.Model
public import Mathlib.Algebra.Field.ZMod

/-!
# Every finite binary task, including powered modular division, is representable

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E8.
The theorem concerns the full supplied real-arithmetic forward: tied
embedding/readout, arbitrary fixed RoPE tables, causal softmax, clipped
XSA, Peri-RMSNorm and SwiGLU. It is an explicit sufficient finite capacity
theorem, independent of whether an optimizer discovers these weights.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini

/-- E8: every binary finite function has a single strict classifier in the
actual source-shaped Transformer, at every finite padding and positive depth.
The same weights work for every phase table because their Q/K are zero;
the attention value/XSA/context path remains active. -/
theorem exists_binary_classifier (p : ℕ) (hp : 0 < p) (padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) (f : Fin p → Fin p → Fin p) :
    ∃ w : Parameters (config p hp padding extraDepth),
      ∀ angle : Fin (extraDepth + 1) → Fin 2 →
        Fin ((config p hp padding extraDepth).head_dim / 2) → ℝ,
      ∀ a b q : Fin p, q ≠ f a b →
        score (config p hp padding extraDepth) eps xeps angle 1 w ![a, b] q <
        score (config p hp padding extraDepth) eps xeps angle 1 w ![a, b] (f a b) := by
  let cfg := config p hp padding extraDepth
  let e := banks p hp padding extraDepth
  let layout := hiddenLayout p hp padding extraDepth
  have hw : 3 < cfg.d_model := (width_room p hp padding).1
  refine ⟨encode cfg (taskModel cfg rfl eps e layout f), ?_⟩
  intro angle a b q hq
  rw [task_score cfg rfl hw heps hx, task_score cfg rfl hw heps hx]
  simp only [hq, ↓reduceIte, mul_zero, mul_one]
  exact mul_pos (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num)) (amplitude_pos cfg hw heps)

/-- Natural residue labels and their field-valued powered quotient. Division
at zero is total in Lean; the mathematical task uses nonzero denominators. -/
def powerLabel (p : ℕ) [Fact p.Prime] (k : ℕ) (a b : Fin p) : Fin p :=
  (ZMod.finEquiv p).symm (((ZMod.finEquiv p a) / (ZMod.finEquiv p b)) ^ k)

/-- E8: for every prime and every natural exponent, a fixed finite Transformer
strictly classifies every numerator and nonzero denominator. The stronger
binary theorem also specifies harmless totalized outputs at denominator zero. -/
theorem exists_power_division_classifier (p : ℕ) [Fact p.Prime] (k padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) :
    ∃ w : Parameters (config p (Fact.out : p.Prime).pos padding extraDepth),
      ∀ angle : Fin (extraDepth + 1) → Fin 2 →
        Fin ((config p (Fact.out : p.Prime).pos padding extraDepth).head_dim / 2) → ℝ,
      ∀ a b q : Fin p, q ≠ powerLabel p k a b →
        score (config p (Fact.out : p.Prime).pos padding extraDepth) eps xeps angle 1 w ![a, b] q <
        score (config p (Fact.out : p.Prime).pos padding extraDepth) eps xeps angle 1 w ![a, b]
          (powerLabel p k a b) := by
  exact exists_binary_classifier p (Fact.out : p.Prime).pos padding extraDepth heps hx (powerLabel p k)

example := exists_binary_classifier 3 (by decide) 0 0 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) (fun a b => a + 2 * b)
example := exists_binary_classifier 3 (by decide) 100 7 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) (fun a b => if a = b then 0 else 1)
example := exists_power_division_classifier 2 0 0 0 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
example := exists_power_division_classifier 3 17 0 4 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)

private abbrev cfg := config 3 (by decide) 0 2
private abbrev e := banks 3 (by decide) 0 2
private abbrev layout := hiddenLayout 3 (by decide) 0 2
private def phases := sourceAngles cfg (T := 2) (fun _ => true) (fun j => j.val)

example := embedding_normalize cfg rfl (by decide) (eps := 1 / 8388608) (by norm_num) e 2
example := amplitude_pos cfg (by decide) (eps := 1 / 8388608) (by norm_num)
example := task_hidden cfg rfl (by decide) (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) e layout (fun a b => a + 2 * b) phases 1 2 2
example := task_score cfg rfl (by decide) (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) e layout (fun a b => a + 2 * b) phases 1 2 2

section
local instance : Fact (Nat.Prime 97) := ⟨by decide⟩
example := exists_power_division_classifier 97 5 0 7 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
end

section
local instance : Fact (Nat.Prime 193) := ⟨by norm_num⟩
example := exists_power_division_classifier 193 0 256 2 (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
end

end Transformer.SequenceModel.Attention.Expressivity

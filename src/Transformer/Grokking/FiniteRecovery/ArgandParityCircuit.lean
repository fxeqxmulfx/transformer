module

public import Transformer.Grokking.FiniteRecovery.ArgandWeightCoordinates
public import Transformer.Grokking.FiniteRecovery.ArgandParityTree

/-! Native product FFNs with the width-64 stream and width-256 hidden layer
of ARGAND.md. Each layer writes to unused parent coordinates. All three RMS
operations and the residual scale preserve a common positive coefficient. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Parity

open Transformer.PhasorV2

abbrev Stream := CSpace (4 * 16)

noncomputable def offset : ℝ := 1 / 1000000

noncomputable def gain (x : Stream) : ℝ := Real.sqrt 64 / (‖x‖ + offset)

theorem gain_pos (x : Stream) : 0 < gain x := by
  dsimp [gain, offset]
  positivity

theorem rms_coord (x : Stream) (i : Fin 64) :
    argandRMS offset x i = (gain x : ℂ) * x i := rfl

def idx (i : ℕ) : Fin 64 := ⟨i % 64, Nat.mod_lt _ (by decide)⟩

noncomputable def coord (i : Fin 64) : Stream →ₗ[ℂ] ℂ := PiLp.projₗ 2 _ i

noncomputable def assemble {E : Type*} [AddCommMonoid E] [Module ℂ E]
    (p : Fin 64 → E →ₗ[ℂ] ℂ) : E →ₗ[ℂ] Stream :=
  (WithLp.linearEquiv 2 ℂ (Fin 64 → ℂ)).symm.toLinearMap.comp (LinearMap.pi p)

theorem assemble_apply {E : Type*} [AddCommMonoid E] [Module ℂ E]
    (p : Fin 64 → E →ₗ[ℂ] ℂ) (x : E) (i : Fin 64) : assemble p x i = p i x := rfl

/-- The populated band shares one positive scale. All coordinates below it are free. -/
def Band (b r m : ℕ) (x : Stream) : Prop :=
  (∀ i : Fin 64, i.val < b → x i = 0) ∧
  ∃ a : ℝ, 0 < a ∧ ∀ i : Fin 64, b ≤ i.val → i.val < 2*b →
    x i = ((a * value m r i.val : ℝ) : ℂ)

noncomputable def children (side : ℕ) : Stream →ₗ[ℂ] Fin 256 → ℂ :=
  LinearMap.pi fun j => coord (idx (2*j.val+side))

/-- The FFN multiplies the two child coordinates of each requested parent. -/
noncomputable def parents (b : ℕ) : FFN 64 256 where
  Wg := children 0
  Wu := children 1
  WF := assemble fun i => if b ≤ i.val ∧ i.val < 2*b then
    LinearMap.proj (R := ℂ) (φ := fun _ : Fin 256 => ℂ) ⟨i.val, by omega⟩ else 0

private theorem parents_apply (b : ℕ) (x : Stream) (i : Fin 64) :
    let f := parents b
    f.WF (f.Wg x * f.Wu x) i =
      if b ≤ i.val ∧ i.val < 2*b then x (idx (2*i.val)) * x (idx (2*i.val+1)) else 0 := by
  change (if b ≤ i.val ∧ i.val < 2*b then
    LinearMap.proj (R := ℂ) (φ := fun _ : Fin 256 => ℂ) ⟨i.val, by omega⟩ else 0)
      ((children 0) x * (children 1) x) = _
  split <;> rfl

noncomputable def productLayer (b : ℕ) : Layer 4 16 where
  attentionProjections := ⟨0,0,0,0⟩
  feedForward := parents b
  scoreMultiplier := 0
  attentionGates := ⟨0,0⟩
  feedForwardGates := ⟨0,0⟩

/-- One native layer doubles the number of factors represented at an active node. -/
theorem productLayer_band {n b r m : ℕ} (hb : 4*b ≤ 64) {scale : ℝ} (hs : 0 < scale)
    (x : Fin n → Stream) (i : Fin n) (hx : Band (2*b) r m (x i)) :
    Band b (r+1) m ((productLayer b).forward offset scale 0 x i) := by
  let z := argandRMS offset (x i)
  let f := parents b
  let out := f.WF (f.Wg z * f.Wu z)
  have he : (productLayer b).forward offset scale 0 x i =
      x i + (scale : ℂ) • argandRMS offset out := by
    simp [Layer.forward, Layer.forwardWith, productLayer, sublayer, attentionWith, argandRMS, z, out, f]
  rw [he]
  obtain ⟨hz, a, ha, hx⟩ := hx
  have hcoord (j : Fin 64) : (x i + (scale : ℂ) • argandRMS offset out) j =
      x i j + (scale : ℂ) * (gain out * out j) := rfl
  constructor
  · intro j hj
    rw [hcoord, hz j (by omega)]
    simp [out, f, parents_apply, show ¬ b ≤ j.val by omega]
  · refine ⟨scale * gain out * (gain (x i) * a)^2,
      by positivity [gain_pos out, gain_pos (x i)], ?_⟩
    intro j hj hj'
    have hc (d : ℕ) (hd : d ≤ 1) : (idx (2*j.val+d)).val = 2*j.val+d :=
      Nat.mod_eq_of_lt (by omega)
    have hv (d : ℕ) (hd : d ≤ 1) : z (idx (2*j.val+d)) =
        (gain (x i) : ℂ) * ((a * value m r (2*j.val+d) : ℝ) : ℂ) := by
      rw [show z = argandRMS offset (x i) from rfl, rms_coord, hx _ (by rw [hc d hd]; omega)
        (by rw [hc d hd]; omega), hc d hd]
    rw [hcoord, hz j hj']
    change _ + _ * (_ * ((parents b).WF _ j)) = _
    rw [parents_apply, ite_eq_left ⟨hj, hj'⟩, hv 1 le_rfl]
    have hv0 := hv 0 (by decide)
    simp only [Nat.add_zero] at hv0
    rw [hv0, value]
    push_cast
    ring

example {n m : ℕ} (x : Fin n → Stream) (i : Fin n) (h : Band 32 0 m (x i)) :=
  productLayer_band (b := 16) (by decide) (by norm_num : (0:ℝ) < 1) x i h

end Transformer.Grokking.FiniteRecovery.Argand.Parity

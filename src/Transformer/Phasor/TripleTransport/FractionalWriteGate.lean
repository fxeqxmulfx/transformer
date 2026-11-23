module

public import Transformer.Phasor.TripleTransport.LocalMemory

/-!
# Write strength and interference at independently chosen addresses

Source: Phasor's two-cell adaptation of Percepta's *Spotlight Memory*,
“How Spotlight works”. The write address may differ from the read address.
A scalar from the causal hidden state scales the overwrite. The stored
payload is an arbitrary full-width vector.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.FractionalWriteGate

open LocalMemory
variable {d : ℕ}

/-- Request the fraction g of the current read error at a write address.
The existing least-change update distributes that request across two cells. -/
noncomputable def store (m : Memory d) (a : Address) (g : ℝ) (v : V d) : Memory d :=
  write m (some a) (LocalMemory.mix g (sample m a) v)

/-- Even an overlapping distractor leaves every memory cell intact when
its hidden-state gate is zero. A unit gate effects a complete overwrite. -/
theorem endpoints (m : Memory d) (a : Address) (v : V d) :
    store m a 0 v = m ∧ sample (store m a 1 v) a = v := by
  constructor
  · funext i j
    simp [store,write,LocalMemory.mix]
  · rw [store,read_after_write]
    funext j
    simp [LocalMemory.mix]

/-- Rereading the exact write address moves by the gated residual. -/
theorem same_address (m : Memory d) (a : Address) (g : ℝ) (v : V d) (j : Fin d) :
    sample (store m a g v) a j - sample m a j = g*(v j-sample m a j) := by
  rw [store,read_after_write]
  simp [LocalMemory.mix]

/-- With one shared cell, the measured change in a neighboring read is
the payload error times the gate and the overlap of the two interpolation
weights, divided by the source stencil's squared norm. -/
theorem adjacent (m : Memory d) (i : ℤ) (s t g : ℝ) (v : V d) (j : Fin d) :
    sample (store m ⟨i,s⟩ g v) ⟨i+1,t⟩ j - sample m ⟨i+1,t⟩ j =
      g*(1-t)*s / energy ⟨i,s⟩ * (v j - sample m ⟨i,s⟩ j) := by
  have ne₁ : i+1 ≠ i := by omega
  have ne₂ : i+1+1 ≠ i := by omega
  have ne₃ : i+1+1 ≠ i+1 := by omega
  simp [store,sample,write,LocalMemory.mix,ne₁,ne₂,ne₃]
  ring

/-- Nonoverlapping read addresses are unaffected at every gate strength. -/
theorem apart (m : Memory d) (a b : Address) (g : ℝ) (v : V d)
    (separate : a.left+1 < b.left ∨ b.left+1 < a.left) :
    sample (store m a g v) b = sample m b :=
  disjoint_read _ _ _ _ separate

example : sample (store (fun _ => 0 : Memory 1) ⟨0,1/2⟩ (1/2) (fun _ => 4))
    ⟨1,1/2⟩ 0 - sample (fun _ => 0 : Memory 1) ⟨1,1/2⟩ 0 = 1 := by
  have h := adjacent (fun _ => 0 : Memory 1) 0 (1/2) (1/2) (1/2) (fun _ => 4) (0 : Fin 1)
  norm_num [energy,sample,LocalMemory.mix] at h ⊢
  exact h

example : sample (store (fun _ => 0 : Memory 2) ⟨1,1/3⟩ (1/4) (fun _ => 8))
    ⟨3,1/2⟩ = sample (fun _ => 0 : Memory 2) ⟨3,1/2⟩ := by
  apply apart
  left
  decide

end Transformer.Phasor.TripleTransport.FractionalWriteGate

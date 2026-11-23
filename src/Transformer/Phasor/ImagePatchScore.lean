module

public import Transformer.Phasor.DifferencedDivisionScore

/-! DiffGPT at 5a37c515, research/current_grokking/probes.py, fashion.encode.
The row-major prompt starts with BOS and contains the nearest-even means of
196 four-pixel patches. Supervision is restricted to the last output. -/
@[expose] public section

namespace Transformer.Phasor.ImagePatchScore

def rounded (s : ℕ) : ℕ :=
  let q := s / 4
  if 2 < s%4 ∨ (s%4 = 2 ∧ q%2 = 1) then q+1 else q

/-- The reported integer intensity is within half a count of the exact mean.
If the source sum is halfway between two multiples of four, its result is even. -/
theorem nearest_even (s : ℕ) : s ≤ 4*rounded s+2 ∧ 4*rounded s ≤ s+2 ∧
    (s%4 = 2 → rounded s % 2 = 0) := by
  dsimp only [rounded]
  split <;> omega

/-- Every valid four-pixel patch stays in the 256-class intensity vocabulary. -/
theorem bounded (a b c d : Fin 256) : rounded (a.val+b.val+c.val+d.val) < 256 := by
  obtain ⟨_,h,_⟩ := nearest_even (a.val+b.val+c.val+d.val)
  have := a.isLt
  have := b.isLt
  have := c.isLt
  have := d.isLt
  omega

/-- Flatten the 14-by-14 block grid. Within a block, i and j select the
two rows and two columns from the original 28-by-28 image. -/
def offset (r c : Fin 14) (i j : Fin 2) : Fin 784 :=
  ⟨(2*r.val+i.val)*28+(2*c.val+j.val), by omega⟩

theorem block_unique (r c r' c' : Fin 14) (i j i' j' : Fin 2)
    (same : offset r c i j = offset r' c' i' j') :
    r=r' ∧ c=c' ∧ i=i' ∧ j=j' := by
  have eq := congrArg Fin.val same
  simp only [offset] at eq
  refine ⟨Fin.ext ?_,Fin.ext ?_,Fin.ext ?_,Fin.ext ?_⟩ <;> omega

/-- Every pixel occurs at the corresponding block and within-block offset. -/
theorem covers (k : Fin 784) : ∃ r c i j, offset r c i j = k := by
  let row := k.val / 28
  let col := k.val % 28
  refine ⟨⟨row/2,by dsimp [row]; omega⟩,⟨col/2,by dsimp [col]; omega⟩,
    ⟨row%2,by omega⟩,⟨col%2,by omega⟩,Fin.ext ?_⟩
  dsimp [offset,row,col]
  omega

def patch (image : Fin 784 → Fin 256) (r c : Fin 14) : Fin 256 :=
  ⟨rounded ((image (offset r c 0 0)).val+(image (offset r c 0 1)).val+
    (image (offset r c 1 0)).val+(image (offset r c 1 1)).val),bounded _ _ _ _⟩

def target (label : Fin 10) (atPos : Fin 197) : Option ℕ :=
  if atPos.val=196 then some (256+label.val) else none

/-- Scoring all supervised positions reduces to predicting the single
shifted class at the final patch, for arbitrary earlier predictions. -/
theorem masked_exact (p : Fin 197 → ℕ) (y : Fin 10) :
    (∀ t v, target y t = some v → p t = v) ↔ p 196 = 256+y.val := by
  constructor
  · intro h
    exact h 196 _ (by rfl)
  · intro h t v eq
    by_cases last : t.val=196
    · have ht : t=196 := Fin.ext last
      subst t
      simpa [target] using (h.trans (by simpa [target] using eq))
    · simp [target,last] at eq

example : (rounded 6,rounded 10,rounded 14,rounded 18)=(2,2,4,4) := by decide
example : offset 13 13 1 1 = 783 := by decide
example : (13:Fin 14)=13 ∧ (13:Fin 14)=13 ∧ (1:Fin 2)=1 ∧ (1:Fin 2)=1 :=
  block_unique 13 13 13 13 1 1 1 1 rfl
example : ∀ t v, target 3 t=some v → (fun _ : Fin 197 => 259) t=v :=
  (masked_exact _ 3).2 rfl
end Transformer.Phasor.ImagePatchScore

module

public import Transformer.Phasor.TripleTransport.DenseInduction

/-!
# Exact overwrite at a learned fractional address

This Phasor experiment takes inspiration from Percepta's *Spotlight Memory*,
“How Spotlight works”, and the latest-value read in the Append-only Lookup
Machine. Its one-dimensional address interpolates two vector cells. A write
fits the new payload to this same interpolation using the least squared
change in the two stored vectors, coordinate by coordinate.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.LocalMemory
variable {d : ℕ}
abbrev V (d : ℕ) := Fin d → ℝ
abbrev Memory (d : ℕ) := ℤ → V d

structure Address where
  left : ℤ
  fraction : ℝ

noncomputable def locate (s : ℝ) : Address := ⟨⌊s⌋, s - ⌊s⌋⟩

/-- Both writes and reads use one projection of a normalized dense vector. -/
noncomputable def project (w x : V d) : Option Address :=
  if DenseInduction.product x x = 0 then none else
    some (locate (d * DenseInduction.product w x / Real.sqrt (DenseInduction.product x x)))

def mix (k : ℝ) (x y : V d) : V d := fun j => x j + k * (y j - x j)

def sample (m : Memory d) (a : Address) : V d := mix a.fraction (m a.left) (m (a.left+1))

def energy (a : Address) : ℝ := (1-a.fraction)^2+a.fraction^2

theorem energy_pos (a : Address) : 0 < energy a := by
  unfold energy
  nlinarith [sq_nonneg (a.fraction-1/2)]

noncomputable def write (m : Memory d) (p : Option Address) (value : V d) : Memory d :=
  match p with
  | none => m
  | some a => fun i =>
    let weight := if i = a.left then 1-a.fraction else if i = a.left+1 then a.fraction else 0
    fun j => m i j + weight / energy a * (value j - sample m a j)

def get (m : Memory d) (q : V d → Option Address) (x : V d) : V d :=
  match q x with
  | none => 0
  | some a => sample m a

theorem integral_write (m : Memory d) (i : ℤ) (v : V d) :
    write m (some ⟨i,0⟩) v = Function.update m i v := by
  funext j k
  by_cases h : j=i <;> by_cases h₁ : j=i+1 <;> simp [write,sample,energy,mix,Function.update,h,h₁]

theorem untouched (m : Memory d) (a : Address) (v : V d) (i : ℤ)
    (left : i ≠ a.left) (right : i ≠ a.left+1) :
    write m (some a) v i = m i := by funext j; simp [write,left,right]

/-- Reading a fractional address immediately after writing returns exactly
the new vector for every incoming memory and payload. -/
theorem read_after_write (m : Memory d) (a : Address) (v : V d) :
    sample (write m (some a) v) a = v := by
  funext j
  have units := ne_of_gt (energy_pos a)
  have next : a.left+1≠a.left := by omega
  simp [write,sample,mix,next]
  field_simp
  unfold energy
  ring

/-- Nonoverlapping two-cell supports preserve one another's reads. -/
theorem disjoint_read (m : Memory d) (a b : Address) (v : V d)
    (apart : a.left+1 < b.left ∨ b.left+1 < a.left) :
    sample (write m (some a) v) b = sample m b := by
  unfold sample
  rw [untouched m a v b.left (by omega) (by omega),
    untouched m a v (b.left+1) (by omega) (by omega)]

noncomputable def applyWrites (m : Memory d) : List (Address × V d) → Memory d
  | [] => m
  | (a,v) :: later => applyWrites (write m (some a) v) later

/-- The value at a fractional address survives any subsequent stream of
writes whose two-cell supports stay disjoint from that address. -/
theorem sample_after_apart_writes (m : Memory d) (a : Address) (later : List (Address × V d))
    (away : ∀ e ∈ later, e.1.left+1 < a.left ∨ a.left+1 < e.1.left) :
    sample (applyWrites m later) a = sample m a := by
  induction later generalizing m with
  | nil => rfl
  | cons e es ih =>
    obtain ⟨b,v⟩ := e
    rw [applyWrites, ih _ (fun e he => away e (List.mem_cons_of_mem _ he)),
      disjoint_read _ b a _ (away _ List.mem_cons_self)]

theorem integral_read (m : Memory d) (q : V d → Option Address) (x : V d) (i : ℤ)
    (key : q x = some ⟨i,0⟩) : get m q x = m i := by
  funext j
  simp [get,key,sample,mix]

/-- The last completed integral write replaces either previous stored value. -/
theorem latest (m : Memory d) (q : V d → Option Address) (x a b : V d) (i : ℤ)
    (key : q x=some ⟨i,0⟩) :
    get (write (write m (q x) a) (q x) b) q x = b := by
  rw [integral_read _ _ _ _ key,key,integral_write, Function.update_self]

def touched : List (Option Address) → Finset ℤ
  | [] => ∅
  | none :: xs => touched xs
  | some a :: xs => insert a.left (insert (a.left+1) (touched xs))

theorem writes_space (xs : List (Option Address)) : (touched xs).card ≤ 2*xs.length := by
  induction xs with
  | nil => simp [touched]
  | cons a xs ih =>
    cases a with
    | none => simp only [touched,List.length_cons]; omega
    | some a =>
      have h := Finset.card_insert_le (a.left+1) (touched xs)
      have k := Finset.card_insert_le a.left (insert (a.left+1) (touched xs))
      simp only [touched,List.length_cons]
      omega

/-- Pairing the interpolation differential with a scalar cotangent gives the
address and both value cotangents used by the sparse reverse pass. -/
theorem mix_adjoint (k x y dk dx dy g : ℝ) :
    g*(dx+dk*(y-x)+k*(dy-dx)) =
      (g*(y-x))*dk+((1-k)*g)*dx+(k*g)*dy := by ring

/-- With a required read change r, the two cell adjustments u and v have
squared cost at least r²/energy. The excess is the squared component
perpendicular to the interpolation weights, divided by energy. -/
theorem movement_lower_bound (a : Address) (r u v : ℝ)
    (fits : (1-a.fraction)*u+a.fraction*v=r) :
    r^2 / energy a ≤ u^2+v^2 := by
  apply (div_le_iff₀ (energy_pos a)).2
  rw [←fits]
  unfold energy
  nlinarith [sq_nonneg (a.fraction*u-(1-a.fraction)*v)]

/-- The normalized delta used in the write attains the lower bound for every
positive or negative payload residual, including at cell boundaries. -/
theorem movement_attained (a : Address) (r : ℝ) :
    ((1-a.fraction)/energy a*r)^2+(a.fraction/energy a*r)^2 = r^2/energy a := by
  have positive := ne_of_gt (energy_pos a)
  field_simp
  unfold energy
  ring

private def zeroStore : Memory 2 := fun _ => 0
private noncomputable def address := project ![1,3]
private theorem key_at : address ![0,1] = some ⟨6,0⟩ := by
  norm_num [address,project,DenseInduction.product,Fin.sum_univ_two,locate]

example : write zeroStore (some ⟨9,1/3⟩) ![1,2] 3 = zeroStore 3 :=
  untouched _ _ _ _ (by decide) (by decide)
example : sample (write zeroStore (some ⟨3,1/3⟩) ![1,4]) ⟨9,2/7⟩ =
    sample zeroStore ⟨9,2/7⟩ := disjoint_read _ _ _ _ (by left; decide)
example : sample (applyWrites (write zeroStore (some ⟨3,1/3⟩) ![1,4])
    [(⟨-2,1/4⟩,![0,1]),(⟨9,2/7⟩,![2,3])]) ⟨3,1/3⟩ = ![1,4] := by
  rw [sample_after_apart_writes, read_after_write]
  intro e he
  rw [List.mem_cons, List.mem_singleton] at he
  rcases he with rfl | rfl <;> norm_num
example : get zeroStore address ![0,1]=zeroStore 6 := integral_read _ _ _ _ key_at
example : get (write (write zeroStore (address ![0,1]) ![1,0]) (address ![0,1]) ![2,3])
    address ![0,1]=![2,3] := latest _ _ _ _ _ _ key_at
example : (5:ℝ)^2/energy ⟨9,1/2⟩ ≤ 3^2+7^2 :=
  movement_lower_bound ⟨9,1/2⟩ 5 3 7 (by norm_num)

end Transformer.Phasor.TripleTransport.LocalMemory

module

public import Transformer.Phasor.TripleTransport.LocalMemory

/-!
# Chaining a learned sparse read at one causal prefix

Source: the Phasor lookup experiment. Thinking Machines Lab's *Introducing
Inkling 2 Mini*, “Looping design”, motivates sharing weights and gating an
earlier pass's message. Here the repeated block is a single memory read.
Each pass uses the same projection and the same prefix of completed writes.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.LocalReadLoop

open LocalMemory
variable {d : ℕ}
/-- Each pass queries the same memory using the value returned by the last pass. -/
def loop (m : Memory d) (q : V d → Option Address) (x : V d) : ℕ → V d
  | 0 => x
  | k+1 => LocalMemory.get m q (loop m q x k)

/-- A path follows its stored links for the stated number of shared reads. -/
theorem follow (m : Memory d) (q : V d → Option Address) (x : ℕ → V d) (cell : ℕ → ℤ)
    (n : ℕ) (keys : ∀ i, i<n → q (x i)=some ⟨cell i,0⟩)
    (links : ∀ i, i<n → m (cell i)=x (i+1)) : loop m q (x 0) n=x n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [loop,ih (fun i h => keys i (by omega)) (fun i h => links i (by omega)),
      integral_read _ _ _ _ (keys n (by omega)),links n (by omega)]

def context (c : ℝ) (x h : V d) : V d := fun j => x j+c*h j

noncomputable def buildMemory (q : V d → Option Address) (x v : ℕ → V d) : ℕ → Memory d
  | 0 => fun _ => 0
  | n+1 => write (buildMemory q x v n) (q (x n)) (v (n+1))

theorem prefix_eq (q : V d → Option Address) (x y vx vy : ℕ → V d) (n : ℕ)
    (same : ∀ i, i≤n → x i=y i) (values : ∀ i, i≤n → vx i=vy i) :
    buildMemory q x vx n=buildMemory q y vy n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [buildMemory, ih (fun i h => same i (by omega)) (fun i h => values i (by omega)),
      same n (by omega),values (n+1) (by omega)]

/-- At t only predecessors with a successor strictly before t have written. -/
theorem causal (q : V d → Option Address) (c : ℝ) (x y hx hy : ℕ → V d) (t passes : ℕ)
    (same : ∀ i, i≤t → x i=y i) (state : ∀ i, i≤t → hx i=hy i) :
    loop (buildMemory q (fun i => context c (x i) (hx i)) x (t-1)) q (context c (x t) (hx t)) passes =
    loop (buildMemory q (fun i => context c (y i) (hy i)) y (t-1)) q (context c (y t) (hy t)) passes := by
  have keys (i : ℕ) (hi : i≤t) : context c (x i) (hx i)=context c (y i) (hy i) := by
    rw [same i hi,state i hi]
  rw [prefix_eq q _ _ _ _ _ (fun i h => keys i (by omega)) (fun i h => same i (by omega)),keys t (by omega)]

/-- The second pass reuses the first query at zero learned message gain
and follows the first retrieved value at unit gain. -/
def gated (m : Memory d) (q : V d → Option Address) (x : V d) (g : ℝ) : V d :=
  LocalMemory.get m q (LocalMemory.mix g x (LocalMemory.get m q x))

theorem gate_ends (m : Memory d) (q : V d → Option Address) (x : V d) :
    gated m q x 0=loop m q x 1 ∧ gated m q x 1=loop m q x 2 := by
  have zero : LocalMemory.mix 0 x (LocalMemory.get m q x)=x := by funext j; simp [LocalMemory.mix]
  have one : LocalMemory.mix 1 x (LocalMemory.get m q x)=LocalMemory.get m q x := by funext j; simp [LocalMemory.mix]
  simp [gated,zero,one,loop]

private def firstKey : V 3 := ![1,0,0]
private def secondKey : V 3 := ![0,1,0]
private def third : V 3 := ![0,0,1]
private noncomputable def address := project ![1,3,5]
private noncomputable def store (v : V 3) : Memory 3 :=
  write (write (fun _ => 0) (address firstKey) secondKey) (address secondKey) v

private theorem first_at : address firstKey = some ⟨3,0⟩ := by
  norm_num [address,project,firstKey,DenseInduction.product,Fin.sum_univ_three,locate]
private theorem second_at : address secondKey = some ⟨9,0⟩ := by
  norm_num [address,project,secondKey,DenseInduction.product,Fin.sum_univ_three,locate]

/-- Two stores produced by the specified projection and writes give the same
first value. A rule seeing only that value cannot return both second answers. -/
theorem one_read_insufficient :
    ¬ ∃ f : V 3 → V 3 → V 3 → V 3, ∀ w x y z : V 3,
      let q := project w
      let m := write (write (fun _ => 0) (q x) y) (q y) z
      f w x (LocalMemory.get m q x) = loop m q x 2 := by
  rintro ⟨f,works⟩
  have h : f ![1,3,5] firstKey (LocalMemory.get (store third) address firstKey) =
      loop (store third) address firstKey 2 := works _ _ secondKey third
  have k : f ![1,3,5] firstKey (LocalMemory.get (store (-third)) address firstKey) =
      loop (store (-third)) address firstKey 2 := works _ _ secondKey (-third)
  have go (v : V 3) : LocalMemory.get (store v) address firstKey=secondKey := by
    rw [integral_read _ _ _ _ first_at]
    simp [store,first_at,second_at,integral_write]
  have next (v : V 3) : LocalMemory.get (store v) address secondKey=v := by
    rw [integral_read _ _ _ _ second_at]
    simp [store,second_at,integral_write]
  simp only [loop,go,next] at h k
  rw [h] at k
  have impossible := congrFun k 2
  norm_num [third] at impossible

example : loop (store third) address firstKey 1 = secondKey := by
  let path : ℕ → V 3 := fun i => if i=0 then firstKey else secondKey
  apply follow (store third) address path (fun _ => 3) 1
  · intro i hi
    have : i=0 := by omega
    subst i
    exact first_at
  · intro i hi
    have : i=0 := by omega
    subst i
    simp [store,first_at,second_at,integral_write,path]

private def later (i : ℕ) : V 3 := if i≤4 then firstKey else secondKey
example : buildMemory address (fun _ => firstKey) later 3 =
    buildMemory address later (fun _ => firstKey) 3 := by
  apply prefix_eq <;> intro i hi <;> simp [later,show i≤4 by omega]
example : loop (buildMemory address (fun _ => context 1 firstKey firstKey) (fun _ => firstKey) 3)
    address (context 1 firstKey firstKey) 2 =
    loop (buildMemory address (fun i => context 1 (later i) (later i)) later 3)
      address (context 1 (later 4) (later 4)) 2 := by
  apply causal address 1 (fun _ => firstKey) later (fun _ => firstKey) later 4 2
  all_goals intro i hi; simp [later,hi]

/-- The learned gains may use the whole current causal hidden state. Prefix
agreement preserves both passes and the final residual for any such gains. -/
theorem gated_causal (q : V d → Option Address) (c : ℝ) (msg gain : V d → ℝ)
    (x y hx hy : ℕ → V d) (t : ℕ)
    (same : ∀ i, i≤t → x i=y i) (state : ∀ i, i≤t → hx i=hy i) :
    context (gain (hx t)) (hx t) (gated (buildMemory q (fun i => context c (x i) (hx i)) x (t-1))
      q (context c (x t) (hx t)) (msg (hx t))) =
    context (gain (hy t)) (hy t) (gated (buildMemory q (fun i => context c (y i) (hy i)) y (t-1))
      q (context c (y t) (hy t)) (msg (hy t))) := by
  have eq := prefix_eq q (fun i => context c (x i) (hx i)) (fun i => context c (y i) (hy i))
    x y (t-1) (fun i hi => by rw [same i (by omega),state i (by omega)]) (fun i hi => same i (by omega))
  rw [eq,state t (by omega),same t (by omega)]

example : context (firstKey 0) firstKey (gated (buildMemory address
      (fun _ => context 1 firstKey firstKey) (fun _ => firstKey) 3)
    address (context 1 firstKey firstKey) (firstKey 1)) =
    context (later 4 0) (later 4) (gated (buildMemory address
      (fun i => context 1 (later i) (later i)) later 3)
    address (context 1 (later 4) (later 4)) (later 4 1)) := by
  apply gated_causal address 1 (fun v => v 1) (fun v => v 0)
    (fun _ => firstKey) later (fun _ => firstKey) later 4
  all_goals intro i hi; simp [later,hi]

end Transformer.Phasor.TripleTransport.LocalReadLoop

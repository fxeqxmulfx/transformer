module

public import Transformer.Phasor.TripleTransport.LocalReadLoop

/-!
# Keeping the query and a gated message in a shared read loop

Source: the Phasor depth study, adapting the shared prelude and previous
message in Thinking Machines Lab's *Introducing Inkling 2 Mini*, “Looping
design”. Each pass proposes a value using the original query and its last
message. The proposal also returns a match strength. The acceptance map sees
the query, message and strength. All passes use the same parameters.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.MessageLoop

open LocalMemory
variable {d : ℕ}

def next (read : V d → V d → V d × ℝ) (accept : V d → V d → ℝ → ℝ)
    (q m : V d) : V d :=
  let p := read q m
  LocalMemory.mix (accept q m p.2) m p.1

def run (read : V d → V d → V d × ℝ) (accept : V d → V d → ℝ → ℝ)
    (q m : V d) : ℕ → V d
  | 0 => m
  | n+1 => next read accept q (run read accept q m n)

/-- A completed state persists for any further budget when its gate rejects
the current match, even if the read still proposes a different value. -/
theorem retain (read : V d → V d → V d × ℝ) (accept : V d → V d → ℝ → ℝ)
    (q m : V d) (n extra : ℕ)
    (closed : accept q (run read accept q m n) (read q (run read accept q m n)).2 = 0) :
    run read accept q m (n+extra) = run read accept q m n := by
  induction extra with
  | zero => rfl
  | succ k ih =>
    rw [Nat.add_succ,run,ih]
    funext j
    simp [next,closed,LocalMemory.mix]

/-- Every pass may choose a new relation using the query and the latest
message. Unit acceptance on a path gives the proposed value at each
depth, including when consecutive edges use different relations. -/
theorem follow {R : Type*} (choose : V d → V d → R)
    (lookup : R → V d → V d × ℝ) (accept : V d → V d → ℝ → ℝ)
    (q : V d) (path : ℕ → V d) (n : ℕ)
    (edges : ∀ i<n, (lookup (choose q (path i)) (path i)).1 = path (i+1))
    (openGate : ∀ i<n, accept q (path i) (lookup (choose q (path i)) (path i)).2 = 1) :
    run (fun x y => lookup (choose x y) y) accept q (path 0) n = path n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [run,ih (fun i hi => edges i (by omega)) (fun i hi => openGate i (by omega))]
    funext j
    simp [next,LocalMemory.mix,edges n (by omega),openGate n (by omega)]

/-- If each subsequent update moves one coordinate by at most ε, its drift
over k extra shared reads is at most k ε. The gate bound uses the actual
match and the distance from the current message to the proposed value. -/
theorem drift (read : V d → V d → V d × ℝ) (accept : V d → V d → ℝ → ℝ)
    (q m : V d) (j : Fin d) (n k : ℕ) (ε : ℝ)
    (small : ∀ i<k, let s := run read accept q m (n+i)
      |accept q s (read q s).2 * ((read q s).1 j - s j)| ≤ ε) :
    |run read accept q m (n+k) j - run read accept q m n j| ≤ k*ε := by
  induction k with
  | zero => simp
  | succ k ih =>
    have short := ih (fun i hi => small i (by omega))
    have bound := small k (by omega)
    let s := run read accept q m (n+k)
    let change := accept q s (read q s).2 * ((read q s).1 j - s j)
    change |change| ≤ ε at bound
    calc |run read accept q m (n+(k+1)) j - run read accept q m n j|
        = |(s j - run read accept q m n j) + change| := by
          rw [Nat.add_succ,run]
          unfold next LocalMemory.mix change s
          congr 1
          ring
      _ ≤ |s j - run read accept q m n j| + |change| := abs_add_le _ _
      _ ≤ k*ε + ε := add_le_add short bound
      _ = (k+1 : ℕ)*ε := by push_cast; ring

private def alternating (r : Bool) (x : V 1) : V 1 × ℝ :=
  (fun j => if r then 2*x j else x j+1, 1)

example : run (fun (q s : V 1) => alternating (decide (s 0 = q 0)) s)
    (fun _ _ _ => 1) (fun _ => 3) (fun _ => 2) 2 = fun _ => 6 := by
  let path : ℕ → V 1 := fun i _ => if i=0 then 2 else if i=1 then 3 else 6
  apply follow (fun q s : V 1 => decide (s 0=q 0)) alternating _ _ path 2
  · intro i hi
    interval_cases i <;> funext j <;> norm_num [path,alternating]
  · intro i hi
    rfl

example : run (fun (_ _ : V 1) => ((fun _ => 9), (0:ℝ)))
    (fun _ _ strength => strength) (fun _ => 0) (fun _ => 3) (2+40)
    = run (fun (_ _ : V 1) => ((fun _ => 9), (0:ℝ)))
      (fun _ _ strength => strength) (fun _ => 0) (fun _ => 3) 2 :=
  retain _ _ _ _ _ _ rfl

example : |run (fun (_ m : V 1) => ((fun j => m j + 1), (1:ℝ)))
    (fun _ _ g => g/2) (fun _ => 0) (fun _ => 0) (0+20) 0 -
    run (fun (_ m : V 1) => ((fun j => m j + 1), (1:ℝ)))
    (fun _ _ g => g/2) (fun _ => 0) (fun _ => 0) 0 0| ≤ (20:ℕ)*(1/2:ℝ) := by
  apply drift
  intro i hi
  norm_num

/-- For a finite uniform budget distribution, the mean of the usual answer-loss
slopes is the derivative of the mean answer loss at the shared parameter. -/
theorem budget_derivative {B : Type*} [Fintype B] (loss : B → ℝ → ℝ) (slope : B → ℝ)
    (w : ℝ) (each : ∀ b, HasDerivAt (loss b) (slope b) w) :
    HasDerivAt (fun x => (∑ b, loss b x) / Fintype.card B)
      ((∑ b, slope b) / Fintype.card B) w := (HasDerivAt.fun_sum (fun i _ => each i)).div_const _

example : HasDerivAt (fun x : ℝ => ((∑ b : Fin 3, (b:ℝ)*x)) / Fintype.card (Fin 3))
    ((∑ b : Fin 3, (b:ℝ)) / Fintype.card (Fin 3)) 1 := by
  apply budget_derivative
  intro b
  simpa using (hasDerivAt_id (1:ℝ)).const_mul (b:ℝ)

end Transformer.Phasor.TripleTransport.MessageLoop

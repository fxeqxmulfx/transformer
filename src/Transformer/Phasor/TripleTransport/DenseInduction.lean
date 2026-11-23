module

public import Transformer.Phasor.TripleTransport.StructuralGate

/-!
# A causal induction read over dense input vectors

Source: the dense induction experiment in `experiments/phasor_value_transport/README.md`.
An earlier key must have a positive dot product and squared cosine above
one half with the current key. Its successor must precede the current
position. A matching read supplies an existing learned gate with the
successor's normalized input vector. All other inputs use the static proposal.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.DenseInduction

open scoped BigOperators

/-- The angular test uses only the current dense key and a past dense key.
Its squared form handles arbitrary norms and abstains for a zero query. -/
def product {d : ℕ} (x y : Fin d → ℝ) : ℝ := ∑ j, x j * y j

def eligible {d : ℕ} (x y : Fin d → ℝ) : Prop :=
  0 < product x y ∧ product x x * product y y < 2 * (product x y)^2

theorem eligible_self {d : ℕ} (x : Fin d → ℝ) (energy : 0 < product x x) :
    eligible x x := by
  refine ⟨energy, ?_⟩
  nlinarith [sq_pos_of_pos energy]

theorem reject_orthogonal {d : ℕ} (x y : Fin d → ℝ) (orth : product x y = 0) :
    ¬ eligible x y := by simp [eligible,orth]

/-- Search the first n source positions backwards and return the successor
index of the most recent qualifying source. The selection never reads values. -/
def findDonor {K : Type*} (accept : K → K → Bool) (keys : ℕ → K) (query : K) : ℕ → Option ℕ
  | 0 => none
  | n+1 => if accept query (keys n) then some (n+1) else findDonor accept keys query n

def donor {K : Type*} (accept : K → K → Bool) (keys : ℕ → K) (t : ℕ) : Option ℕ :=
  findDonor accept keys (keys t) (t-1)

/-- Every previous matching key with no more recent completed match supplies
its successor, independently of prefix length or the number of distinct keys. -/
theorem latest {K : Type*} (accept : K → K → Bool) (keys : ℕ → K) (q : K) (i n : ℕ)
    (hi : i < n) (hit : accept q (keys i) = true)
    (last : ∀ j, i < j → j < n → accept q (keys j) = false) :
    findDonor accept keys q n = some (i+1) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : i = n
    · subst n
      simp [findDonor, hit]
    · have hlt : i < n := by omega
      simp [findDonor,last n hlt (by omega),ih hlt (fun j hj hk => last j hj (by omega))]

theorem bounded {K : Type*} (accept : K → K → Bool) (keys : ℕ → K) (q : K) (i n : ℕ)
    (found : findDonor accept keys q n = some i) : 0 < i ∧ i ≤ n := by
  induction n with
  | zero => simp [findDonor] at found
  | succ n ih =>
    simp only [findDonor] at found
    split at found
    · simp only [Option.some.injEq] at found
      omega
    · obtain ⟨pos,le⟩ := ih found
      omega

private theorem find_congr {K : Type*} (accept : K → K → Bool) (x y : ℕ → K) (q : K) (n : ℕ)
    (agree : ∀ i, i < n → x i = y i) : findDonor accept x q n = findDonor accept y q n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [findDonor,agree n (by omega),ih (fun i hi => agree i (by omega))]

/-- The donor index and its value depend only on the input prefix; arbitrary
changes after the query cannot affect which dense vector enters the mixture. -/
theorem prefix_congr {K V : Type*} (accept : K → K → Bool) (x y : ℕ → K) (vx vy : ℕ → V) (t : ℕ)
    (keys : ∀ i, i ≤ t → x i = y i) (vals : ∀ i, i < t → vx i = vy i) :
    (donor accept x t).map vx = (donor accept y t).map vy := by
  have eq : donor accept x t = donor accept y t := by
    unfold donor
    rw [keys t (by omega)]
    exact find_congr _ _ _ _ _ (fun i hi => keys i (by omega))
  rw [eq]
  cases h : donor accept y t with
  | none => rfl
  | some j =>
    obtain ⟨pos,le⟩ := bounded _ _ _ _ _ h
    simp [vals j (by omega)]

/-- The same learned structural gate can mix either the current static
proposal or the selected earlier dense value. At the first two positions
the existing mixture is recovered for all parameters and arbitrary inputs. -/
theorem two_positions {K V : Type*} (accept : K → K → Bool) (keys : ℕ → K) (v : ℕ → V)
    (fallback : V) (mix : V → V) (t : ℕ) (short : t ≤ 1) :
    mix (((donor accept keys t).map v).getD fallback) = mix fallback := by
  have empty : t-1=0 := by omega
  simp [donor,empty,findDonor]

example : eligible (fun _ : Fin 1 => 1) (fun _ : Fin 1 => 1) :=
  eligible_self _ (by norm_num [product, Fin.sum_univ_one])

example : ¬ eligible (![1,0] : Fin 2 → ℝ) ![0,1] :=
  reject_orthogonal _ _ (by norm_num [product, Fin.sum_univ_two])

example : findDonor (· == ·) (fun i => i % 3) 1 8 = some 8 :=
  latest _ _ _ 7 8 (by decide) (by decide) (by intro j h k; omega)

example : (donor (· == ·) (fun i => i % 3) 7).map (·+5) = some 10 := by decide

/-- On orthogonal unit directions the angular test recovers equality of
directions. Their coordinates can be permuted without changing the test. -/
def axis {d : ℕ} (i j : Fin d) : ℝ := if j = i then 1 else 0

private theorem axis_product {d : ℕ} (i j : Fin d) :
    product (axis i) (axis j) = if i=j then 1 else 0 := by
  simp [product,axis,eq_comm]

noncomputable def accepts {d : ℕ} (x y : Fin d → ℝ) : Bool := by
  classical exact decide (eligible x y)

theorem accepts_axis {d : ℕ} (i j : Fin d) : accepts (axis i) (axis j) = decide (i=j) := by
  classical
  by_cases h : i=j <;> simp [accepts,eligible,axis_product,h]

private def stream (v : Bool) : ℕ → Fin 4
  | 0 => 0
  | 1 => if v then 1 else 2
  | 2 => 3
  | _ => 0

/-- A read of the current and preceding input alone cannot realize the
whole induction class with shared dense keys and values. The witness uses
four orthogonal directions and two streams with the same final pair. -/
theorem dense_history_needed :
    ¬ ∃ g : (Fin 4 → ℝ) → (Fin 4 → ℝ) → Option (Fin 4 → ℝ), ∀ xs : ℕ → Fin 4 → ℝ,
        g (xs 2) (xs 3) = (donor accepts xs 3).map xs := by
  rintro ⟨g,h⟩
  have left := h (fun i => axis (stream false i))
  have right := h (fun i => axis (stream true i))
  simp [donor,findDonor,stream,accepts_axis] at left right
  have disagree := congrFun (Option.some.inj (left.symm.trans right)) (1 : Fin 4)
  norm_num [axis] at disagree

example : 0 < 8 ∧ 8 ≤ 8 := bounded (· == ·) (fun i => i % 3) 1 8 8 (by decide)

example : (donor (· == ·) (fun i => i % 3) 7).map id =
    (donor (· == ·) (fun i => if i ≤ 7 then i % 3 else 12) 7).map
      (fun i => if i < 7 then i else 13) := by
  apply prefix_congr <;> intro i hi <;> simp [hi]

example : (fun n : ℕ => n+1)
    (((donor (· == ·) (fun i => i % 3) 1).map id).getD 5) = 6 :=
  two_positions (· == ·) (fun i : ℕ => i % 3) id 5 (fun n => n+1) 1 (by decide)

/-- The residual includes the contextual, direct and phase reads. Prefix
induction supplies one candidate to the existing structural gate. -/
noncomputable def feature {d : ℕ} (xs residual : ℕ → Fin d → ℝ)
    (fixed : ℕ → Option (Fin d → ℝ)) (gate : ℕ → ℝ) (t : ℕ) (c : Fin d) : ℝ :=
  residual t c + gate t * SharedStructuralGate.value id
    (((donor accepts xs t).map xs).or (fixed t)) c

/-- Every most recent angular match contributes its actual successor
vector to the learned mixture at every coordinate. -/
theorem feature_of_latest {d : ℕ} (xs residual : ℕ → Fin d → ℝ)
    (fixed : ℕ → Option (Fin d → ℝ)) (gate : ℕ → ℝ) (i t : ℕ) (c : Fin d)
    (complete : i+1 < t) (hit : accepts (xs t) (xs i) = true)
    (last : ∀ j, i < j → j < t-1 → accepts (xs t) (xs j) = false) :
    feature xs residual fixed gate t c = residual t c + gate t * xs (i+1) c := by
  unfold feature donor
  rw [latest accepts xs (xs t) i (t-1) (by omega) hit last]
  rfl

example : feature (fun i => axis (stream false i)) (fun _ _ => 0) (fun _ => none)
    (fun _ => 1) 3 2 = 1 := by
  rw [feature_of_latest (fun i => axis (stream false i)) (fun _ _ => 0) (fun _ => none)
    (fun _ => 1) 0 3 2 (by decide) (by simp [stream,accepts_axis])
    (by
      intro j lo hi
      obtain rfl : j = 1 := by omega
      simp [stream,accepts_axis])]
  norm_num [axis,stream]

end Transformer.Phasor.TripleTransport.DenseInduction

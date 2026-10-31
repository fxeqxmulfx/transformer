module

public import Transformer.CRASP.Rotary.Cells

/-!
# Position-sensitive finite-state attention without a position code

Extension of arXiv:2506.16055v3, Appendix B.2/F;
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U2.
The same tables apply at all natural query/source positions. Their values
are fixed-point coefficients, but their dependence on positions need not be
periodic. Prefix sums use unbounded integers as in the paper's `sumdiv`.
-/

@[expose] public section

namespace Transformer.CRASP.Rotary

universe u v w
variable {Q : Type u} {C : Type v} {σ : Type w} {p s d : ℕ}

/-- Finite hidden states and a position-sensitive coefficient table per head
coordinate. `finish` performs only token-local operations. -/
structure Layer (Q : Type u) (C : Type v) (p s : ℕ) where
  denominator : Q → C → Q → ℕ → ℕ → Fx p s
  numerator : Q → C → Q → ℕ → ℕ → Fx p s
  finish : Q → (C → Fx p s) → Q

/-- Softmax coefficients are nonnegative, independently of position. -/
def Layer.Nonnegative (L : Layer Q C p s) : Prop :=
  ∀ q c a i j, 0 ≤ (L.denominator q c a i j).m

/-- Aggregate actual prefix states, rounding once after integer summation. -/
noncomputable def Layer.aggregate (L : Layer Q C p s) (r : Q)
    (q : ℕ → Q) (i : ℕ) (c : C) : Fx p s :=
  Fx.round p s
    (((∑ j ∈ Finset.Icc 1 i, (L.numerator r c (q j) i j).m : ℤ) : ℝ) /
      ((∑ j ∈ Finset.Icc 1 i, (L.denominator r c (q j) i j).m : ℤ) : ℝ))

/-- One attention layer and its local postprocessing. -/
noncomputable def Layer.step (L : Layer Q C p s) (q : ℕ → Q) (i : ℕ) : Q :=
  L.finish (q i) (L.aggregate (q i) q i)

/-- A kernel name records a layer, numerator/denominator choice, fixed query
state, head coordinate and fixed source state. It contains no word data. -/
abbrev Key (Q : Type u) (C : Type v) := ℕ × Bool × Q × C × Q

/-- Interpret names by the model's fixed position-dependent coefficient tables. -/
def kernel (layers : ℕ → Layer Q C p s) : Key Q C → ℕ → ℕ → ℤ
  | (l, numerator, q, c, a), i, j =>
      if numerator then ((layers l).numerator q c a i j).m
      else ((layers l).denominator q c a i j).m

/-- Exact nonperiodic positional dependence does not require an unbounded
coefficient alphabet: every coefficient remains a fixed-point mantissa. -/
theorem kernel_bounds (layers : ℕ → Layer Q C p s) (name : Key Q C) (i j : ℕ) :
    -2 ^ (p - 1) ≤ kernel layers name i j ∧ kernel layers name i j < 2 ^ (p - 1) := by
  rcases name with ⟨l, b, q, c, a⟩
  cases b with
  | false => exact ⟨((layers l).denominator q c a i j).lo,
      ((layers l).denominator q c a i j).hi⟩
  | true => exact ⟨((layers l).numerator q c a i j).lo,
      ((layers l).numerator q c a i j).hi⟩

variable [Fintype Q]

/-- A coordinate cell uses actual weighted-count syntax, not an attention oracle. -/
noncomputable def coordinate (l : ℕ) (ψ : Q → Form σ (Key Q C))
    (r : Q) (c : C) (y : Fx p s) : Form σ (Key Q C) :=
  cell (stateSum (fun a => (l, true, r, c, a)) ψ)
    (stateSum (fun a => (l, false, r, c, a)) ψ) y

/-- Every coordinate of every head costs at most one counting level (U2). -/
theorem coordinate_depth (l : ℕ) (ψ : Q → Form σ (Key Q C))
    (hψ : ∀ a, (ψ a).depth ≤ d) (r : Q) (c : C) (y : Fx p s) :
    (coordinate l ψ r c y).depth ≤ d + 1 :=
  cell_depth _ _ (stateSum_depth _ ψ hψ) (stateSum_depth _ ψ hψ) y

variable [DecidableEq Q] [DecidableEq σ]

/-- A coordinate formula recovers the rounded attention on every finite
prefix, using the same kernels at all lengths (U2). -/
theorem coordinate_sat (layers : ℕ → Layer Q C p s) (l : ℕ)
    (hL : (layers l).Nonnegative) (ψ : Q → Form σ (Key Q C))
    (q : ℕ → Q) (w : List σ) (i : ℕ)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a,
      (ψ a).sat (kernel layers) w j = decide (q j = a))
    (r : Q) (c : C) (y : Fx p s) :
    (coordinate l ψ r c y).sat (kernel layers) w i = true ↔
      (layers l).aggregate r q i c = y := by
  have hd := stateSum_val (fun a => (l, false, r, c, a)) ψ (kernel layers) q w i hψ
  have hn := stateSum_val (fun a => (l, true, r, c, a)) ψ (kernel layers) q w i hψ
  have hnonneg : 0 ≤ (stateSum (fun a => (l, false, r, c, a)) ψ).val
      (kernel layers) w i := by
    rw [hd]
    apply Finset.sum_nonneg
    intro j hj
    exact hL r c (q j) i j
  rw [coordinate, cell_sat _ _ _ _ _ hnonneg, hd, hn]
  rfl


end Transformer.CRASP.Rotary

namespace Transformer.CRASP.Rotary

universe u v w
variable {Q : Type u} {C : Type v} {σ : Type w} {p s d : ℕ}
variable [Fintype Q] [Fintype C]

open scoped Classical in
/-- Enumerate current and aggregate states, then test the actual local update. -/
noncomputable def nextFormula (layers : ℕ → Layer Q C p s) (l : ℕ)
    (ψ : Q → Form σ (Key Q C)) (r : Q) : Form σ (Key Q C) :=
  Form.any (Finset.univ.toList.map fun z : Q × (C → Fx p s) =>
    .and (ψ z.1) (.and (.truth (decide ((layers l).finish z.1 z.2 = r)))
      (Form.all (Finset.univ.toList.map fun c : C => coordinate l ψ z.1 c (z.2 c)))))

/-- One layer adds at most one counting level, at every width and head count (U2). -/
theorem nextFormula_depth (layers : ℕ → Layer Q C p s) (l : ℕ)
    (ψ : Q → Form σ (Key Q C)) (hψ : ∀ a, (ψ a).depth ≤ d) (r : Q) :
    (nextFormula layers l ψ r).depth ≤ d + 1 := by
  classical
  apply Form.depth_any
  rintro φ hφ
  obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
  apply max_le ((hψ z.1).trans (Nat.le_succ d))
  apply max_le (Nat.zero_le _)
  apply Form.depth_all
  rintro φ hφ
  obtain ⟨c, _, rfl⟩ := List.mem_map.mp hφ
  exact coordinate_depth l ψ hψ z.1 c (z.2 c)

variable [DecidableEq Q] [DecidableEq σ]

/-- Exact simulation of one complete layer at any positive position (U2). -/
theorem nextFormula_sat (layers : ℕ → Layer Q C p s) (l : ℕ)
    (hL : (layers l).Nonnegative) (ψ : Q → Form σ (Key Q C))
    (q : ℕ → Q) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a,
      (ψ a).sat (kernel layers) w j = decide (q j = a)) (r : Q) :
    (nextFormula layers l ψ r).sat (kernel layers) w i = true ↔
      (layers l).step q i = r := by
  classical
  have hcur := hψ i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩)
  have hcell := coordinate_sat layers l hL ψ q w i hψ
  rw [nextFormula, Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
    simp only [Form.sat, Bool.and_eq_true, decide_eq_true_eq] at hs
    have hq : q i = z.1 := of_decide_eq_true (by rw [← hcur]; exact hs.1)
    have ha : (layers l).aggregate (q i) q i = z.2 := by
      funext c
      have hc := (Form.sat_all _ _ w i).mp hs.2.2 _
        (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ c)))
      simpa only [hq] using (hcell z.1 c (z.2 c)).mp hc
    rw [Layer.step, ha, hq]
    exact hs.2.1
  · intro hr
    let a := (layers l).aggregate (q i) q i
    have hcoords : (Form.all (Finset.univ.toList.map fun c : C =>
        coordinate l ψ (q i) c (a c))).sat (kernel layers) w i = true := by
      apply (Form.sat_all _ _ w i).mpr
      rintro φ hφ
      obtain ⟨c, _, rfl⟩ := List.mem_map.mp hφ
      exact (hcell (q i) c (a c)).mpr rfl
    refine ⟨_, List.mem_map_of_mem (a := (q i, a))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [Form.sat, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨by rw [hcur]; simp, hr, hcoords⟩

private def witness : Layer Bool Unit 2 0 :=
  ⟨fun _ _ _ _ _ => 0, fun _ _ _ _ _ => 0, fun q _ => q⟩

/-- All layer/stack premises have a concrete nonempty binary-word instance. -/
example : (∀ l : ℕ, ((fun _ => witness) l).Nonnegative) ∧ 0 < 1 ∧
    (∀ a : Bool, (Form.sym a : Form Bool (Key Bool Unit)).depth ≤ 0) ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Bool,
      (Form.sym a : Form Bool (Key Bool Unit)).sat (kernel (fun _ => witness))
        [true] j = decide (true = a)) := by
  refine ⟨by simp [Layer.Nonnegative, witness], by decide, fun _ => le_rfl, ?_⟩
  intro j hj a
  have : j = 1 := by have := Finset.mem_Icc.mp hj; omega
  subst j
  simp [Form.sat]

end Transformer.CRASP.Rotary

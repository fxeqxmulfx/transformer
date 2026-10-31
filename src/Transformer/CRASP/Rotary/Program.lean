module

public import Transformer.CRASP.Rotary.Layer

/-!
# A single classifier formula for all finite token words

Extension of arXiv:2506.16055v3, Appendix B.2/F;
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U2.
Only a first-position bit is stored alongside the finite residual. No position
number, periodic approximation, or length-dependent token alphabet is used.
-/

@[expose] public section

namespace Transformer.CRASP.Rotary

universe u v w
variable {Q : Type u} {C : Type v} {σ : Type w} {κ : Type*} {p s : ℕ}
section Stack
variable {d : ℕ} [Fintype Q] [Fintype C] [DecidableEq Q] [DecidableEq σ]

/-- Execute a finite stack on the entire causal stream. -/
noncomputable def run (layers : ℕ → Layer Q C p s) (initial : ℕ → Q) : ℕ → ℕ → Q
  | 0 => initial
  | n + 1 => (layers n).step (run layers initial n)

/-- Compile the stack once; neither the syntax nor kernels depend on word length. -/
noncomputable def formula (layers : ℕ → Layer Q C p s)
    (initial : Q → Form σ (Key Q C)) : ℕ → Q → Form σ (Key Q C)
  | 0 => initial
  | n + 1 => nextFormula layers n (formula layers initial n)

omit [DecidableEq Q] [DecidableEq σ] in
/-- A stack of n blocks has counting depth at most its initial depth plus n (U2). -/
theorem formula_depth (layers : ℕ → Layer Q C p s) (initial : Q → Form σ (Key Q C))
    (hinit : ∀ a, (initial a).depth ≤ d) (n : ℕ) (r : Q) :
    (formula layers initial n r).depth ≤ d + n := by
  induction n generalizing r with
  | zero => exact hinit r
  | succ n ih => exact nextFormula_depth layers n _ ih r

/-- Uniform prefix simulation for every finite number of layers (U2). -/
theorem formula_sat (layers : ℕ → Layer Q C p s) (hL : ∀ l, (layers l).Nonnegative)
    (initial : Q → Form σ (Key Q C)) (q : ℕ → Q) (w : List σ) (n : ℕ)
    {i : ℕ} (hi : 0 < i)
    (hinit : ∀ j ∈ Finset.Icc 1 i, ∀ a,
      (initial a).sat (kernel layers) w j = decide (q j = a)) (r : Q) :
    (formula layers initial n r).sat (kernel layers) w i = decide (run layers q n i = r) := by
  induction n generalizing i r with
  | zero => exact hinit i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩) r
  | succ n ih =>
      rw [Bool.eq_iff_iff, decide_eq_true_iff]
      apply nextFormula_sat layers n (hL n) _ _ w hi
      intro j hj a
      have hjb := Finset.mem_Icc.mp hj
      apply ih (by omega)
      intro t ht b
      have htb := Finset.mem_Icc.mp ht
      exact hinit t (Finset.mem_Icc.mpr ⟨htb.1, htb.2.trans hjb.2⟩) b

end Stack

variable [Fintype Q] [Fintype σ] [DecidableEq Q] [DecidableEq σ]

/-- The only extra unary predicate needed for the source's first-token XSA rule. -/
def firstChoice (b : Bool) : Form σ κ := if b then .first else .neg .first

/-- Letter embedding and the first-position flag are tested locally. -/
noncomputable def initialFormula (embed : Bool → σ → Q) (r : Q) : Form σ κ :=
  Form.any (Finset.univ.toList.map fun z : Bool × σ =>
    .and (firstChoice z.1) (.and (.sym z.2) (.truth (decide (embed z.1 z.2 = r)))))

omit [Fintype Q] [DecidableEq σ] in
/-- Initial finite embedding and the first-position bit have counting depth zero. -/
theorem initialFormula_depth (embed : Bool → σ → Q) (r : Q) :
    (initialFormula (κ := κ) embed r).depth ≤ 0 := by
  apply Form.depth_any
  rintro φ hφ
  obtain ⟨⟨b, a⟩, _, rfl⟩ := List.mem_map.mp hφ
  cases b <;> exact le_rfl

omit [Fintype Q] in
/-- The initial formula defines the actual tokenwise state for any word length. -/
theorem initialFormula_sat (embed : Bool → σ → Q) (r : Q)
    (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) (a : σ)
    (ha : w[i - 1]? = some a) :
    (initialFormula embed r).sat K w i = decide (embed (decide (i = 1)) a = r) := by
  have hfirst (b : Bool) : (firstChoice b : Form σ κ).sat K w i = true ↔
      b = decide (i = 1) := by
    cases b <;> by_cases h : i = 1 <;> simp [firstChoice, Form.sat, h]
  rw [Bool.eq_iff_iff, decide_eq_true_iff, initialFormula, Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
    change (firstChoice z.1).sat K w i &&
      (decide (w[i - 1]? = some z.2) && decide (embed z.1 z.2 = r)) = true at hs
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hs
    have hf := (hfirst z.1).mp hs.1
    have hletter : a = z.2 := Option.some.inj (ha.symm.trans hs.2.1)
    rw [← hf, hletter]
    exact hs.2.2
  · intro h
    refine ⟨_, List.mem_map_of_mem (a := (decide (i = 1), a))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    change (firstChoice (decide (i = 1))).sat K w i &&
      (decide (w[i - 1]? = some a) && decide (embed (decide (i = 1)) a = r)) = true
    simp only [Bool.and_eq_true, ha, h, decide_true, and_true]
    exact (hfirst _).mpr rfl

/-- Embed the original token word, supplying a default only outside its positions. -/
def input (embed : Bool → σ → Q) (default : Q) (w : List σ) (i : ℕ) : Q :=
  match w[i - 1]? with
  | some a => embed (decide (i = 1)) a
  | none => default

/-- Compile any final local output test, including normalized tied decoding. -/
noncomputable def outputFormula (accept : Q → Bool) (ψ : Q → Form σ κ) : Form σ κ :=
  Form.any (Finset.univ.toList.map fun r : Q => .and (ψ r) (.truth (accept r)))

omit [Fintype σ] [DecidableEq Q] [DecidableEq σ] in
/-- A final finite-state output test adds no counting depth (U2). -/
theorem outputFormula_depth (accept : Q → Bool) (ψ : Q → Form σ κ) (d : ℕ)
    (hψ : ∀ r, (ψ r).depth ≤ d) : (outputFormula accept ψ).depth ≤ d := by
  apply Form.depth_any
  rintro φ hφ
  obtain ⟨r, _, rfl⟩ := List.mem_map.mp hφ
  exact max_le (hψ r) (Nat.zero_le _)

omit [Fintype σ] in
/-- The output formula agrees exactly with the local decision (U2). -/
theorem outputFormula_sat (accept : Q → Bool) (ψ : Q → Form σ κ)
    (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) (q : Q)
    (hψ : ∀ r, (ψ r).sat K w i = decide (q = r)) :
    (outputFormula accept ψ).sat K w i = accept q := by
  rw [Bool.eq_iff_iff, outputFormula, Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨r, _, rfl⟩ := List.mem_map.mp hφ
    simp only [Form.sat, Bool.and_eq_true, hψ, decide_eq_true_eq] at hs
    exact hs.1 ▸ hs.2
  · intro h
    refine ⟨_, List.mem_map_of_mem (a := q)
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [Form.sat, hψ, decide_true, Bool.true_and, h]

variable [Fintype C]

/-- The complete classifier syntax is independent of the word being classified. -/
noncomputable def classifier (layers : ℕ → Layer Q C p s) (embed : Bool → σ → Q)
    (accept : Q → Bool) (n : ℕ) : Form σ (Key Q C) :=
  outputFormula accept (formula layers (initialFormula embed) n)

omit [DecidableEq σ] in
/-- One fixed depth-n classifier represents the whole depth-n stack (U2). -/
theorem classifier_depth (layers : ℕ → Layer Q C p s) (embed : Bool → σ → Q)
    (accept : Q → Bool) (n : ℕ) : (classifier layers embed accept n).depth ≤ n := by
  apply outputFormula_depth
  intro r
  simpa only [Nat.zero_add] using formula_depth layers _ (initialFormula_depth embed) n r

/-- The same formula and kernel family work on all finite token words, with
no context bound or periodicity hypothesis (U2). -/
theorem classifier_correct (layers : ℕ → Layer Q C p s)
    (hL : ∀ l, (layers l).Nonnegative) (embed : Bool → σ → Q) (default : Q)
    (accept : Q → Bool) (n : ℕ) (w : List σ) {i : ℕ} (hi : 0 < i) (hw : i ≤ w.length) :
    (classifier layers embed accept n).sat (kernel layers) w i =
      accept (run layers (input embed default w) n i) := by
  apply outputFormula_sat
  intro r
  apply formula_sat layers hL _ _ w n hi
  intro j hj a
  have hjb := Finset.mem_Icc.mp hj
  have hidx : j - 1 < w.length := by omega
  have htok := List.getElem?_eq_getElem hidx
  rw [input, htok]
  exact initialFormula_sat embed a (kernel layers) w j w[j - 1] htok

/-- Initial, output and readable-prefix hypotheses have concrete token witnesses. -/
example : 0 < 1 ∧ 1 ≤ [true].length ∧
    (∀ r : Bool × Bool, (initialFormula (κ := Unit) Prod.mk r).depth ≤ 0) ∧
    (∀ r : Bool × Bool, (initialFormula (κ := Unit) Prod.mk r).sat
      (fun _ _ _ => 1) [true] 1 = decide ((true, true) = r)) := by
  refine ⟨by decide, by decide, initialFormula_depth Prod.mk, ?_⟩
  intro r
  exact initialFormula_sat Prod.mk r _ [true] 1 true rfl

end Transformer.CRASP.Rotary

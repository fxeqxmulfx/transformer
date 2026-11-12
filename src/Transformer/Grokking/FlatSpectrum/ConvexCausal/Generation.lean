module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Recovery

/-!
# Free causal generation after finite learning

Source: `review/convex_causal.md`, CCa7. The next input is the actual prefix
extended by the model's own emitted token. Generation stops at EOS or a fixed
fuel budget. The recovery theorem is therefore stronger than agreement on
teacher-forced training tokens. It assumes a complete causal target rule in
the fixed bank and separation of that bank, not arbitrary length extrapolation
from the universal bounded-prefix head.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

variable {C : Type*} [Fintype C] [Nonempty C] [DecidableEq C]

/-- A maximum-probability class; only the finite comparison specification is used (CCa7). -/
noncomputable def best (q : C → ℝ) : C :=
  (Finset.exists_max_image Finset.univ q Finset.univ_nonempty).choose

omit [DecidableEq C] in
/-- The chosen class has maximal probability over the full vocabulary (CCa7). -/
theorem best_max (q : C → ℝ) (c : C) : q c ≤ q (best q) :=
  (Finset.exists_max_image Finset.univ q Finset.univ_nonempty).choose_spec.2 c
    (Finset.mem_univ c)

/-- Any maximum readout of an identified deterministic distribution returns its answer (CCa7). -/
theorem best_point {X : Type*} (f : X → C) {ε : ℝ} (hε : ε < 1) (x : X) :
    best (smooth ε (point f) x) = f x := by
  by_contra hne
  have ht := point_strict f hε x (best (smooth ε (point f) x)) hne
  exact (not_lt_of_ge (best_max (smooth ε (point f) x) (f x))) ht

/-- Free generation includes EOS and feeds back only its own outputs (CCa7). -/
def generate (next : List C → C) (eos : C) : ℕ → List C → List C
  | 0, _ => []
  | n + 1, history =>
    let c := next history
    if c = eos then [c] else c :: generate next eos n (history ++ [c])

omit [Fintype C] [Nonempty C] in
/-- Equality of causal next-token rules propagates through actual free generation (CCa7). -/
theorem generate_congr (f g : List C → C) (h : ∀ xs, f xs = g xs)
    (eos : C) (n : ℕ) (history : List C) : generate f eos n history = generate g eos n history := by
  induction n generalizing history with
  | zero => rfl
  | succ n ih =>
    simp only [generate, h history]
    split_ifs
    · rfl
    · rw [ih]

variable {H S : Type*} [Fintype H] [DecidableEq H] [Fintype S]

/-- Finite training recovers the entire generated sequence, including its termination (CCa7). -/
theorem trained_generation (rule : H → List C → C) (input : S → List C) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input)
    {ε : ℝ} (hε : ε < 1) (eos : C) (n : ℕ) (history : List C) :
    generate (fun xs => best (smooth ε (trained rule input label) xs)) eos n history =
      generate (rule g) eos n history := by
  apply generate_congr
  intro xs
  rw [trained_target rule input label g hfit hsep]
  exact best_point (rule g) hε xs

/-- End-to-end causal learning: computed rule, feasible parameters, global CE minimum,
and exact free generation, under observable separation and bank realizability (CCa7). -/
theorem learns_causal [LinearOrder H] (rule : H → List C → C) (input : S → List C)
    (label : S → C) (g : H) (hfit : ∀ s, rule g (input s) = label s)
    (hsep : Separates rule input) {ε : ℝ} (hε : 0 < ε) (hε1 : ε < 1) :
    learn rule input label = .learned g ∧ Rows (trained rule input label) ∧
      IsMinOn (loss ε input label) {Q | Rows Q} (trained rule input label) ∧
      ∀ eos n history,
        generate (fun xs => best (smooth ε (trained rule input label) xs)) eos n history =
          generate (rule g) eos n history := by
  have hs : (survivors rule input label).Nonempty :=
    ⟨g, target_survives rule input label g hfit⟩
  obtain ⟨hr, hm⟩ := trained_minimum rule input label hs hε hε1.le
  exact ⟨learn_target rule input label g hfit hsep, hr, hm,
    fun eos n history => trained_generation rule input label g hfit hsep hε1 eos n history⟩

/-! An unknown orientation is learned from one prefix; free generation visits new prefixes. -/

/-- A two-rule causal bank; the next token also depends on prefix length (CCa7 witness). -/
def alternating (h : Bool) (xs : List Bool) : Bool := xor h (xs.length % 2 == 1)

/-- The empty prefix distinguishes both causal rules (CCa7 witness). -/
theorem alternating_separates : Separates alternating (fun _ : Unit => []) := by
  intro h k hne
  refine ⟨(), ?_⟩
  simpa [alternating] using hne

example := best_max (fun c : Bool => if c then (3 / 4 : ℝ) else 1 / 4) true
example := best_point (id : Bool → Bool) (ε := 1 / 4) (by norm_num) true
example := generate_congr (alternating false) (alternating false)
  (fun xs => Eq.refl (alternating false xs)) true 4 []
example := learns_causal alternating (fun _ : Unit => []) (fun _ => false) false
  (by intro s; rfl) alternating_separates (ε := 1 / 4) (by norm_num) (by norm_num)

/-- The learned rule emits a fresh token and then EOS, using its own new prefix (CCa7 witness). -/
example : generate
    (fun xs => best (smooth (1 / 4) (trained alternating (fun _ : Unit => [])
      (fun _ => false)) xs)) true 4 [] = [false, true] := by
  rw [trained_generation alternating (fun _ : Unit => []) (fun _ => false) false
    (by intro s; rfl) alternating_separates (by norm_num)]
  decide

end Transformer.Grokking.FlatSpectrum.ConvexCausal

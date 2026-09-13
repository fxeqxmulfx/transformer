/-
# RASP: the worked programs of §3

Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like Transformers",
§3 ("simple select-aggregate examples").

Two programs are given there in full, and both are checked here against the
sequence they are claimed to produce:

    flip    = select(indices, length-indices-1, ==);
    reverse = aggregate(flip, tokens);          -- reverse("hey") = "yeh"

    select_all = select(1, 1, ==);
    frac_as    = aggregate(select_all, 1 if tokens == "a" else 0);

`reverse` is the smallest program in the paper that needs the footnote to §3:
each of its rows selects exactly one position, and the value there is passed
through rather than averaged, so it transports tokens rather than numbers.
-/

import Transformer.RASP.Basic

namespace Transformer
namespace RASP

variable {n : ℕ} {α : Type*}

/-- `flip = select(indices, length-indices-1, ==)` (§3). -/
noncomputable def flip (n : ℕ) : Selector n :=
  sel (indices n) (fun i => length n i - indices n i - 1) (fun a b => decide (a = b))

/-- Each query position of `flip` selects the position at the opposite end of
the sequence, and only it. -/
lemma selected_flip (i : Fin n) : selected (flip n) i = {i.rev} := by
  have hle : (i : ℕ) + 1 ≤ n := i.isLt
  have hcast : ((n - ((i : ℕ) + 1) : ℕ) : ℝ) = (n : ℝ) - (i : ℕ) - 1 := by
    push_cast [hle]
    ring
  ext j
  have key : (((j : ℕ) : ℝ) = (n : ℝ) - (i : ℕ) - 1) ↔ ((j : ℕ) = n - ((i : ℕ) + 1)) := by
    rw [← hcast, Nat.cast_inj]
  simp [mem_selected, flip, sel, indices, length, Fin.ext_iff, Fin.val_rev, key]

/-- `reverse = aggregate(flip, tokens)` (§3). -/
noncomputable def reverse (t : Seq n α) (d : α) : Seq n α := aggregateOne (flip n) t d

/-- **`reverse` reverses.**  §3: `reverse("hey") = "yeh"`. -/
theorem reverse_apply (t : Seq n α) (d : α) (i : Fin n) : reverse t d i = t i.rev :=
  aggregateOne_of_selected_eq_singleton (selected_flip i) t d

/-- And it is an involution, so nothing is lost: the program is a
permutation of the input, not a lossy summary of it. -/
theorem reverse_reverse (t : Seq n α) (d : α) : reverse (reverse t d) d = t := by
  funext i
  rw [reverse_apply, reverse_apply, Fin.rev_rev]

/-- **`frac_as` is the frequency of a token.**  §3: aggregating the indicator
of `tokens == "a"` over `select_all` divides the number of occurrences by the
input length. -/
theorem fracVal_eq [DecidableEq α] (hn : 0 < n) (t : Seq n α) (a : α) (i : Fin n) :
    aggregate (selectAll n) (ind fun j => decide (t j = a)) 0 i
      = (({j | t j = a} : Finset (Fin n)).card : ℝ) / (n : ℝ) := by
  rw [aggregate_selectAll hn]
  congr 1
  simp [ind, Finset.sum_boole]

/-- **The frequency is a probability.**  Between `0` and `1`, so a downstream
elementwise comparison against a constant is meaningful — which is how §3
uses it. -/
theorem fracVal_mem_Icc [DecidableEq α] (hn : 0 < n) (t : Seq n α) (a : α) (i : Fin n) :
    aggregate (selectAll n) (ind fun j => decide (t j = a)) 0 i ∈ Set.Icc (0 : ℝ) 1 := by
  rw [fracVal_eq hn]
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  constructor
  · positivity
  · rw [div_le_one hn']
    exact_mod_cast Finset.card_filter_le _ _ |>.trans_eq (by simp)

/-- The hypotheses are satisfiable: `reverse("hey") = "yeh"`, and `"a"`
occupies two of the four positions of `"aabc"`. -/
example :
    let t : Seq 3 (Fin 3) := ![0, 1, 2]
    let u : Seq 4 (Fin 3) := ![0, 0, 1, 2]
    reverse t 0 = ![2, 1, 0] ∧
      aggregate (selectAll 4) (ind fun j => decide (u j = 0)) 0 0 = 2 / 4 := by
  intro t u
  refine ⟨?_, ?_⟩
  · funext i
    fin_cases i <;> simp [reverse_apply, t]
  · rw [fracVal_eq (by norm_num)]
    have h : ({j | u j = 0} : Finset (Fin 4)) = {0, 1} := by decide
    have hc : ({0, 1} : Finset (Fin 4)).card = 2 := by decide
    rw [h, hc]
    norm_num

end RASP
end Transformer

module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.Geometry
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.Margins

/-!
# An inhabited two-hop margin with a repeated intermediate object

Source: `review/flat_spectrum.md`, CM113. The example chooses values of
the learned parameters for a proof. Original training always starts from
the prospectively specified random initialization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

/-- CM113. Two angular coordinates distinguish three opaque objects. -/
noncomputable def chainAngle (o : Fin 3) : Fin 2 → ℝ := ![![0,0], ![Real.pi,0], ![Real.pi,Real.pi]] o

/-- Both interpolation gates are strictly inside the physical interval.
The learned predecessor potential suppresses the ambiguous earlier copy
of the intermediate object, CM113. -/
noncomputable def chain : Parameters (Fin 3) 1 :=
  ⟨chainAngle, chainAngle, chainAngle, 0, fun b => if b = some 0 then -64 else 0,
    64, 0, fun δ => if δ = 1 then 64 else 0, 31/32, 1/32⟩

/-- Two successive writes sharing their intermediate object, CM113. -/
def memory : Fin 4 → Fin 3 := ![0,1,1,2]

/-- The source and destination in the first genuine lookup, CM113. -/
def first : Hop 4 := ⟨0,1,64,64⟩

/-- The source and destination in the second genuine lookup, CM113. -/
def second : Hop 4 := ⟨2,3,64,64⟩

private theorem chain_average (u : Fin 2 → ℝ) :
    average (d := 1) u = (u 0 + u 1) / 2 := by
  change (∑ f : Fin 2, u f) / (2 : ℝ) = _
  rw [Fin.sum_univ_two]

/-- The physical witness satisfies both gate constraints, CM113. -/
theorem chain_admissible : Admissible chain := by norm_num [Admissible, chain]

/-- CM113. Repetitions with the same predecessor have identical learned
content and context terms. The learned recency coefficient is precisely
what distinguishes their source scores. -/
theorem repeated_key_difference {O : Type*} {d n : ℕ} (p : Parameters O d)
    (x : Fin n → O) (q : O) (i j : Fin n) (hx : x i = x j)
    (hprev : previous x i = previous x j) :
    keyScore p x q j - keyScore p x q i = p.recency * ((j.val : ℝ) - i.val) := by
  simp only [keyScore, hx, hprev]
  ring

/-- With recency zero, a strict source gap against a repeated object in the
same preceding context is impossible, CM113. The output can still combine
several paths to a common destination. -/
theorem repeated_zero_recency {O : Type*} {d n : ℕ} (p : Parameters O d)
    (x : Fin n → O) (q : O) (i j : Fin n) (hx : x i = x j)
    (hprev : previous x i = previous x j) (ht : p.recency = 0) :
    ¬ keyScore p x q i < keyScore p x q j := by
  have h := repeated_key_difference p x q i j hx hprev
  rw [ht, zero_mul] at h
  linarith

/-- CM113. Source positions zero and two beat all rivals for their respective
queries under the same learned match, including both occurrences of object one. -/
theorem chain_separation : first.Separated chain memory 0 ∧
    second.Separated chain memory (memory first.dest) := by
  constructor
  · constructor
    · intro j hj
      fin_cases j <;> norm_num [first, keyScore, chain, previous, chainAngle, memory,
        chain_average] at *
    · intro j hj
      fin_cases j <;> norm_num [first, linkScore, chain] at *
  · constructor
    · intro j hj
      fin_cases j <;> norm_num [first, second, keyScore, chain, previous, chainAngle, memory,
        chain_average] at *
    · intro j hj
      fin_cases j <;> norm_num [second, linkScore, chain] at *

/-- Shared value phases at object two have decoder gap at least one against
every other original class in the example, CM113. -/
theorem terminal_separation (c : Fin 3) (hc : c ≠ memory second.dest) :
    1 ≤ 1 - decode chain (chain.value (memory second.dest)) 0 c := by
  fin_cases c <;> norm_num [decode, chain, chainAngle, memory, second, chain_average] at *

private theorem exp_bound : Real.exp (-64) ≤ 1/65 := by
  rw [Real.exp_neg]
  have hi := (inv_le_inv₀ (Real.exp_pos (64 : ℝ)) (by norm_num : (0 : ℝ) < 65)).mpr
    (by linarith [Real.add_one_le_exp (64 : ℝ)])
  simpa using hi

/-- The sufficient mixed margin is strictly positive with both gates
interior and all three routes active, CM113. -/
theorem chain_positive : 0 < guaranteed chain first second 1 := by
  norm_num [guaranteed, Hop.error, first, second, chain]
  linarith [exp_bound]

/-- On the concrete original-object prefix, the first read alone prefers
the intermediate object and the full two-read model returns the terminal
object against every class, CM113. -/
theorem chain_identifies : readOne chain memory 0 0 2 < readOne chain memory 0 0 1 ∧
    ∀ c : Fin 3, c ≠ 2 → score chain [0,1,1,2,0] c < score chain [0,1,1,2,0] 2 := by
  constructor
  · have hc : 1 ≤ 1 - decode chain (chain.value (memory first.dest)) 0 2 := by
      norm_num [decode, chain, chainAngle, memory, first, chain_average]
    have h := readOne_margin chain memory 0 2 first 1 (by norm_num) chain_separation.1 hc
    change 1 - (1+2)*first.error ≤ readOne chain memory 0 0 1 - readOne chain memory 0 0 2 at h
    norm_num [Hop.error, first] at h
    linarith [exp_bound]
  · intro c hc
    have hh := correct_object chain chain_admissible memory 0 first second 1 (by norm_num)
      chain_separation.1 chain_separation.2 terminal_separation chain_positive c hc
    have hxs : (fun j : Fin 4 => [0,1,1,2][j]) = memory := by
      funext j
      fin_cases j <;> rfl
    change logit chain (fun j : Fin 4 => [0,1,1,2][j]) 0 c <
      logit chain (fun j : Fin 4 => [0,1,1,2][j]) 0 2
    rw [hxs]
    exact hh

example : ∑ k, jump chain memory 0 k = 1 := sum_jump (by decide) _ _ _
example : ∑ kl, trail chain memory 0 kl = 1 := sum_trail (by decide) _ _ _
example (q : Fin 3) : keyScore chain (![0,1,0,1] : Fin 4 → Fin 3) q 3 -
    keyScore chain (![0,1,0,1] : Fin 4 → Fin 3) q 1 = chain.recency * (((3 : Fin 4).val : ℝ) - (1 : Fin 4).val) :=
  repeated_key_difference chain (![0,1,0,1] : Fin 4 → Fin 3) q 1 3 rfl rfl
example (q : Fin 3) : ¬ keyScore chain (![0,1,0,1] : Fin 4 → Fin 3) q 1 <
    keyScore chain (![0,1,0,1] : Fin 4 → Fin 3) q 3 :=
  repeated_zero_recency chain (![0,1,0,1] : Fin 4 → Fin 3) q 1 3 rfl rfl rfl
example : |score chain [0,1,1,2,0] 2| ≤ 1 := score_bound chain chain_admissible _ _
example : 1 - jump chain memory 0 first.dest ≤ first.error :=
  jump_error chain memory 0 first chain_separation.1
example : 1 - trail chain memory 0 (first.dest,second.dest) ≤ first.error + second.error :=
  trail_error chain memory 0 first second chain_separation.1 chain_separation.2
example : 1 ≤ 1 - decode chain (chain.value (memory second.dest)) 0 1 :=
  terminal_separation 1 (by decide)
example : 1 - (1+2)*first.error ≤ readOne chain memory 0 0 1 - readOne chain memory 0 0 2 :=
  readOne_margin chain memory 0 2 first 1 (by norm_num) chain_separation.1
    (by norm_num [decode, chain, chainAngle, memory, first, chain_average])
example : logit chain memory 0 1 < logit chain memory 0 2 :=
  correct_object chain chain_admissible memory 0 first second 1 (by norm_num)
    chain_separation.1 chain_separation.2 terminal_separation chain_positive 1 (by decide)
example : 1 - (1+2)*(first.error+second.error) ≤
    readTwo chain memory 0 2 - readTwo chain memory 0 1 :=
  readTwo_margin chain memory 0 1 first second 1 (by norm_num) chain_separation.1
    chain_separation.2 (terminal_separation 1 (by decide))
example : guaranteed chain first second 1 ≤ logit chain memory 0 2 - logit chain memory 0 1 :=
  logit_margin chain chain_admissible memory 0 1 first second 1 (by norm_num)
    chain_separation.1 chain_separation.2 (terminal_separation 1 (by decide))
example : Landscape.Neural.crossEntropy (fun c => 16 * logit chain memory 0 c) 2 ≤
    Real.log (1 + (Fintype.card (Fin 3) - 1) * Real.exp (-(16 * guaranteed chain first second 1))) :=
  crossEntropy_bound chain chain_admissible memory 0 first second 1 16 (by norm_num)
    (by norm_num) chain_separation.1 chain_separation.2 terminal_separation

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

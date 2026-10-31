module

public import Transformer.CRASP.Normalized.Counting
public import Transformer.CRASP.PositionalTransformers
public import Transformer.SequenceModel.Attention.Grid.Head
public import Transformer.SequenceModel.Attention.Grid.Separation

/-!
# Depth of a stack of normalized counting layers

Extension of arXiv:2506.16055v3, Appendix B.2/F, `eq:phi_h`;
`review/flat_spectrum.md`, *Depth, width and precision*, R3.
The theorem is uniform over prefixes and layer count for a fixed finite state
space. Width changes the state space and formula size, not counting depth.
R5 proves the scope boundary: Appendix F assumes periodic angles, whereas
the exact source first frequency admits no finite positional code at all lengths.
-/

@[expose] public section

namespace Transformer.CRASP.Normalized

universe u v w
variable {Q : Type u} {C : Type v} {σ : Type w} {p s k : ℕ}

/-- Every layer reads the previous layer at every causal position. -/
noncomputable def run (layers : ℕ → Layer Q C p s) (bos : Q)
    (initial : ℕ → Q) : ℕ → ℕ → Q
  | 0 => initial
  | n + 1 => fun i => if i = 0 then bos else
      (layers n).step bos (run layers bos initial n) i

variable [Fintype Q] [Fintype C]

/-- Compile the entire stack by substituting the previous state predicates. -/
noncomputable def formula (layers : ℕ → Layer Q C p s) (bos : Q)
    (initial : Q → FormP σ) : ℕ → Q → FormP σ
  | 0 => initial
  | n + 1 => (layers n).formula bos (formula layers bos initial n)

/-- A depth-n stack adds at most n counting levels, regardless of the finite
number of parallel heads and arbitrary local nonlinear maps (B.2/F extension). -/
theorem formula_mem (layers : ℕ → Layer Q C p s) (bos : Q)
    (initial : Q → FormP σ) (hinit : ∀ r, initial r ∈ TLClMod σ k)
    (n : ℕ) (r : Q) : formula layers bos initial n r ∈ TLClMod σ (k + n) := by
  induction n generalizing r with
  | zero => exact hinit r
  | succ n ih =>
      exact (layers n).formula_mem bos (fun q => ih q) r

variable [DecidableEq Q] [DecidableEq σ]

/-- The compiled stack agrees with execution on every positive prefix. No
approximation or training-success assumption enters this simulation theorem. -/
theorem sat_formula (layers : ℕ → Layer Q C p s) (bos : Q)
    (hL : ∀ n, (layers n).Nonnegative) (initial : Q → FormP σ)
    (q : ℕ → Q) (w : List σ) (n : ℕ) {i : ℕ} (hi : 0 < i)
    (hinit : ∀ j ∈ Finset.Icc 1 i, ∀ a, (initial a).sat w j = decide (q j = a))
    (r : Q) : (formula layers bos initial n r).sat w i =
      decide (run layers bos q n i = r) := by
  induction n generalizing i r with
  | zero => exact hinit i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩) r
  | succ n ih =>
      rw [Bool.eq_iff_iff, decide_eq_true_iff]
      have hprev : ∀ j ∈ Finset.Icc 1 i, ∀ a,
          (formula layers bos initial n a).sat w j = decide (run layers bos q n j = a) := by
        intro j hj a
        have hjb := Finset.mem_Icc.mp hj
        apply ih (by omega)
        intro t ht b
        have htb := Finset.mem_Icc.mp ht
        exact hinit t (Finset.mem_Icc.mpr ⟨htb.1, htb.2.trans hjb.2⟩) b
      simpa only [formula, run, ite_eq_right (Nat.ne_of_gt hi)] using
        (layers n).sat_formula bos (hL n) (formula layers bos initial n)
          (run layers bos q n) w hi hprev r

/-- Tokenwise embedding and rounding have a depth-zero formula over any fixed
finite input alphabet. Position annotations, when used, are part of that alphabet. -/
noncomputable def initialFormula [Fintype σ] (embed : σ → Q) (r : Q) : FormP σ :=
  FormP.any (Finset.univ.toList.map fun a : σ =>
    .and (.sym a) (FormP.truth (decide (embed a = r))))

omit [Fintype Q] [DecidableEq σ] in
/-- Tokenwise finite embedding costs no counting level (B.2/F extension). -/
theorem initialFormula_mem [Fintype σ] (embed : σ → Q) (r : Q) :
    initialFormula embed r ∈ TLClMod σ 0 := by
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨a, _, rfl⟩ := List.mem_map.mp hφ
  exact FormP.and_mem ⟨rfl, le_rfl⟩ (FormP.truth_mem _ _)

omit [Fintype Q] in
/-- The initial formula tests the actual embedded token, rather than an
assumed hidden-state predicate (B.2/F extension). -/
theorem sat_initialFormula [Fintype σ] (embed : σ → Q) (r : Q)
    (w : List σ) (i : ℕ) (a : σ) (ha : w[i - 1]? = some a) :
    (initialFormula embed r).sat w i = decide (embed a = r) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff, initialFormula, FormP.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨b, _, rfl⟩ := List.mem_map.mp hφ
    simp only [FormP.sat, ha, Option.some.injEq, Bool.and_eq_true,
      decide_eq_true_eq, FormP.sat_truth] at hs
    exact hs.1 ▸ hs.2
  · intro h
    refine ⟨_, List.mem_map_of_mem (a := a)
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [FormP.sat, ha, FormP.sat_truth, h, decide_true, Bool.and_self]

private def identityLayer : Layer Bool Unit 2 0 :=
  ⟨fun _ _ _ => 0, fun _ _ _ => 0, fun q _ => q⟩

/-- Stack assumptions have a concrete binary embedding and readable prefix. -/
example : (∀ n : ℕ, ((fun _ => identityLayer) n).Nonnegative) ∧
    (∀ r : Bool, initialFormula id r ∈ TLClMod Bool 0) ∧ 0 < 1 ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Bool,
      (initialFormula id a).sat [true] j = decide (true = a)) := by
  refine ⟨by simp [Layer.Nonnegative, identityLayer], initialFormula_mem id, by decide, ?_⟩
  intro j hj a
  have : j = 1 := by have := Finset.mem_Icc.mp hj; omega
  subst j
  exact sat_initialFormula id a [true] 1 true rfl

open GPTMini SequenceModel.Attention

/-- The source's first rotary frequency is not a rational multiple of pi,
independently of base and the other rotary pairs (Appendix F scope boundary). -/
theorem first_frequency_not_rational (d : ℕ) (hd : 1 < d) (base : ℝ) :
    ¬ ∃ q : ℚ, invFreq d base ⟨0, by omega⟩ = Real.pi * q := by
  rw [Grid.first_frequency d hd base]
  rintro ⟨q, hq⟩
  have hne : q ≠ 0 := by intro h; simp [h] at hq
  have hi := irrational_mul_ratCast_iff.mpr ⟨hne, irrational_pi⟩
  exact hi.ne_rat 1 (by simpa using hq.symm)

/-- No extension of the source frequency vector to natural indices satisfies
the existing positional hierarchy's `RationalAngles` hypothesis. -/
theorem source_not_rationalAngles (d : ℕ) (hd : 1 < d) (base : ℝ)
    (θ : ℕ → ℝ) (hθ : θ 0 = invFreq d base ⟨0, by omega⟩) :
    ¬ PosEnc.RationalAngles (.rope θ) := by
  rintro ⟨q, hq⟩
  exact first_frequency_not_rational d hd base ⟨q 0, hθ ▸ hq 0⟩

/-- Exact integer-position first-pair rotations have no positive period. -/
theorem first_cosine_not_periodic :
    ¬ ∃ period : ℕ, 0 < period ∧ ∀ i : ℕ,
      Real.cos ((i + period : ℕ) : ℝ) = Real.cos (i : ℝ) := by
  rintro ⟨period, hp, h⟩
  have he := Grid.cos_nat_injective (h 0)
  omega

/-- Any finite code loses an exact first-pair cosine at some integer position.
This rules out an exact residue-table reduction for unbounded source positions. -/
theorem no_finite_exact_position_code {Q : Type*} [Finite Q]
    (code : ℕ → Q) (decode : Q → ℝ) :
    ∃ i : ℕ, decode (code i) ≠ Real.cos (i : ℝ) := by
  by_contra h
  push Not at h
  apply not_injective_infinite_finite code
  intro i j hij
  apply Grid.cos_nat_injective
  change Real.cos (i : ℝ) = Real.cos (j : ℝ)
  rw [← h i, ← h j, hij]

/-- The nonperiodic source assumptions are inhabited by a standard head size,
base 10000 and the actual geometric frequency sequence. -/
example : (1 : ℕ) < 4 ∧
    (fun c : ℕ => (10000 : ℝ) ^ (-((2 * c : ℕ) : ℝ) / 4)) 0 =
      invFreq 4 10000 ⟨0, by decide⟩ := ⟨by decide, rfl⟩

end Transformer.CRASP.Normalized

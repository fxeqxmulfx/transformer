module

public import Transformer.SequenceModel.Attention.Finite.Codes
public import Mathlib.Analysis.Real.Pi.Irrational
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
public import Mathlib.LinearAlgebra.LinearIndependent.Basic
public import Mathlib.Analysis.Analytic.Uniqueness
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Bounded query gain separates subsets on the fixed integer RoPE grid

Source: `review/flat_spectrum.md`, *Prescribed RoPE capacity*, P1--P2.
Irrationality of pi distinguishes integer cosine offsets. Dedekind character
independence and analytic uniqueness then give a separating query gain in
any positive open interval. No phase table is chosen or trained.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Grid

open GPTMini
open scoped BigOperators Topology RealInnerProductSpace

/-- P1: nonnegative integer offsets have distinct cosine values. The
irrationality of pi rules out both possible periodic cosine collisions. -/
theorem cos_nat_injective : Function.Injective (fun n : ℕ => Real.cos (n : ℝ)) := by
  intro a b h
  obtain ⟨k, h | h⟩ := Real.cos_eq_cos_iff.mp h
  · by_cases hk : k = 0
    · simp only [hk, Int.cast_zero, mul_zero, zero_mul, zero_add] at h
      exact_mod_cast h.symm
    · have hi := irrational_pi.intCast_mul (m := 2 * k) (mul_ne_zero (by norm_num) hk)
      exfalso
      apply hi.ne_int ((b : ℤ) - a)
      push_cast
      linarith
  · by_cases hk : k = 0
    · simp only [hk, Int.cast_zero, mul_zero, zero_mul, zero_sub] at h
      have ha : (0 : ℝ) ≤ a := Nat.cast_nonneg _
      have hb : (0 : ℝ) ≤ b := Nat.cast_nonneg _
      have he : (a : ℝ) = b := by linarith
      exact_mod_cast he
    · have hi := irrational_pi.intCast_mul (m := 2 * k) (mul_ne_zero (by norm_num) hk)
      exfalso
      apply hi.ne_int ((b : ℤ) + a)
      push_cast
      linarith

/-- The source first-pair cosine scores seen from the last position. -/
def integerSlopes (N : ℕ) (j : Fin N) : ℝ := Real.cos ((N : ℝ) - j.val)

/-- P1: all operand positions have distinct fixed source rotary slopes. -/
theorem integerSlopes_injective (N : ℕ) : Function.Injective (integerSlopes N) := by
  intro i j h
  have hi : i.val ≤ N := by omega
  have hj : j.val ≤ N := by omega
  have he : Real.cos ((N - i.val : ℕ) : ℝ) = Real.cos ((N - j.val : ℕ) : ℝ) := by
    simpa only [integerSlopes, Nat.cast_sub hi, Nat.cast_sub hj] using h
  have hn := cos_nat_injective he
  apply Fin.ext
  omega

/-- A real exponential score is a multiplicative character of the additive reals. -/
def expCharacter (a : ℝ) : Multiplicative ℝ →* ℝ where
  toFun x := Real.exp (a * x.toAdd)
  map_one' := by simp
  map_mul' x y := by simp [mul_add, Real.exp_add]

/-- P2: different slopes give different characters, witnessed already at one. -/
theorem expCharacter_injective : Function.Injective expCharacter := by
  intro a b h
  have he := congrArg (fun χ : Multiplicative ℝ →* ℝ => χ (Multiplicative.ofAdd 1)) h
  change Real.exp (a * 1) = Real.exp (b * 1) at he
  have hx : Real.exp a = Real.exp b := by simpa only [mul_one] using he
  exact Real.exp_injective hx

/-- P2: Dedekind character independence makes every nontrivial finite
combination of distinct exponential scores nonzero at some query gain. -/
theorem sum_exp_nonzero {A : Type*} [Fintype A] (a c : A → ℝ)
    (ha : Function.Injective a) (hc : ∃ j, c j ≠ 0) :
    ∃ t : ℝ, (∑ j, c j * Real.exp (a j * t)) ≠ 0 := by
  have hi := (linearIndependent_monoidHom (Multiplicative ℝ) ℝ).comp
    (fun j => expCharacter (a j)) (expCharacter_injective.comp ha)
  by_contra h
  push Not at h
  have hz : (∑ j, c j • (expCharacter (a j) : Multiplicative ℝ → ℝ)) = 0 := by
    funext t
    rw [Finset.sum_apply]
    change (∑ j, c j * Real.exp (a j * t.toAdd)) = 0
    exact h t.toAdd
  have hzero := (Fintype.linearIndependent_iff.mp hi) c hz
  obtain ⟨j, hj⟩ := hc
  exact hj (hzero j)

example : Real.cos (7 : ℝ) ≠ Real.cos (11 : ℝ) := by
  intro h
  have hn := cos_nat_injective (a₁ := 7) (a₂ := 11) h
  omega
example := integerSlopes_injective 10
example := sum_exp_nonzero (fun a : Fin 3 => (a.val : ℝ)) ![1, -2, 1]
  (by intro a b h; change (a.val : ℝ) = b.val at h; apply Fin.ext; exact_mod_cast h)
  ⟨0, by norm_num⟩

/-- P2: a globally analytic real function that is nonzero somewhere cannot
vanish on any open interval, by the proved analytic identity principle. -/
theorem dense_nonzero (f : ℝ → ℝ) (hf : AnalyticOnNhd ℝ f Set.univ)
    (hn : ∃ t, f t ≠ 0) : Dense {t | f t ≠ 0} := by
  have hi : interior {t | f t = 0} = ∅ := by
    by_contra h
    obtain ⟨t, ht⟩ := Set.nonempty_iff_ne_empty.mpr h
    have he : f =ᶠ[𝓝 t] (fun _ => 0) := mem_interior_iff_mem_nhds.mp ht
    have hz : Set.EqOn f (fun _ => 0) Set.univ :=
      hf.eqOn_of_preconnected_of_eventuallyEq (fun _ _ => analyticAt_const)
        isPreconnected_univ (Set.mem_univ t) he
    obtain ⟨s, hs⟩ := hn
    exact hs (hz (Set.mem_univ s))
  exact interior_eq_empty_iff_dense_compl.mp hi

/-- P2: finitely many such analytic constraints can simultaneously avoid
zero in any nonempty open gain interval. -/
theorem exists_nonzero_gain {B : Type*} [Fintype B] (f : B → ℝ → ℝ)
    (hf : ∀ b, AnalyticOnNhd ℝ (f b) Set.univ) (hn : ∀ b, ∃ t, f b t ≠ 0)
    {lo hi : ℝ} (h : lo < hi) : ∃ t ∈ Set.Ioo lo hi, ∀ b, f b t ≠ 0 := by
  have hd : Dense (⋂ b, {t | f b t ≠ 0}) :=
    dense_iInter_of_isOpen
      (fun b => isOpen_ne_fun (continuousOn_univ.mp (hf b).continuousOn) continuous_const)
      (fun b => dense_nonzero (f b) (hf b) (hn b))
  obtain ⟨t, ht, hn⟩ := hd.inter_open_nonempty (Set.Ioo lo hi) isOpen_Ioo (Set.nonempty_Ioo.mpr h)
  exact ⟨t, ht, Set.mem_iInter.mp hn⟩

/-- The exponentiated positional scores at a fixed ordinary query gain. -/
def exponentialWeights {N : ℕ} (a : Fin N → ℝ) (t : ℝ) : EucSpace N :=
  WithLp.toLp 2 (fun j => Real.exp (a j * t))

/-- Difference of two positional subset sums, as a real analytic function of gain. -/
def scoreDifference {N : ℕ} (a : Fin N → ℝ) (x y : Fin N → Bool) (t : ℝ) : ℝ :=
  ∑ j, ((if x j then 1 else 0) - (if y j then 1 else 0)) * Real.exp (a j * t)

/-- P2: the scalar analytic constraint is precisely equality of the
context code's actual exponentiated positional subset sums. -/
theorem scoreDifference_eq {N : ℕ} (a : Fin N → ℝ) (x y : Fin N → Bool) (t : ℝ) :
    scoreDifference a x y t =
      ⟪exponentialWeights a t, Finite.bits N x⟫ - ⟪exponentialWeights a t, Finite.bits N y⟫ := by
  rw [Finite.inner_bits, Finite.inner_bits, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  cases x j <;> cases y j <;> simp [exponentialWeights]

/-- P2: all constraints are analytic on the entire real gain line. -/
theorem analytic_scoreDifference {N : ℕ} (a : Fin N → ℝ) (x y : Fin N → Bool) :
    AnalyticOnNhd ℝ (scoreDifference a x y) Set.univ := by
  intro t _
  apply Finset.analyticAt_fun_sum
  intro j _
  exact analyticAt_const.fun_mul ((analyticAt_const.fun_mul analyticAt_id).rexp')

/-- P2: distinct fixed slopes make every unequal-subset constraint nontrivial. -/
theorem scoreDifference_nonzero {N : ℕ} (a : Fin N → ℝ) (ha : Function.Injective a)
    (x y : Fin N → Bool) (hxy : x ≠ y) : ∃ t, scoreDifference a x y t ≠ 0 := by
  apply sum_exp_nonzero a _ ha
  by_contra h
  push Not at h
  apply hxy
  funext j
  have hj := h j
  cases hx : x j <;> cases hy : y j <;> simp_all

/-- P2: a bounded positive trainable gain separates every ordered positional
subset at once. The prescribed slopes and positions are never changed. -/
theorem exists_separating_gain {N : ℕ} (a : Fin N → ℝ) (ha : Function.Injective a)
    {upper : ℝ} (hu : 0 < upper) :
    ∃ t : ℝ, 0 < t ∧ t < upper ∧
      Function.Injective (fun b : Fin N → Bool => ⟪exponentialWeights a t, Finite.bits N b⟫) := by
  classical
  let B := {xy : (Fin N → Bool) × (Fin N → Bool) // xy.1 ≠ xy.2}
  let f : B → ℝ → ℝ := fun b => scoreDifference a b.val.1 b.val.2
  obtain ⟨t, ht, hn⟩ := exists_nonzero_gain f
    (fun b => analytic_scoreDifference a b.val.1 b.val.2)
    (fun b => scoreDifference_nonzero a ha b.val.1 b.val.2 b.property) hu
  refine ⟨t, ht.1, ht.2, ?_⟩
  intro x y h
  change ⟪exponentialWeights a t, Finite.bits N x⟫ = ⟪exponentialWeights a t, Finite.bits N y⟫ at h
  by_contra hxy
  apply hn ⟨(x, y), hxy⟩
  rw [show f ⟨(x, y), hxy⟩ t = scoreDifference a x y t from rfl, scoreDifference_eq, h, sub_self]

example := dense_nonzero (fun t => t - 1)
  (by intro t ht; exact analyticAt_id.sub analyticAt_const) ⟨2, by norm_num⟩
example := exists_nonzero_gain (fun a : Fin 3 => fun t => t - a.val)
  (by intro a t ht; exact analyticAt_id.sub analyticAt_const)
  (by intro a; refine ⟨a.val + 1, ?_⟩; norm_num) (lo := 0) (hi := 1 / 1000) (by norm_num)
example := scoreDifference_nonzero (integerSlopes 3) (integerSlopes_injective 3)
  ![true, false, true] ![false, true, false] (by decide)
example := exists_separating_gain (integerSlopes 17) (integerSlopes_injective 17)
  (upper := 1 / 1000000) (by norm_num)

end Transformer.SequenceModel.Attention.Grid

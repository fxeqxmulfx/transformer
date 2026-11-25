module

public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# SSMax reads in a loop

Source: `review/flat_spectrum.md`, *Loops*, item LP9. A loop that follows a chain of pointers
reads one link per pass. Position `j` of a context of length `n` holds the key `e_j` and the value
`e_{f j}` of a table `f`; the state of the loop is a query `c`, which scores the key of `j` by
`c j`. One read at the scale `β` averages the values with the weights `e^{β c_j}` (`softmaxHop`),
and SSMax reads at the scale `s log n` (`ssmaxHop`). Each pass reads with the query that the pass
before wrote, so after `N` passes the query is `(ssmaxHop s f)^[N] c`.

A query within `η` of the one-hot `e_v` (`NearOneHot`) leads the other keys by `1 − 2η`, so by
`Recall.abs_logScaleAttention_sub_le` one SSMax read is within `2 (n − 1) n^{−s(1−2η)}` of
`e_{f v}` (`nearOneHot_ssmaxHop`). Where that is at most `η`, the tolerance does not accumulate:
the loop follows the chain for any number of passes (`nearOneHot_iterate_ssmaxHop`). For
`s (1 − 2η) > 1` this holds from some length on, for every table, start and number of passes
(`exists_forall_nearOneHot_iterate_ssmaxHop`).

At a fixed scale `β` the exact query `e_v` on an injective table keeps the weight
`e^β/(e^β + n − 1)` on its link (`softmaxHop_single_apply`), so from some length on one pass
already loses the chain (`exists_forall_not_nearOneHot_softmaxHop`). SSMax at `s = 1` keeps the
weight `n/(2n − 1)`, which tends to `1/2` (`ssmaxHop_one_single_apply`): a lead of `1` is the
threshold itself.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Filter Topology
open Transformer.Grokking.FlatSpectrum.Recall (softmaxAttention logScaleAttention
  abs_logScaleAttention_sub_le)

variable {ι : Type*} [DecidableEq ι]

/-- `c` lies within `η` of the one-hot vector `e_v`, coordinate by coordinate. -/
def NearOneHot (η : ℝ) (c : ι → ℝ) (v : ι) : Prop :=
  ∀ w, |c w - (Pi.single v 1 : ι → ℝ) w| ≤ η

/-- The one-hot vector `e_v` is within every `η ≥ 0` of itself. -/
lemma nearOneHot_single {η : ℝ} (hη : 0 ≤ η) (v : ι) : NearOneHot η (Pi.single v 1) v :=
  fun w => by simpa using hη

example := nearOneHot_single (ι := Fin 3) zero_le_one 0

variable [Fintype ι]

/-- **One read at the scale `β`**: position `j` holds the key `e_j` and the value `e_{f j}`, the
query `c` scores the key of `j` by `β c_j`, and the read is the softmax average of the values. -/
noncomputable def softmaxHop (β : ℝ) (f : ι → ι) (c : ι → ℝ) : ι → ℝ :=
  fun w => softmaxAttention (fun j => β * c j) fun j => (Pi.single (f j) 1 : ι → ℝ) w

/-- **One SSMax read**: the scale `s log n` over the `n` positions. -/
noncomputable def ssmaxHop (s : ℝ) (f : ι → ι) : (ι → ℝ) → ι → ℝ :=
  softmaxHop (s * Real.log (Fintype.card ι)) f

/-- **One SSMax read keeps the chain**: if the query is within `η` of `e_v`, the read is within
`2 (n − 1) n^{−s(1−2η)}` of `e_{f v}`, the next link. -/
theorem nearOneHot_ssmaxHop {s η : ℝ} (hs : 0 ≤ s) (f : ι → ι) {c : ι → ℝ} {v : ι}
    (hc : NearOneHot η c v) :
    NearOneHot (2 * (Fintype.card ι - 1) * (Fintype.card ι : ℝ) ^ (-(s * (1 - 2 * η))))
      (ssmaxHop s f c) (f v) := by
  intro w
  have hz : ∀ j, j ≠ v → c j + (1 - 2 * η) ≤ c v := fun j hj => by
    have h1 := hc j
    have h2 := hc v
    rw [Pi.single_eq_of_ne hj, sub_zero] at h1
    rw [Pi.single_eq_same] at h2
    linarith [(abs_le.1 h1).2, (abs_le.1 h2).1]
  have h := abs_logScaleAttention_sub_le (C := 1) hs hz
    (v := fun j => (Pi.single (f j) 1 : ι → ℝ) w) fun j => by
      rw [Pi.single_apply]
      split_ifs <;> simp
  rw [mul_one] at h
  exact h

example := nearOneHot_ssmaxHop (ι := Fin 2) (s := 1) zero_le_one id (nearOneHot_single le_rfl 0)

/-- **Every pass keeps the tolerance**: where the error `2 (n − 1) n^{−s(1−2η)}` of one read is at
most `η`, the query after `N` passes is within `η` of `e_{f^N v}`. -/
theorem nearOneHot_iterate_ssmaxHop {s η : ℝ} (hs : 0 ≤ s)
    (hη : 2 * (Fintype.card ι - 1) * (Fintype.card ι : ℝ) ^ (-(s * (1 - 2 * η))) ≤ η)
    (f : ι → ι) {c : ι → ℝ} {v : ι} (hc : NearOneHot η c v) (N : ℕ) :
    NearOneHot η ((ssmaxHop s f)^[N] c) (f^[N] v) := by
  induction N with
  | zero => exact hc
  | succ N ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    exact fun w => (nearOneHot_ssmaxHop hs f ih w).trans hη

/-- Two positions at `s = 6` keep the tolerance `1/4`: one read errs by `2 · 2^{−3} = 1/4`. -/
example (f : Fin 2 → Fin 2) (N : ℕ) :=
  nearOneHot_iterate_ssmaxHop (s := 6) (η := 1 / 4) (by norm_num) (by
    rw [Fintype.card_fin, show (-(6 * (1 - 2 * (1 / 4 : ℝ)))) = ((-3 : ℤ) : ℝ) by norm_num,
      Real.rpow_intCast]
    norm_num) f (nearOneHot_single (by norm_num) 0) N

/-- **SSMax follows the chain at every length**: for `s (1 − 2η) > 1` there is a length from which
on every query within `η` of `e_v` stays, after any number `N` of passes, within `η` of
`e_{f^N v}`, for every table `f`. -/
theorem exists_forall_nearOneHot_iterate_ssmaxHop {s η : ℝ} (hs : 0 ≤ s) (hη : 0 < η)
    (hsη : 1 < s * (1 - 2 * η)) :
    ∃ n₀ : ℕ, ∀ n ≥ n₀, ∀ (f : Fin n → Fin n) (c : Fin n → ℝ) (v : Fin n),
      NearOneHot η c v → ∀ N, NearOneHot η ((ssmaxHop s f)^[N] c) (f^[N] v) := by
  have ht : Tendsto (fun n : ℕ => 2 * (n : ℝ) ^ (-(s * (1 - 2 * η) - 1))) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < s * (1 - 2 * η) - 1)).comp
      tendsto_natCast_atTop_atTop).const_mul 2
    rw [mul_zero] at h
    exact h
  obtain ⟨n₀, hn₀⟩ := eventually_atTop.1 (ht.eventually (ge_mem_nhds hη))
  refine ⟨n₀, fun n hn f c v hc => nearOneHot_iterate_ssmaxHop hs ?_ f hc⟩
  have hpos : (0 : ℝ) < n := by exact_mod_cast v.pos
  rw [Fintype.card_fin]
  calc 2 * ((n : ℝ) - 1) * (n : ℝ) ^ (-(s * (1 - 2 * η)))
      ≤ 2 * (n : ℝ) * (n : ℝ) ^ (-(s * (1 - 2 * η))) :=
        mul_le_mul_of_nonneg_right (by linarith) (Real.rpow_nonneg hpos.le _)
    _ = 2 * (n : ℝ) ^ (-(s * (1 - 2 * η) - 1)) := by
        rw [show -(s * (1 - 2 * η) - 1) = 1 + -(s * (1 - 2 * η)) by ring, Real.rpow_add hpos,
          Real.rpow_one, mul_assoc]
    _ ≤ η := hn₀ n hn

example := exists_forall_nearOneHot_iterate_ssmaxHop (s := 3) (η := 1 / 4) (by norm_num)
  (by norm_num) (by norm_num)

/-- **A fixed scale keeps the weight `e^β/(e^β + n − 1)`**: on an injective table, the read of the
exact query `e_v` holds at the next link `f v` the weight of the key of `v`. -/
theorem softmaxHop_single_apply {f : ι → ι} (hf : Function.Injective f) (β : ℝ) (v : ι) :
    softmaxHop β f (Pi.single v 1) (f v) = Real.exp β / (Real.exp β + (Fintype.card ι - 1)) := by
  have hval (j : ι) : (Pi.single (f j) 1 : ι → ℝ) (f v) = (Pi.single v 1 : ι → ℝ) j := by
    by_cases h : j = v
    · subst h
      simp
    · rw [Pi.single_eq_of_ne (hf.ne (Ne.symm h)), Pi.single_eq_of_ne h]
  have hnum : ∑ j, Real.exp (β * (Pi.single v 1 : ι → ℝ) j) * (Pi.single v 1 : ι → ℝ) j =
      Real.exp β := by
    rw [Fintype.sum_eq_single v]
    · simp
    · intro j hj
      simp [Pi.single_eq_of_ne hj]
  have hsplit : ∑ j, Real.exp (β * (Pi.single v 1 : ι → ℝ) j) =
      Real.exp (β * (Pi.single v 1 : ι → ℝ) v) +
        ∑ j ∈ univ.erase v, Real.exp (β * (Pi.single v 1 : ι → ℝ) j) :=
    (add_sum_erase _ _ (mem_univ v)).symm
  have hone : ∀ j ∈ univ.erase v, Real.exp (β * (Pi.single v 1 : ι → ℝ) j) = 1 := fun j hj => by
    rw [Pi.single_eq_of_ne (ne_of_mem_erase hj), mul_zero, Real.exp_zero]
  simp only [softmaxHop, softmaxAttention, hval]
  rw [hnum, hsplit, sum_congr rfl hone, Pi.single_eq_same, mul_one, sum_const,
    card_erase_of_mem (mem_univ v), card_univ, nsmul_eq_mul, mul_one,
    Nat.cast_sub (Fintype.card_pos_iff.2 ⟨v⟩), Nat.cast_one]

example := softmaxHop_single_apply (ι := Fin 3) Function.injective_id 1 0

/-- **A fixed scale loses the chain**: for every scale `β` and every tolerance `η < 1`, from some
length on, one read of the exact query `e_v` on an injective table is farther than `η` from
`e_{f v}`. -/
theorem exists_forall_not_nearOneHot_softmaxHop (β : ℝ) {η : ℝ} (hη : η < 1) :
    ∃ n₀ : ℕ, ∀ n ≥ n₀, ∀ f : Fin n → Fin n, Function.Injective f → ∀ v : Fin n,
      ¬ NearOneHot η (softmaxHop β f (Pi.single v 1)) (f v) := by
  obtain ⟨n₀, hn₀⟩ := exists_nat_gt (Real.exp β * η / (1 - η) + 1)
  refine ⟨n₀, fun n hn f hf v h => ?_⟩
  have h1 := h (f v)
  rw [softmaxHop_single_apply hf, Pi.single_eq_same, Fintype.card_fin] at h1
  have he := Real.exp_pos β
  have hn' : (n₀ : ℝ) ≤ n := by exact_mod_cast hn
  have hv : (1 : ℝ) ≤ n := by exact_mod_cast v.pos
  have hd : 0 < Real.exp β + ((n : ℝ) - 1) := by linarith
  have h2 := (abs_le.1 h1).1
  rw [le_sub_iff_add_le, le_div_iff₀ hd] at h2
  have h3 : Real.exp β * η / (1 - η) < (n : ℝ) - 1 := by linarith
  rw [div_lt_iff₀ (by linarith)] at h3
  nlinarith

example := exists_forall_not_nearOneHot_softmaxHop 10 (η := 1 / 4) (by norm_num)

/-- **At `s = 1` a lead of `1` is the threshold**: on an injective table, the exact query `e_v`
keeps the weight `n/(2n − 1)` on its link, which tends to `1/2`. -/
theorem ssmaxHop_one_single_apply {f : ι → ι} (hf : Function.Injective f) (v : ι) :
    ssmaxHop 1 f (Pi.single v 1) (f v) = Fintype.card ι / (2 * Fintype.card ι - 1) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨v⟩
  rw [ssmaxHop, one_mul, softmaxHop_single_apply hf, Real.exp_log hn]
  congr 1
  ring

example := ssmaxHop_one_single_apply (ι := Fin 3) Function.injective_id 0

end Transformer.Grokking.FlatSpectrum

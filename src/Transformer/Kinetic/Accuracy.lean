/-
# Kinetic theory for Transformers — the prediction accuracy

Formalization of `eq:Acc-def` of arXiv:2605.09213v1, *Kinetic theory for
Transformers and the lost-in-the-middle phenomenon*, §1.1: the retrieval task's
positions, and the identity that turns "the decoder returns the source word"
into a radius test on the hidden state.

The identity is proved, on the codeword geometry of `Codewords.lean`.  It holds
off the tie set `|θ_N(t) - θ_{i_*}(0)|_𝕋 = π/M`, where the nearest codeword is
not unique, the indicator on the right is `1`, and an `argmin` may legitimately
return the other word.  The source does not mention ties; excluding them is the
one hypothesis added here, and it is added rather than assumed away because
`argmin` over a finite set always returns *some* minimizer and the source's `=`
is otherwise false.
-/

import Transformer.Kinetic.Codewords

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The last token of a prompt of length `N`. -/
def lastIdx (N : ℕ) (hN : 0 < N) : Idx N := ⟨N - 1, by omega⟩

/-- The source position `i_* = ⌊σ₀N⌋` of the retrieval task, as a `0`-based
index.

**Where this differs from the source.**  The source writes
`1 ≤ i_* = ⌊σ₀N⌋ ≤ N`, which is an assumption on `σ₀` and `N` and not a
consequence of `σ₀ ∈ (0,1)`: at `σ₀ < 1/N` the floor is `0`.  A `Fin N` has to
be total, so both ends are clamped — to the first token when `⌊σ₀N⌋ = 0`, and
to the last when `σ₀ ≥ 1`.  Under the hypothesis `σ₀ ∈ (0,1)` that every
theorem here carries, `σ₀N < N` gives `⌊σ₀N⌋ ≤ N - 1` and only the lower clamp
can act; it makes `σ₀ ↦ 𝒜_N(t,σ₀)` constant on `(0, 1/N)` at the value of the
first position, which is the position the source's own convention puts there.

Source: arXiv:2605.09213v1, `eq:Acc-def`. -/
noncomputable def sourceIdx (N : ℕ) (hN : 0 < N) (σ₀ : ℝ) : Idx N :=
  ⟨min (⌊σ₀ * N⌋₊ - 1) (N - 1), by omega⟩

/-- **Equation (eq:Acc-def).**  The prediction accuracy of the retrieval task:
for a distinguished source position `i_* = ⌊σ₀N⌋`,

  `P[m̂_N(t) = m_{i_*}] = E[1_{|θ_N(t) - θ_{i_*}(0)|_𝕋 ≤ π/M}]`,

the expectation being over the random ensemble of prompts.

**What the source says and what is added here.**  The decoder `m̂_N(t)` is an
`argmin`, so it is data, not a formula: it is carried as a function `dec` that
returns a nearest codeword.  The encoding `θ_i(0) = ϑ_{m_i}` of the input
tokens is carried as `henc`.  The one hypothesis with no counterpart in the
source is `hties`: on the tie set `|θ_N(t) - θ_{i_*}(0)|_𝕋 = π/M` the nearest
codeword is not unique, the indicator on the right is `1`, and an `argmin` may
legitimately return the other word — so the identity is false there, for every
decoder that resolves the tie the other way.  Measurability of the event on the
right is the remaining datum.

Source: arXiv:2605.09213v1, `eq:Acc-def`. -/
theorem acc_def (P : Measure Ω) (M N : ℕ) (hM : 0 < M) (hN : 0 < N) (t σ₀ : ℝ)
    (ϑ : Ω → ℝ → Idx N → ℝ) (msg : Idx N → ℕ) (hmsg : ∀ i, msg i < M)
    (henc : ∀ ω i, ϑ ω 0 i = codewordAngle M (msg i))
    (dec : Ω → ℕ) (hdec : ∀ ω, IsNearestCodeword M (ϑ ω t (lastIdx N hN)) (dec ω))
    (hties : ∀ᵐ ω ∂P,
      torusDist (ϑ ω t (lastIdx N hN)) (ϑ ω 0 (sourceIdx N hN σ₀)) ≠ π / M)
    (hmeas : MeasurableSet
      {ω | torusDist (ϑ ω t (lastIdx N hN)) (ϑ ω 0 (sourceIdx N hN σ₀)) ≤ π / M}) :
    (P {ω | dec ω = msg (sourceIdx N hN σ₀)}).toReal =
      ∫ ω, Set.indicator
        {ω | torusDist (ϑ ω t (lastIdx N hN)) (ϑ ω 0 (sourceIdx N hN σ₀)) ≤ π / M}
        (1 : Ω → ℝ) ω ∂P := by
  have hcongr : {ω | dec ω = msg (sourceIdx N hN σ₀)} =ᵐ[P]
      {ω | torusDist (ϑ ω t (lastIdx N hN)) (ϑ ω 0 (sourceIdx N hN σ₀)) ≤ π / M} := by
    rw [Filter.eventuallyEqSet_iff]
    filter_upwards [hties] with ω hω
    rw [henc ω (sourceIdx N hN σ₀)] at hω ⊢
    exact isNearestCodeword_iff hM (hmsg _) (hdec ω) hω
  rw [MeasureTheory.integral_indicator_one hmeas, MeasureTheory.measureReal_def,
    measure_congr hcongr]

/-- The hypotheses of `acc_def` are satisfiable, all five at once: a vocabulary
of one word, a prompt of one token sitting on its codeword `ϑ_0 = 0`, the
decoder that returns that word, no tie — the distance is `0`, the tie radius is
`π` — and the retrieval event is everything. -/
example :
    (∀ _i : Idx 1, (0 : ℕ) < 1) ∧
      (∀ (_ω : Unit) (_i : Idx 1), (0 : ℝ) = codewordAngle 1 0) ∧
      (∀ _ω : Unit, IsNearestCodeword 1 0 0) ∧
      (∀ᵐ _ω ∂(Measure.dirac ()), torusDist (0 : ℝ) 0 ≠ π / ((1 : ℕ) : ℝ)) ∧
      MeasurableSet {_ω : Unit | torusDist (0 : ℝ) 0 ≤ π / ((1 : ℕ) : ℝ)} := by
  have hzero : torusDist (0 : ℝ) 0 = 0 := by simp [torusDist]
  refine ⟨fun _ => Nat.one_pos, fun _ _ => by simp [codewordAngle],
    fun _ => ⟨Nat.one_pos, fun k hk => by interval_cases k; exact le_refl _⟩,
    Filter.Eventually.of_forall fun _ => ?_, ?_⟩
  · rw [hzero, Nat.cast_one, div_one]
    exact fun h => absurd h.symm (ne_of_gt Real.pi_pos)
  · have huniv : {_ω : Unit | torusDist (0 : ℝ) 0 ≤ π / ((1 : ℕ) : ℝ)} = Set.univ := by
      ext _
      simp [hzero, Real.pi_pos.le]
    rw [huniv]
    exact MeasurableSet.univ

end Kinetic
end Transformer

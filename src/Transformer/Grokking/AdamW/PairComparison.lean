import Transformer.Grokking.AdamW.PairEnvelope

/-!
# Two retained factor-pair masses under shared current feedback

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
circuits and appendix C's product partials; retained native pair
envelopes at lab commit f9d9eae and original feedback at 7977534.

Compare parameter mass and negative first-moment mass separately.
The efficient pair inserts gainRatio times the other pair's current
feedback. Its complete denominators have an upper ceiling; those of
the other pair have a lower floor. Choose their numerical ratio equal
to gainRatio. The cone pGen >= factor*pMem, mGen >= factor*gainRatio*mMem
is then preserved by both simultaneous recurrences.

All recurrences and denominator bounds remain explicit inputs here.
They must be generated from the actual CE/AdamW path in the application.
In particular this lemma does not assert that an arbitrary trained pair
has small variance, that Gen is already stronger, or that the parameters
converge. It keeps retained first moments rather than replacing them
by current gradients, and allows the shared feedback to vary with time.

This comparison prepares an exclusion of Gen-only boundary collapse:
under that hypothesis Gen's denominator tends to epsilon, whereas
every native denominator has an eventual floor arbitrarily near
epsilon. A positive cone factor need not give a winning logit margin.
Fixed gained tables and uniform native decay still differ from the
source's coupled-cost GD and learned floating-point GPTMini.
-/

namespace Transformer.Grokking.AdamW

/-- The current numerical retained-pair comparison cone is preserved.
Sources: appendix C partner partials and native denominator/mass laws
at f9d9eae; both first moments and both parameter bounds are retained.
The shared coefficient is current feedback, without a future limit. -/
theorem retained_pair_comparison_step
    (beta keep rate ceiling floor gainRatio factor coefficient
      pGen pMem mGen mMem pGenNext pMemNext mGenNext mMemNext : ℝ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hkeep : 0 ≤ keep) (hrate : 0 ≤ rate)
    (hceiling : 0 < ceiling) (hfloor : 0 < floor)
    (hratio : 0 ≤ gainRatio) (hfactor : 0 ≤ factor) (hcoefficient : 0 ≤ coefficient)
    (hdenominators : ceiling = gainRatio * floor)
    (hp : factor * pMem ≤ pGen) (hm : factor * gainRatio * mMem ≤ mGen)
    (hmGen : mGenNext = beta * mGen + (1 - beta) * gainRatio * coefficient * pGen)
    (hmMem : mMemNext = beta * mMem + (1 - beta) * coefficient * pMem)
    (hpGen : keep * pGen + rate * (mGenNext / ceiling) ≤ pGenNext)
    (hpMem : pMemNext ≤ keep * pMem + rate * (mMemNext / floor)) :
    factor * pMemNext ≤ pGenNext ∧ factor * gainRatio * mMemNext ≤ mGenNext := by
  have hmemory := mul_nonneg hb (show 0 ≤ mGen - factor * gainRatio * mMem by linarith only [hm])
  have hparameter := mul_nonneg
    (mul_nonneg (mul_nonneg (show 0 ≤ 1 - beta by linarith only [h1]) hratio) hcoefficient)
    (show 0 ≤ pGen - factor * pMem by linarith only [hp])
  have hmNext : factor * gainRatio * mMemNext ≤ mGenNext := by
    rw [hmGen, hmMem]
    nlinarith only [hmemory, hparameter]
  have hcancel : factor * (mMemNext / floor) * ceiling = factor * gainRatio * mMemNext := by
    rw [hdenominators]
    field_simp [ne_of_gt hfloor]
  have hquotient : factor * (mMemNext / floor) ≤ mGenNext / ceiling := by
    apply (le_div_iff₀ hceiling).mpr
    rw [hcancel]
    exact hmNext
  have hown := mul_nonneg hkeep (show 0 ≤ pGen - factor * pMem by linarith only [hp])
  have hincrement := mul_le_mul_of_nonneg_left hquotient hrate
  have hscaled := mul_le_mul_of_nonneg_left hpMem hfactor
  exact ⟨by nlinarith only [hpGen, hscaled, hown, hincrement], hmNext⟩

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 999 / 1000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) < 3 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 3 / 2 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 ∧ (3 / 2 : ℝ) = (3 / 2) * 1 ∧
    (1 / 4 : ℝ) * 1 ≤ 3 ∧ (1 / 4 : ℝ) * (3 / 2) * 1 ≤ 9 / 2 ∧
    (9 / 2 : ℝ) = (9 / 10) * (9 / 2) + (1 - 9 / 10) * (3 / 2) * 1 * 3 ∧
    (1 : ℝ) = (9 / 10) * 1 + (1 - 9 / 10) * 1 * 1 ∧
    (999 / 1000 : ℝ) * 3 + (1 / 1000) * ((9 / 2) / (3 / 2)) ≤ 3 ∧
    (1 : ℝ) ≤ (999 / 1000) * 1 + (1 / 1000) * (1 / 1) := by norm_num

/-- Positive current efficient-pair parameter and moment masses
give a positive cone factor against any nonnegative competing pair.
The factor is selected from the present numerical masses and can
be arbitrarily small. It certifies a relative lower comparison,
without claiming dominance of the efficient pair or a logit margin.
Sources: section 3's nonzero formation and retained native masses at
f9d9eae; no future comparison or winning output is supplied. -/
theorem retained_pair_comparison_initial_factor (pGen pMem mGen mMem gainRatio : ℝ)
    (hpGen : 0 < pGen) (hmGen : 0 < mGen)
    (hpMem : 0 ≤ pMem) (hmMem : 0 ≤ mMem) (hratio : 0 ≤ gainRatio) :
    ∃ factor : ℝ, 0 < factor ∧ factor * pMem ≤ pGen ∧ factor * gainRatio * mMem ≤ mGen := by
  have hdP : 0 < pMem + 1 := by linarith only [hpMem]
  have hdM : 0 < gainRatio * mMem + 1 := by
    have hh := mul_nonneg hratio hmMem
    linarith only [hh]
  let factor := min (pGen / (pMem + 1)) (mGen / (gainRatio * mMem + 1))
  have hf : 0 < factor := lt_min (div_pos hpGen hdP) (div_pos hmGen hdM)
  have hpf : factor * (pMem + 1) ≤ pGen :=
    (le_div_iff₀ hdP).mp (min_le_left _ _)
  have hmf : factor * (gainRatio * mMem + 1) ≤ mGen :=
    (le_div_iff₀ hdM).mp (min_le_right _ _)
  exact ⟨factor, hf, by nlinarith only [hpf, hf], by nlinarith only [hmf, hf]⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 9 / 2 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 3 / 2 := by norm_num

/-- A numerical comparison cone at a finite retained clock persists
on its whole subsequent tail, even with varying shared feedback.
Sources: section 3 competition, appendix C product derivatives and
retained pair laws at f9d9eae; no future success or limit is assumed. -/
theorem retained_pair_comparison_tail
    (beta keep rate ceiling floor gainRatio factor : ℝ)
    (coefficient pGen pMem mGen mMem : ℕ → ℝ) (start : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hkeep : 0 ≤ keep) (hrate : 0 ≤ rate)
    (hceiling : 0 < ceiling) (hfloor : 0 < floor)
    (hratio : 0 ≤ gainRatio) (hfactor : 0 ≤ factor)
    (hcoefficient : ∀ n, 0 ≤ coefficient n)
    (hdenominators : ceiling = gainRatio * floor)
    (hp : factor * pMem start ≤ pGen start) (hm : factor * gainRatio * mMem start ≤ mGen start)
    (hmGen : ∀ n, start ≤ n → mGen (n + 1) = beta * mGen n + (1 - beta) * gainRatio * coefficient n * pGen n)
    (hmMem : ∀ n, start ≤ n → mMem (n + 1) = beta * mMem n + (1 - beta) * coefficient n * pMem n)
    (hpGen : ∀ n, start ≤ n → keep * pGen n + rate * (mGen (n + 1) / ceiling) ≤ pGen (n + 1))
    (hpMem : ∀ n, start ≤ n → pMem (n + 1) ≤ keep * pMem n + rate * (mMem (n + 1) / floor)) :
    ∀ k, factor * pMem (start + k) ≤ pGen (start + k) ∧
      factor * gainRatio * mMem (start + k) ≤ mGen (start + k) := by
  intro k
  induction k with
  | zero => simpa only [Nat.add_zero] using And.intro hp hm
  | succ k ih =>
    have hn : start ≤ start + k := by omega
    have hh := retained_pair_comparison_step beta keep rate ceiling floor gainRatio factor (coefficient (start + k))
      (pGen (start + k)) (pMem (start + k)) (mGen (start + k)) (mMem (start + k))
      (pGen (start + k + 1)) (pMem (start + k + 1)) (mGen (start + k + 1)) (mMem (start + k + 1))
      hb h1 hkeep hrate hceiling hfloor hratio hfactor (hcoefficient _) hdenominators ih.1 ih.2
      (hmGen _ hn) (hmMem _ hn) (hpGen _ hn) (hpMem _ hn)
    simpa only [Nat.add_assoc] using hh

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 999 / 1000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) < 3 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 3 / 2 ∧ (0 : ℝ) ≤ 1 / 4 ∧
    (∀ _ : ℕ, (0 : ℝ) ≤ 1) ∧ (3 / 2 : ℝ) = (3 / 2) * 1 ∧
    (1 / 4 : ℝ) * 1 ≤ 3 ∧ (1 / 4 : ℝ) * (3 / 2) * 1 ≤ 9 / 2 ∧
    (∀ n : ℕ, 7 ≤ n → (9 / 2 : ℝ) = (9 / 10) * (9 / 2) + (1 - 9 / 10) * (3 / 2) * 1 * 3) ∧
    (∀ n : ℕ, 7 ≤ n → (1 : ℝ) = (9 / 10) * 1 + (1 - 9 / 10) * 1 * 1) ∧
    (∀ n : ℕ, 7 ≤ n → (999 / 1000 : ℝ) * 3 + (1 / 1000) * ((9 / 2) / (3 / 2)) ≤ 3) ∧
    (∀ n : ℕ, 7 ≤ n → (1 : ℝ) ≤ (999 / 1000) * 1 + (1 / 1000) * (1 / 1)) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, fun _ => by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ _ => by norm_num, fun _ _ => by norm_num, fun _ _ => by norm_num, fun _ _ => by norm_num⟩

end Transformer.Grokking.AdamW

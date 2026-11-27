module

public import Transformer.Grokking.FiniteRecovery.ArgandPhase
public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Transformer.Grokking.FiniteRecovery.CompletionStability
public import Transformer.Grokking.FiniteRecovery.Examples

/-! At finite weights the softmax phase is a strictly weighted average of
the codes. Approximate nuclear completion gives a close phase expectation, which
forces a strict probability margin and thus a strict logit margin. -/

@[expose] public section

open Filter
open scoped Topology ComplexConjugate

namespace Transformer.Grokking.FiniteRecovery

variable {A : Type*} [Fintype A]

private theorem exists_phase_gap (code : A → ℂ) (hinj : Function.Injective code) :
    ∃ gap > 0, ∀ a b, b ≠ a → gap ≤ ‖code b - code a‖ ^ 2 := by
  have hlim := tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ)
  have h : ∀ᶠ n : ℕ in atTop, ∀ a b, b ≠ a → 1 / (n : ℝ) ≤ ‖code b - code a‖ ^ 2 := by
    apply eventually_all.mpr
    intro a
    apply eventually_all.mpr
    intro b
    by_cases hba : b = a
    · exact Eventually.of_forall fun _ hn => (hn hba).elim
    have hpos := pow_pos (norm_pos_iff.mpr (sub_ne_zero.mpr (fun he => hba (hinj he)))) 2
    exact (hlim.eventually (gt_mem_nhds hpos)).mono fun _ hn _ => hn.le
  obtain ⟨n, hn, hn'⟩ := (h.and (eventually_ge_atTop 1)).exists
  have hpos : 0 < n := by omega
  exact ⟨1 / (n : ℝ), one_div_pos.mpr (by exact_mod_cast hpos), hn⟩

/-- A small error of the average phase controls the actual softmax
classifier. Distinct unit phase codes have a positive separation. -/
theorem exists_phase_error_strict_probability (code : A → ℂ) (hinj : Function.Injective code)
    (hunit : ∀ a, ‖code a‖ = 1) :
    ∃ ε > 0, ∀ (p : A → ℝ) (a : A), (∀ b, 0 ≤ p b) → (∑ b, p b) = 1 →
      ‖phaseExpectation code p - code a‖ < ε → ∀ b ≠ a, p b < p a := by
  classical
  obtain ⟨gap, hgap, hsep⟩ := exists_phase_gap code hinj
  refine ⟨gap / 4, by positivity, fun p a hp hm herr b hba => ?_⟩
  have hn (t : A) : Complex.normSq (code t) = 1 := by
    rw [Complex.normSq_eq_norm_sq, hunit, one_pow]
  have hsqa (t : A) : ‖code t - code a‖ ^ 2 = 2 * (1 - (code t * conj (code a)).re) := by
    rw [Complex.sq_norm, Complex.normSq_sub, hn, hn]
    ring
  let z := phaseExpectation code p
  have hsum :
      (∑ t, p t * ‖code t - code a‖ ^ 2) = 2 * ((code a - z) * conj (code a)).re := by
    calc
      _ = ∑ t, (2 * p t - 2 * (((p t : ℂ) * code t) * conj (code a)).re) := by
        apply Finset.sum_congr rfl
        intro t _
        rw [hsqa, mul_assoc]
        simp only [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero,
          Complex.mul_im]
        ring
      _ = _ := by
        rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hm]
        simp only [sub_mul, Complex.sub_re, Complex.mul_conj, hn, Complex.ofReal_one, Complex.one_re]
        rw [show z * conj (code a) = ∑ t, ((p t : ℂ) * code t) * conj (code a) by
          simp only [z, phaseExpectation, Finset.sum_mul], Complex.re_sum]
        ring
  have hupper : (∑ t, p t * ‖code t - code a‖ ^ 2) ≤ 2 * ‖z - code a‖ := by
    rw [hsum]
    have h := Complex.re_le_norm ((code a - z) * conj (code a))
    rw [norm_mul, Complex.norm_conj, hunit, mul_one, norm_sub_rev] at h
    linarith
  have hmass := Finset.sum_erase_add Finset.univ p (Finset.mem_univ a)
  rw [hm] at hmass
  have hlower : gap * (1 - p a) ≤ ∑ t, p t * ‖code t - code a‖ ^ 2 := by
    have hz (t : A) (ht : t ∈ Finset.univ.erase a) :
        gap * p t ≤ p t * ‖code t - code a‖ ^ 2 := by
      rw [mul_comm gap]
      exact mul_le_mul_of_nonneg_left (hsep a t (Finset.ne_of_mem_erase ht)) (hp t)
    have hl := Finset.sum_le_sum hz
    rw [← Finset.mul_sum, show ∑ t ∈ Finset.univ.erase a, p t = 1 - p a by linarith] at hl
    simpa only [sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0), mul_zero, add_zero] using
      hl.trans_eq (by
        have h := Finset.sum_erase_add Finset.univ (fun t => p t * ‖code t - code a‖ ^ 2)
          (Finset.mem_univ a)
        simp at h ⊢)
  by_contra! hab
  have hb := Finset.single_le_sum (f := p) (fun t _ => hp t)
    (Finset.mem_erase.mpr ⟨hba, Finset.mem_univ b⟩)
  have ha : 1 / 2 ≤ 1 - p a := by linarith
  have hh := mul_le_mul_of_nonneg_left ha hgap.le
  change ‖z - code a‖ < gap / 4 at herr
  linarith only [hlower, hupper, hh, herr]

namespace Argand

open FlatSpectrum

variable {U V : Type*} {T H d layers : ℕ} [Fintype U] [Fintype V] [DecidableEq V]
  [DecidableEq A] [Nonempty A]

/-- A finite, positive training loss and a finite nuclear norm slack are
enough to certify strict logits on the entire word domain, including the
hidden cells. The norm condition concerns the Argand phase table itself. -/
theorem exists_observedLoss_nuclear_slack_for_realizes (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hinj : Function.Injective code)
    (hunit : ∀ a, ‖code a‖ = 1) (target : Word A (T + 1) → A)
    (inputs : Finset (Word A (T + 1)))
    (hunique : NuclearRecovers (observedCells cut inputs) (targetKernel cut code target)) :
    ∃ ε > 0, ∀ model : Model A H d layers,
      model.observedLoss repeats (observe inputs target) ≤ ε →
      nuclearNorm (model.phaseKernel repeats cut code) ≤ nuclearNorm (targetKernel cut code target) + ε →
      model.realizes repeats target := by
  obtain ⟨ε, hε, hp⟩ := exists_phase_error_strict_probability code hinj hunit
  obtain ⟨δ, hδ, hstable⟩ := hunique.exists_error_threshold hε
  let part := observe inputs target
  let den := 2 * (part.card : ℝ) + 1
  have hden : 0 < den := by dsimp [den]; positivity
  have hden1 : 1 ≤ den := by dsimp [den]; linarith [Nat.cast_nonneg (α := ℝ) part.card]
  refine ⟨δ / den, div_pos hδ hden, fun model hfit hnorm => ?_⟩
  have hc : 2 * (part.card : ℝ) * model.observedLoss repeats part ≤ δ := by
    have h := (le_div_iff₀ hden).mp hfit
    have hnon : 0 ≤ model.observedLoss repeats part := by
      unfold Model.observedLoss
      exact div_nonneg (Finset.sum_nonneg fun p _ => model.labelLoss_nonneg repeats p.1 p.2)
        (Nat.cast_nonneg _)
    change model.observedLoss repeats part * (2 * (part.card : ℝ) + 1) ≤ δ at h
    linarith only [h, hnon]
  have hobs (c : U × V) (hm : c ∈ observedCells cut inputs) :
      ‖model.phaseKernel repeats cut code c.1 c.2 - targetKernel cut code target c.1 c.2‖ ≤ δ := by
    have hmem : (cut c, target (cut c)) ∈ part := mem_observe.mpr ⟨hm, rfl⟩
    have hl := model.labelLoss_le_card_mul_observedLoss repeats part hmem
    exact (model.phaseError_le_labelLoss repeats cut code hunit target c.1 c.2).trans
      ((mul_le_mul_of_nonneg_left hl (by norm_num)).trans (by linarith only [hc]))
  have hslack : δ / den ≤ δ := (div_le_iff₀ hden).mpr (by nlinarith [hden1, hδ.le])
  have hclose := hstable _ hobs (hnorm.trans (add_le_add (le_refl _) hslack))
  intro word b hb
  obtain ⟨⟨u,v⟩, rfl⟩ := cut.surjective word
  let w := cut (u, v)
  let p := model.probability repeats w (Fin.last T)
  have hprob := hp p (target w) (fun a => (model.probability_pos repeats w (Fin.last T) a).le)
    (model.sum_probability repeats w (Fin.last T)) (hclose u v) b hb
  unfold p Model.probability at hprob
  have hs : 0 < ∑ a, Real.exp (model.logits repeats w (Fin.last T) a) :=
    Finset.sum_pos' (fun a _ => (Real.exp_pos _).le) ⟨b, Finset.mem_univ b, Real.exp_pos _⟩
  exact Real.exp_lt_exp.mp ((div_lt_div_iff_of_pos_right hs).mp hprob)

/-- The certificate applies to the three-cell mask for the two-variable
binary modular sum, with a genuine hidden fourth cell. -/
example :=
  exists_observedLoss_nuclear_slack_for_realizes (H := 4) (d := 16) (layers := 1) (T := 1)
    1 (enumeratedCut (wordEnumeration 2 1)) ZMod.stdAddChar ZMod.injective_stdAddChar
    (fun _ => by simp) (modularSum (wordEnumeration 2 1))
    (inputsAtCells (enumeratedCut (wordEnumeration 2 1)) (staircase 1))
    (by simpa only [observedCells_inputsAtCells] using modularSum_nuclear_recovers (wordEnumeration 2 1))

end Argand

end Transformer.Grokking.FiniteRecovery

module

public import Transformer.Grokking.FiniteRecovery.FlatSpectrum

/-! Nearest-phase decoding turns convergence of fitted entries into
simultaneous exact forecasts on the finite word domain. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Filter
open scoped Topology

variable {A : Type*} [Finite A] [Nonempty A]

theorem exists_nearestPhase (code : A → ℂ) (z : ℂ) :
    ∃ a, ∀ b, ‖z - code a‖ ≤ ‖z - code b‖ :=
  Set.exists_lower_bound_image Set.univ (fun a => ‖z - code a‖) (Set.toFinite _)
    |>.imp fun _ ha b => ha b (Set.mem_univ _)

/-- Decode to a nearest codeword, with a fixed selection at ties. -/
noncomputable def phaseDecode (code : A → ℂ) (z : ℂ) : A :=
  Classical.choose (exists_nearestPhase code z)

/-- An injective finite code has a positive distance to every competing token. -/
theorem phaseDecode_eventually_of_tendsto (code : A → ℂ) (hinj : Function.Injective code)
    (a : A) {values : ℕ → ℂ} (hlimit : Tendsto values atTop (𝓝 (code a))) :
    ∀ᶠ step in atTop, phaseDecode code (values step) = a := by
  have hstrict : ∀ᶠ step in atTop, ∀ b ≠ a, ‖values step - code a‖ < ‖values step - code b‖ := by
    apply eventually_all.mpr
    intro b
    by_cases hb : b = a
    · exact Eventually.of_forall fun _ hne => (hne hb).elim
    · have ha : Tendsto (fun k => ‖values k - code a‖) atTop (𝓝 0) := by
        simpa only [sub_self, norm_zero] using (hlimit.sub_const (code a)).norm
      have hpos : 0 < ‖code a - code b‖ :=
        norm_pos_iff.mpr (sub_ne_zero.mpr fun he => hb (hinj he).symm)
      exact (ha.eventually_lt (hlimit.sub_const (code b)).norm hpos).mono fun _ h _ => h
  filter_upwards [hstrict] with step hs
  by_contra hb
  exact (not_le_of_gt (hs _ hb)) (Classical.choose_spec (exists_nearestPhase code (values step)) a)

/-- One finite training index suffices for exact recovery on every word. -/
theorem phaseDecodeKernel_eventually {U V : Type*} {T : ℕ} (cut : U × V ≃ Word A T)
    (code : A → ℂ) (hinj : Function.Injective code) (target : Word A T → A) (P : ℕ → Matrix U V ℂ)
    (hlim : Tendsto P atTop (𝓝 (targetKernel cut code target))) :
    ∀ᶠ step in atTop, (fun word => phaseDecode code (P step (cut.symm word).1 (cut.symm word).2)) =
      target := by
  have hw : ∀ᶠ step in atTop, ∀ word : Word A T,
      phaseDecode code (P step (cut.symm word).1 (cut.symm word).2) = target word := by
    apply eventually_all.mpr
    intro word
    apply phaseDecode_eventually_of_tendsto code hinj
    have hc := tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hlim (cut.symm word).1) (cut.symm word).2
    simpa [targetKernel] using hc
  exact hw.mono fun _ h => funext h

example (a : ZMod 3) :=
  phaseDecode_eventually_of_tendsto ZMod.stdAddChar ZMod.injective_stdAddChar a tendsto_const_nhds

example (z : ℂ) := exists_nearestPhase (A := ZMod 3) ZMod.stdAddChar z

example (target : Word (ZMod 3) 4 → ZMod 3) :=
  phaseDecodeKernel_eventually (appendCut (ZMod 3) 2 2) ZMod.stdAddChar ZMod.injective_stdAddChar
    target (fun _ => targetKernel (appendCut (ZMod 3) 2 2) ZMod.stdAddChar target) tendsto_const_nhds

end Transformer.Grokking.FiniteRecovery

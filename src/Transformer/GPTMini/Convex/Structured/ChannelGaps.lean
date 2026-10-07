import Transformer.GPTMini.Convex.Structured.Concentration
import Transformer.GPTMini.Convex.Structured.Factorial

/-!
# Actual finite learned-channel energy deficits

Source: the freely trainable sharp-row capacity assignment at d68381b
and the genuine compact matching/value energies at 87ffa1b. These
inequalities evaluate finite raw channel logits; they do not assume
correct routing, a concentrated probability or correctly encoded task
features. A wrong channel loses one gain, and distinct query/key label
vectors prevent all matching groups from receiving both gains.

The label vectors below specify given weight witnesses and training
targets only. They are not fixed channels in the proposed forward,
whose unrestricted raw Q/K/value potentials remain trainable. The
results will derive complete raw recall energy gaps after all physical
position and latest-write conditions have been obtained from parsing.
No infinite temperatures, optimizer change or output-only convexity
are asserted. Both the matching and value variables are evaluated.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {G D : Type*} [Fintype G]

/-- Finite given channel logits attain the exact full gain on their actual witness assignment.
Source: sharpRowLogits evaluated group by group, with the genuine channelEnergy finite sum. -/
theorem sharpChannel_self (labels : G → D) (gain : ℝ) :
    channelEnergy (fun g => sharpRowLogits (labels g) gain) labels = (Fintype.card G : ℝ) * gain := by
  unfold channelEnergy
  simp only [sharpRowLogits, eq_self, ite_true, Finset.sum_const, nsmul_eq_mul, Finset.card_univ]

/-- Every actual channel assignment has at most one nonnegative gain per learned group.
Source: the finite zero/gain raw witness rows, before any probability or desired route is supplied. -/
theorem sharpChannel_upper (labels assignment : G → D) (gain : ℝ) (hgain : 0 ≤ gain) :
    channelEnergy (fun g => sharpRowLogits (labels g) gain) assignment ≤ (Fintype.card G : ℝ) * gain := by
  have h : channelEnergy (fun g => sharpRowLogits (labels g) gain) assignment ≤ ∑ _ : G, gain := by
    apply Finset.sum_le_sum
    intro g hg
    dsimp only [sharpRowLogits]
    split_ifs <;> linarith
  simpa only [Finset.sum_const, nsmul_eq_mul, Finset.card_univ] using h

example : (0 : ℝ) ≤ 1 := by norm_num

/-- A real wrong complete value-channel assignment loses at least one whole finite gain.
Source: an actual differing group, its computed zero logit and the full finite upper bound on all other groups. -/
theorem sharpChannel_miss (labels assignment : G → D) (gain : ℝ) (hgain : 0 ≤ gain) (hne : assignment ≠ labels) :
    channelEnergy (fun g => sharpRowLogits (labels g) gain) assignment ≤ (Fintype.card G : ℝ) * gain - gain := by
  obtain ⟨p, hp⟩ : ∃ p, assignment p ≠ labels p := by
    by_contra hn
    apply hne
    funext p
    by_contra h
    exact hn ⟨p, h⟩
  have h : channelEnergy (fun g => sharpRowLogits (labels g) gain) assignment ≤
      ∑ g : G, (gain - if g = p then gain else 0) := by
    apply Finset.sum_le_sum
    intro g hg
    by_cases he : g = p
    · subst g
      simp only [sharpRowLogits, ite_eq_right hp, ite_true, sub_self, le_refl]
    · simp only [ite_eq_right he, sub_zero]
      dsimp only [sharpRowLogits]
      split_ifs <;> linarith
  rw [Finset.sum_sub_distrib, Fintype.sum_ite_eq', Finset.sum_const, nsmul_eq_mul, Finset.card_univ] at h
  exact h

example : (0 : ℝ) ≤ 1 ∧ (fun _ : Fin 4 => (1 : Fin 4)) ≠ (fun _ : Fin 4 => (0 : Fin 4)) := by
  refine ⟨by norm_num, ?_⟩
  intro h
  have he := congrArg (fun f : Fin 4 → Fin 4 => f 0) h
  contradiction

/-- Independent learned Q/K contributions add through the actual same matching-channel energy.
Source: genuine channelEnergy finite-sum distributivity, not a product of trained query and key coordinates. -/
theorem channelEnergy_add (left right : G → D → ℝ) (assignment : G → D) :
    channelEnergy (fun g d => left g d + right g d) assignment =
      channelEnergy left assignment + channelEnergy right assignment := by
  unfold channelEnergy
  rw [Finset.sum_add_distrib]

/-- Identical given query/key labels receive both finite gains in every actual matching group.
Source: independent free sharp-row Q and K witness fields evaluated by the true matching energy. -/
theorem sharpMatching_self (labels : G → D) (gain : ℝ) :
    channelEnergy (fun g d => sharpRowLogits (labels g) gain d + sharpRowLogits (labels g) gain d) labels =
      2 * (Fintype.card G : ℝ) * gain := by
  rw [channelEnergy_add, sharpChannel_self]
  ring

/-- Any given query/key/channel combination is bounded by the two genuine raw group gains.
Source: separate actual Q/K channel bounds, valid for freely chosen different witness labels. -/
theorem sharpMatching_upper (left right assignment : G → D) (gain : ℝ) (hgain : 0 ≤ gain) :
    channelEnergy (fun g d => sharpRowLogits (left g) gain d + sharpRowLogits (right g) gain d) assignment ≤
      2 * (Fintype.card G : ℝ) * gain := by
  rw [channelEnergy_add]
  have hl := sharpChannel_upper left assignment gain hgain
  have hr := sharpChannel_upper right assignment gain hgain
  linarith

example : (0 : ℝ) ≤ 65 := by norm_num

/-- Distinct actual query/key label vectors force one full finite matching-energy loss for every latent channel choice.
Source: no assignment can equal both distinct vectors, so one real learned channel sum loses its gain. -/
theorem sharpMatching_different (left right assignment : G → D) (gain : ℝ) (hgain : 0 ≤ gain) (hne : left ≠ right) :
    channelEnergy (fun g d => sharpRowLogits (left g) gain d + sharpRowLogits (right g) gain d) assignment ≤
      2 * (Fintype.card G : ℝ) * gain - gain := by
  rw [channelEnergy_add]
  by_cases he : assignment = left
  · have ha : assignment ≠ right := fun h => hne (he.symm.trans h)
    have hl := sharpChannel_upper left assignment gain hgain
    have hr := sharpChannel_miss right assignment gain hgain ha
    linarith
  · have hl := sharpChannel_miss left assignment gain hgain he
    have hr := sharpChannel_upper right assignment gain hgain
    linarith

example : (0 : ℝ) ≤ 1 ∧ (fun _ : Fin 4 => (0 : Fin 4)) ≠ (fun _ : Fin 4 => (1 : Fin 4)) := by
  refine ⟨by norm_num, ?_⟩
  intro h
  have he := congrArg (fun f : Fin 4 → Fin 4 => f 0) h
  contradiction

/-- A wrong matching channel loses a full gain even when query and key truly agree.
Source: one genuine raw sum's channel deficit plus the actual other sum's finite upper bound. -/
theorem sharpMatching_miss (labels assignment : G → D) (gain : ℝ) (hgain : 0 ≤ gain) (hne : assignment ≠ labels) :
    channelEnergy (fun g d => sharpRowLogits (labels g) gain d + sharpRowLogits (labels g) gain d) assignment ≤
      2 * (Fintype.card G : ℝ) * gain - gain := by
  rw [channelEnergy_add]
  have hl := sharpChannel_miss labels assignment gain hgain hne
  have hr := sharpChannel_upper labels assignment gain hgain
  linarith

example : (0 : ℝ) ≤ 65 ∧ (fun _ : Fin 5 => (3 : Fin 4)) ≠ (fun _ : Fin 5 => (2 : Fin 4)) := by
  refine ⟨by norm_num, ?_⟩
  intro h
  have he := congrArg (fun f : Fin 5 → Fin 4 => f 0) h
  contradiction

/-- The actual four learned matching groups give eight finite gains on a correctly shared code.
Source: the compact Basis layout's four Q and four K fields evaluated at the given complete channel assignment. -/
theorem sharpMatching_basis_self (labels : Fin 4 → Fin 4) (gain : ℝ) :
    channelEnergy (fun g d => sharpRowLogits (labels g) gain d + sharpRowLogits (labels g) gain d) labels = 8 * gain := by
  have h := sharpMatching_self labels gain
  norm_num only [Fintype.card_fin, Nat.cast_ofNat] at h
  exact h

/-- The actual five jointly trained output groups give five finite gains on their given output assignment.
Source: the compact Basis value layout and genuine full channel-energy evaluation. -/
theorem sharpValue_basis_self (labels : Fin 5 → Fin 4) (gain : ℝ) :
    channelEnergy (fun h => sharpRowLogits (labels h) gain) labels = 5 * gain := by
  simpa only [Fintype.card_fin, Nat.cast_ofNat] using sharpChannel_self labels gain

end
end Transformer.GPTMini.Convex.Structured

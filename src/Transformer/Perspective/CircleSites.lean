module

public import Transformer.Perspective.Section6_Circle

/-!
# Sorted positions of a configuration on the circle

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3.

The proof of `thm:main` reads a configuration `x ∈ 𝕋^n` through its distinct
positions `0 ≤ θ_1 < ⋯ < θ_K < 2π`, continued by `θ_{K+1} = 2π + θ_1`.
Counted from `0`, they are

* `circlePos a`: the point of `𝕋 = ℝ/2πℤ` that the angle `a` stands for, read
  in `[0, 2π)`;
* `circleSites x`: the distinct positions of `x`, a `Finset` of `[0, 2π)`;
* `circleSite x m`: the positions in increasing order, continued periodically
  by `circleSite x (m + K) = circleSite x m + 2π`.

The gaps between consecutive positions are in `Perspective.CircleGaps`.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- The point of `𝕋 = ℝ/2πℤ` that the angle `a` stands for, read in `[0, 2π)`. -/
noncomputable def circlePos (a : ℝ) : ℝ := toIcoMod Real.two_pi_pos 0 a

theorem circlePos_mem_Ico (a : ℝ) : circlePos a ∈ Set.Ico 0 (2 * π) := by
  simpa [circlePos] using toIcoMod_mem_Ico Real.two_pi_pos 0 a

/-- A `2π`-periodic function does not tell two angles from their points on `𝕋`. -/
theorem periodic_circlePos_sub {g : ℝ → ℝ} (hg : Function.Periodic g (2 * π)) (a b : ℝ) :
    g (circlePos a - circlePos b) = g (a - b) := by
  have ha := self_sub_toIcoMod Real.two_pi_pos 0 a
  have hb := self_sub_toIcoMod Real.two_pi_pos 0 b
  have hab : a - b = (circlePos a - circlePos b) +
      (toIcoDiv Real.two_pi_pos 0 a - toIcoDiv Real.two_pi_pos 0 b) • (2 * π) := by
    rw [sub_smul, ← ha, ← hb]
    simp only [circlePos]
    ring
  rw [hab, hg.zsmul _ _]

/-- The distinct positions of the configuration `x` on `𝕋`, in `[0, 2π)`. -/
noncomputable def circleSites (x : Angles n) : Finset ℝ :=
  Finset.univ.image fun i => circlePos (x i)

theorem mem_Ico_of_mem_circleSites {x : Angles n} {s : ℝ} (hs : s ∈ circleSites x) :
    s ∈ Set.Ico 0 (2 * π) := by
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hs
  exact circlePos_mem_Ico _

/-- **The positions in increasing order**, `θ_0 < θ_1 < ⋯ < θ_{K-1}` where
`K = #(circleSites x)`, continued periodically: `θ_{m+K} = θ_m + 2π`.

These are the source's `0 ≤ θ_1 < ⋯ < θ_K < 2π` and `θ_{K+1} = 2π + θ_1`,
counted from `0`.

Source: arXiv:2507.22857v1, §3, before `lemma:twoconsecutive_general`. -/
noncomputable def circleSite (x : Angles n) (m : ℕ) : ℝ :=
  if h : 0 < (circleSites x).card then
    (circleSites x).orderEmbOfFin rfl ⟨m % (circleSites x).card, Nat.mod_lt m h⟩ +
      2 * π * ((m / (circleSites x).card : ℕ) : ℝ)
  else 0

theorem circleSite_of_lt {x : Angles n} {m : ℕ} (hm : m < (circleSites x).card) :
    circleSite x m = (circleSites x).orderEmbOfFin rfl ⟨m, hm⟩ := by
  have h : 0 < (circleSites x).card := lt_of_le_of_lt (Nat.zero_le m) hm
  simp [circleSite, h, Nat.mod_eq_of_lt hm, Nat.div_eq_of_lt hm]

theorem circleSite_mem {x : Angles n} {m : ℕ} (hm : m < (circleSites x).card) :
    circleSite x m ∈ circleSites x := by
  rw [circleSite_of_lt hm]
  exact Finset.orderEmbOfFin_mem _ _ _

theorem circleSite_add_card (x : Angles n) (h : 0 < (circleSites x).card) (m : ℕ) :
    circleSite x (m + (circleSites x).card) = circleSite x m + 2 * π := by
  simp only [circleSite, h, ↓reduceDIte, Nat.add_mod_right, Nat.add_div_right m h]
  push_cast
  ring

theorem strictMono_circleSite (x : Angles n) (h : 0 < (circleSites x).card) :
    StrictMono (circleSite x) := by
  intro m m' hm
  have he : ∀ r, (circleSites x).orderEmbOfFin rfl r ∈ Set.Ico 0 (2 * π) := fun r =>
    mem_Ico_of_mem_circleSites (Finset.orderEmbOfFin_mem _ _ r)
  simp only [circleSite, h, ↓reduceDIte]
  rcases (Nat.div_le_div_right hm.le : m / (circleSites x).card ≤ m' / _).lt_or_eq with hq | hq
  · have hq' : ((m / (circleSites x).card : ℕ) : ℝ) + 1 ≤
        ((m' / (circleSites x).card : ℕ) : ℝ) := by
      exact_mod_cast hq
    have h1 := he ⟨m % (circleSites x).card, Nat.mod_lt m h⟩
    have h2 := he ⟨m' % (circleSites x).card, Nat.mod_lt m' h⟩
    nlinarith [Real.pi_pos, h1.1, h1.2, h2.1, h2.2]
  · rw [hq]
    have hr : m % (circleSites x).card < m' % (circleSites x).card := by
      have h1 := Nat.div_add_mod m (circleSites x).card
      have h2 := Nat.div_add_mod m' (circleSites x).card
      rw [hq] at h1
      omega
    have := ((circleSites x).orderEmbOfFin rfl).strictMono
      (show (⟨_, Nat.mod_lt m h⟩ : Fin (circleSites x).card) < ⟨_, Nat.mod_lt m' h⟩ from hr)
    linarith

/-- Every particle sits at one of the sorted positions. -/
theorem exists_circleSite_eq (x : Angles n) (i : Idx n) :
    ∃ a < (circleSites x).card, circleSite x a = circlePos (x i) := by
  have hmem : circlePos (x i) ∈ Set.range ((circleSites x).orderEmbOfFin rfl) := by
    rw [Finset.range_orderEmbOfFin]
    exact Finset.mem_image_of_mem _ (Finset.mem_univ i)
  obtain ⟨r, hr⟩ := hmem
  exact ⟨r, r.2, by rw [circleSite_of_lt r.2]; exact hr⟩

/-- Every sorted position is occupied by some particle. -/
theorem exists_circlePos_eq (x : Angles n) {m : ℕ} (hm : m < (circleSites x).card) :
    ∃ i, circlePos (x i) = circleSite x m := by
  obtain ⟨i, -, hi⟩ := Finset.mem_image.mp (circleSite_mem hm)
  exact ⟨i, hi⟩


/-- The hypotheses above are satisfiable: `cos` is `2π`-periodic, and a single
particle has one position, so `K = 1`, `m = 0` and `s = circlePos 0`. -/
example : Function.Periodic Real.cos (2 * π) ∧
    circlePos 0 ∈ circleSites (fun _ : Idx 1 => (0 : ℝ)) ∧
    0 < (circleSites (fun _ : Idx 1 => (0 : ℝ))).card :=
  ⟨Real.cos_periodic, Finset.mem_image_of_mem _ (Finset.mem_univ 0),
    Finset.card_pos.mpr ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ 0)⟩⟩

end Perspective
end Transformer

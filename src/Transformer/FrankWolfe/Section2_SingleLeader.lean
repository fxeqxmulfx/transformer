/-
# Attention's forward pass and Frank-Wolfe — almost-sure single leaders

A proof of arXiv:2508.09628v1, §2.1, lem:singleLeader that allows token
mergers. It uses the finite linear branches and their null sets of ties,
rather than requiring an invertible update.
-/

import Transformer.FrankWolfe.Section2_Merger

open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n : ℕ}

/-- A trajectory of eq:hardmax.dynamics.V. The average over the leader set
defines the update even before uniqueness is known.
Source: arXiv:2508.09628v1, §1.2, eq:hardmax.dynamics.V. -/
def AverageFlow (P B : ℕ → ParamMatrix d) (x : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℕ) (i : Idx n), x (t + 1) i = hardmaxAverageStep (P t) (B t) (x t) i

/-- **Lemma (lem:singleLeader).** If every Bᵗ is invertible, then for almost
every initial configuration the leader set consists of exactly one point,
at every time and every token.

The source argues that a piecewise affine step cannot map a set of positive
measure onto a null set. This is false: at V = I₁, P = I₁/2 and B = -I₁,
two tokens x₁ > 0 > x₂ lead each other and merge in one step on the diagonal
(hardmaxAverageStep_maps_open_set_to_null).
The lemma counts points rather than indices and survives such mergers.

Instead, every fixed sequence of leader selections gives a linear branch
fixing consensus tuples. Translating all initial tokens leaves their image
differences unchanged. Every tie between distinct images is null by Fubini
and the hyperplane argument in Section2_LeaderNull. A countable intersection
over time and finite intersections over branches give a common full-measure
set. Induction then represents every actual trajectory by those branches.

The proof works for every linear value map P, and in particular for the
source's Pᵗ = (I + Vᵗ)⁻¹Vᵗ whenever I + Vᵗ is invertible.

Source: arXiv:2508.09628v1, §2.1, lem:singleLeader. -/
theorem singleLeader (P B : ℕ → ParamMatrix d)
    (hB : ∀ t : ℕ, Function.Bijective (B t)) :
    ∀ᵐ X₀ : Idx n → EucSpace d,
      ∀ x : ℕ → Idx n → EucSpace d, x 0 = X₀ → AverageFlow P B x →
        ∀ (t : ℕ) (i : Idx n), (leaderSet (B t) (x t) i).card = 1 := by
  classical
  have hbranches (t : ℕ) : ∀ᵐ X : Idx n → EucSpace d,
      ∀ L ∈ leaderBranches P t, ∀ i j k : Idx n, L X j ≠ L X k →
        inner (𝕜 := ℝ) (B t (L X i)) (L X j) ≠
          inner (𝕜 := ℝ) (B t (L X i)) (L X k) := by
    have hsub : ∀ᵐ X : Idx n → EucSpace d,
        ∀ L : leaderBranches P t, ∀ i j k : Idx n, L.val X j ≠ L.val X k →
          inner (𝕜 := ℝ) (B t (L.val X i)) (L.val X j) ≠
            inner (𝕜 := ℝ) (B t (L.val X i)) (L.val X k) :=
      ae_all_iff.mpr fun L => ae_all_iff.mpr fun i =>
        ae_all_iff.mpr fun j => ae_all_iff.mpr fun k =>
          ae_linearBranch_scores_ne L.val (leaderBranches_consensus P t L.val L.property)
            (B t) (hB t).2 i j k
    filter_upwards [hsub] with X hX
    intro L hL
    exact hX ⟨L, hL⟩
  filter_upwards [ae_all_iff.mpr hbranches] with X₀ hX₀
  intro x hx₀ hflow
  have hrep : ∀ t : ℕ, ∃ L ∈ leaderBranches P t, x t = L X₀ := by
    intro t
    induction t with
    | zero => exact ⟨LinearMap.id, by simp [leaderBranches], hx₀⟩
    | succ t ih =>
        obtain ⟨L, hL, hxt⟩ := ih
        have hunique (i : Idx n) : (leaderSet (B t) (x t) i).card = 1 := by
          rw [hxt]
          exact leaderSet_card_eq_one_of_scores_ne (B t) (L X₀) i (hX₀ t L hL i)
        have hsel : ∀ i : Idx n, ∃ j : Idx n, leaderSet (B t) (x t) i = {x t j} := by
          intro i
          obtain ⟨a, ha⟩ := Finset.card_eq_one.mp (hunique i)
          have hamem : a ∈ leaderSet (B t) (x t) i := by rw [ha]; simp
          obtain ⟨j, _, hj⟩ := Finset.mem_image.mp (Finset.mem_filter.mp hamem).1
          exact ⟨j, by rwa [← hj] at ha⟩
        choose σ hσ using hsel
        refine ⟨(leaderStepLinearMap (P t) σ).comp L,
          mem_leaderBranches_succ P t L hL σ, ?_⟩
        funext i
        rw [hflow t i, hardmaxAverageStep, hσ i]
        simp only [Finset.card_singleton, Nat.cast_one, inv_one, Finset.sum_singleton, one_smul]
        rw [hxt]
        rfl
  intro t i
  obtain ⟨L, hL, hxt⟩ := hrep t
  rw [hxt]
  exact leaderSet_card_eq_one_of_scores_ne (B t) (L X₀) i (hX₀ t L hL i)

/-- The hypotheses and the trajectory condition are satisfiable: the
identity key and zero value map keep two unit tokens fixed.
Source: arXiv:2508.09628v1, §2.1, lem:singleLeader. -/
example :
    (∀ t : ℕ, Function.Bijective
      ((fun _ : ℕ => ContinuousLinearMap.id ℝ (EucSpace 1)) t)) ∧
    AverageFlow (fun _ => (0 : ParamMatrix 1))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ => fun _ : Idx 2 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) := by
  refine ⟨fun _ => Function.bijective_id, ?_⟩
  intro t i
  simp [hardmaxAverageStep]

end Transformer.FrankWolfe

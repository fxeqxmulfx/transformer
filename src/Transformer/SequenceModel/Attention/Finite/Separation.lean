module

public import Transformer.SequenceModel.Attention.Expressivity.Normalization
public import Transformer.Interpolation.Section4_GenericQuadrantPoints
public import Mathlib.Topology.Baire.Lemmas

/-!
# Finite codes admit bounded separating linear measurements

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F1.
The coefficients can be chosen inside any nonempty open ball. This matters
for normalized Q/K attention: positional scores have a bounded range, so a
construction must not assume arbitrarily concentrated attention weights.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini
open scoped RealInnerProductSpace BigOperators

variable {A : Type*} [Fintype A] {d : ℕ}

/-- F1: finitely many nonzero linear forms can all be nonzero at one point
of any open ball. The proper-subspace density lemma is from the repository's
formalization of arXiv:2411.04551v3, §3.2. -/
theorem exists_nonzero_forms (L : A → EucSpace d →L[ℝ] ℝ)
    (hL : ∀ a, ∃ x, L a x ≠ 0) (center : EucSpace d) {r : ℝ} (hr : 0 < r) :
    ∃ w ∈ Metric.ball center r, ∀ a, L a w ≠ 0 := by
  let P := fun a => (L a).toLinearMap.ker
  have hp (a : A) : P a ≠ ⊤ := by
    intro ht
    obtain ⟨x, hx⟩ := hL a
    have hm : x ∈ P a := by rw [ht]; trivial
    exact hx (LinearMap.mem_ker.mp hm)
  have hd : Dense (⋂ a, (P a : Set (EucSpace d))ᶜ) :=
    dense_iInter_of_isOpen
      (fun a => (P a).closed_of_finiteDimensional.isOpen_compl)
      (fun a => Interpolation.dense_compl_proper_subspace (P a) (hp a))
  obtain ⟨w, hw, hpw⟩ := hd.inter_open_nonempty (Metric.ball center r)
    Metric.isOpen_ball (Metric.nonempty_ball.mpr hr)
  refine ⟨w, hw, ?_⟩
  intro a ha
  exact (Set.mem_iInter.mp hpw a) (LinearMap.mem_ker.mpr ha)

/-- F1: a single bounded linear measurement separates any finite collection
of distinct vectors. Neither the vectors nor their order is changed. -/
theorem exists_separating_inner (z : A → EucSpace d) (hz : Function.Injective z)
    (center : EucSpace d) {r : ℝ} (hr : 0 < r) :
    ∃ w ∈ Metric.ball center r, Function.Injective (fun a => ⟪w, z a⟫) := by
  classical
  let B := {ab : A × A // ab.1 ≠ ab.2}
  let L : B → EucSpace d →L[ℝ] ℝ := fun ab => innerSL ℝ (z ab.val.1 - z ab.val.2)
  have hL (ab : B) : ∃ x, L ab x ≠ 0 := by
    refine ⟨z ab.val.1 - z ab.val.2, ?_⟩
    intro h
    have he : z ab.val.1 - z ab.val.2 = 0 := inner_self_eq_zero.mp h
    exact ab.property (hz (sub_eq_zero.mp he))
  obtain ⟨w, hw, hn⟩ := exists_nonzero_forms L hL center hr
  refine ⟨w, hw, ?_⟩
  intro a b hab
  by_contra hne
  apply hn ⟨(a, b), hne⟩
  change ⟪z a - z b, w⟫ = 0
  rw [inner_sub_left, real_inner_comm w (z a), real_inner_comm w (z b)]
  exact sub_eq_zero.mpr hab

/-- Zero/one coordinate code of a subset of positions. -/
def bits (N : ℕ) (b : Fin N → Bool) : EucSpace N :=
  WithLp.toLp 2 (fun j => if b j then 1 else 0)

/-- F1: the zero/one representation retains every positional membership bit. -/
theorem bits_injective (N : ℕ) : Function.Injective (bits N) := by
  intro a b hab
  funext j
  have h := congrArg (fun x : EucSpace N => x j) hab
  change (if a j then (1 : ℝ) else 0) = (if b j then 1 else 0) at h
  cases ha : a j <;> cases hb : b j <;> simp_all

/-- F1: there are strictly positive, arbitrarily near-uniform coefficients
that distinguish every subset of a finite set of positions. This is a
finite exact-arithmetic claim, not a lower bound on numerical separation. -/
theorem exists_bounded_subset_weights (N : ℕ) {lo hi : ℝ} (h : lo < hi) :
    ∃ w : EucSpace N, (∀ j, lo < w j ∧ w j < hi) ∧
      Function.Injective (fun b : Fin N → Bool => ⟪w, bits N b⟫) := by
  let center : EucSpace N := WithLp.toLp 2 (fun _ => (lo + hi) / 2)
  obtain ⟨w, hw, hinj⟩ := exists_separating_inner (bits N) (bits_injective N)
    center (r := (hi - lo) / 2) (by linarith)
  refine ⟨w, ?_, hinj⟩
  intro j
  have hn : ‖w - center‖ < (hi - lo) / 2 := by simpa only [Metric.mem_ball, dist_eq_norm] using hw
  have hc := lt_of_le_of_lt (PiLp.norm_apply_le (w - center) j) hn
  change ‖w j - (lo + hi) / 2‖ < (hi - lo) / 2 at hc
  rw [Real.norm_eq_abs, abs_lt] at hc
  constructor <;> linarith [hc.1, hc.2]

example := exists_nonzero_forms (fun (_ : Fin 1) => innerSL ℝ (WithLp.toLp 2 ![1, 0]))
  (by intro a; refine ⟨WithLp.toLp 2 ![1, 0], ?_⟩; norm_num [PiLp.inner_apply, Fin.sum_univ_two])
  (WithLp.toLp 2 ![1, 1]) (r := 1 / 100) (by norm_num)
example := exists_separating_inner (fun a : Fin 3 => WithLp.toLp 2 ![(a.val : ℝ), 1])
  (by
    intro a b h
    have hc : (a.val : ℝ) = b.val := congrArg (fun x : EucSpace 2 => x 0) h
    apply Fin.ext
    exact_mod_cast hc)
  (WithLp.toLp 2 ![1, 1]) (r := 1 / 100) (by norm_num)
example := exists_bounded_subset_weights 7 (lo := 1) (hi := 1001 / 1000) (by norm_num)

end Transformer.SequenceModel.Attention.Finite

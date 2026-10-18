/-
# Saddle avoidance — sequences

Facts on real numbers and on sequences, with no gradient in them, that the
convergence of gradient descent and of descent curves rests on.

* **Drops of a power.**  For `0 ≤ a ≤ 1` the concavity of `s ↦ s ^ (1 - a)`, a
  weighted AM-GM inequality, bounds a drop of `s` by `s ^ a` times the drop of
  `s ^ (1 - a)` (`mul_sub_le_rpow_mul_sub`).
* **Steps bounded by drops.**  They add up (`mul_norm_sub_le_of_step`); a sequence
  that starts close to `z`, and whose steps from inside `B(z, r)` are so bounded,
  never leaves `B(z, r)` (`mem_ball_of_step`).
* **Values near a level.**  The values of a decreasing sequence that comes close to
  a value `c` infinitely often stay above `c` and converge to it
  (`le_and_tendsto_of_frequently`).  Along a sequence with a limit point `z` up to
  translations leaving `f` invariant, `f` comes close to `f z` infinitely often
  where it is continuous at `z` (`frequently_abs_sub_lt`), and a bound on `f`
  along the sequence holds at `z` (`le_of_frequently_norm_add_sub_lt`).
* **Limit points up to translations.**  One exists when every term can be
  translated into one compact set (`exists_frequently_norm_add_sub_lt`), as on a
  torus; if the sequence converges to `w` and the translations form a closed set
  `Λ`, then `z - w ∈ Λ` (`sub_mem_of_tendsto_of_frequently`).
-/

import Mathlib.Analysis.MeanInequalities
import Mathlib.Topology.Sequences

open Filter Metric
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

/-- Concavity of `s ↦ s ^ (1 - a)` as the Łojasiewicz argument uses it:
`(1 - a)(s - t) ≤ s ^ a (s ^ (1 - a) - t ^ (1 - a))`, the weighted AM-GM
inequality `s ^ a t ^ (1 - a) ≤ a s + (1 - a) t`. -/
theorem mul_sub_le_rpow_mul_sub {a s t : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hs : 0 ≤ s)
    (ht : 0 ≤ t) : (1 - a) * (s - t) ≤ s ^ a * (s ^ (1 - a) - t ^ (1 - a)) := by
  have hamgm := Real.geom_mean_le_arith_mean2_weighted ha0 (sub_nonneg.2 ha1) hs ht
    (by ring)
  have hss : s ^ a * s ^ (1 - a) = s := by
    rw [← Real.rpow_add' hs (by norm_num), add_sub_cancel, Real.rpow_one]
  have hlin : (1 - a) * (s - t) = s - (a * s + (1 - a) * t) := by ring
  rw [hlin, mul_sub, hss]
  linarith

variable {E : Type*} [NormedAddCommGroup E]

/-- **Steps bounded by drops add up.**  If `κ ‖u (k + 1) - u k‖ ≤
φ k - φ (k + 1)` for every `k`, then `κ ‖u k - u n‖ ≤ φ n - φ k` for
`n ≤ k`. -/
theorem mul_norm_sub_le_of_step {u : ℕ → E} {φ : ℕ → ℝ} {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ k, κ * ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1)) {n k : ℕ} (hnk : n ≤ k) :
    κ * ‖u k - u n‖ ≤ φ n - φ k := by
  induction k, hnk using Nat.le_induction with
  | base => simp
  | succ k _ ih =>
    have htri := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u k) (u n)) hκ
    linarith [h k]

/-- **A trap.**  If every step of `u` from inside `B(z, r)` is bounded by the
drop of a nonnegative sequence `φ`, `κ ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1)`
with `κ > 0`, and `κ ‖u 0 - z‖ + φ 0 < κ r`, then `u` never leaves
`B(z, r)`. -/
theorem mem_ball_of_step {u : ℕ → E} {φ : ℕ → ℝ} {κ r : ℝ} {z : E} (hκ : 0 < κ)
    (hφ : ∀ k, 0 ≤ φ k) (h : ∀ k, u k ∈ ball z r → κ * ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1))
    (h0 : κ * ‖u 0 - z‖ + φ 0 < κ * r) (k : ℕ) : u k ∈ ball z r := by
  suffices H : ∀ k, u k ∈ ball z r ∧ κ * ‖u k - u 0‖ ≤ φ 0 - φ k from (H k).1
  intro k
  induction k with
  | zero =>
    refine ⟨?_, by simp⟩
    rw [mem_ball, dist_eq_norm]
    exact lt_of_mul_lt_mul_left (by linarith [hφ 0]) hκ.le
  | succ k ih =>
    have htri := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u k) (u 0)) hκ.le
    have hlen : κ * ‖u (k + 1) - u 0‖ ≤ φ 0 - φ (k + 1) := by linarith [h k ih.1, ih.2]
    have hz := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u 0) z) hκ.le
    refine ⟨?_, hlen⟩
    rw [mem_ball, dist_eq_norm]
    exact lt_of_mul_lt_mul_left (by linarith [hφ (k + 1)]) hκ.le

/-- **The values of a decreasing sequence near a value it comes close to
infinitely often.**  They stay above it and converge to it.  The index may be
discrete, as for gradient descent, or continuous, as for gradient flows. -/
theorem le_and_tendsto_of_frequently {ι : Type*} [SemilatticeSup ι] [Nonempty ι] {u : ι → ℝ}
    (hu : Antitone u) {c : ℝ}
    (hc : ∀ ε > 0, ∃ᶠ k in atTop, |u k - c| < ε) :
    (∀ k, c ≤ u k) ∧ Tendsto u atTop (𝓝 c) := by
  have hle : ∀ k, c ≤ u k := by
    intro m
    by_contra hm
    push Not at hm
    obtain ⟨k, hk, hkm⟩ := ((hc (c - u m) (by linarith)).and_eventually
      (eventually_ge_atTop m)).exists
    have := hu hkm
    rw [abs_lt] at hk
    linarith [hk.1]
  refine ⟨hle, Metric.tendsto_atTop.2 fun ε hε => ?_⟩
  obtain ⟨N, hN⟩ := (hc ε hε).exists
  refine ⟨N, fun n hn => ?_⟩
  rw [Real.dist_eq, abs_lt] at *
  constructor <;> linarith [hle n, hu hn]

/-- If infinitely many points of a sequence lie near `z` up to translations
in `Λ`, and `f` is continuous at `z` and invariant under `Λ`, then infinitely
many values of `f` along the sequence lie near `f z`. -/
theorem frequently_abs_sub_lt {f : E → ℝ} {z : E} (hcont : ContinuousAt f z) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {ι : Type*} [Preorder ι] {u : ι → E}
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε) :
    ∀ ε > 0, ∃ᶠ k in atTop, |f (u k) - f z| < ε := by
  intro ε hε
  obtain ⟨δ, hδ, hball⟩ := Metric.continuousAt_iff.1 hcont ε hε
  refine (hcl δ hδ).mono fun k ⟨v, hv, hk⟩ => ?_
  have h := hball (x := u k + v) (by rwa [dist_eq_norm])
  rwa [hΛ v hv, Real.dist_eq] at h

/-- **A limit point up to translations.**  If every term of a sequence can
be translated by a vector of `Λ` into a compact set `K`, some point of `K` is
approached by infinitely many terms up to such translations. -/
theorem exists_frequently_norm_add_sub_lt {K : Set E} (hK : IsCompact K) {Λ : Set E}
    {u : ℕ → E} (hu : ∀ k, ∃ v ∈ Λ, u k + v ∈ K) :
    ∃ z ∈ K, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε := by
  choose v hv hvK using hu
  obtain ⟨z, hzK, φ, hφ, hlim⟩ := hK.tendsto_subseq hvK
  refine ⟨z, hzK, fun ε hε => frequently_atTop.2 fun a => ?_⟩
  obtain ⟨j, hja, hj⟩ := ((hφ.tendsto_atTop.eventually (eventually_ge_atTop a)).and
    (Metric.tendsto_nhds.1 hlim ε hε)).exists
  exact ⟨φ j, hja, v (φ j), hv _, by rwa [← dist_eq_norm]⟩

/-- **A limit point up to translations differs from the limit by a
translation**: if `u` converges to `w` and comes arbitrarily close to `z` up
to translations in a closed set `Λ`, then `z - w ∈ Λ`. -/
theorem sub_mem_of_tendsto_of_frequently {Λ : Set E} (hΛ : IsClosed Λ) {u : ℕ → E}
    {w z : E} (hw : Tendsto u atTop (𝓝 w))
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε) : z - w ∈ Λ := by
  rw [← hΛ.closure_eq, Metric.mem_closure_iff]
  intro ε hε
  obtain ⟨k, ⟨v, hv, hk⟩, hk'⟩ := ((hz (ε / 2) (half_pos hε)).and_eventually
    (Metric.tendsto_nhds.1 hw (ε / 2) (half_pos hε))).exists
  refine ⟨v, hv, ?_⟩
  rw [dist_eq_norm] at hk' ⊢
  calc ‖z - w - v‖ = ‖(u k - w) - (u k + v - z)‖ := by
        congr 1
        abel
    _ ≤ ‖u k - w‖ + ‖u k + v - z‖ := norm_sub_le _ _
    _ < ε / 2 + ε / 2 := add_lt_add hk' hk
    _ = ε := add_halves ε

/-- **A bound on `f` along a sequence holds at its limit points up to
translations** leaving `f` invariant, where `f` is continuous: if
`f (u i) ≤ b` eventually along `l` and, for every `ε > 0`, frequently some
`u i + v` with `v ∈ Λ` lies within `ε` of `z`, then `f z ≤ b`. -/
theorem le_of_frequently_norm_add_sub_lt {f : E → ℝ} {ι : Type*} {l : Filter ι} {u : ι → E}
    {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {z : E} (hc : ContinuousAt f z)
    (hz : ∀ ε > 0, ∃ᶠ i in l, ∃ v ∈ Λ, ‖u i + v - z‖ < ε) {b : ℝ}
    (hb : ∀ᶠ i in l, f (u i) ≤ b) : f z ≤ b := by
  refine le_of_forall_pos_lt_add fun δ hδ => ?_
  obtain ⟨ε, hε, hball⟩ := Metric.continuousAt_iff.1 hc δ hδ
  obtain ⟨i, ⟨v, hv, hi⟩, hbi⟩ := ((hz ε hε).and_eventually hb).exists
  have h := hball (show dist (u i + v) z < ε by rwa [dist_eq_norm])
  rw [Real.dist_eq, hΛ v hv] at h
  linarith [(abs_lt.1 h).1]

/-- `(1 - 1/2)(1 - 0) ≤ 1 ^ (1/2) (1 ^ (1/2) - 0 ^ (1/2))`, with `a = 1/2`. -/
example := mul_sub_le_rpow_mul_sub (a := 1 / 2) (s := 1) (t := 0) (by norm_num) (by norm_num)
  zero_le_one le_rfl

/-- The constant sequence `0`, `φ = 0` and `κ = 1`: the steps add up and stay in `B(0, 1)`. -/
example := mul_norm_sub_le_of_step (u := fun _ => (0 : ℝ)) (φ := fun _ => 0) zero_le_one
  (fun _ => by simp) (Nat.zero_le 1)

example := mem_ball_of_step (u := fun _ => (0 : ℝ)) (φ := fun _ => 0) (z := 0) (r := 1)
  one_pos (fun _ => le_rfl) (fun _ _ => by simp) (by norm_num) 0

/-- The constant sequence `1` decreases and comes close to `1` infinitely often. -/
example := le_and_tendsto_of_frequently (ι := ℕ) (u := fun _ => (1 : ℝ)) antitone_const
  (c := 1) fun ε hε => Frequently.of_forall fun _ => by simpa using hε

/-- The constant sequence `0` in `ℝ` comes close to `0` up to the translations `{0}`,
which form a closed set leaving the identity invariant, and lies in `[-1, 1]`. -/
example := frequently_abs_sub_lt (f := fun x : ℝ => x) (z := 0) continuousAt_id (Λ := {0})
  (by simp) (ι := ℕ) (u := fun _ => 0) fun ε hε =>
    Frequently.of_forall fun _ => ⟨0, rfl, by simpa using hε⟩

example := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : ℝ) 1) (Λ := {0})
  (u := fun _ => 0) fun _ => ⟨0, rfl, by simp⟩

example := sub_mem_of_tendsto_of_frequently isClosed_singleton (u := fun _ => (0 : ℝ)) (w := 0)
  (z := 0) tendsto_const_nhds fun ε hε => Frequently.of_forall fun _ => ⟨0, rfl, by simpa using hε⟩

example := le_of_frequently_norm_add_sub_lt (f := fun x : ℝ => x) (l := atTop)
  (u := fun _ : ℕ => 0) (Λ := {0}) (by simp) (z := 0) continuousAt_id
  (fun ε hε => Frequently.of_forall fun _ => ⟨0, rfl, by simpa using hε⟩) (b := 0)
  (Eventually.of_forall fun _ => le_rfl)

end Transformer.Landscape.SaddleAvoidance

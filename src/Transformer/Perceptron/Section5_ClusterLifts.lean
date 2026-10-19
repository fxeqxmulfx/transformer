module

public import Transformer.Perceptron.Section5_KernelClusterBounds
public import Mathlib.Data.Finset.Max
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Coherent real representatives of a short circular cluster

ArXiv:2601.21366v2, §5.4, `eq:dist.cond`. The distance hypothesis
allows a different integer turn for each pair of atoms. To compare
the actual force at two extreme cluster points, we need one choice
of representative for each atom, all lying in a single interval of
the same stated width. This must also cover a cluster crossing zero.

Choose every representative within the cluster width of one fixed
atom. Their pairwise real differences are at most twice that width.
If the width is at most one half, the original circular-distance
condition forces these differences to be at most the original width:
any additional nonzero turn would have absolute size at least `2π`.
The distinct original representatives in `[0,2π)` remain distinct.
The finite set therefore has two distinct extreme representatives.
-/

@[expose] public section

open Real

namespace Transformer.Perceptron

/-- **Two representatives in `[0,2π)` cannot differ by a nonzero turn.**
Source: arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`, and
§5.4, `eq:dist.cond`. This preserves the source's distinct atoms. -/
theorem angle_eq_of_int_period (u v : ℝ) (hu : u ∈ Set.Ico 0 (2 * π))
    (hv : v ∈ Set.Ico 0 (2 * π)) (k : ℤ)
    (heq : u + 2 * π * (k : ℝ) = v) : u = v := by
  by_cases hk : k = 0
  · simpa [hk] using heq
  rcases le_total k 0 with hneg | hpos
  · have hki : k ≤ -1 := by omega
    have hkr : (k : ℝ) ≤ -1 := by exact_mod_cast hki
    nlinarith [hu.2, hv.1, Real.pi_pos]
  · have hki : 1 ≤ k := by omega
    have hkr : (1 : ℝ) ≤ k := by exact_mod_cast hki
    nlinarith [hu.1, hv.2, Real.pi_pos]

/-- **A short real difference cannot use an additional nonzero turn.**
Source: arXiv:2601.21366v2, §5.4, `eq:dist.cond`.
The two-width preliminary bound and the circular one-width bound
identify the same real difference when `δ ≤ 1/2`. -/
theorem abs_le_of_short_periodic_shift (d δ : ℝ) (k : ℤ)
    (hδ : δ ≤ 1 / 2) (hd : |d| ≤ 2 * δ)
    (hk : |d + 2 * π * (k : ℝ)| ≤ δ) : |d| ≤ δ := by
  by_cases hk0 : k = 0
  · simpa [hk0] using hk
  have hki : (1 : ℝ) ≤ |(k : ℝ)| := by
    rcases le_total k 0 with hneg | hpos
    · have hkr : (k : ℝ) ≤ -1 := by exact_mod_cast (show k ≤ -1 by omega)
      linarith [neg_le_abs (k : ℝ)]
    · have hkr : (1 : ℝ) ≤ k := by exact_mod_cast (show 1 ≤ k by omega)
      linarith [le_abs_self (k : ℝ)]
  have hturn : 2 * π ≤ |2 * π * (k : ℝ)| := by
    rw [abs_mul, abs_of_pos (show 0 < 2 * π by positivity)]
    nlinarith [Real.pi_pos]
  have htriangle : |2 * π * (k : ℝ)| ≤ |d + 2 * π * (k : ℝ)| + |d| := by
    have h := abs_sub_le (d + 2 * π * (k : ℝ)) 0 d
    simpa only [show d + 2 * π * (k : ℝ) - d = 2 * π * (k : ℝ) by ring,
      sub_zero, zero_sub, abs_neg] using h
  nlinarith [Real.pi_gt_three]

/-- **The source's pairwise circular condition gives one short real interval.**
Source: arXiv:2601.21366v2, §5.4, `eq:dist.cond`.
Integer turns are chosen coherently for each atom; the two interval
endpoints are distinct atoms from the cluster itself. The original
cluster width is retained exactly, including across angle zero. -/
theorem exists_short_cluster_lifts {ι : Type*} (S : Finset ι) (θ : ι → ℝ)
    (hmem : ∀ i, θ i ∈ Set.Ico 0 (2 * π)) (hinj : Function.Injective θ)
    (δ : ℝ) (hδ : δ ≤ 1 / 2) (hcard : 2 ≤ S.card)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ, |θ i - θ j + 2 * π * (k : ℝ)| ≤ δ) :
    ∃ k : ι → ℤ, ∃ lo ∈ S, ∃ hi ∈ S,
      let α := fun i => θ i + 2 * π * (k i : ℝ)
      α lo < α hi ∧ α hi - α lo ≤ δ ∧ ∀ i ∈ S, α lo ≤ α i ∧ α i ≤ α hi := by
  classical
  have hne : S.Nonempty := Finset.card_pos.mp (by omega)
  obtain ⟨b, hb⟩ := hne
  have hne : S.Nonempty := ⟨b, hb⟩
  have hchoose : ∀ i, ∃ k : ℤ, i ∈ S → |θ i - θ b + 2 * π * (k : ℝ)| ≤ δ := by
    intro i
    by_cases hi : i ∈ S
    · obtain ⟨k, hk⟩ := hpair i hi b hb
      exact ⟨k, fun _ => hk⟩
    · exact ⟨0, fun hi' => (hi hi').elim⟩
  choose k hk using hchoose
  let α := fun i => θ i + 2 * π * (k i : ℝ)
  have hnear (i : ι) (hi : i ∈ S) : |α i - θ b| ≤ δ := by
    change |θ i + 2 * π * (k i : ℝ) - θ b| ≤ δ
    rw [show θ i + 2 * π * (k i : ℝ) - θ b = θ i - θ b + 2 * π * (k i : ℝ) by ring]
    exact hk i hi
  have hwidth (i : ι) (hi : i ∈ S) (j : ι) (hj : j ∈ S) : |α i - α j| ≤ δ := by
    have htri := abs_sub_le (α i) (θ b) (α j)
    rw [abs_sub_comm (θ b) (α j)] at htri
    have htwo : |α i - α j| ≤ 2 * δ := by linarith [hnear i hi, hnear j hj]
    obtain ⟨p, hp⟩ := hpair i hi j hj
    have hshift : |α i - α j + 2 * π * ((p - k i + k j : ℤ) : ℝ)| ≤ δ := by
      dsimp [α]
      rw [Int.cast_add, Int.cast_sub]
      convert hp using 1
      congr 1
      ring
    exact abs_le_of_short_periodic_shift (α i - α j) δ _ hδ htwo hshift
  have hαinj (i j : ι) (heq : α i = α j) : i = j := by
    apply hinj
    apply angle_eq_of_int_period (θ i) (θ j) (hmem i) (hmem j) (k i - k j)
    rw [Int.cast_sub]
    dsimp [α] at heq
    linarith
  obtain ⟨lo, hlo, hmin⟩ := S.exists_min_image α hne
  obtain ⟨hi, hhi, hmax⟩ := S.exists_max_image α hne
  have hstrict : α lo < α hi := by
    by_contra hnot
    have heq : α lo = α hi := le_antisymm (hmin hi hhi) (le_of_not_gt hnot)
    have hsame : ∀ i ∈ S, ∀ j ∈ S, i = j := by
      intro i hi' j hj
      apply hαinj
      linarith [hmin i hi', hmax i hi', hmin j hj, hmax j hj]
    have hone : S.card ≤ 1 := Finset.card_le_one.mpr hsame
    omega
  exact ⟨k, lo, hlo, hi, hhi, hstrict, (abs_le.mp (hwidth hi hhi lo hlo)).2,
    fun i hi' => ⟨hmin i hi', hmax i hi'⟩⟩

/-- The representative and periodic-shift inputs are realized at
the actual angle zero and positive width `1/4`, with zero extra turn. -/
example : (0 : ℝ) ∈ Set.Ico 0 (2 * π) ∧
    (0 : ℝ) + 2 * π * (0 : ℤ) = 0 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (1 : ℝ) / 4 ≤ 1 / 2 ∧
    |(0 : ℝ)| ≤ 2 * (1 / 4) ∧ |(0 : ℝ) + 2 * π * (0 : ℤ)| ≤ 1 / 4 := by
  exact ⟨⟨le_rfl, by positivity⟩, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

/-- All coherent-lift hypotheses are jointly realized by two distinct
actual angles `0,1/4`, the full two-element cluster, and width `1/4`.
This is the source's cluster scale at `β=4`. -/
example : let θ := fun i : Idx 2 => ((i : ℕ) : ℝ) / 4
    (∀ i, θ i ∈ Set.Ico 0 (2 * π)) ∧ Function.Injective θ ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (1 : ℝ) / 4 ≤ 1 / 2 ∧
    2 ≤ (Finset.univ : Finset (Idx 2)).card ∧
    (∀ i ∈ (Finset.univ : Finset (Idx 2)), ∀ j ∈ Finset.univ, ∃ k : ℤ,
      |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / 4) := by
  intro θ
  have hbounds (i : Idx 2) : 0 ≤ θ i ∧ θ i ≤ 1 / 4 := by
    have hi : ((i : ℕ) : ℝ) ≤ 1 := by exact_mod_cast i.is_le
    constructor
    · positivity
    · dsimp [θ]; linarith
  refine ⟨fun i => ⟨(hbounds i).1, by linarith [(hbounds i).2, Real.pi_gt_three]⟩,
    ?_, by norm_num, by norm_num, by norm_num, ?_⟩
  · intro i j hij
    apply Fin.ext
    exact_mod_cast (show ((i : ℕ) : ℝ) = ((j : ℕ) : ℝ) by dsimp [θ] at hij; linarith)
  · intro i hi j hj
    refine ⟨0, ?_⟩
    simp only [Int.cast_zero, mul_zero, add_zero]
    exact abs_sub_le_of_nonneg_of_le (hbounds i).1 (hbounds i).2 (hbounds j).1 (hbounds j).2

end Transformer.Perceptron

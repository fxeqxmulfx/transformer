module

public import Transformer.Causal.Section4_TwoTokenPositiveLimit
public import Transformer.Causal.Section4_FollowerDrift

/-!
# The actual last-token leading drift for arbitrary prefix length

ArXiv:2411.04990v2, §4, `thm2`, under its original spectral assumptions.
The Rayleigh ceiling makes the original self contribution nonnegative
in the leading positive hemisphere. Every original predecessor force
has its leading consensus value up to its actual value-matrix error.
The positive predecessor mass and vanishing original prefix errors
give a lower bound on the actual overlap derivative, for any number
of original predecessors and arbitrary original query and key matrices.
-/

@[expose] public section

open Real Filter
open scoped Topology BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual last leading overlap has positive consensus drift minus
the full original prefix error in the nonnegative hemisphere.
Source: arXiv:2411.04990v2, §4, `thm2`, using the original overlap
argument of `thm1`; the self weight remains in the partition. -/
theorem csaLastLeadingOverlap_lower (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 ≤ lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (Y : SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (hr : 0 ≤ inner ℝ (u : EucSpace d) (Y (Fin.last n) : EucSpace d)) :
    lam * csaWeightFloor β Q K (n + 1) *
        (1 - inner ℝ (u : EucSpace d) (Y (Fin.last n) : EucSpace d) ^ 2) -
      ‖V‖ * ∑ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖ ≤
        inner ℝ (u : EucSpace d)
          (csaAmbientField β Q K V (fun j => (Y j : EucSpace d)) (Fin.last n)) := by
  let q := 1 - inner ℝ (u : EucSpace d) (Y (Fin.last n) : EucSpace d) ^ 2
  let a : Idx (n + 1) → ℝ :=
    csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) (Fin.last n)
  have hq : 0 ≤ q := by
    have h := csaSphereOverlap_bounds u (Y (Fin.last n))
    dsimp [q]
    nlinarith
  have hmass : csaWeightFloor β Q K (n + 1) ≤ ∑ j : Idx n, a j.castSucc := by
    have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have h := csaPredecessorWeight_lower β Q K (fun j => (Y j : EucSpace d))
      (fun j => mem_sphere_zero_iff_norm.mp (Y j).2)
    have hp := csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)
    dsimp [a]
    nlinarith
  have hs : lam * csaWeightFloor β Q K (n + 1) * q -
      ‖V‖ * ∑ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖ ≤
        ∑ j : Idx n, a j.castSucc * inner ℝ (u : EucSpace d)
          (proj d (Y (Fin.last n) : EucSpace d) (V (Y j.castSucc : EucSpace d))) := by
    calc
      _ ≤ (∑ j : Idx n, a j.castSucc) * (lam * q) -
          ‖V‖ * ∑ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖ := by
        have h := mul_le_mul_of_nonneg_right hmass (mul_nonneg hlam hq)
        nlinarith
      _ = ∑ j : Idx n, (a j.castSucc * (lam * q) -
          ‖V‖ * ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖) := by
        rw [Finset.sum_sub_distrib, Finset.sum_mul, Finset.mul_sum]
      _ ≤ _ := by
        apply Finset.sum_le_sum
        intro j _
        have h0 := csaTokenWeight_nonneg β Q K
          (fun k => (Y k : EucSpace d)) (Fin.last n) j.castSucc
        have h1 := csaTokenWeight_le_one β Q K
          (fun k => (Y k : EucSpace d)) (Fin.last n) j.castSucc
        have hi := (abs_le.mp (leadingOverlapInteraction_approx V lam u (Y (Fin.last n))
          (Y j.castSucc) (heig _ hu))).1
        have he : 0 ≤ ‖V‖ * ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖ := by
          positivity
        have hf : lam * q - ‖V‖ * ‖(Y j.castSucc : EucSpace d) - (u : EucSpace d)‖ ≤
            inner ℝ (u : EucSpace d)
              (proj d (Y (Fin.last n) : EucSpace d) (V (Y j.castSucc : EucSpace d))) := by
          dsimp [q]
          linarith
        have hm := mul_le_mul_of_nonneg_left hf h0
        have herr := mul_le_mul_of_nonneg_right h1 he
        change a j.castSucc * (lam * q) - _ ≤ a j.castSucc * _
        change 0 ≤ a j.castSucc at h0
        change a j.castSucc ≤ 1 at h1
        change a j.castSucc * _ ≤ _ at herr
        change a j.castSucc * _ ≤ a j.castSucc * _ at hm
        nlinarith
  have hself : 0 ≤ inner ℝ (u : EucSpace d)
      (proj d (Y (Fin.last n) : EucSpace d) (V (Y (Fin.last n) : EucSpace d))) := by
    have hceil :=
      rayleigh_le_leading V L lam heig hinv hdecay (Y (Fin.last n) : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp (Y (Fin.last n)).2, one_pow, mul_one] at hceil
    rw [proj, inner_sub_right, real_inner_smul_right,
      inner_value_of_mem_leading V L lam heig hinv _ hu,
      real_inner_comm (V (Y (Fin.last n) : EucSpace d)) (Y (Fin.last n) : EucSpace d)]
    nlinarith
  have hself' := mul_nonneg
    (csaTokenWeight_nonneg β Q K (fun k => (Y k : EucSpace d)) (Fin.last n) (Fin.last n)) hself
  rw [csaAmbientField_eq_sum, Fin.sum_univ_castSucc, inner_add_right, inner_sum,
    real_inner_smul_right]
  simp only [real_inner_smul_right]
  exact hs.trans (le_add_of_nonneg_right hself')

/-- The exact actual last-token field is the derivative of its original
leading overlap. Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
theorem hasDerivAt_csaLast_fieldOverlap (β : ℝ) (Q K V : ParamMatrix d)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (u : EucSpace d) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ u (X s (Fin.last n) : EucSpace d))
      (inner ℝ u (csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) (Fin.last n))) t := by
  simpa! only [inner_zero_left, add_zero] using
    (hasDerivAt_const t u).inner ℝ ((CSA_iff_csaAmbientField β Q K V X).mp hX t (Fin.last n))

/-- The original prefix error vanishes when each original predecessor
has the asserted ambient limit. Source: arXiv:2411.04990v2, §4, the
converging-prefix argument of `thm1` applied to `thm2`. -/
theorem tendsto_csaLeadingPrefixError (X : ℝ → SphereTuple d (n + 1)) (u : EucSpace d)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop (nhds u)) :
    Tendsto (fun t => ∑ j : Idx n, ‖(X t j.castSucc : EucSpace d) - u‖) atTop (nhds 0) := by
  have hj (j : Idx n) : Tendsto
      (fun t => ‖(X t j.castSucc : EucSpace d) - u‖) atTop (nhds 0) := by
    simpa only [sub_self, norm_zero] using ((hp j).sub_const u).norm
  simpa only [Finset.sum_const_zero] using tendsto_finsetSum Finset.univ (fun j _ => hj j)

/-- Actual consensus realizes the original spectral, nonnegative-overlap,
flow and prefix-limit hypotheses. Source: arXiv:2411.04990v2, §4, `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
      inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    0 ≤ inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) ∧
    (∀ j : Idx 1, Tendsto (fun _ : ℝ =>
      ((fun _ : Idx 2 => basePoint 1) j.castSucc : EucSpace 2)) atTop
        (nhds (basePoint 1 : EucSpace 2))) := by
  refine ⟨one_pos, le_rfl, fun v _ => by simp, fun z hz => hz, ?_,
    CSA_const_consensus β Q K (basePoint 1), ?_, fun _ => tendsto_const_nhds⟩
  · intro z hz hz0
    exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim
  · rw [real_inner_self_eq_norm_sq]
    positivity

end Transformer.Causal

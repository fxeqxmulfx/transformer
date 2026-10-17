/-
# The bound on second differences of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3, `eq:est-Glaub-2nd`:
the mixed second difference of the flow in the initial angles of tokens `l` and
`m` is at most `2e^{2|λ|} L₂ |h||h'| e^{2L₁t}` times Duhamel's supersolution
`crossProfile` (`abs_flowMap_second_diff_le`).
-/

import Transformer.Kinetic.Section3_FieldSecondDifference

open Real

namespace Transformer
namespace Kinetic

/-- **Second differences (`eq:est-Glaub-2nd`).** Moving token `l` by `h` and token
`m` by `h'`, the mixed second difference of token `a` at time `t ∈ [0, T]` is at
most `2e^{2|λ|} L₂ |h||h'| e^{2L₁t} V_a(t)`, with `V = crossProfile`. -/
theorem abs_flowMap_second_diff_le (lam β : ℝ) (N : ℕ) {L₁ L₂ α γ T : ℝ}
    (hd : ∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hL₂ : ∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (x : Idx N → ℝ) (l m : Idx N) (h h' : ℝ) (a : Idx N) {t : ℝ} (ht : t ∈ Set.Icc 0 T) :
    |flowMap lam β N t (x + Pi.single l h + Pi.single m h') a -
        flowMap lam β N t (x + Pi.single l h) a - flowMap lam β N t (x + Pi.single m h') a +
        flowMap lam β N t x a| ≤
      2 * Real.exp (2 * |lam|) * L₂ * |h| * |h'| *
        (Real.exp (2 * L₁ * t) * crossProfile α γ T N l m a t) := by
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hL₂0 : 0 ≤ L₂ := by simpa using (abs_nonneg _).trans (hL₂ 0 1)
  set κ := 2 * Real.exp (2 * |lam|) * L₂ * |h| * |h'| with hκ
  have hκ0 : 0 ≤ κ := by positivity
  set Φ := flowMap lam β N
  set y : Idx N → ℝ → ℝ := fun a t => Φ t (x + Pi.single l h + Pi.single m h') a -
    Φ t (x + Pi.single l h) a - Φ t (x + Pi.single m h') a + Φ t x a with hy
  set V : ℕ → ℝ → ℝ := fun a t => crossProfile α γ T N l m a t with hV
  have hdy : ∀ a t, HasDerivAt (y a)
      (gptField lam β N (Φ t (x + Pi.single l h + Pi.single m h')) a -
        gptField lam β N (Φ t (x + Pi.single l h)) a -
        gptField lam β N (Φ t (x + Pi.single m h')) a + gptField lam β N (Φ t x) a) t :=
    fun a t => (((hasDerivAt_flowMap lam β N _ t a).sub (hasDerivAt_flowMap lam β N _ t a)).sub
      (hasDerivAt_flowMap lam β N _ t a)).add (hasDerivAt_flowMap lam β N x t a)
  have hdW : ∀ a : Idx N, ∀ t, HasDerivAt (fun t => κ * (Real.exp (2 * L₁ * t) * V a t))
      (κ * (Real.exp (2 * L₁ * t) * (2 * L₁) * V a t + Real.exp (2 * L₁ * t) *
        (∑ c ∈ Finset.range N, crossForcing α γ T l m c * profile α γ c a t +
          t * ∑ c ∈ Finset.range N, crossForcing α γ T l m c * profileDeriv α γ c a t))) t :=
    fun a t => by
      have he : HasDerivAt (fun t => Real.exp (2 * L₁ * t)) (Real.exp (2 * L₁ * t) * (2 * L₁)) t :=
        by simpa using ((hasDerivAt_id t).const_mul (2 * L₁)).exp
      exact (he.mul (hasDerivAt_crossProfile α hγ0.ne' T N l m a t)).const_mul κ
  have h0 : ∀ a, |y a 0| ≤ κ * (Real.exp (2 * L₁ * 0) * V a 0) := fun a => by
    simp [hy, hV, crossProfile, Φ, flowMap_zero]
  have hbound : ∀ a, ∀ t ∈ Set.Ico 0 T,
      |gptField lam β N (Φ t (x + Pi.single l h + Pi.single m h')) a -
        gptField lam β N (Φ t (x + Pi.single l h)) a -
        gptField lam β N (Φ t (x + Pi.single m h')) a + gptField lam β N (Φ t x) a| ≤
      L₁ * |y a t| + ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * |y b t| +
      κ * Real.exp (2 * L₁ * t) * crossForcing α γ T l m a := fun a t ht' => by
    refine (abs_gptField_second_diff_le lam β N hd hL hL₁ hL₂ _ _ _ _ a).trans
      (add_le_add le_rfl ?_)
    have hti : t ∈ Set.Icc 0 T := Set.Ico_subset_Icc_self ht'
    have hu : ∀ c, |Φ t (x + Pi.single l h + Pi.single m h') c - Φ t (x + Pi.single l h) c| ≤
        |h'| * Real.exp (L₁ * t) * profile α γ m c T := fun c =>
      (abs_flowMap_add_single_sub_le lam β N hL hL₁ hα hγ0 hγ1 _ m h' c hti).trans
        (mul_le_mul_of_nonneg_left (profile_mono hα0 hγ0 m c hti.2) (by positivity))
    have hv : ∀ c, |Φ t (x + Pi.single l h + Pi.single m h') c - Φ t (x + Pi.single m h') c| ≤
        |h| * Real.exp (L₁ * t) * profile α γ l c T := fun c => by
      rw [add_right_comm]
      exact (abs_flowMap_add_single_sub_le lam β N hL hL₁ hα hγ0 hγ1 _ l h c hti).trans
        (mul_le_mul_of_nonneg_left (profile_mono hα0 hγ0 l c hti.2) (by positivity))
    have hp : ∀ c, 0 ≤ (profile α γ m a T + profile α γ m c T) *
        (profile α γ l a T + profile α γ l c T) := fun c => by
      have := fun b c => profile_nonneg hα0 hγ0 b c hT
      exact mul_nonneg (add_nonneg (this _ _) (this _ _)) (add_nonneg (this _ _) (this _ _))
    have hsum := sum_alibiWeight_mul_le lam a hp
    have hE : Real.exp (L₁ * t) * Real.exp (L₁ * t) = Real.exp (2 * L₁ * t) := by
      rw [← Real.exp_add]
      ring_nf
    calc L₂ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), alibiWeight lam N a b *
          ((|Φ t (x + Pi.single l h + Pi.single m h') a - Φ t (x + Pi.single l h) a| +
            |Φ t (x + Pi.single l h + Pi.single m h') b - Φ t (x + Pi.single l h) b|) *
          (|Φ t (x + Pi.single l h + Pi.single m h') a - Φ t (x + Pi.single m h') a| +
            |Φ t (x + Pi.single l h + Pi.single m h') b - Φ t (x + Pi.single m h') b|))
        ≤ L₂ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), alibiWeight lam N a b *
          ((|h'| * Real.exp (L₁ * t) * profile α γ m a T +
            |h'| * Real.exp (L₁ * t) * profile α γ m b T) *
          (|h| * Real.exp (L₁ * t) * profile α γ l a T +
            |h| * Real.exp (L₁ * t) * profile α γ l b T)) := by
          refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun b _ =>
            mul_le_mul_of_nonneg_left ?_ (alibiWeight_nonneg lam N a b)) hL₂0
          exact mul_le_mul (add_le_add (hu a) (hu b)) (add_le_add (hv a) (hv b))
            (by positivity) (add_nonneg
              (mul_nonneg (by positivity) (profile_nonneg hα0 hγ0 _ _ hT))
              (mul_nonneg (by positivity) (profile_nonneg hα0 hγ0 _ _ hT)))
      _ = L₂ * |h| * |h'| * Real.exp (2 * L₁ * t) *
          ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a), alibiWeight lam N a b *
            ((profile α γ m a T + profile α γ m b T) * (profile α γ l a T + profile α γ l b T)) := by
          rw [← hE, Finset.mul_sum, Finset.mul_sum]
          exact Finset.sum_congr rfl fun b _ => by ring
      _ ≤ L₂ * |h| * |h'| * Real.exp (2 * L₁ * t) * (2 * Real.exp (2 * |lam|) / ((a : ℝ) + 1) *
          ∑ c ∈ Finset.range a, (profile α γ m a T + profile α γ m c T) *
            (profile α γ l a T + profile α γ l c T)) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = κ * Real.exp (2 * L₁ * t) * crossForcing α γ T l m a := by
          rw [hκ, crossForcing]
          ring
  have hsuper : ∀ a : Idx N, ∀ t ∈ Set.Ico 0 T, L₁ * (κ * (Real.exp (2 * L₁ * t) * V a t)) +
      ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
        L₁ * alibiWeight lam N a b * (κ * (Real.exp (2 * L₁ * t) * V b t)) +
      κ * Real.exp (2 * L₁ * t) * crossForcing α γ T l m a ≤
      κ * (Real.exp (2 * L₁ * t) * (2 * L₁) * V a t + Real.exp (2 * L₁ * t) *
        (∑ c ∈ Finset.range N, crossForcing α γ T l m c * profile α γ c a t +
          t * ∑ c ∈ Finset.range N, crossForcing α γ T l m c * profileDeriv α γ c a t)) :=
    fun a t ht' => by
      have hVn : ∀ c, 0 ≤ V c t := fun c => crossProfile_nonneg hα0 hγ0 hT N l m c ht'.1
      have hs := crossProfile_super hα0 hγ0 hγ1 hT (n := N) l m a.2 ht'.1
      have hsum := sum_alibiWeight_mul_le lam a hVn
      have h3 : L₁ * (2 * Real.exp (2 * |lam|) / ((a : ℝ) + 1)) ≤ α / ((a : ℝ) + 1) := by
        rw [mul_div_assoc', div_le_div_iff_of_pos_right (by positivity)]
        linarith
      have hsumV : 0 ≤ ∑ c ∈ Finset.range a, V c t := Finset.sum_nonneg fun c _ => hVn c
      have hB : L₁ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
          alibiWeight lam N a b * V b t ≤ α / ((a : ℝ) + 1) * ∑ c ∈ Finset.range a, V c t :=
        (mul_le_mul_of_nonneg_left hsum hL₁).trans (by rw [← mul_assoc]; gcongr)
      have hκe : 0 ≤ κ * Real.exp (2 * L₁ * t) := by positivity
      have hrw : ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
          L₁ * alibiWeight lam N a b * (κ * (Real.exp (2 * L₁ * t) * V b t)) =
          κ * Real.exp (2 * L₁ * t) * (L₁ * ∑ b ∈ Finset.univ.filter (fun b : Idx N => b < a),
            alibiWeight lam N a b * V b t) := by
        rw [Finset.mul_sum, Finset.mul_sum]
        exact Finset.sum_congr rfl fun b _ => by ring
      rw [hrw]
      have h1 := mul_le_mul_of_nonneg_left hB hκe
      have h2 := mul_le_mul_of_nonneg_left hs hκe
      have h4 : 0 ≤ κ * Real.exp (2 * L₁ * t) * (L₁ * V a t) :=
        mul_nonneg hκe (mul_nonneg hL₁ (hVn a))
      simp only [hV] at h1 h2 h4 ⊢
      nlinarith
  exact abs_le_of_triangular (fun a b => mul_nonneg hL₁ (alibiWeight_nonneg lam N a b))
    hdy hdW h0 hbound hsuper a t ht

/-- The hypotheses of `abs_gptField_second_diff_le` are satisfiable: the kernel's
own constants. -/
example (β : ℝ) : ∃ L₁ L₂ : ℝ, 0 ≤ L₁ ∧
    (∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x) ∧
    (∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) ∧
    (∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|) := by
  obtain ⟨L₁, L₂, hL₁, -, hd, h₁, h₂⟩ := exists_lipschitz_wBetaDeriv β
  exact ⟨L₁, L₂, hL₁, hd, h₁, h₂⟩

end Kinetic
end Transformer

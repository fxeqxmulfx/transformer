/-
# Differences of a periodic test of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, `eq:chain-rule-Glaub` and
`eq:chain-rule-Glaub-2nd`: for a `2π`-periodic test `φ` with `|φ'| ≤ κ` and
`|φ'|, |φ''| ≤ K'`, the observable `x ↦ φ(θ_a(t))` of the flow from `x` moves,
when token `l` alone moves, by at most `κ π e^{L₁T} Ĝ^l_a(T)`
(`abs_obs_update_sub_le`). Its mixed second differences are at most
`K' π² e^{2L₁T} (2e^{2|λ|} L₂ V_a(T) + Ĝ^l_a(T) Ĝ^m_a(T))`
(`abs_obs_second_diff_le`). Periodicity reduces every move to one by at most
`π` (`exists_periodic_update`, `exists_periodic_update_two`).
-/

import Transformer.Kinetic.Section3_SecondDifferenceBound

open Real

namespace Transformer
namespace Kinetic

/-- The mean value inequality for a test with `|φ'| ≤ K`. -/
theorem abs_sub_le_of_abs_deriv_le {φ dφ : ℝ → ℝ} {K : ℝ} (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hK : ∀ x, |dφ x| ≤ K) (x y : ℝ) : |φ x - φ y| ≤ K * |x - y| := by
  have := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (s := Set.univ)
    (fun z _ => (hφ z).hasDerivWithinAt) (fun z _ => by simpa using hK z) convex_univ
    (Set.mem_univ y) (Set.mem_univ x)
  simpa [Real.norm_eq_abs] using this

/-- A move by at most `π` of token `l` moves token `a` at time `t ∈ [0, T]` by at
most `π e^{L₁T} Ĝ^l_a(T)`. -/
theorem abs_flowMap_add_single_sub_le_pi (lam β : ℝ) (N : ℕ) {L₁ α γ T : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (x : Idx N → ℝ) (l : Idx N) {h : ℝ} (hh : |h| ≤ π) (a : Idx N) {t : ℝ}
    (ht : t ∈ Set.Icc 0 T) :
    |flowMap lam β N t (x + Pi.single l h) a - flowMap lam β N t x a| ≤
      π * Real.exp (L₁ * T) * profile α γ l a T := by
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  refine (abs_flowMap_add_single_sub_le lam β N hL hL₁ hα hγ0 hγ1 x l h a ht).trans ?_
  exact mul_le_mul (mul_le_mul hh (Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left ht.2 hL₁))
    (by positivity) Real.pi_pos.le) (profile_mono hα0 hγ0 l a ht.2)
    (profile_nonneg hα0 hγ0 l a ht.1) (by positivity)

/-- **First differences of an observable (`eq:chain-rule-Glaub`).** -/
theorem abs_obs_update_sub_le (lam β : ℝ) (N : ℕ) {L₁ α γ T κ : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    {φ dφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π)) (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hκ : ∀ x, |dφ x| ≤ κ) {t : ℝ} (ht : t ∈ Set.Icc 0 T) (a : Idx N)
    (x : Idx N → ℝ) (l : Idx N) (z : ℝ) :
    |φ (flowMap lam β N t x a) - φ (flowMap lam β N t (Function.update x l z) a)| ≤
      κ * π * Real.exp (L₁ * T) * profile α γ l a T := by
  have hκ0 : 0 ≤ κ := (abs_nonneg _).trans (hκ 0)
  obtain ⟨h, hh, hrep⟩ := exists_periodic_update lam β N hper x l z
  rw [hrep, abs_sub_comm]
  refine (abs_sub_le_of_abs_deriv_le hφ hκ _ _).trans ?_
  rw [mul_assoc κ, mul_assoc κ]
  exact mul_le_mul_of_nonneg_left
    (by simpa only [mul_assoc] using
      abs_flowMap_add_single_sub_le_pi lam β N hL hL₁ hα hγ0 hγ1 x l hh a ht) hκ0

/-- Moving two tokens, a periodic test sees moves by at most `π`. -/
theorem exists_periodic_update_two (lam β : ℝ) (N : ℕ) {φ : ℝ → ℝ}
    (hper : Function.Periodic φ (2 * π)) (x : Idx N → ℝ) {l m : Idx N} (hml : m ≠ l) (z z' : ℝ) :
    ∃ h h' : ℝ, |h| ≤ π ∧ |h'| ≤ π ∧ ∀ t a,
      φ (flowMap lam β N t (Function.update x l z) a) =
        φ (flowMap lam β N t (x + Pi.single l h) a) ∧
      φ (flowMap lam β N t (Function.update x m z') a) =
        φ (flowMap lam β N t (x + Pi.single m h') a) ∧
      φ (flowMap lam β N t (Function.update (Function.update x l z) m z') a) =
        φ (flowMap lam β N t (x + Pi.single l h + Pi.single m h') a) := by
  obtain ⟨h, hh, n, hn⟩ := exists_shift_le_pi (z - x l)
  obtain ⟨h', hh', n', hn'⟩ := exists_shift_le_pi (z' - x m)
  refine ⟨h, h', hh, hh', fun t a => ⟨?_, ?_, ?_⟩⟩
  · rw [update_eq_add_single, hn, periodic_flowMap_add_single lam β N hper]
  · rw [update_eq_add_single, hn', periodic_flowMap_add_single lam β N hper]
  · have e1 : Function.update (Function.update x l z) m z' =
        Function.update x l z + Pi.single m (h' + n' * (2 * π)) := by
      rw [update_eq_add_single, Function.update_of_ne hml, hn']
    have e2 : Function.update x l z + Pi.single m h' =
        x + Pi.single m h' + Pi.single l (h + n * (2 * π)) := by
      rw [← hn, update_eq_add_single]
      funext k
      simp only [Pi.add_apply]
      by_cases hk : k = l
      · subst hk
        simp [Pi.single_eq_of_ne' hml]
      · simp [Pi.single_eq_of_ne hk]
    rw [e1, periodic_flowMap_add_single lam β N hper, e2, periodic_flowMap_add_single lam β N hper,
      add_right_comm]

/-- **Mixed second differences of an observable (`eq:chain-rule-Glaub-2nd`).** For
`m ≠ l` and `|φ'|, |φ''| ≤ K'`, at time `t ∈ [0, T]`. -/
theorem abs_obs_second_diff_le (lam β : ℝ) (N : ℕ) {L₁ L₂ α γ T K' : ℝ}
    (hd : ∀ x, HasDerivAt (wBetaDeriv β) (deriv (wBetaDeriv β) x) x)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (hL₁ : 0 ≤ L₁)
    (hL₂ : ∀ x y, |deriv (wBetaDeriv β) x - deriv (wBetaDeriv β) y| ≤ L₂ * |x - y|)
    (hα : 2 * Real.exp (2 * |lam|) * L₁ ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    {φ dφ ddφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π)) (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    (hdφ : ∀ x, HasDerivAt dφ (ddφ x) x) (hK₁ : ∀ x, |dφ x| ≤ K') (hK₂ : ∀ x, |ddφ x| ≤ K')
    {t : ℝ} (ht : t ∈ Set.Icc 0 T) (a : Idx N) (x : Idx N → ℝ) {l m : Idx N} (hml : m ≠ l)
    (z z' : ℝ) :
    |φ (flowMap lam β N t x a) - φ (flowMap lam β N t (Function.update x l z) a) -
        φ (flowMap lam β N t (Function.update x m z') a) +
        φ (flowMap lam β N t (Function.update (Function.update x l z) m z') a)| ≤
      K' * π ^ 2 * Real.exp (2 * L₁ * T) * (2 * Real.exp (2 * |lam|) * L₂ *
        crossProfile α γ T N l m a T + profile α γ l a T * profile α γ m a T) := by
  have hα0 : 0 ≤ α := le_trans (by positivity) hα
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hK0 : 0 ≤ K' := (abs_nonneg _).trans (hK₁ 0)
  have hL₂0 : 0 ≤ L₂ := by simpa using (abs_nonneg _).trans (hL₂ 0 1)
  obtain ⟨h, h', hh, hh', hrep⟩ := exists_periodic_update_two lam β N hper x hml z z'
  obtain ⟨e1, e2, e3⟩ := hrep t a
  rw [e1, e2, e3]
  set Φ := flowMap lam β N t
  have h4 := abs_four_point_le hφ (abs_sub_le_of_abs_deriv_le hφ hK₁)
    (abs_sub_le_of_abs_deriv_le hdφ hK₂) (Φ x a) (Φ (x + Pi.single l h) a)
    (Φ (x + Pi.single m h') a) (Φ (x + Pi.single l h + Pi.single m h') a)
  set V := crossProfile α γ T N l m a T
  set E := Real.exp (L₁ * T)
  have hE : Real.exp (2 * L₁ * T) = E * E := by rw [← Real.exp_add]; ring_nf
  have hV0 : 0 ≤ V := crossProfile_nonneg hα0 hγ0 hT N l m a hT
  have hs : |Φ x a - Φ (x + Pi.single l h) a - Φ (x + Pi.single m h') a +
      Φ (x + Pi.single l h + Pi.single m h') a| ≤
      2 * Real.exp (2 * |lam|) * L₂ * (π * π * (Real.exp (2 * L₁ * T) * V)) := by
    have h2 := abs_flowMap_second_diff_le lam β N hd hL hL₁ hL₂ hα hγ0 hγ1 x l m h h' a ht
    rw [show Φ x a - Φ (x + Pi.single l h) a - Φ (x + Pi.single m h') a +
      Φ (x + Pi.single l h + Pi.single m h') a = Φ (x + Pi.single l h + Pi.single m h') a -
      Φ (x + Pi.single l h) a - Φ (x + Pi.single m h') a + Φ x a by ring]
    refine h2.trans ?_
    have hEt : Real.exp (2 * L₁ * t) ≤ Real.exp (2 * L₁ * T) :=
      Real.exp_le_exp.2 (by have := mul_le_mul_of_nonneg_left ht.2 hL₁; linarith)
    have hb : |h| * |h'| * (Real.exp (2 * L₁ * t) * crossProfile α γ T N l m a t) ≤
        π * π * (Real.exp (2 * L₁ * T) * V) :=
      mul_le_mul (mul_le_mul hh hh' (abs_nonneg _) Real.pi_pos.le)
        (mul_le_mul hEt (crossProfile_mono hα0 hγ0 hT N l m a ht.1 ht.2)
          (crossProfile_nonneg hα0 hγ0 hT N l m a ht.1) (Real.exp_pos _).le)
        (mul_nonneg (Real.exp_pos _).le (crossProfile_nonneg hα0 hγ0 hT N l m a ht.1))
        (by positivity)
    calc 2 * Real.exp (2 * |lam|) * L₂ * |h| * |h'| *
          (Real.exp (2 * L₁ * t) * crossProfile α γ T N l m a t) =
          2 * Real.exp (2 * |lam|) * L₂ *
            (|h| * |h'| * (Real.exp (2 * L₁ * t) * crossProfile α γ T N l m a t)) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hb (by positivity)
  have hl : |Φ x a - Φ (x + Pi.single l h) a| ≤ π * E * profile α γ l a T := by
    rw [abs_sub_comm]
    exact abs_flowMap_add_single_sub_le_pi lam β N hL hL₁ hα hγ0 hγ1 x l hh a ht
  have hm : |Φ x a - Φ (x + Pi.single m h') a| ≤ π * E * profile α γ m a T := by
    rw [abs_sub_comm]
    exact abs_flowMap_add_single_sub_le_pi lam β N hL hL₁ hα hγ0 hγ1 x m hh' a ht
  have hp : |Φ x a - Φ (x + Pi.single l h) a| * |Φ x a - Φ (x + Pi.single m h') a| ≤
      π * E * profile α γ l a T * (π * E * profile α γ m a T) :=
    mul_le_mul hl hm (abs_nonneg _) ((abs_nonneg _).trans hl)
  have hfin : K' * (2 * Real.exp (2 * |lam|) * L₂ * (π * π * (Real.exp (2 * L₁ * T) * V))) +
      K' * (π * E * profile α γ l a T * (π * E * profile α γ m a T)) =
      K' * π ^ 2 * Real.exp (2 * L₁ * T) * (2 * Real.exp (2 * |lam|) * L₂ * V +
        profile α γ l a T * profile α γ m a T) := by
    rw [hE]; ring
  linarith [mul_le_mul_of_nonneg_left hs hK0, mul_le_mul_of_nonneg_left hp hK0]

/-- A test of the flow is measurable and bounded. -/
theorem measurable_obs (lam β : ℝ) (N : ℕ) {φ dφ : ℝ → ℝ} (hφ : ∀ x, HasDerivAt φ (dφ x) x)
    {t : ℝ} (ht : 0 ≤ t) (a : Idx N) : Measurable fun x => φ (flowMap lam β N t x a) :=
  (continuous_iff_continuousAt.2 fun x => (hφ x).continuousAt).measurable.comp
    (measurable_flowMap lam β N ht a)

theorem exists_abs_le_of_periodic {φ dφ : ℝ → ℝ} (hper : Function.Periodic φ (2 * π))
    (hφ : ∀ x, HasDerivAt φ (dφ x) x) : ∃ C, ∀ x, |φ x| ≤ C := by
  obtain ⟨C, hC⟩ := isBounded_iff_forall_norm_le.1 (hper.isBounded_of_continuous
    (by positivity) (continuous_iff_continuousAt.2 fun x => (hφ x).continuousAt))
  exact ⟨C, fun x => by simpa using hC (φ x) ⟨x, rfl⟩⟩

/-- The hypotheses of `abs_obs_update_sub_le` and `abs_obs_second_diff_le` are
satisfiable: `sin`, with `κ = K' = 1`. -/
example : Function.Periodic Real.sin (2 * π) ∧ (∀ x, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x, HasDerivAt Real.cos (-Real.sin x) x) ∧ (∀ x, |Real.cos x| ≤ 1) ∧
    ∀ x, |-Real.sin x| ≤ 1 :=
  ⟨Real.sin_periodic, Real.hasDerivAt_sin, Real.hasDerivAt_cos, Real.abs_cos_le_one,
    fun x => by rw [abs_neg]; exact Real.abs_sin_le_one x⟩

end Kinetic
end Transformer

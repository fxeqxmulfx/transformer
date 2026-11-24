module

public import Transformer.GainMatroid.Section2_ExchangeSwitching

/-!
# Switching equivalence across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.10. For an
exchange pair `T`, `S` on `X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star
`X` at `v` in `S`:

* a switching makes the gains of the star one (`IsExchange.exists_switch_eq_one`), as its three
  edges join `v` to three distinct vertices (`Joins.switch_eq_one`);
* gain functions with value one on `X` are switching equivalent in `T` exactly when they are in `S`
  (`IsExchange.switchEquiv_iff`): a switching between them takes one value on the ends of `X`, in
  `T` and in `S`, so it moves from one graph to the other;
* gain functions with value zero on `X` are switching-and-scaling equivalent in `T` exactly when
  they are in `S` (`IsExchange.switchScaleEquiv_iff`), as switching-and-scaling equivalence is
  switching equivalence to a nonzero multiple (`switchScaleEquiv_iff_exists_addSwitchEquiv`).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- Switching-and-scaling equivalence is switching equivalence to a nonzero multiple. -/
theorem switchScaleEquiv_iff_exists_addSwitchEquiv {V E F : Type*} [Field F]
    {G : OrientedGraph V E} {φ ψ : E → F} :
    SwitchScaleEquiv G φ ψ ↔ ∃ r : Fˣ, AddSwitchEquiv G φ ((r : F) • ψ) := by
  constructor
  · rintro ⟨η, r, rfl⟩
    exact ⟨r⁻¹, η, by rw [smul_smul, Units.inv_mul, one_smul]⟩
  · rintro ⟨r, η, h⟩
    exact ⟨η, r⁻¹, by rw [h, smul_smul, Units.inv_mul, one_smul]⟩

namespace OrientedGraph

variable {V W E Γ : Type*} [Group Γ]

/-- A switching with value one at `s` makes the gain of an edge `e` joining `s` and `t` one when its
value at `t` is `γ(e)⁻¹` if `e` leaves `s`, and `γ(e)` if `e` enters `s`. -/
@[to_additive Joins.addSwitch_eq_zero /-- A switching with value zero at `s` makes the gain of an
edge `e` joining `s` and `t` zero when its value at `t` is `-γ(e)` if `e` leaves `s`, and `γ(e)` if
`e` enters `s`. -/]
theorem Joins.switch_eq_one {G : OrientedGraph V E} {γ : E → Γ} {η : V → Γ} {e : E} {s t : V}
    (h : G.Joins e s t) (hs : η s = 1) (hout : G.tail e = s → η t = (γ e)⁻¹)
    (hin : G.head e = s → η t = γ e) : switch G η γ e = 1 := by
  rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rw [switch_apply, h₁, h₂, hs, hout h₁, inv_one, one_mul, mul_inv_cancel]
  · rw [switch_apply, h₁, h₂, hs, hin h₂, mul_one, inv_mul_cancel]

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- A switching makes the gains of the star `X` at `v` one. -/
@[to_additive exists_addSwitch_eq_zero /-- A switching makes the additive gains of the star `X` at
`v` zero. -/]
theorem exists_switch_eq_one (h : IsExchange T S ι v a b c x y z) (γ : E → Γ) :
    ∃ η : W → Γ, ∀ e ∈ ({a, b, c} : Set E), switch S η γ e = 1 := by
  classical
  have hx : ι x ≠ v := h.ne_of_joins h.triangle.joins_b
  have hy : ι y ≠ v := h.ne_of_joins h.triangle.joins_a
  have hz : ι z ≠ v := h.ne_of_joins h.triangle.joins_a.symm
  have hxy : ι y ≠ ι x := h.inj.ne h.triangle.ne_xy.symm
  have hxz : ι z ≠ ι x := h.inj.ne h.triangle.ne_xz.symm
  have hyz : ι z ≠ ι y := h.inj.ne h.triangle.ne_yz.symm
  let g : E → Γ := fun e => if S.tail e = v then (γ e)⁻¹ else γ e
  let η : W → Γ := fun w =>
    if w = ι x then g a else if w = ι y then g b else if w = ι z then g c else 1
  have hv : η v = 1 := by simp [η, hx.symm, hy.symm, hz.symm]
  have key : ∀ {e : E} {u : V}, e ∈ ({a, b, c} : Set E) → S.Joins e v (ι u) → η (ι u) = g e →
      switch S η γ e = 1 := fun {e} {_} he hj hu =>
    hj.switch_eq_one hv (fun ht => by rw [hu]; simp [g, ht]) fun hh => by
      have ht : ¬S.tail e = v := fun ht => h.isLink_of_mem he (ht.trans hh.symm)
      rw [hu]
      simp [g, ht]
  refine ⟨η, ?_⟩
  rintro e (rfl | rfl | rfl)
  · exact key (Or.inl rfl) h.joins_a (by simp [η])
  · exact key (Or.inr (Or.inl rfl)) h.joins_b (by simp [η, hxy])
  · exact key (Or.inr (Or.inr rfl)) h.joins_c (by simp [η, hxz, hyz])

/-- Gain functions with value one on `X` are switching equivalent in `T` exactly when they are in
`S`. -/
@[to_additive addSwitchEquiv_iff /-- Gain functions with value zero on `X` are additively switching
equivalent in `T` exactly when they are in `S`. -/]
theorem switchEquiv_iff (h : IsExchange T S ι v a b c x y z) {γ γ' : E → Γ}
    (hγ : ∀ e ∈ ({a, b, c} : Set E), γ e = 1) (hγ' : ∀ e ∈ ({a, b, c} : Set E), γ' e = 1) :
    SwitchEquiv T γ γ' ↔ SwitchEquiv S γ γ' := by
  have ha := hγ a (Or.inl rfl)
  have hb := hγ b (Or.inr (Or.inl rfl))
  have hc := hγ c (Or.inr (Or.inr rfl))
  constructor
  · rintro ⟨η, rfl⟩
    have hyz : η y = η z :=
      (h.triangle.joins_a.switch_eq_one_iff_of_eq_one η ha).1 (hγ' a (Or.inl rfl))
    have hxz : η x = η z :=
      (h.triangle.joins_b.switch_eq_one_iff_of_eq_one η hb).1 (hγ' b (Or.inr (Or.inl rfl)))
    classical
    let η' : W → Γ := fun w => if w = v then η x else Function.extend ι η 1 w
    have hη₀ : ∀ u, ι u ≠ v → η' (ι u) = η u := fun u hu => by
      simp [η', hu, h.inj.extend_apply]
    refine ⟨η', funext fun e => ?_⟩
    by_cases heX : e ∈ ({a, b, c} : Set E)
    · rw [hγ' e heX]
      rcases heX with rfl | rfl | rfl
      · rw [h.joins_a.switch_eq_one_iff_of_eq_one η' ha, hη₀ x (h.ne_of_joins h.triangle.joins_b)]
        simp [η']
      · rw [h.joins_b.switch_eq_one_iff_of_eq_one η' hb, hη₀ y (h.ne_of_joins h.triangle.joins_a)]
        simp [η', hxz, hyz]
      · rw [h.joins_c.switch_eq_one_iff_of_eq_one η' hc,
          hη₀ z (h.ne_of_joins h.triangle.joins_a.symm)]
        simp [η', hxz]
    · exact h.switch_eq γ hη₀ heX
  · rintro ⟨η', rfl⟩
    have hx := (h.joins_a.switch_eq_one_iff_of_eq_one η' ha).1 (hγ' a (Or.inl rfl))
    have hy := (h.joins_b.switch_eq_one_iff_of_eq_one η' hb).1 (hγ' b (Or.inr (Or.inl rfl)))
    have hz := (h.joins_c.switch_eq_one_iff_of_eq_one η' hc).1 (hγ' c (Or.inr (Or.inr rfl)))
    refine ⟨η' ∘ ι, funext fun e => ?_⟩
    by_cases heX : e ∈ ({a, b, c} : Set E)
    · rw [hγ' e heX]
      rcases heX with rfl | rfl | rfl
      · exact (h.triangle.joins_a.switch_eq_one_iff_of_eq_one _ ha).2 (hy.symm.trans hz)
      · exact (h.triangle.joins_b.switch_eq_one_iff_of_eq_one _ hb).2 (hx.symm.trans hz)
      · exact (h.triangle.joins_c.switch_eq_one_iff_of_eq_one _ hc).2 (hx.symm.trans hy)
    · exact (h.switch_eq γ (η := η' ∘ ι) (fun _ _ => rfl) heX).symm

/-- Gain functions with value zero on `X` are switching-and-scaling equivalent in `T` exactly when
they are in `S`. -/
theorem switchScaleEquiv_iff {F : Type*} [Field F] (h : IsExchange T S ι v a b c x y z)
    {φ ψ : E → F} (hφ : ∀ e ∈ ({a, b, c} : Set E), φ e = 0)
    (hψ : ∀ e ∈ ({a, b, c} : Set E), ψ e = 0) :
    SwitchScaleEquiv T φ ψ ↔ SwitchScaleEquiv S φ ψ := by
  rw [switchScaleEquiv_iff_exists_addSwitchEquiv, switchScaleEquiv_iff_exists_addSwitchEquiv]
  exact exists_congr fun r => h.addSwitchEquiv_iff hφ fun e he => by
    rw [Pi.smul_apply, hψ e he, smul_zero]

end IsExchange

example := K4_isLabelledTriangle.joins_a.switch_eq_one (γ := (1 : Fin 6 → ℚˣ)) (η := 1) rfl
  (fun _ => by simp) fun _ => by simp
example := K4_isLabelledTriangle.joins_a.addSwitch_eq_zero (γ := (0 : Fin 6 → ℚ)) (η := 0) rfl
  (fun _ => by simp) fun _ => by simp
example := K4_isLabelledTriangle.isExchange.exists_switch_eq_one (1 : Fin 6 → ℚˣ)
example := K4_isLabelledTriangle.isExchange.exists_addSwitch_eq_zero (0 : Fin 6 → ℚ)
example := K4_isLabelledTriangle.isExchange.switchEquiv_iff (γ := (1 : Fin 6 → ℚˣ)) (γ' := 1)
  (fun _ _ => rfl) fun _ _ => rfl
example := K4_isLabelledTriangle.isExchange.addSwitchEquiv_iff (γ := (0 : Fin 6 → ℚ)) (γ' := 0)
  (fun _ _ => rfl) fun _ _ => rfl
example := K4_isLabelledTriangle.isExchange.switchScaleEquiv_iff (φ := (0 : Fin 6 → ℚ)) (ψ := 0)
  (fun _ _ => rfl) fun _ _ => rfl

end OrientedGraph

end Transformer.GainMatroid

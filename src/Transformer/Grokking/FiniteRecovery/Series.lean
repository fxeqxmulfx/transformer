module

public import Transformer.Grokking.FiniteRecovery.Basic
public import Transformer.Grokking.FlatSpectrum.TablePart.Defs

/-! Recovery from windows belonging to several series. Observation sites are
labeled by series and window, and the predictor sees their encoded contexts.
A decoder may also use the grid and prefix available at its own site. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Transformer.Grokking.FlatSpectrum.TablePart

/-- Each series may contribute its own number of prediction windows. -/
abbrev SeriesSite (S : Type*) (windows : S → ℕ) := Σ s, Fin (windows s)

/-- The number of sites counts every window, including repeated contexts. -/
theorem card_seriesSite {S : Type*} [Fintype S] (windows : S → ℕ) :
    Fintype.card (SeriesSite S windows) = ∑ s, windows s := by
  rw [Fintype.card_sigma]
  simp only [Fintype.card_fin]

/-- The context goes to the shared predictor. The decoder at a site can use
the local raw prefix and grid. A partial decoder can take values in an option type. -/
structure SeriesEncoding (I X A Y : Type*) where
  context : I → X
  decode : I → A → Y

variable {I X A Y : Type*}

def SeriesEncoding.decoded (encoding : SeriesEncoding I X A Y) (predict : X → A) : I → Y :=
  fun i => encoding.decode i (predict (encoding.context i))

/-- A shared predictor exists exactly when one token works for every site
with each given context. The same token may decode to different values at
different sites. -/
theorem SeriesEncoding.exists_decoded_eq_iff (encoding : SeriesEncoding I X A Y) (target : I → Y) :
    (∃ predict, encoding.decoded predict = target) ↔
      ∀ x : X, ∃ a : A, ∀ i, encoding.context i = x → encoding.decode i a = target i := by
  constructor
  · rintro ⟨predict, rfl⟩ x
    exact ⟨predict x, fun i h => by rw [SeriesEncoding.decoded, h]⟩
  · intro h
    choose predict hp using h
    exact ⟨predict, funext fun i => hp (encoding.context i) i rfl⟩

/-- For exact token labels, the compatibility condition is constancy on the
fibers of the encoder. -/
theorem exists_token_predictor_iff [Nonempty A] (context : I → X) (target : I → A) :
    (∃ predict : X → A, ∀ i, predict (context i) = target i) ↔
      ∀ i j, context i = context j → target i = target j := by
  constructor
  · rintro ⟨predict, hpredict⟩ i j hij
    rw [← hpredict i, ← hpredict j, hij]
  · intro h
    let codec : SeriesEncoding I X A A := ⟨context, fun _ => id⟩
    obtain ⟨predict, hp⟩ := (codec.exists_decoded_eq_iff target).2 (by
      intro x
      by_cases hx : ∃ i, context i = x
      · obtain ⟨i, hi⟩ := hx
        exact ⟨target i, fun j hj => h i j (hi.trans hj.symm)⟩
      · exact ⟨Classical.choice ‹Nonempty A›, fun j hj => (hx ⟨j, hj⟩).elim⟩)
    exact ⟨predict, congrFun hp⟩

/-- FiniteRecovery observes the image of the chosen sites under the
encoder. Repeated compatible labels give one pair in the observed graph. -/
theorem observe_series_eq_iff {T : ℕ} [DecidableEq A] (context : I → Word A T)
    (sites : Finset I) (f g : Word A T → A) :
    observe (sites.image context) f = observe (sites.image context) g ↔
      ∀ i ∈ sites, f (context i) = g (context i) := by
  rw [observe_eq_iff_eqOn]
  constructor
  · exact fun h i hi => h (Finset.mem_image_of_mem context hi)
  · intro h x hx
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hx
    exact h i hi

/-- A mask identifies a class on the contexts that actually occur. -/
def DeterminesOnContexts (context : I → X) (family : Set (X → A)) (sites : Set I) : Prop :=
  ∀ f ∈ family, ∀ g ∈ family, Set.EqOn f g (context '' sites) → Set.EqOn f g (Set.range context)

/-- For unrestricted token-valued predictors, every occurring context must
be observed. A missing context permits changing its label independently. -/
theorem determinesOnContexts_univ_iff [Nontrivial A] (context : I → X) (sites : Set I) :
    DeterminesOnContexts context (Set.univ : Set (X → A)) sites ↔
      context '' sites = Set.range context := by
  constructor
  · intro h
    apply Set.Subset.antisymm (Set.image_subset_range _ _)
    intro x hx
    by_contra hmissing
    classical
    obtain ⟨a, b, hab⟩ := exists_pair_ne A
    let f : X → A := fun _ => a
    let g := Function.update f x b
    have hagree : Set.EqOn f g (context '' sites) := by
      intro y hy
      have hne : y ≠ x := fun heq => hmissing (heq ▸ hy)
      exact (Function.update_of_ne hne b f).symm
    have hlabel := h f (Set.mem_univ _) g (Set.mem_univ _) hagree hx
    exact hab (by simpa [f, g] using hlabel)
  · intro h _ _ _ _ hagree
    rwa [← h]

/-- With injective per-site decoders, identifying the entire decoded profile
of an unrestricted predictor has the same coverage condition. -/
theorem SeriesEncoding.determines_decoded_iff [Nontrivial A] (encoding : SeriesEncoding I X A Y)
    (hdecode : ∀ i, Function.Injective (encoding.decode i)) (sites : Set I) :
    Determines (Set.range encoding.decoded) sites ↔
      encoding.context '' sites = Set.range encoding.context := by
  constructor
  · intro h
    apply (determinesOnContexts_univ_iff (A := A) _ _).1
    intro f _ g _ hagree x hx
    obtain ⟨i, rfl⟩ := hx
    have heq := h _ (Set.mem_range_self f) _ (Set.mem_range_self g) fun j hj =>
      congrArg (encoding.decode j) (hagree (Set.mem_image_of_mem _ hj))
    exact hdecode i (congrFun heq i)
  · intro h
    rintro _ ⟨f, rfl⟩ _ ⟨g, rfl⟩ hagree
    funext i
    have hmem : encoding.context i ∈ encoding.context '' sites :=
      h.symm ▸ Set.mem_range_self i
    obtain ⟨j, hj, hij⟩ := hmem
    have ht := hdecode j (hagree j hj)
    apply congrArg (encoding.decode i)
    simpa only [hij] using ht

/-- For a finite collection of windows, the least number of sites that
identify an unrestricted predictor on its occurring contexts is the number
of distinct encoded contexts. A representative of each fiber attains it. -/
theorem isLeast_unrestricted_series_observations [Fintype I] [DecidableEq X] [Nontrivial A]
    (context : I → X) :
    IsLeast {n : ℕ | ∃ sites : Finset I, sites.card = n ∧
      DeterminesOnContexts context (Set.univ : Set (X → A)) sites}
      (Finset.univ.image context).card := by
  classical
  obtain ⟨u, hu, hInj⟩ := Set.exists_image_eq_and_injOn (Set.univ : Set I) context
  let sites := (Set.toFinite u).toFinset
  have hcover (s : Finset I) :
      s.image context = Finset.univ.image context ↔ context '' (s : Set I) = Set.range context := by
    rw [← Finset.coe_inj, Finset.coe_image, Finset.coe_image, Finset.coe_univ, Set.image_univ]
  have hImage : context '' (sites : Set I) = Set.range context := by simpa [sites] using hu
  refine ⟨⟨sites, ?_, (determinesOnContexts_univ_iff _ _).2 hImage⟩, ?_⟩
  · rw [← (hcover sites).2 hImage]
    exact (Finset.card_image_iff.mpr (by simpa [sites] using hInj)).symm
  · rintro _ ⟨observed, rfl, hdet⟩
    rw [← (hcover observed).2 ((determinesOnContexts_univ_iff _ _).1 hdet)]
    exact Finset.card_image_le

example : Fintype.card (SeriesSite (Fin 2) (fun _ => 3)) = 6 := by decide

example : ¬ ∃ predict : Unit → Bool, ∀ i : Bool, predict () = i := by
  intro h
  exact Bool.false_ne_true ((exists_token_predictor_iff (fun _ : Bool => ()) id).1 h false true rfl)

private def shiftedDecoder : SeriesEncoding Bool Unit ℤ ℤ :=
  ⟨fun _ => (), fun r a => (if r then 100 else 10) + a⟩

example : ∃ predict, shiftedDecoder.decoded predict = fun r => if r then 103 else 13 := by
  apply (shiftedDecoder.exists_decoded_eq_iff _).2
  exact fun x => ⟨3, fun r _ => by cases r <;> decide⟩

/-- The two series share both contexts, so two labeled sites suffice for
every predictor on these contexts, even though there are six sites. -/
example : IsLeast {n : ℕ | ∃ sites : Finset (SeriesSite (Fin 2) (fun _ => 3)), sites.card = n ∧
    DeterminesOnContexts (fun i : SeriesSite (Fin 2) (fun _ => 3) => decide (i.2.val % 2 = 1))
      (Set.univ : Set (Bool → Bool)) sites} 2 := by
  have h := isLeast_unrestricted_series_observations (A := Bool)
    (fun i : SeriesSite (Fin 2) (fun _ => 3) => decide (i.2.val % 2 = 1))
  have hcard : (Finset.univ.image
      (fun i : SeriesSite (Fin 2) (fun _ => 3) => decide (i.2.val % 2 = 1))).card = 2 := by decide
  rw [hcard] at h
  exact h

end Transformer.Grokking.FiniteRecovery

module

public import Transformer.Grokking.Protocol.AdamSymmetry
public import Transformer.Grokking.Protocol.OrbitAnchors
public import Transformer.Grokking.Protocol.ReadoutSymmetry

/-!
# On a split that leaves a residue unlabelled, exchanging two readout rows is a symmetry

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; openai/grok at 3d64b1d8, `grok/data.py`, `ArithmeticTokenizer.get_tokens`, and
`grok/transformer.py`, where the readout is not tied to the embedding;
`experiments/grok_division/PROTOCOL.md`, *The task and its tokens* and *The split*.

A split leaves the residue `r` *unlabelled* when `r` is the quotient of no training equation:
all 96 equations with the answer `r` are held out (`unlabelled`). The token of `r` is then the
target of no training example, like the 141 tokens that are no residue and not `<|eos|>`;
`spareToken` is one of those, the first of the operator names, which division does not read
either. The token of `r` may still be read, as a numerator or a denominator.

`relabel r` exchanges the readout rows of the token of `r` and of the spare token and moves
nothing else. The supervision of an equation with another quotient has no target that it moves
(`targetsFixed_supervised`), so on such a split every gradient of the run commutes with it
(`divisionGrad_comp_relabel`) and the run from the exchanged initial readout is the exchanged
run (`trajectory_relabel`), under Adam and under AdamW alike. The rows of the readout have one
initialization law, so the exchange preserves the law of the initial parameters and of a
sample (`measurePreserving_comp_of_initLaw`, `measurePreserving_relabelSample`).

At most 96 equations have a given quotient (`card_quotient_eq_le`), fewer than the 4656 held
out, so some split leaves `r` unlabelled (`exists_unlabelled`) and such splits have positive
probability (`uniformPerm_unlabelled_pos`); for one residue it is
`∏_{j<96} (4656 − j)/(9312 − j)`, about `7.7 · 10⁻³⁰` (`uniformPerm_unlabelled_eq` and
`uniformPerm_unlabelled_mem_Icc` of `LabelBound`).
`LabelCoverage` draws the consequence for the held-out answers.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- A token that division never shows: index `2`, the first of the operator names. -/
def spareToken : Fin 239 := ⟨2, by norm_num⟩

/-- The renaming that exchanges the token of the residue `r` with the spare token. -/
def labelSwap (r : Fin 97) : Equiv.Perm (Fin 239) := Equiv.swap (residueToken r) spareToken

/-- **The relabelling**: the readout rows of the two tokens change places; no other
parameter moves. -/
abbrev relabel (r : Fin 97) : Equiv.Perm (Param paperSizes) :=
  outPerm (S := paperSizes) (labelSwap r)

lemma labelSwap_residueToken {r s : Fin 97} (hs : s ≠ r) :
    labelSwap r (residueToken s) = residueToken s := by
  refine Equiv.swap_apply_of_ne_of_ne (fun h => hs (residueToken_injective h)) fun h => ?_
  have := congrArg Fin.val h
  simp only [residueToken, spareToken] at this
  omega

lemma labelSwap_eosToken (r : Fin 97) : labelSwap r eosToken = eosToken := by
  refine Equiv.swap_apply_of_ne_of_ne (fun h => ?_) (by decide)
  have := congrArg Fin.val h
  simp only [residueToken, eosToken] at this
  omega

/-- **The supervision of an equation with another quotient has no target that the relabelling
moves.** -/
lemma targetsFixed_supervised {r : Fin 97} {x : Equation} (hx : x.1 ≠ r) :
    (supervised x).TargetsFixed (labelSwap r) := fun pc hpc => by
  simp only [supervised, List.mem_cons, List.mem_nil_iff, or_false] at hpc
  rcases hpc with rfl | rfl
  · exact labelSwap_residueToken hx
  · exact labelSwap_eosToken r

/-- **On a batch of equations with other quotients the gradient commutes with the
relabelling.** -/
theorem divisionGrad_comp_relabel {r : Fin 97} {B : List Equation} (hB : ∀ x ∈ B, x.1 ≠ r)
    (θ : Param paperSizes → ℝ) :
    divisionGrad B (θ ∘ relabel r) = divisionGrad B θ ∘ relabel r := by
  funext p
  exact sourceGrad_comp_outPerm (fun e he => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he
    exact targetsFixed_supervised (hB x hx)) θ p

/-- **The splits that leave the residue `r` unlabelled**: no training equation has the
quotient `r`. -/
def unlabelled (r : Fin 97) : Set (Equiv.Perm (Fin (97 * 96))) :=
  {π | ∀ i : Fin 4656, (trainEq π i).1 ≠ r}

/-- **On a split that leaves `r` unlabelled the run from the relabelled initial parameters is
the relabelled run**, at every weight decay. -/
theorem trajectory_relabel (decay : ℝ) {r : Fin 97} {π : Equiv.Perm (Fin (97 * 96))}
    (hπ : π ∈ unlabelled r) (θ₀ : Param paperSizes → ℝ) (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀ ∘ relabel r, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ relabel r :=
  (adamRun_comp (relabel r) (fun s θ => divisionGrad_comp_relabel (fun x hx => by
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
    exact hπ i) θ) θ₀ t).1

/-- The relabelling keeps the kind of a parameter, hence its initialization law. -/
lemma initLaw_relabel (r : Fin 97) (p : Param paperSizes) :
    initLaw paperSizes (relabel r p) = initLaw paperSizes p := by
  cases p <;> rfl

/-- **A permutation of the parameters that keeps the law of each preserves the law of the
initial parameters.** -/
theorem measurePreserving_comp_of_initLaw (ρ : Equiv.Perm (Param paperSizes))
    (hρ : ∀ p, initLaw paperSizes (ρ p) = initLaw paperSizes p) :
    MeasurePreserving (fun θ : Param paperSizes → ℝ => θ ∘ ρ) (initMeasure paperSizes)
      (initMeasure paperSizes) := by
  have hm : Measurable fun θ : Param paperSizes → ℝ => θ ∘ ρ :=
    measurable_pi_iff.2 fun p => measurable_pi_apply (ρ p)
  refine ⟨hm, ?_⟩
  unfold initMeasure
  refine (Measure.pi_eq fun s hs => ?_).symm
  have hpre : (fun θ : Param paperSizes → ℝ => θ ∘ ρ) ⁻¹' Set.pi Set.univ s =
      Set.pi Set.univ fun p => s (ρ.symm p) := by
    ext θ
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, true_implies, Function.comp_apply]
    exact ⟨fun h p => by simpa using h (ρ.symm p), fun h p => by simpa using h (ρ p)⟩
  rw [Measure.map_apply hm (MeasurableSet.univ_pi hs), hpre, Measure.pi_pi, ← Equiv.prod_comp ρ]
  refine Finset.prod_congr rfl fun p _ => ?_
  rw [Equiv.symm_apply_apply, hρ]

/-- The relabelling of a sample: of the readout of its initial parameters. -/
def relabelSample (r : Fin 97) : Sample (97 * 96) 4656 (Param paperSizes) →
    Sample (97 * 96) 4656 (Param paperSizes) :=
  Prod.map id (Prod.map (fun θ => θ ∘ relabel r) id)

/-- **The relabelling preserves the law of a run.** -/
theorem measurePreserving_relabelSample (r : Fin 97) :
    MeasurePreserving (relabelSample r) (law (97 * 96) 4656 (initMeasure paperSizes))
      (law (97 * 96) 4656 (initMeasure paperSizes)) :=
  (MeasurePreserving.id _).prod
    ((measurePreserving_comp_of_initLaw _ (initLaw_relabel r)).prod (MeasurePreserving.id _))

/-- At most 96 equations have the quotient `r`. -/
lemma card_quotient_eq_le (r : Fin 97) :
    (Finset.univ.filter fun j : Fin (97 * 96) => (equations j).1 = r).card ≤ 96 := by
  have h := Finset.card_le_card_of_injOn
    (s := Finset.univ.filter fun j : Fin (97 * 96) => (equations j).1 = r)
    (t := (Finset.univ : Finset (Fin 96))) (fun j => (equations j).2)
    (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)) fun i hi j hj hij =>
      equations_bijective.injective (Prod.ext
        (((Finset.mem_filter.1 hi).2 : (equations i).1 = r).trans
          ((Finset.mem_filter.1 hj).2 : (equations j).1 = r).symm) hij)
  rwa [Finset.card_univ, Fintype.card_fin] at h

/-- **Some split leaves the residue `r` unlabelled**: it holds out all 96 equations with the
quotient `r`. -/
theorem exists_unlabelled (r : Fin 97) : ∃ π, π ∈ unlabelled r := by
  have hcard : 4656 ≤
      (Finset.univ.filter fun j : Fin (97 * 96) => ¬(equations j).1 = r).card := by
    have h := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin (97 * 96))))
      fun j => (equations j).1 = r
    rw [Finset.card_univ, Fintype.card_fin] at h
    have h96 := card_quotient_eq_le r
    omega
  exact exists_perm_forall_castLE (by norm_num : 4656 ≤ 97 * 96)
    (fun j => ¬(equations j).1 = r) hcard

/-- **The splits that leave `r` unlabelled have positive probability.** -/
theorem uniformPerm_unlabelled_pos (r : Fin 97) : 0 < uniformPerm (97 * 96) (unlabelled r) := by
  obtain ⟨π, hπ⟩ := exists_unlabelled r
  exact uniformPerm_pos_of_mem hπ

/-! The witnesses: another residue's token is not moved; `3 / 5 = 20` has no target that the
relabelling of `7` moves; the exchange of two parameters that have one law. -/

example : labelSwap 7 (residueToken 20) = residueToken 20 := labelSwap_residueToken (by decide)

example : (supervised (20, 4)).TargetsFixed (labelSwap 7) := targetsFixed_supervised (by decide)

example (θ : Param paperSizes → ℝ) :=
  divisionGrad_comp_relabel (r := 7) (B := [(20, 4)]) (fun x hx => by
    rw [List.mem_singleton.1 hx]
    decide) θ

example := measurePreserving_comp_of_initLaw (relabel 7) (initLaw_relabel 7)

/-- Some split leaves `7` unlabelled, and on it the run from relabelled parameters is the
relabelled run. -/
example (decay : ℝ) (θ₀ : Param paperSizes → ℝ) (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    ∃ π, trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀ ∘ relabel 7, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ relabel 7 := by
  obtain ⟨π, hπ⟩ := exists_unlabelled 7
  exact ⟨π, trajectory_relabel decay hπ θ₀ σ t⟩

end Transformer.Grokking.Protocol

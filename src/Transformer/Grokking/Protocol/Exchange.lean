module

public import Transformer.Grokking.Protocol.AdamSymmetry
public import Transformer.Grokking.Protocol.Symmetry
public import Transformer.Grokking.Protocol.Uncovered

/-!
# On an uncovered split, exchanging the two hidden tokens is a symmetry of the run

Source: arXiv:2201.02177v1 (Power et al.), section 1, "distinct abstract symbols"; openai/grok
at 3d64b1d8; `experiments/grok_division/PROTOCOL.md`. The split is `uncovered` of
`Transformer.Grokking.Protocol.Uncovered`: no training equation shows the residue `1` or `2`.

`hiddenSwap` exchanges the tokens of those two residues, and `exchange` is the permutation of
the parameters it induces: two rows of the embedding and two rows of the readout change
places. The supervision of an equation that avoids both residues is left alone
(`fixed_supervised`), so on an uncovered split every gradient of the run commutes with the
exchange (`divisionGrad_comp_exchange`) and the run from the exchanged initial parameters is
the exchanged run (`trajectory_exchange`), under Adam and under AdamW alike.

The initialization law does not see the exchange: the rows of the embedding have one law and
the rows of the readout another (`initLaw_exchange`), so the exchange preserves the law of the
initial parameters (`measurePreserving_comp_exchange`) and the law of a run
(`measurePreserving_exchangeSample`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- The renaming that exchanges the tokens of the hidden residues `1` and `2`. -/
def hiddenSwap : Equiv.Perm (Fin 239) := Equiv.swap (residueToken 1) (residueToken 2)

/-- **The exchange**: the permutation of the parameters induced by `hiddenSwap`. -/
abbrev exchange : Equiv.Perm (Param paperSizes) := paramPerm (S := paperSizes) hiddenSwap

lemma hiddenSwap_residueToken {r : Fin 97} (hr : r ∉ hidden) :
    hiddenSwap (residueToken r) = residueToken r := by
  refine Equiv.swap_apply_of_ne_of_ne (fun h => hr ?_) (fun h => hr ?_)
  · rw [residueToken_injective h]
    decide
  · rw [residueToken_injective h]
    decide

/-- **The supervision of an equation that avoids the hidden residues is left alone.** -/
lemma fixed_supervised {x : Equation} (hx : Avoids x) : (supervised x).Fixed hiddenSwap := by
  obtain ⟨ha, hb, hc⟩ := hx
  have h0 : hiddenSwap eosToken = eosToken := by decide
  have h1 : hiddenSwap eqToken = eqToken := by decide
  have h2 : hiddenSwap divToken = divToken := by decide
  refine ⟨?_, fun pc hpc => ?_⟩
  · change ∀ t : Fin 6, hiddenSwap (read x t) = read x t
    intro t
    fin_cases t <;>
      simp [read, h0, h1, h2, hiddenSwap_residueToken ha, hiddenSwap_residueToken hb,
        hiddenSwap_residueToken hc]
  · simp only [supervised, List.mem_cons, List.mem_nil_iff, or_false] at hpc
    rcases hpc with rfl | rfl
    · exact hiddenSwap_residueToken hc
    · exact h0

/-- **On a batch of equations that avoid the hidden residues the gradient commutes with the
exchange.** -/
theorem divisionGrad_comp_exchange {B : List Equation} (hB : ∀ x ∈ B, Avoids x)
    (θ : Param paperSizes → ℝ) :
    divisionGrad B (θ ∘ exchange) = divisionGrad B θ ∘ exchange := by
  funext p
  exact sourceGrad_comp_paramPerm (fun e he => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he
    exact fixed_supervised (hB x hx)) θ p

/-- **On an uncovered split the run from the exchanged initial parameters is the exchanged
run**, at every weight decay. -/
theorem trajectory_exchange (decay : ℝ) {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ uncovered)
    (θ₀ : Param paperSizes → ℝ) (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀ ∘ exchange, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ exchange :=
  (adamRun_comp exchange (fun s θ => divisionGrad_comp_exchange (fun x hx => by
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
    exact hπ i) θ) θ₀ t).1

/-- The exchange keeps the kind of a parameter, hence its initialization law. -/
lemma initLaw_exchange (p : Param paperSizes) :
    initLaw paperSizes (exchange p) = initLaw paperSizes p := by
  cases p <;> rfl

/-- **The exchange preserves the law of the initial parameters.** -/
theorem measurePreserving_comp_exchange :
    MeasurePreserving (fun θ : Param paperSizes → ℝ => θ ∘ exchange) (initMeasure paperSizes)
      (initMeasure paperSizes) := by
  have hm : Measurable fun θ : Param paperSizes → ℝ => θ ∘ exchange :=
    measurable_pi_iff.2 fun p => measurable_pi_apply (exchange p)
  refine ⟨hm, ?_⟩
  unfold initMeasure
  refine (Measure.pi_eq fun s hs => ?_).symm
  have hpre : (fun θ : Param paperSizes → ℝ => θ ∘ exchange) ⁻¹' Set.pi Set.univ s =
      Set.pi Set.univ fun p => s (exchange.symm p) := by
    ext θ
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, true_implies, Function.comp_apply]
    exact ⟨fun h p => by simpa using h (exchange.symm p), fun h p => by simpa using h (exchange p)⟩
  rw [Measure.map_apply hm (MeasurableSet.univ_pi hs), hpre, Measure.pi_pi,
    ← Equiv.prod_comp exchange]
  refine Finset.prod_congr rfl fun p _ => ?_
  rw [Equiv.symm_apply_apply, initLaw_exchange]

/-- The exchange of a sample: of its initial parameters. -/
def exchangeSample : Sample (97 * 96) 4656 (Param paperSizes) →
    Sample (97 * 96) 4656 (Param paperSizes) :=
  Prod.map id (Prod.map (fun θ => θ ∘ exchange) id)

/-- **The exchange preserves the law of a run.** -/
theorem measurePreserving_exchangeSample :
    MeasurePreserving exchangeSample (law (97 * 96) 4656 (initMeasure paperSizes))
      (law (97 * 96) 4656 (initMeasure paperSizes)) :=
  (MeasurePreserving.id _).prod (measurePreserving_comp_exchange.prod (MeasurePreserving.id _))

/-! The witnesses: the residue `3` is not hidden, and `5 / 4 = 50` avoids the hidden ones. -/

example : hiddenSwap (residueToken 3) = residueToken 3 := hiddenSwap_residueToken (by decide)

example : (supervised (50, 3)).Fixed hiddenSwap := fixed_supervised (by decide)

example (θ : Param paperSizes → ℝ) :=
  divisionGrad_comp_exchange (B := [(50, 3)]) (fun x hx => by
    rw [List.mem_singleton.1 hx]
    decide) θ

/-- Some split is uncovered, and on it the run from exchanged parameters is the exchanged run. -/
example (decay : ℝ) (θ₀ : Param paperSizes → ℝ) (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    ∃ π, trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀ ∘ exchange, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ exchange := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨π, trajectory_exchange decay hπ θ₀ σ t⟩

end Transformer.Grokking.Protocol

module

public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.DivTasks
public import Transformer.Grokking.FlatSpectrum.TablePart.E6DivParts

/-!
# PT6: the 15 tasks on the tables without the column `b = 0`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. At each of `p = 17`,
`97` and `193` the arithmetic data hold the five tasks `(a / b)^k`, `k = 0, 1, 2, 3, 5`, on a table
without the column `b = 0`. The kernel reads a certificate from each training mask
(`certificate17`, `certificate97`, `certificate193`). Inside the class the part then gives the
constant `(a / b)^0` on the whole table (`constant17`, `constant97`, `constant193`). For
`k = 1, 2, 3, 5` no function of the class fits the training cells (`no_function17`, `no_function97`,
`no_function193`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

attribute [local instance] fact_prime_97

local instance fact_prime_17 : Fact (Nat.Prime 17) := ⟨by norm_num⟩

local instance fact_prime_193 : Fact (Nat.Prime 193) := ⟨by norm_num⟩

/-- The training part at `p = 17` has a hub, no empty column and a four-cycle. -/
theorem certificate17 : DivCertificate 17 mask17 0 1 4 11 := by decide +kernel

/-- The training part at `p = 97` has a hub, no empty column and a four-cycle. -/
theorem certificate97 : DivCertificate 97 mask97 0 1 0 1 := by decide +kernel

/-- The training part at `p = 193` has a hub, no empty column and a four-cycle. -/
theorem certificate193 : DivCertificate 193 mask193 0 1 1 9 := by decide +kernel

/-- **Inside the class the half at `p = 17` gives `(a / b)^0` on the whole table**
(`review/flat_spectrum.md`, PT6). -/
theorem constant17 {f : (Fin 1 → ZMod 17) × (Fin 1 → ZMod 17) → ZMod 17}
    (hf : f ∈ flatClass (ZMod 17) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ divPart 17 mask17, f c = divTable 17 0 c) (u : Fin 1 → ZMod 17)
    {v : Fin 1 → ZMod 17} (hv : v ≠ 0) : f (u, v) = divTable 17 0 (u, v) :=
  constant_off_column certificate17 hf hag u hv

example : divTable 17 0 (0, 1) = divTable 17 0 (0, 1) :=
  constant17 divTable_zero_mem (fun _ _ => rfl) 0 one_ne_zero

/-- **Inside the class the half at `p = 97` gives `(a / b)^0` on the whole table**
(`review/flat_spectrum.md`, PT6). -/
theorem constant97 {f : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97}
    (hf : f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ divPart 97 mask97, f c = divTable 97 0 c) (u : Fin 1 → ZMod 97)
    {v : Fin 1 → ZMod 97} (hv : v ≠ 0) : f (u, v) = divTable 97 0 (u, v) :=
  constant_off_column certificate97 hf hag u hv

example : divTable 97 0 (0, 1) = divTable 97 0 (0, 1) :=
  constant97 divTable_zero_mem (fun _ _ => rfl) 0 one_ne_zero

/-- **Inside the class the half at `p = 193` gives `(a / b)^0` on the whole table**
(`review/flat_spectrum.md`, PT6). -/
theorem constant193 {f : (Fin 1 → ZMod 193) × (Fin 1 → ZMod 193) → ZMod 193}
    (hf : f ∈ flatClass (ZMod 193) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ divPart 193 mask193, f c = divTable 193 0 c) (u : Fin 1 → ZMod 193)
    {v : Fin 1 → ZMod 193} (hv : v ≠ 0) : f (u, v) = divTable 193 0 (u, v) :=
  constant_off_column certificate193 hf hag u hv

example : divTable 193 0 (0, 1) = divTable 193 0 (0, 1) :=
  constant193 divTable_zero_mem (fun _ _ => rfl) 0 one_ne_zero

/-- The rectangles at `p = 17` are training cells with the labels of their tasks. -/
theorem divOutside17_valid :
    divOutside17.all (fun w => divWitnessOk 17 mask17 w.2.1 w.2.2.1 w.2.2.2) = true := by
  decide +kernel

/-- The rectangles at `p = 97` are training cells with the labels of their tasks. -/
theorem divOutside97_valid :
    divOutside97.all (fun w => divWitnessOk 97 mask97 w.2.1 w.2.2.1 w.2.2.2) = true := by
  decide +kernel

/-- The rectangles at `p = 193` are training cells with the labels of their tasks. -/
theorem divOutside193_valid :
    divOutside193.all (fun w => divWitnessOk 193 mask193 w.2.1 w.2.2.1 w.2.2.2) = true := by
  decide +kernel

/-- **No function of the class fits `(a / b)^k` on the half at `p = 17`**, `k = 1, 2, 3, 5`
(`review/flat_spectrum.md`, PT6). -/
theorem no_function17 {k : ℕ} (hk : k ∈ [1, 2, 3, 5]) :
    ¬ ∃ f ∈ flatClass (ZMod 17) (Fin 1) (Fin 1), ∀ c ∈ divPart 17 mask17, f c = divTable 17 k c :=
  no_div_function_of_all divOutside17_valid hk

example : ¬ ∃ f ∈ flatClass (ZMod 17) (Fin 1) (Fin 1),
    ∀ c ∈ divPart 17 mask17, f c = divTable 17 1 c := no_function17 (by simp)

/-- **No function of the class fits `(a / b)^k` on the half at `p = 97`**, `k = 1, 2, 3, 5`
(`review/flat_spectrum.md`, PT6). -/
theorem no_function97 {k : ℕ} (hk : k ∈ [1, 2, 3, 5]) :
    ¬ ∃ f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1), ∀ c ∈ divPart 97 mask97, f c = divTable 97 k c :=
  no_div_function_of_all divOutside97_valid hk

example : ¬ ∃ f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1),
    ∀ c ∈ divPart 97 mask97, f c = divTable 97 1 c := no_function97 (by simp)

/-- **No function of the class fits `(a / b)^k` on the half at `p = 193`**, `k = 1, 2, 3, 5`
(`review/flat_spectrum.md`, PT6). -/
theorem no_function193 {k : ℕ} (hk : k ∈ [1, 2, 3, 5]) :
    ¬ ∃ f ∈ flatClass (ZMod 193) (Fin 1) (Fin 1),
      ∀ c ∈ divPart 193 mask193, f c = divTable 193 k c :=
  no_div_function_of_all divOutside193_valid hk

example : ¬ ∃ f ∈ flatClass (ZMod 193) (Fin 1) (Fin 1),
    ∀ c ∈ divPart 193 mask193, f c = divTable 193 1 c := no_function193 (by simp)

end Transformer.Grokking.FlatSpectrum.TablePart.E6

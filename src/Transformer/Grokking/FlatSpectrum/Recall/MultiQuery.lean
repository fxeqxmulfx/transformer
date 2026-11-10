module

public import Transformer.Grokking.FlatSpectrum.Recall.Width
public import Transformer.Zoology.Appendix_Attention

/-!
# Recall with one query is an input of the multi-query task

Source: `review/flat_spectrum.md`, *Families of dictionaries and the multi-query task* (added),
item FD5, for the task of Arora et al., arXiv:2312.04927v1, Appendix `sec: intro-general-ar`,
Setup: the input is a sequence of triples `(k_i, v_i, q_i)` of tokens, and "Our goal is then to
check, for each `1 ≤ i ≤ N/3 − 1`, whether there exists `0 ≤ j < i` such that `q_i ≡ k_j`, and
if so, output `v_j`."

A dictionary `π` of `m` keys and a query `q` are such an input of `m + 1` triples over `c > m`
tokens (`mqarOfDictionary`). The triple `j < m` is `(j, π j, j)`: it binds the key `j` and
asks for a key that no earlier triple binds (`not_priorAnswer_mqarOfDictionary`). The last
triple is `(m, m, q)`: a new key, and the query. The keys are distinct
(`uniqueKeys_mqarOfDictionary`), the first `m` triples depend on `π` alone and the last on `q`
alone, and the answer at the last triple is `π q` (`priorAnswer_mqarOfDictionary`).

So a model that answers the last query of every input with distinct keys recalls. If its
answer there has cut width `d` across the cut before the last triple, then `m ≤ d`
(`card_le_of_mqar`), and its squared error on these inputs is that of AR4
(`mqar_recall_error`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.Zoology

variable {m c : ℕ}

/-- **The input of a dictionary and a query** (`review/flat_spectrum.md`, FD5): `m + 1` triples
over a vocabulary of `c > m` tokens. The triple `j < m` is `(j, π j, j)`, and the last triple
is `(m, m, q)`. -/
def mqarOfDictionary (h : m < c) (π : Fin m → Fin c) (q : Fin m) : MQARInstance (m + 1) c where
  key := Fin.snoc (α := fun _ => Fin c) (fun j => Fin.castLE h.le j) ⟨m, h⟩
  value := Fin.snoc (α := fun _ => Fin c) π ⟨m, h⟩
  query := Fin.snoc (α := fun _ => Fin c) (fun j => Fin.castLE h.le j) (Fin.castLE h.le q)

variable (h : m < c) (π : Fin m → Fin c) (q : Fin m)

/-- A key of the dictionary is not the new key of the last triple. -/
lemma castLE_ne_mk (j : Fin m) : Fin.castLE h.le j ≠ ⟨m, h⟩ := fun h0 => by
  have h1 : (j : ℕ) = m := by simpa using congrArg Fin.val h0
  have h2 := j.isLt
  omega

example : Fin.castLE (Nat.le_of_lt (Nat.lt_succ_self 2)) (1 : Fin 2) ≠ ⟨2, Nat.lt_succ_self 2⟩ :=
  castLE_ne_mk (Nat.lt_succ_self 2) 1

/-- **The keys of the input are distinct** (`review/flat_spectrum.md`, FD5). -/
theorem uniqueKeys_mqarOfDictionary : UniqueKeys (mqarOfDictionary h π q) := by
  intro j k hjk
  induction j using Fin.lastCases with
  | last =>
    induction k using Fin.lastCases with
    | last => rfl
    | cast k =>
      simp only [mqarOfDictionary, Fin.snoc_last, Fin.snoc_castSucc] at hjk
      exact absurd hjk.symm (castLE_ne_mk h k)
  | cast j =>
    induction k using Fin.lastCases with
    | last =>
      simp only [mqarOfDictionary, Fin.snoc_last, Fin.snoc_castSucc] at hjk
      exact absurd hjk (castLE_ne_mk h j)
    | cast k =>
      simp only [mqarOfDictionary, Fin.snoc_castSucc] at hjk
      rw [Fin.castLE_injective h.le hjk]

/-- **The answer at the last triple is the bound value** (`review/flat_spectrum.md`, FD5). -/
theorem priorAnswer_mqarOfDictionary (v : Fin c) :
    PriorAnswer (mqarOfDictionary h π q) (Fin.last m) v ↔ v = π q := by
  constructor
  · rintro ⟨j, hj, hk, hv⟩
    obtain ⟨j', rfl⟩ := Fin.exists_castSucc_eq.2 hj.ne
    simp only [mqarOfDictionary, Fin.snoc_castSucc, Fin.snoc_last] at hk hv
    rw [← hv, Fin.castLE_injective h.le hk]
  · rintro rfl
    exact ⟨Fin.castSucc q, Fin.castSucc_lt_last q, by simp [mqarOfDictionary],
      by simp [mqarOfDictionary]⟩

/-- **The first `m` triples ask for keys that are not bound yet**
(`review/flat_spectrum.md`, FD5): queries without a match. -/
theorem not_priorAnswer_mqarOfDictionary (j : Fin m) (v : Fin c) :
    ¬ PriorAnswer (mqarOfDictionary h π q) (Fin.castSucc j) v := by
  rintro ⟨j', hj', hk, -⟩
  obtain ⟨i, rfl⟩ := Fin.exists_castSucc_eq.2 (hj'.trans (Fin.castSucc_lt_last j)).ne
  simp only [mqarOfDictionary, Fin.snoc_castSucc] at hk
  rw [Fin.castLE_injective h.le hk] at hj'
  exact lt_irrefl _ hj'

variable {h} {d : ℕ} {G : MQARInstance (m + 1) c → ℝ} {φ : Fin c → ℝ}

/-- **A model of the multi-query task recalls** (`review/flat_spectrum.md`, FD5): if the
answer `G` at the last triple is `φ` of the value bound to the query on every input with
distinct keys, and it has cut width `d` across the cut before the last triple, then `m ≤ d`. -/
theorem card_le_of_mqar {v v' : Fin c} (hφ : φ v ≠ φ v')
    (hG : HasCutWidth d fun (q : Fin m) (π : Fin m → Fin c) => G (mqarOfDictionary h π q))
    (hsolve : ∀ x : MQARInstance (m + 1) c, UniqueKeys x →
      ∀ u, PriorAnswer x (Fin.last m) u → G x = φ u) : m ≤ d := by
  simpa using hG.card_le_of_recall hφ fun q π =>
    hsolve _ (uniqueKeys_mqarOfDictionary h π q) _ ((priorAnswer_mqarOfDictionary h π q _).2 rfl)

/-- **The error at the last triple** (`review/flat_spectrum.md`, FD5): over the inputs of the
dictionaries and the queries, an answer of cut width `d` is at least
`(m − d) c^{m−1} ∑_v (φ(v) − μ)²` from `φ` of the bound value. -/
theorem mqar_recall_error {μ : ℝ} (hμ : ∑ v, (φ v - μ) = 0)
    (hG : HasCutWidth d fun (q : Fin m) (π : Fin m → Fin c) => G (mqarOfDictionary h π q)) :
    ((m : ℝ) - d) * ((c : ℝ) ^ (m - 1) * ∑ v, (φ v - μ) ^ 2) ≤
      ∑ q, ∑ π : Fin m → Fin c, (φ (π q) - G (mqarOfDictionary h π q)) ^ 2 := by
  simpa using hG.mul_le_sum_sq hμ

/-- One key over two tokens: the answer "the first value" has cut width one across the cut
before the last triple. -/
lemma hasCutWidth_firstValue : HasCutWidth 1 fun (q : Fin 1) (π : Fin 1 → Fin 2) =>
    (((mqarOfDictionary (c := 2) one_lt_two π q).value 0 : Fin 2) : ℝ) :=
  ⟨fun _ => 0, fun _ _ => 1, fun π _ => ((π 0 : Fin 2) : ℝ), fun q π => by
    show ((Fin.snoc (α := fun _ => Fin 2) π ⟨1, one_lt_two⟩ (Fin.castSucc (0 : Fin 1)) :
      Fin 2) : ℝ) = 0 + ∑ _r : Fin 1, 1 * ((π 0 : Fin 2) : ℝ)
    rw [Fin.snoc_castSucc]
    simp⟩

/-- The model that reads the first value answers the last query of every input of two
triples, within `m ≤ d` at `m = d = 1`. -/
example : (1 : ℕ) ≤ 1 :=
  card_le_of_mqar (h := one_lt_two) (G := fun x => (x.value 0 : ℝ))
    (φ := fun u => (u : ℝ)) (v := 0) (v' := 1) (by norm_num) hasCutWidth_firstValue
    fun x _ u ⟨j, hj, _, hv⟩ => by
      obtain rfl : j = 0 := by
        have h1 : j.val < 1 := hj
        exact Fin.ext (by simpa using h1)
      rw [hv]

example := mqar_recall_error (h := one_lt_two) (G := fun x => (x.value 0 : ℝ))
  (φ := fun u => (u : ℝ)) (μ := 1 / 2) (by simp [Fin.sum_univ_two]) hasCutWidth_firstValue

/-- Two keys over three tokens, every key bound to the token `0`, and the query `1`. -/
example : PriorAnswer (mqarOfDictionary (m := 2) (c := 3) (by norm_num) (fun _ => 0) 1)
    (Fin.last 2) 0 :=
  (priorAnswer_mqarOfDictionary _ _ _ _).2 rfl

example := uniqueKeys_mqarOfDictionary (m := 2) (c := 3) (by norm_num) (fun _ => 0) 1

example := not_priorAnswer_mqarOfDictionary (m := 2) (c := 3) (by norm_num) (fun _ => 0) 1 0 0

end Transformer.Grokking.FlatSpectrum.Recall

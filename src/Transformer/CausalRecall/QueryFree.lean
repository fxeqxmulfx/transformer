module

public import Transformer.CausalRecall.MQAR

/-!
# Attention that ignores the query reads the past through summaries

In no paper. `Transformer.CRASP.QueryFree` shows that a future-masked rounded
transformer (arXiv:2506.16055v3, Appendix B.1, `def:transformer`) whose
attention ignores the query (`RTfr.QueryFree`) computes its last activation on
`⊲ · u · b` from `b` and a message about `u`: per layer `2d + 1` integers, each
one of `2^p (|u| + 1) + 1` (`RTfr.actAt_last`, `RTfr.msg_mem`). So it is a
model of `Transformer.CausalRecall.Summary` on contexts of `n` tokens
(`queryFreeModel`): a state holds a token and an activation, the summary of
layer `ℓ` codes the message of layer `ℓ` about the tokens before the position,
one of `(2^p (n + 1) + 1)^{2d + 1}` (`card_msgCodes`), and the position takes
Bob's step (`RTfr.bob`). The model accepts `u · b` exactly when the
transformer accepts `⊲ · u · b` (`accepts_queryFreeModel_iff`), so also on
Zoology's aligned words (`accepts_word_queryFreeModel_iff`); its `k` summaries
bound recall by `(2^p (n + 1) + 1)^{k (2d + 1)}`, the bound of
`Transformer.CRASP.two_pow_le_of_queryFree`, which the class bound
`Model.two_pow_le_of_recall` gives again.
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset Zoology CRASP

/-- Whether a transformer accepts a word is decidable, classically: its output
is a real number. -/
noncomputable instance decidableAcceptsRTfr {τ : Type*} {p s d k : ℕ} (T : RTfr τ p s d k)
    (w : List τ) : Decidable (T.Accepts w) :=
  inferInstanceAs (Decidable (0 < (T.out w).val))

variable {σ : Type*} {p s d k : ℕ}

/-- An integer of a message about `n` tokens at `p` bits, shifted by
`(n + 1) 2^{p-1}` into `[0, 2^p (n + 1)]` (`RTfr.msg_mem`). -/
def msgCode (p n : ℕ) (z : ℤ) : Fin (2 ^ p * (n + 1) + 1) :=
  ⟨(z + (n + 1) * 2 ^ (p - 1)).toNat % (2 ^ p * (n + 1) + 1), Nat.mod_lt _ (Nat.succ_pos _)⟩

/-- The integer that the code `r` stands for. -/
def msgDecode (p n : ℕ) (r : Fin (2 ^ p * (n + 1) + 1)) : ℤ := r.val - (n + 1) * 2 ^ (p - 1)

theorem msgDecode_msgCode {p n : ℕ} {z : ℤ} (h₀ : 0 ≤ z + (n + 1) * 2 ^ (p - 1))
    (h₁ : z + (n + 1) * 2 ^ (p - 1) < 2 ^ p * (n + 1) + 1) :
    msgDecode p n (msgCode p n z) = z := by
  have h₂ : (z + (n + 1) * 2 ^ (p - 1)).toNat < 2 ^ p * (n + 1) + 1 :=
    (Int.toNat_lt h₀).mpr (by push_cast; exact h₁)
  simp only [msgDecode, msgCode, Nat.mod_eq_of_lt h₂, Int.toNat_of_nonneg h₀]
  ring

/-- The hypotheses of `msgDecode_msgCode` are satisfiable: `0` at one bit. -/
example : msgDecode 1 0 (msgCode 1 0 0) = 0 := msgDecode_msgCode (by norm_num) (by norm_num)

/-- The summaries of `queryFreeModel` on contexts of `n` tokens at `p` bits: a
code for each of the `2d + 1` parts of a layer. -/
abbrev MsgCodes (d p n : ℕ) : Type := Option (Fin d ⊕ Fin d) → Fin (2 ^ p * (n + 1) + 1)

theorem card_msgCodes (d p n : ℕ) :
    Fintype.card (MsgCodes d p n) = (2 ^ p * (n + 1) + 1) ^ (2 * d + 1) := by
  simp only [MsgCodes, Fintype.card_fun, Fintype.card_fin, Fintype.card_option,
    Fintype.card_sum]
  rw [two_mul]

/-- Layer `ℓ` of `queryFreeModel`: a state holds a token and an activation; the
summary codes the message of layer `ℓ` about the tokens before the position,
and the position `i` takes Bob's step over the `i + 2` positions of `⊲` and
the tokens up to it. -/
noncomputable def queryFreeLayer (T : RTfr (Option σ) p s d k) (n ℓ : ℕ) :
    Layer (σ × (Fin d → Fx p s)) (MsgCodes d p n) where
  read zs o := msgCode p n (T.msg (zs.map Prod.fst) ℓ o)
  combine i z r := (z.1, T.mix ℓ (fun o => (msgDecode p n (r o) : ℝ) / 2 ^ s +
    (T.part ℓ o z.2).val) (i + 2) z.2)

/-- The layers `ℓ, …, ℓ + j - 1` of `queryFreeModel`. -/
noncomputable def queryFreeLayers (T : RTfr (Option σ) p s d k) (n : ℕ) :
    ℕ → ℕ → List (Layer (σ × (Fin d → Fx p s)) (MsgCodes d p n))
  | _, 0 => []
  | ℓ, j + 1 => queryFreeLayer T n ℓ :: queryFreeLayers T n (ℓ + 1) j

/-- **A transformer whose attention ignores the query, as a model of the
class**, on contexts of `n` tokens: a token enters with its embedding, the `k`
layers are those of `queryFreeLayer`, and the output is whether `W_out` is
positive. -/
noncomputable def queryFreeModel (T : RTfr (Option σ) p s d k) (n : ℕ) :
    Model σ (σ × (Fin d → Fx p s)) (MsgCodes d p n) Bool where
  embed _ x := (x, T.E (some x))
  layers := queryFreeLayers T n 0 k
  out z := decide (0 < (T.Wout z.2).val)

theorem length_queryFreeLayers (T : RTfr (Option σ) p s d k) (n ℓ j : ℕ) :
    (queryFreeLayers T n ℓ j).length = j := by
  induction j generalizing ℓ with
  | zero => rfl
  | succ j ih => simp [queryFreeLayers, ih]

@[simp] theorem fst_combine_queryFreeLayer (T : RTfr (Option σ) p s d k) (n ℓ i : ℕ)
    (z : σ × (Fin d → Fx p s)) (r : MsgCodes d p n) :
    ((queryFreeLayer T n ℓ).combine i z r).1 = z.1 := rfl

/-- A layer of `queryFreeModel` keeps the tokens. -/
theorem map_fst_apply_queryFreeLayer (T : RTfr (Option σ) p s d k) (n ℓ : ℕ)
    (zs : List (σ × (Fin d → Fx p s))) :
    ((queryFreeLayer T n ℓ).apply zs).map Prod.fst = zs.map Prod.fst := by
  suffices h : ∀ pre, ((queryFreeLayer T n ℓ).scan pre zs).map Prod.fst = zs.map Prod.fst from
    h []
  induction zs with
  | nil => exact fun _ => rfl
  | cons z zs ih => exact fun pre => by simp [Layer.scan, ih]

/-- **The layers of `queryFreeModel` take Bob's steps**: after a context of
tokens `u`, `|u| = n`, the layers `ℓ, …, ℓ + j - 1` lead the token `b` from
Bob's activation at depth `ℓ` to his activation at depth `ℓ + j`. -/
theorem finish_queryFreeLayers (T : RTfr (Option σ) p s d k) (u : List σ) {n : ℕ}
    (hu : u.length = n) (b : σ) (j ℓ : ℕ) (zs : List (σ × (Fin d → Fx p s)))
    (hzs : zs.map Prod.fst = u) :
    finish (queryFreeLayers T n ℓ j) n (summaries (queryFreeLayers T n ℓ j) zs)
        (b, T.bob n (T.msg u) b ℓ) = (b, T.bob n (T.msg u) b (ℓ + j)) := by
  induction j generalizing ℓ zs with
  | zero => simp [queryFreeLayers, summaries, finish]
  | succ j ih =>
    have hM (o : Option (Fin d ⊕ Fin d)) :
        msgDecode p n (msgCode p n (T.msg u ℓ o)) = T.msg u ℓ o := by
      have h := T.msg_mem u ℓ o
      rw [hu] at h
      exact msgDecode_msgCode h.1 h.2
    have hstep : (queryFreeLayer T n ℓ).combine n (b, T.bob n (T.msg u) b ℓ)
        ((queryFreeLayer T n ℓ).read zs) = (b, T.bob n (T.msg u) b (ℓ + 1)) := by
      simp only [queryFreeLayer, hzs, hM]
      rfl
    simp only [queryFreeLayers, summaries, finish, hstep]
    rw [ih (ℓ + 1) _ (by rw [map_fst_apply_queryFreeLayer, hzs]),
      show ℓ + 1 + j = ℓ + (j + 1) by omega]

/-- The hypotheses of `finish_queryFreeLayers` are satisfiable: a context of one
token. -/
example (T : RTfr (Option σ) p s d k) (a b : σ) :
    finish (queryFreeLayers T 1 0 k) 1 (summaries (queryFreeLayers T 1 0 k) [(a, T.E (some a))])
      (b, T.bob 1 (T.msg [a]) b 0) = (b, T.bob 1 (T.msg [a]) b (0 + k)) :=
  finish_queryFreeLayers T [a] rfl b k 0 _ rfl

/-- **The model decides as the transformer**: `queryFreeModel T n` accepts
`u · b`, `|u| = n`, exactly when `T` accepts `⊲ · u · b`. -/
theorem accepts_queryFreeModel_iff [Fintype σ] [DecidableEq σ] (T : RTfr (Option σ) p s d k)
    (hT : T.QueryFree) (u : List σ) {n : ℕ} (hu : u.length = n) (b : σ) :
    (queryFreeModel T n).Accepts (u ++ [b]) ↔ T.Accepts (bos (u ++ [b])) := by
  subst hu
  have h := finish_queryFreeLayers T u rfl b k 0 (u.mapIdx fun _ x => (x, T.E (some x)))
    (map_fst_mapIdx _ u)
  rw [zero_add] at h
  rw [Model.Accepts, Model.getLast?_outputs_append_singleton, Option.some_inj, RTfr.Accepts,
    RTfr.out_bos, List.length_append, List.length_singleton, T.actAt_last hT u b k]
  change decide (0 < (T.Wout (finish (queryFreeLayers T u.length 0 k) u.length
    (summaries (queryFreeLayers T u.length 0 k) (u.mapIdx fun _ x => (x, T.E (some x))))
    (b, T.bob u.length (T.msg u) b 0)).2).val) = true ↔ _
  rw [h, decide_eq_true_iff]

/-- The hypotheses of `accepts_queryFreeModel_iff` hold for the transformer of
`thm:TLCl_to_rtfr`, which ignores the query. -/
example [Fintype σ] [DecidableEq σ] (φ : Form σ) (u : List σ) (b : σ) :
    (queryFreeModel (TemporalProgram.model φ k) u.length).Accepts (u ++ [b]) ↔
      (TemporalProgram.model φ k).Accepts (bos (u ++ [b])) :=
  accepts_queryFreeModel_iff _ (RTfr.queryFree_model φ k) u rfl b

/-- On Zoology's aligned words of `n + 1` rows, `queryFreeModel T n` decides as
`T` on `⊲ · w`. -/
theorem accepts_word_queryFreeModel_iff {c n : ℕ}
    (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k) (hT : T.QueryFree)
    (x : MQARInstance (n + 1) c) :
    (queryFreeModel T n).Accepts (word x) ↔ T.Accepts (bos (word x)) := by
  rw [word, List.ofFn_succ', List.concat_eq_append]
  exact accepts_queryFreeModel_iff T hT _ List.length_ofFn _

/-- The hypothesis of `accepts_word_queryFreeModel_iff` holds for the
transformer of `thm:TLCl_to_rtfr`. -/
example {c n : ℕ} (φ : Form (Fin c × Fin c × Fin c)) (x : MQARInstance (n + 1) c) :
    (queryFreeModel (TemporalProgram.model φ k) n).Accepts (word x) ↔
      (TemporalProgram.model φ k).Accepts (bos (word x)) :=
  accepts_word_queryFreeModel_iff _ (RTfr.queryFree_model φ k) x

/-- **The class bound gives `Transformer.CRASP.two_pow_le_of_queryFree`
again**: the `k` summaries of `queryFreeModel T n` take
`(2^p (n + 1) + 1)^{2d + 1}` values each. -/
example {c n : ℕ} (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k) (hT : T.QueryFree)
    (v : Fin c) (hn : c ≤ n + 1)
    (hrec : ∀ x : MQARInstance (n + 1) c, Consistent x →
      (T.Accepts (bos (word x)) ↔ PriorAnswer x (Fin.last n) v)) :
    2 ^ (c - 1) ≤ (2 ^ p * (n + 1) + 1) ^ (k * (2 * d + 1)) := by
  have h := (queryFreeModel T n).two_pow_le_of_recall v hn fun x hx =>
    (accepts_word_queryFreeModel_iff T hT x).trans (hrec x hx)
  have hL : (queryFreeModel T n).layers.length = k := length_queryFreeLayers T n 0 k
  rwa [card_msgCodes, hL, ← pow_mul, mul_comm (2 * d + 1)] at h

end Transformer.CausalRecall

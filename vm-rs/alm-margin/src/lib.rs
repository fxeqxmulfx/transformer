//! How far a hard-attention query may miss its integer value and still
//! retrieve the same key.
//!
//! `ALM.DriftMargin` answers this with `order_survives_drift`: a query `eps`
//! off its integer value moves the score of a key bounded by `K` by at most
//! `2*K*eps`, distinct integer keys are a whole unit apart, so `4*K*eps < 1`
//! keeps the winner.  That bound is sound, and it is enormously conservative
//! -- at the `K = 10^7` the file's own closing example uses, it asks for
//! `eps < 2.5e-8` where the true margin is `eps < 0.5`, a factor of
//! twenty million.
//!
//! `drift` measures the true one.  The score difference between two integer
//! keys at an integer query factors as `(j - k) * (j + k - 2q)`, and a drift
//! `d` changes it by `2*(k - j)*d`; the common factor `(j - k)` cancels, so
//! the winner flips exactly at `d = |j + k - 2q| / 2`.  On integers that
//! quantity is a non-zero integer unless the two keys are symmetric about the
//! query -- the tie `ALM.DriftMargin.symmetric_tie_broken_by_drift` already
//! isolates -- so the margin is `1/2`, and `K` is not in it.
//!
//! `ceiling` is the constraint that does bite, and it is not about the query
//! at all.  The score squares the key, so a format with `m` mantissa bits
//! holds the scores of keys up to `ceil(2^(m/2))` and no further.  For
//! float64 that is `94 906 266`, the wall `alm_hull::grid` already names.  For
//! float32 it is `4096`, and nothing in the Lean or in either reference
//! runtime says so.
//!
//! The two together decide whether an exact head can be an ordinary layer of a
//! transformer: float32 is enough for the query (the margin is `1/2`, and one
//! rounding of an address is far below it) and nowhere near enough for the
//! score (`4096` addresses).  Rebasing the keys to the current position is
//! what reconciles them, and it is `KEY_OFFSET` in
//! `transformer_vm/graph/core.py:9`, a live parameter set to zero.

pub mod ceiling;
pub mod drift;
pub mod rewrite;

pub use ceiling::{first_failing_key_f32, score_wall, wall_f32, wall_f64, Format};
pub use drift::{flip_point, margin_of, winner_at, Margin, DRIFT_MARGIN};
pub use rewrite::{capacity, last_resolving_position, levels, recency_span, RECENCY_ALPHA};

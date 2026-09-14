//! A 2D hard-attention key-value cache, answering `argmax_k q . k` in
//! `O(log n)` per query.
//!
//! A port of `transformer_vm/attention/hull2d_cht.h` from the Percepta
//! `transformer-vm` release, with the breakpoint arithmetic rebuilt: the
//! original stores every hull breakpoint as a rounded `long double`, a type
//! Rust does not have, so breakpoints here are exact rationals compared by
//! cross-multiplication.  See `breakpoint` and `exact`.

pub mod breakpoint;
pub mod cht;
pub mod exact;
pub mod grid;
pub mod head;
pub mod meta;

pub use breakpoint::Break;
pub use grid::{grid_ratio, off_the_grid, ulp, Crossing, GridWitness, GRID_LIMIT};
pub use head::{BruteAttentionHead, HardAttentionHead, HullHalf};
pub use meta::{HullMeta, TieBreak};

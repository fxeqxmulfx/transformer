//! Compact complex FFN storage, with the retained coordinates drawn from
//! exactly the same seeded full-width model. See Transformer.Phasor.CompactFFN.
use crate::{Config, head::Head, layout};
use std::ops::Range;

pub fn hidden(cfg: &Config) -> usize {
    if cfg.head == Head::CompactStream {
        2 * cfg.width
    } else {
        layout::hidden(cfg.width, &cfg.architecture)
    }
}

pub fn base_count(cfg: &Config) -> usize {
    let width = cfg.width;
    cfg.vocab * width + cfg.layers * (4 * width * width + 3 * width * hidden(cfg))
}

pub fn wide_config(cfg: &Config) -> Config {
    let mut wide = cfg.clone();
    wide.head = Head::Stream;
    wide
}

// Each pair of slices occupies the same mathematical coordinates. Full tensors
// store the real and imaginary halves separately for both complex input maps.
fn retained(compact: &Config, wide: &Config, mut copy: impl FnMut(Range<usize>, Range<usize>)) {
    let d = compact.width;
    let (small, large) = (hidden(compact) / 2, hidden(wide) / 2);
    let mut c = compact.vocab * d;
    let mut w = c;
    copy(0..c, 0..w);
    for _ in 0..compact.layers {
        let attention = 4 * d * d;
        copy(c..c + attention, w..w + attention);
        c += attention;
        w += attention;
        // Gate real, gate imaginary, value real, value imaginary.
        for _ in 0..4 {
            copy(c..c + small * d, w..w + small * d);
            c += small * d;
            w += large * d;
        }
        // In every output row, retain the same complex units in both halves.
        for _ in 0..2 * d {
            copy(c..c + small, w..w + small);
            c += small;
            w += large;
        }
    }
    // Both models have identical router, rotary, score and stream-read layouts.
    copy(
        c..layout::total_count(compact),
        w..layout::total_count(wide),
    );
}

pub fn restrict(cfg: &Config, wide: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; layout::total_count(cfg)];
    retained(cfg, &wide_config(cfg), |c, w| {
        out[c].copy_from_slice(&wide[w])
    });
    out
}

pub fn pad(cfg: &Config, compact: &[f32]) -> Vec<f32> {
    let wide = wide_config(cfg);
    let mut out = vec![0.; layout::total_count(&wide)];
    retained(cfg, &wide, |c, w| out[w].copy_from_slice(&compact[c]));
    out
}

pub fn initialize(cfg: &Config, seed: u32) -> Vec<f32> {
    let wide = layout::initialize_model(&wide_config(cfg), seed);
    restrict(cfg, &wide)
}

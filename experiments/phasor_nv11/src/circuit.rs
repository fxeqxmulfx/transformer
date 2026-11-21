//! Encode the fitted CM121 shifts and the bilinear term into ordinary Phasor weights.
use crate::{Config, Result, layout};
use arithmetic::dataset::Training;
use std::f64::consts::{LN_2, TAU};
use table_part::solve::{Completion, complete};

pub const P: usize = 97;
const WIDTH: usize = 128;
const HEADS: usize = 4;
const BITS: usize = 7;

// Complex coordinates in the residual stream. Scratch coordinates have zero embeddings.
const BIAS: usize = 0;
const ROW_POWERS: usize = 1;
const ROW_SHIFT: usize = 8;
const COL_SHIFT: usize = 9;
const COL_BITS: usize = 10;
const ANSWER: usize = 17;
const COPIED: usize = 18; // seven powers, the row shift, then 1
const LEAVES: usize = 27; // seven selected powers and the product of the two shifts
const PAIRS: usize = 35;
const QUARTETS: usize = 39;

const SIDE: f32 = 1.0 / 32.0;
const CONSTANT: f32 = 256.0;
const EMBEDDING_SCALE: f32 = 64.0;

pub fn config() -> Config {
    Config {
        width: WIDTH,
        heads: HEADS,
        layers: 4,
        vocab: P,
        gain: HEADS as f32 / WIDTH as f32,
        architecture: "complex-phase-score-lerope".into(),
        phase_optimizer: false,
    }
}

/// Real and imaginary indices in the native head-grouped storage.
fn components(slot: usize, width: usize, groups: usize) -> (usize, usize) {
    let bands = width / (2 * groups);
    let real = slot / bands * 2 * bands + slot % bands;
    (real, real + bands)
}

/// One native complex-linear projection. A real coefficient goes on both diagonal blocks.
struct Projection<'a> {
    weights: &'a mut [f32],
    rows: usize,
    cols: usize,
    out_groups: usize,
    in_groups: usize,
}

impl Projection<'_> {
    fn add(&mut self, dst: usize, src: usize, weight: f32) {
        let (r, i) = components(dst, self.rows, self.out_groups);
        let (c, j) = components(src, self.cols, self.in_groups);
        self.weights[r * self.cols + c] += weight;
        self.weights[i * self.cols + j] += weight;
    }
}

fn embedding_phase(row: &mut [f32], slot: usize, residue: usize, magnitude: f32) {
    let theta = TAU * (residue % P) as f64 / P as f64;
    let (real, imag) = components(slot, WIDTH, HEADS);
    row[real] = EMBEDDING_SCALE * magnitude * theta.cos() as f32;
    row[imag] = EMBEDDING_SCALE * magnitude * theta.sin() as f32;
}

fn embeddings(weights: &mut [f32], m: usize) {
    for (token, row) in weights.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        embedding_phase(row, BIAS, 0, CONSTANT);
        embedding_phase(row, ANSWER, token, 1.0);
        embedding_phase(row, ROW_SHIFT, 0, SIDE);
        embedding_phase(row, COL_SHIFT, 0, SIDE);
        for bit in 0..BITS {
            embedding_phase(row, ROW_POWERS + bit, (m * token) << bit, SIDE);
            let sign = if token & (1 << bit) == 0 { -SIDE } else { SIDE };
            embedding_phase(row, COL_BITS + bit, 0, sign);
        }
    }
}

/// Head 0 reads a constant query and key on rotary band 0. Head 1 is uniform.
/// Inverting their two weights at position 1 isolates the preceding token.
fn copy_first_operand(qkv: &mut Projection, output: &mut Projection) {
    let head = WIDTH / (2 * HEADS);
    let key = WIDTH / 2;
    let value = WIDTH;
    qkv.add(0, BIAS, 1.0);
    qkv.add(key, BIAS, 1.0);
    // Unit head RMS gives the first key weight 1/(1 + 2^(1 - cos(1))).
    // Its epsilon correction is below FP32 precision at this bias magnitude.
    let first = 1.0 / (1.0 + (LN_2 * (1.0 - f64::from(1.0_f32.cos()))).exp());
    let decoder = (1.0 / (2.0 * first - 1.0)) as f32;
    for (i, src) in (ROW_POWERS..=ROW_SHIFT).chain([BIAS]).enumerate() {
        let gain = if src == BIAS { SIDE / CONSTANT } else { 1.0 };
        for offset in [0, head] {
            qkv.add(value + offset + i, src, 128.0 * gain);
        }
        output.add(COPIED + i, i, decoder);
        output.add(COPIED + i, head + i, 1.0 - decoder);
    }
}

/// The two units for bit j write either the j-th row phase or the constant 1.
/// The eighth leaf multiplies the row and column phases, implementing the class shift.
fn write_leaves(fc: &mut Projection, out: &mut Projection, units: usize) {
    let gate_scale = 256.0;
    for bit in 0..BITS {
        for (unit, sign, src) in [
            (2 * bit, 1.0, COPIED + bit),
            (2 * bit + 1, -1.0, COPIED + BITS + 1),
        ] {
            fc.add(unit, BIAS, gate_scale * SIDE / CONSTANT);
            fc.add(unit, COL_BITS + bit, sign * gate_scale);
            fc.add(units + unit, src, 1.0);
            out.add(LEAVES + bit, unit, 1.0);
        }
    }
    let unit = 2 * BITS;
    fc.add(unit, COL_SHIFT, 2.0 * gate_scale);
    fc.add(units + unit, COPIED + BITS, 1.0);
    out.add(LEAVES + BITS, unit, 1.0);
}

/// Initial weights for the interaction with coupling m and identity phase shifts.
pub fn parameters(m: usize) -> Vec<f32> {
    let cfg = config();
    let mut weights = vec![0.0; layout::total_count(&cfg)];
    let hidden = layout::hidden(WIDTH, &cfg.architecture);
    let (embedding, layers) = weights.split_at_mut(P * WIDTH);
    embeddings(embedding, m);
    let layer_size = 4 * WIDTH * WIDTH + 3 * WIDTH * hidden;
    // Zero router weights, biases and phases preserve four equal streams.
    for (layer, block) in layers
        .chunks_exact_mut(layer_size)
        .take(cfg.layers)
        .enumerate()
    {
        let (qkv, block) = block.split_at_mut(3 * WIDTH * WIDTH);
        let (attention_out, block) = block.split_at_mut(WIDTH * WIDTH);
        let (fc, ffn_out) = block.split_at_mut(2 * hidden * WIDTH);
        if layer == 0 {
            copy_first_operand(
                &mut Projection {
                    weights: qkv,
                    rows: 3 * WIDTH,
                    cols: WIDTH,
                    out_groups: 3 * HEADS,
                    in_groups: HEADS,
                },
                &mut Projection {
                    weights: attention_out,
                    rows: WIDTH,
                    cols: WIDTH,
                    out_groups: HEADS,
                    in_groups: HEADS,
                },
            );
        }
        let mut fc = Projection {
            weights: fc,
            rows: 2 * hidden,
            cols: WIDTH,
            out_groups: 2,
            in_groups: HEADS,
        };
        let mut out = Projection {
            weights: ffn_out,
            rows: WIDTH,
            cols: hidden,
            out_groups: HEADS,
            in_groups: 1,
        };
        let units = hidden / 2;
        if layer == 0 {
            write_leaves(&mut fc, &mut out, units);
        } else {
            let (src, dst, count) = [
                (LEAVES, PAIRS, 4),
                (PAIRS, QUARTETS, 2),
                (QUARTETS, ANSWER, 1),
            ][layer - 1];
            for unit in 0..count {
                fc.add(unit, src + 2 * unit, 1.0);
                fc.add(units + unit, src + 2 * unit + 1, 1.0);
                out.add(dst + unit, unit, 1.0);
            }
        }
    }
    weights
}

pub fn validate_training(training: &Training) -> Result<()> {
    if training.vocabulary != P || training.rows.iter().flatten().any(|&v| v >= P as u32) {
        return Err("expected triples of residues modulo 97".into());
    }
    Ok(())
}

/// Fit the full class g(a) + h(b) + m a b mod 97 from the observed token triples.
pub fn fit_class(training: &Training) -> Result<Vec<f32>> {
    validate_training(training)?;
    match complete(P as u32, &training.rows, &[true; 2 * P]) {
        Completion::NoMember => Err("training labels contradict the class".into()),
        Completion::NotDetermined => Err("training cells do not determine the class".into()),
        Completion::Determined { m, value } => {
            let (rows, negatives) = value.split_at(P);
            let columns: Vec<_> = negatives
                .iter()
                .map(|&x| (P as u32 - x) % P as u32)
                .collect();
            let mut weights = parameters(m as usize);
            set_shifts(&mut weights, rows, &columns);
            layout::validate(&config(), &weights)?;
            Ok(weights)
        }
    }
}

/// Write one phase per operand into its own embedding plane.
pub fn set_shifts(weights: &mut [f32], rows: &[u32], columns: &[u32]) {
    let embedding = weights[..P * WIDTH].as_chunks_mut::<WIDTH>().0;
    for (token, row) in embedding.iter_mut().enumerate() {
        embedding_phase(row, ROW_SHIFT, rows[token] as usize, SIDE);
        embedding_phase(row, COL_SHIFT, columns[token] as usize, SIDE);
    }
}

//! Strict greedy accuracy from model logits. A strictly correct next token
//! at each true prefix generates the whole reference sequence by induction.
use crate::{Result, engine::Engine, metrics::Packed};

pub(crate) struct RowMargin {
    pub gap: f64,
    pub ties: usize,
    pub labels: usize,
}

pub(crate) fn rows(engine: &mut Engine, packed: &Packed) -> Result<Vec<RowMargin>> {
    let logits = masked_logits(engine, packed)?;
    rows_from_logits(&logits, engine.config.vocab, packed)
}

pub(crate) fn masked_logits(engine: &mut Engine, packed: &Packed) -> Result<Vec<f32>> {
    let vocab = engine.config.vocab;
    let masked = vec![u32::MAX; packed.labels.len()];
    engine.forward(
        &packed.tokens,
        &masked,
        packed.tokens.len() / packed.length,
        packed.length,
    )?;
    let logits = engine
        .logits
        .part(..vocab * masked.len())
        .ok_or("logit capacity")?
        .read()?;
    Ok(logits)
}

pub(crate) fn rows_from_logits(
    logits: &[f32],
    vocab: usize,
    packed: &Packed,
) -> Result<Vec<RowMargin>> {
    if logits.len() != vocab * packed.labels.len() {
        return Err("logit count does not match the packed labels".into());
    }
    let mut rows = Vec::new();
    for (row, targets) in logits
        .chunks_exact(vocab * packed.length)
        .zip(packed.labels.chunks_exact(packed.length))
    {
        let mut out = RowMargin {
            gap: f64::INFINITY,
            ties: 0,
            labels: 0,
        };
        for (z, &target) in row.chunks_exact(vocab).zip(targets) {
            let y = target as usize;
            if y >= vocab {
                continue;
            }
            if z.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite scoring logit".into());
            }
            let best = z
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != y)
                .map(|(_, z)| *z)
                .fold(f32::NEG_INFINITY, f32::max);
            let delta = f64::from(z[y]) - f64::from(best);
            out.gap = out.gap.min(delta);
            out.ties += usize::from(delta == 0.0);
            out.labels += 1;
        }
        if out.labels == 0 {
            return Err("unlabeled evaluation row".into());
        }
        rows.push(out);
    }
    Ok(rows)
}

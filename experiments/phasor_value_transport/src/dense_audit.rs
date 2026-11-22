//! Label-masked forward passes for independently auditing saved dense heads.
use super::*;

pub(super) struct Scores {
    pub logits: Vec<f32>,
    pub predictions: Vec<usize>,
    pub correct: usize,
}

/// Return logits and predictions in original vocabulary order. The callback
/// can inspect a completed forward before the next batch reuses its buffers.
pub(super) fn score(
    engine: &mut Engine,
    rows: &[[u32; 3]],
    names: &[usize],
    mut inspect: impl FnMut(&Engine) -> Result<()>,
) -> Result<Scores> {
    let vocab = names.len();
    let mut inverse = vec![0; vocab];
    for (i, &j) in names.iter().enumerate() {
        inverse[j] = i;
    }
    let mut result = Scores {
        logits: Vec::with_capacity(rows.len() * vocab),
        predictions: Vec::with_capacity(rows.len()),
        correct: 0,
    };
    for examples in rows.chunks(BATCH) {
        let tokens: Vec<_> = examples
            .iter()
            .flat_map(|&[a, b, _]| [names[a as usize] as u32, names[b as usize] as u32])
            .collect();
        engine.forward(&tokens, &vec![u32::MAX; tokens.len()], examples.len(), 2)?;
        inspect(engine)?;
        let scores = engine.logits.read()?;
        let predicted = engine.predictions.read()?;
        for (i, example) in examples.iter().enumerate() {
            let at = (2 * i + 1) * vocab;
            let y = inverse[predicted[2 * i + 1] as usize];
            result.correct += usize::from(y == example[2] as usize);
            result.predictions.push(y);
            result.logits.extend(names.iter().map(|&j| scores[at + j]));
        }
    }
    Ok(result)
}

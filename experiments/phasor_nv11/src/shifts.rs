//! Maximum likelihood for CM121 row translations of a multiplication Phasor's logits.
use crate::{P, Result, answer_logits, circuit, config, forward_pairs, forward_weights, layout};
use arithmetic::dataset::Training;
use rayon::prelude::*;

pub fn fit(training: &Training) -> Result<Vec<f32>> {
    circuit::validate_training(training)?;
    let mut rows = vec![Vec::new(); P];
    for &triple in &training.rows {
        rows[triple[0] as usize].push(triple);
    }
    if rows.iter().any(Vec::is_empty) {
        return Err("an unobserved operand has no fitted row shift".into());
    }
    let cfg = config();
    let mut weights = circuit::parameters(1);
    let base = forward_weights(&weights);
    let offsets: Vec<_> = rows
        .par_iter()
        .map(|examples| {
            let mut scores = [0.0; P];
            let output = forward_pairs(&cfg, &base, examples)?;
            for (logits, &[_, _, label]) in answer_logits(&output).zip(examples) {
                // Shifting classes preserves the softmax denominator. Subtracting the
                // same reference score from every candidate also removes the common bias.
                for (shift, score) in scores.iter_mut().enumerate() {
                    let source = (label as usize + P - shift) % P;
                    *score += logits[source].v - logits[0].v;
                }
            }
            let best = scores.iter().copied().reduce(f64::max).unwrap();
            let mut winners = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score == best);
            let (offset, _) = winners.next().unwrap();
            if winners.next().is_some() {
                return Err("training scores tie for the best row shift");
            }
            Ok(offset as u32)
        })
        .collect::<std::result::Result<_, _>>()?;
    circuit::set_shifts(&mut weights, &offsets, &[0; P]);
    layout::validate(&cfg, &weights)?;
    Ok(weights)
}

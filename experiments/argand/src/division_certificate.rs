//! Identify the label table from observed equations, then score a saved model
//! on all legal prompts against that table. The verifier never reads test labels.
use crate::{
    Result, division, division_equations as eq,
    engine::Engine,
    margins, metrics,
    suite::{load_weights, publish},
};
use convex_relabel_graph_audit::graph::{Pair, Structure, analyze};
use serde_json::{Value, json};
use std::{fs, path::Path};

mod bitplanes;
pub(crate) mod mask;
pub(crate) mod recorded_scores;
pub(crate) mod relabel;
pub(crate) mod row_projection;
mod spectrum;

const N: usize = eq::P - 1;

struct Completion {
    graph: Structure,
    observed: Vec<bool>,
    zero: usize,
    units: Vec<usize>,
}

impl Completion {
    fn original_split() -> Result<Self> {
        let open = eq::split(0, 4656).0;
        if eq::fingerprint(&open) != division::TRAIN_HASH {
            return Err("training fingerprint mismatch".into());
        }
        Self::from_observations(&open)
    }

    fn from_observations(rows: &[eq::Equation]) -> Result<Self> {
        let mut observed = vec![false; eq::P * N];
        let mut zero = None;
        let mut pairs = Vec::new();
        for e in rows {
            let (a, b, y) = (usize::from(e.num), usize::from(e.den), usize::from(e.quo));
            if a >= eq::P
                || !(1..=N).contains(&b)
                || y >= eq::P
                || std::mem::replace(&mut observed[a * N + b - 1], true)
            {
                return Err("invalid or repeated division observation".into());
            }
            if a == 0 {
                if zero.replace(y).is_some_and(|old| old != y) {
                    return Err("observed constant row is inconsistent".into());
                }
            } else {
                pairs.push(Pair {
                    row: a - 1,
                    column: b - 1,
                    label: y.checked_sub(1).ok_or("invalid zero in a unit cell")?,
                });
            }
        }
        let zero = zero.ok_or("no observation of the constant row")?;
        let mut log = [usize::MAX; N];
        let mut residue = 1;
        for power in 0..N {
            log[residue - 1] = power;
            residue = (5 * residue) % eq::P;
        }
        let mut graph = analyze(N, &pairs, &log).map_err(|error| error.to_string())?;
        let units = graph
            .completion_labels
            .take()
            .ok_or("unit observations do not identify a consistent phase table")?;
        Ok(Self {
            graph,
            observed,
            zero,
            units,
        })
    }

    fn label(&self, a: usize, b: usize) -> usize {
        if a == 0 {
            self.zero
        } else {
            1 + self.units[a - 1 + N * (b - 1)]
        }
    }

    fn examples(&self) -> Vec<crate::data::Example> {
        (0..eq::P * N)
            .map(|i| {
                let (a, b) = (i / N, i % N + 1);
                division::example(a, b, self.label(a, b))
            })
            .collect()
    }
}

#[derive(Default)]
struct Counts {
    correct: usize,
    ties: usize,
    total: usize,
    min_gap: Option<f64>,
}

/// Keep the frozen held-out input order and reconstruct every label from
/// observed factors. The verifier reads only the operands of a hidden row.
pub(crate) fn reconstructed_hidden() -> Result<Vec<crate::data::Example>> {
    let completion = Completion::original_split()?;
    Ok(eq::split(0, 4656)
        .1
        .iter()
        .map(|e| {
            let (a, b) = (usize::from(e.num), usize::from(e.den));
            division::example(a, b, completion.label(a, b))
        })
        .collect())
}

impl Counts {
    fn add(&mut self, row: margins::RowMargin) {
        self.correct += usize::from(row.gap > 0.0);
        self.total += 1;
        self.ties += row.ties;
        self.min_gap = Some(self.min_gap.map_or(row.gap, |g| g.min(row.gap)));
    }
    fn json(&self) -> Value {
        json!({"strict_correct":self.correct,"total":self.total,"ties":self.ties,"minimum_gap":self.min_gap})
    }
}

fn checkpoint(dir: &Path, first: bool) -> Result<(Value, Engine)> {
    checkpoint_with_batch(dir, first, division::shape().max_batch)
}

fn checkpoint_with_batch(dir: &Path, first: bool, eval_batch: usize) -> Result<(Value, Engine)> {
    let (metadata, weights) = if first {
        ("first-strict.json", "first-strict.f32")
    } else {
        ("result.json", "weights.f32")
    };
    let record: Value = serde_json::from_slice(&fs::read(dir.join(metadata))?)?;
    let id = &record["identity"];
    let mut spec = division::shape();
    spec.max_batch = eval_batch;
    if id["case"] != division::NAME
        || id["split_sha256"][0] != division::TRAIN_HASH
        || id["width"] != spec.width
        || id["heads"] != spec.heads
        || id["layers"] != spec.layers
        || id["vocab"] != spec.vocab
        || id["model"] != "argand-skip-write-tied-score-init-one"
        || id["repeat"] != 1
    {
        return Err(
            "division checkpoint does not match the canonical corpus and Argand architecture"
                .into(),
        );
    }
    let mut engine = Engine::new(spec, 1, 0)?;
    load_weights(dir, weights, &record["weights_sha256"], &mut engine)?;
    Ok((record, engine))
}

pub(crate) fn check(dir: &Path, first: bool) -> Result<Value> {
    let (record, mut engine) = checkpoint(dir, first)?;
    let id = &record["identity"];
    let completion = Completion::original_split()?;
    let examples = completion.examples();
    let order: Vec<_> = (0..examples.len() as u32).collect();
    let mut counts = [Counts::default(), Counts::default()];
    let mut phases = spectrum::PhaseTable::new();
    let mut batch = metrics::Packed::default();
    for indices in order.chunks(engine.config.max_batch) {
        batch.fill(&examples, indices);
        let vocab = engine.config.vocab;
        let logits = margins::masked_logits(&mut engine, &batch)?;
        let rows = margins::rows_from_logits(&logits, vocab, &batch)?;
        for ((&i, row), scores) in indices
            .iter()
            .zip(rows)
            .zip(logits.chunks_exact(vocab * batch.length))
        {
            let i = i as usize;
            let (a, b) = (i / N, i % N + 1);
            if a > 0 {
                let start = eq::ANSWER_POS * vocab;
                phases.observe(
                    &scores[start..start + vocab],
                    completion.label(a, b),
                    row.gap,
                );
            }
            counts[usize::from(completion.observed[i])].add(row);
        }
    }
    let report = json!({"case":division::NAME,"seed":id["seed"],"step":record["step"],
        "weights_sha256":record["weights_sha256"],"observed_sha256":division::TRAIN_HASH,
        "accept":counts.iter().all(|c| c.correct==c.total),"observed":counts[1].json(),"hidden":counts[0].json(),
        "output_tokens_compared":engine.config.vocab,"observed_unit_cells":completion.graph.observed_cells,
        "unit_graph_components":completion.graph.connected_components,"zero_row_label":completion.zero,
        "phase_generator":5,"row_phases":completion.graph.row_phases,"column_phases":completion.graph.column_phases,
        "matching":completion.graph.perfect_matching,"unit_character_spectrum":phases.report()?, "projection_row_norms":engine.projection_row_norms()?});
    let output = if first {
        "first-strict-certificate.json"
    } else {
        "observed-certificate.json"
    };
    publish(dir, output, &serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_mask_identifies_division_and_unknown_unit_factors() -> Result<()> {
        let (mut open, hidden) = eq::split(0, 4656);
        let actual = Completion::from_observations(&open)?;
        assert_eq!(actual.graph.observed_cells, 4611);
        assert!(actual.graph.perfect_matching.is_some());
        for e in &hidden {
            assert_eq!(actual.label(e.num.into(), e.den.into()), usize::from(e.quo));
        }
        // Replace both sets of factors and the constant class. A verifier
        // hardcoding the quotient cannot recover this second table.
        let other = |a: usize, b: usize| {
            if a == 0 {
                17
            } else {
                a * a * (1 + 7 * b % N) % eq::P
            }
        };
        for e in &mut open {
            e.quo = other(e.num.into(), e.den.into()) as u8;
        }
        let second = Completion::from_observations(&open)?;
        for e in &hidden {
            assert_eq!(
                second.label(e.num.into(), e.den.into()),
                other(e.num.into(), e.den.into())
            );
        }
        let duplicate = open[0];
        open.push(duplicate);
        assert!(Completion::from_observations(&open).is_err());
        Ok(())
    }
}

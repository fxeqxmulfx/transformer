//! Sparse-input port of ConvexChannel.Forward CF1-CF5 (a7c246fe) and the native
//! learner at 3e0e2cb1, experiments/convex_channel_forward/src/model.rs.
//! Keep the candidate order, default class, explicit failure and full scan count.
//! Only storage changes: absent input slots are PAD rather than dense arrays.
use crate::{
    channel::{Channel, QueryChannel},
    data::Example,
    features::Adapter,
    random::Random,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

struct Sample {
    items: Vec<(usize, usize)>,
    label: usize,
}
type LearnedRecord = (usize, Vec<Option<usize>>);
fn learn(
    keys: usize,
    values: usize,
    pad: usize,
    samples: &[Sample],
) -> (Option<LearnedRecord>, usize) {
    let mut visits = 0;
    for key in 0..keys {
        let mut record = vec![None; values];
        let mut alive = true;
        for sample in samples {
            visits += 1;
            if !alive {
                continue;
            }
            let value = sample
                .items
                .binary_search_by_key(&key, |x| x.0)
                .map(|i| sample.items[i].1)
                .unwrap_or(pad);
            match record[value] {
                None => record[value] = Some(sample.label),
                Some(previous) => alive = previous == sample.label,
            }
        }
        if alive {
            return (Some((key, record)), visits);
        }
    }
    (None, visits)
}

pub fn fit(adapter: Adapter, examples: &[Example], seed: u32) -> (Option<Channel>, Value) {
    let mut order: Vec<_> = (0..examples.len()).collect();
    Random::new(seed + 10_000).shuffle(&mut order);
    let mut samples: Vec<Vec<Sample>> = (0..adapter.vocab).map(|_| vec![]).collect();
    for i in order {
        let example = &examples[i];
        for (pos, &label) in example.targets.iter().enumerate() {
            if label < 0 {
                continue;
            }
            let (q, items) = adapter.encode(&example.tokens[..=pos]);
            samples[q].push(Sample {
                items,
                label: label as usize,
            });
        }
    }
    let mut queries = BTreeMap::new();
    let mut rows = vec![];
    let mut failed = vec![];
    let mut visits = 0;
    let observations: usize = samples.iter().map(Vec::len).sum();
    for (q, samples) in samples.iter().enumerate() {
        let (learned, count) = learn(adapter.keys(), adapter.values(), adapter.pad(), samples);
        visits += count;
        match learned {
            None => {
                failed.push(q);
                rows.push(json!({"query":q,"observations":samples.len(),"candidate_visits":count,"status":"no_consistent_key"}));
            }
            Some((key, record)) => {
                let classes: Vec<_> = record
                    .iter()
                    .flatten()
                    .copied()
                    .chain([0])
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let weights: BTreeMap<_, _> = record
                    .iter()
                    .enumerate()
                    .filter_map(|(v, &c)| {
                        c.map(|c| {
                            let mut weights = vec![0.; classes.len()];
                            weights[classes.binary_search(&c).unwrap()] = 1.;
                            (key * adapter.values() + v, weights)
                        })
                    })
                    .collect();
                rows.push(
                    json!({"query":q,"observations":samples.len(),"candidate_visits":count,
                    "status":"learned","key":key,"observed_values":weights.len()}),
                );
                queries.insert(
                    q,
                    QueryChannel {
                        classes,
                        routes: [(key, 1.)].into(),
                        rows: weights,
                    },
                );
            }
        }
    }
    let report = json!({"observations":observations,"candidate_visits":visits,"visit_bound":adapter.keys()*observations,
        "failed_queries":failed,"all_queries_realizable":failed.is_empty(),"groups":rows,
        "mean_loss":if failed.is_empty(){Some(((adapter.vocab-1) as f64+std::f64::consts::E).ln()-1.)}else{None},
        "mean_fw_gap":if failed.is_empty(){Some(0.)}else{None}});
    (
        failed.is_empty().then_some(Channel { adapter, queries }),
        report,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhaustive_partial_boolean_data_match_direct_decoder_enumeration() {
        for code in 0..81usize {
            let mut state = code;
            let mut samples = vec![];
            for a in 0..2 {
                for b in 0..2 {
                    let digit = state % 3;
                    state /= 3;
                    if digit != 0 {
                        samples.push(Sample {
                            items: vec![(0, a), (1, b)],
                            label: digit - 1,
                        });
                    }
                }
            }
            let expected = (0..2).find(|&key| {
                (0..4usize).any(|decoder| {
                    samples
                        .iter()
                        .all(|s| ((decoder >> s.items[key].1) & 1) == s.label)
                })
            });
            let (result, visits) = learn(2, 2, 0, &samples);
            assert_eq!(result.as_ref().map(|x| x.0), expected);
            assert!(visits <= 2 * samples.len());
            if let Some((key, record)) = result {
                for s in &samples {
                    assert_eq!(record[s.items[key].1], Some(s.label));
                }
            }
        }
    }
    #[test]
    fn successful_forward_and_first_fw_choose_the_same_observed_channel() {
        let adapter = Adapter {
            vocab: 6,
            context: 3,
        };
        let examples = vec![
            Example::generation("any", vec![1, 2], vec![4]),
            Example::generation("any", vec![3, 2], vec![5]),
        ];
        let (direct, report) = fit(adapter, &examples, 0);
        assert_eq!(report["all_queries_realizable"], true);
        let fw = crate::train::Trainer::new(adapter, &examples, 0).export();
        let direct = direct.unwrap();
        for prefix in [vec![1, 2], vec![3, 2], vec![4, 2]] {
            assert_eq!(direct.logits(&prefix), fw.logits(&prefix));
        }
    }
}

//! Full-batch Frank-Wolfe on the direct joint tensor with ordinary vocabulary CE.
//! Missing classes remain in the softmax denominator. Sparse rows are an exact
//! storage scheme, not a learned low-rank routing/decoder factorization.
use crate::{
    Result,
    channel::{Channel, QueryChannel},
    data::Example,
    features::Adapter,
    random::Random,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

struct Sample {
    label: usize,
    features: Vec<usize>,
}
pub struct Group {
    query: usize,
    classes: Vec<usize>,
    features: Vec<usize>,
    by_key: Vec<Vec<usize>>,
    pads: Vec<usize>,
    samples: Vec<Sample>,
    model: Vec<f64>,
    routes: Vec<f64>,
    vocab: usize,
    seen: BTreeSet<usize>,
    pub fixed: bool,
    pub realizable: bool,
    pub gap: Option<f64>,
}

impl Group {
    fn new(query: usize, samples: Vec<Sample>, adapter: Adapter) -> Self {
        let classes: Vec<_> = samples
            .iter()
            .map(|x| x.label)
            .chain([0])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let features: Vec<_> = samples
            .iter()
            .flat_map(|x| x.features.iter().copied())
            .chain((0..adapter.keys()).map(|k| k * adapter.values() + adapter.pad()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let positions: std::collections::HashMap<_, _> =
            features.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let samples: Vec<_> = samples
            .into_iter()
            .map(|s| Sample {
                label: classes.binary_search(&s.label).unwrap(),
                features: s.features.into_iter().map(|f| positions[&f]).collect(),
            })
            .collect();
        let pads = (0..adapter.keys())
            .map(|k| positions[&(k * adapter.values() + adapter.pad())])
            .collect::<Vec<_>>();
        let mut by_key = vec![vec![]; adapter.keys()];
        for (r, &f) in features.iter().enumerate() {
            by_key[f / adapter.values()].push(r);
        }
        let mut counts = vec![0u32; features.len() * classes.len()];
        let mut totals = vec![0u32; classes.len()];
        for sample in &samples {
            totals[sample.label] += 1;
        }
        for &pad in &pads {
            counts[pad * classes.len()..(pad + 1) * classes.len()].copy_from_slice(&totals);
        }
        for sample in &samples {
            for &r in &sample.features {
                counts[r * classes.len() + sample.label] += 1;
                counts[pads[features[r] / adapter.values()] * classes.len() + sample.label] -= 1;
            }
        }
        let mut choices = vec![0; features.len()];
        for (r, choice) in choices.iter_mut().enumerate() {
            *choice = (0..classes.len())
                .max_by_key(|&c| (counts[r * classes.len() + c], std::cmp::Reverse(c)))
                .unwrap();
        }
        let correct: Vec<usize> = by_key
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|&r| counts[r * classes.len() + choices[r]] as usize)
                    .sum()
            })
            .collect();
        let key = (0..by_key.len())
            .max_by_key(|&k| (correct[k], std::cmp::Reverse(k)))
            .unwrap();
        // At uniform logits the exact linear oracle is majority counting.
        let mut model = vec![0.; counts.len()];
        for &r in &by_key[key] {
            model[r * classes.len() + choices[r]] = 1.;
        }
        let mut routes = vec![0.; by_key.len()];
        routes[key] = 1.;
        let realizable = correct[key] == samples.len();
        let seen = features
            .iter()
            .enumerate()
            .filter(|(r, _)| {
                counts[r * classes.len()..(r + 1) * classes.len()]
                    .iter()
                    .sum::<u32>()
                    > 0
            })
            .map(|(_, &f)| f)
            .collect();
        Self {
            query,
            classes,
            features,
            by_key,
            pads,
            samples,
            model,
            routes,
            vocab: adapter.vocab,
            seen,
            fixed: realizable,
            realizable,
            gap: realizable.then_some(0.),
        }
    }

    fn gradient(&self, values: usize) -> (Vec<f64>, f64) {
        let width = self.classes.len();
        let mut gradient = vec![0.; self.model.len()];
        let mut base = vec![0.; width];
        for &r in &self.pads {
            for (c, b) in base.iter_mut().enumerate() {
                *b += self.model[r * width + c];
            }
        }
        let mut total = vec![0.; width];
        let mut z = vec![0.; width];
        let mut errors = vec![0.; width];
        let mut loss = 0.;
        for sample in &self.samples {
            z.copy_from_slice(&base);
            for &r in &sample.features {
                let pad = self.pads[self.features[r] / values];
                for (c, x) in z.iter_mut().enumerate() {
                    *x += self.model[r * width + c] - self.model[pad * width + c];
                }
            }
            let denominator = (self.vocab - width) as f64 + z.iter().map(|x| x.exp()).sum::<f64>();
            loss += denominator.ln() - z[sample.label];
            for (c, e) in errors.iter_mut().enumerate() {
                *e = z[c].exp() / denominator - f64::from(c == sample.label);
                total[c] += *e;
            }
            for &r in &sample.features {
                let pad = self.pads[self.features[r] / values];
                for (c, &e) in errors.iter().enumerate() {
                    gradient[r * width + c] += e;
                    gradient[pad * width + c] -= e;
                }
            }
        }
        for &r in &self.pads {
            for (c, &e) in total.iter().enumerate() {
                gradient[r * width + c] += e;
            }
        }
        (gradient, loss / self.samples.len() as f64)
    }

    fn oracle(&self, gradient: &[f64]) -> (usize, Vec<usize>, f64) {
        let width = self.classes.len();
        let choices: Vec<_> = gradient
            .chunks_exact(width)
            .map(|row| {
                (0..width)
                    .min_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)))
                    .unwrap()
            })
            .collect();
        let costs: Vec<f64> = self
            .by_key
            .iter()
            .map(|rows| rows.iter().map(|&r| gradient[r * width + choices[r]]).sum())
            .collect();
        let key = (0..costs.len())
            .min_by(|&a, &b| costs[a].total_cmp(&costs[b]).then(a.cmp(&b)))
            .unwrap();
        (key, choices, costs[key])
    }

    pub fn step(&mut self, iteration: usize, values: usize) -> Result<()> {
        if self.fixed {
            return Ok(());
        }
        let (gradient, _) = self.gradient(values);
        let (key, choices, cost) = self.oracle(&gradient);
        let gap = (gradient
            .iter()
            .zip(&self.model)
            .map(|(&g, &w)| g * w)
            .sum::<f64>()
            - cost)
            / self.samples.len() as f64;
        if !gap.is_finite() || gap < -1e-8 {
            return Err(format!("invalid FW gap {gap}").into());
        }
        self.gap = Some(gap);
        if gap <= 1e-12 {
            self.fixed = true;
            return Ok(());
        }
        let eta = 2. / (iteration + 1) as f64;
        for weight in &mut self.model {
            *weight *= 1. - eta;
        }
        for mass in &mut self.routes {
            *mass *= 1. - eta;
        }
        self.routes[key] += eta;
        for &r in &self.by_key[key] {
            self.model[r * self.classes.len() + choices[r]] += eta;
        }
        self.gap = None; // The gap above belongs to the previous iterate.
        Ok(())
    }

    fn export(&self, values: usize) -> QueryChannel {
        let routes: BTreeMap<_, _> = self
            .routes
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r > 0.)
            .map(|(k, &r)| (k, r))
            .collect();
        let rows = self
            .features
            .iter()
            .enumerate()
            .filter(|&(_, &f)| routes.contains_key(&(f / values)))
            .map(|(r, &f)| {
                (
                    f,
                    self.model[r * self.classes.len()..(r + 1) * self.classes.len()].to_vec(),
                )
            })
            .collect();
        QueryChannel {
            classes: self.classes.clone(),
            routes,
            rows,
        }
    }
}

pub struct Trainer {
    pub adapter: Adapter,
    pub groups: Vec<Group>,
}
impl Trainer {
    pub fn new(adapter: Adapter, train: &[Example], seed: u32) -> Self {
        let mut order: Vec<_> = (0..train.len()).collect();
        Random::new(seed + 10_000).shuffle(&mut order);
        let mut raw: BTreeMap<usize, Vec<Sample>> = BTreeMap::new();
        for i in order {
            let example = &train[i];
            for (pos, &label) in example.targets.iter().enumerate() {
                if label < 0 {
                    continue;
                }
                let (q, features) = adapter.encode(&example.tokens[..=pos]);
                raw.entry(q).or_default().push(Sample {
                    label: label as usize,
                    features: features
                        .into_iter()
                        .map(|(k, v)| k * adapter.values() + v)
                        .collect(),
                });
            }
        }
        let groups = raw
            .into_iter()
            .map(|(q, samples)| Group::new(q, samples, adapter))
            .collect();
        Self { adapter, groups }
    }
    pub fn export(&self) -> Channel {
        Channel {
            adapter: self.adapter,
            queries: self
                .groups
                .iter()
                .map(|g| (g.query, g.export(self.adapter.values())))
                .collect(),
        }
    }
    pub fn step(&mut self, iteration: usize) -> Result<()> {
        for group in &mut self.groups {
            group.step(iteration, self.adapter.values())?;
        }
        Ok(())
    }
    pub fn fixed(&self) -> bool {
        self.groups.iter().all(|g| g.fixed)
    }
    /// A missing coordinate disproves the O2 span certificate; complete coordinate
    /// coverage does not prove a span certificate or prediction identification.
    pub fn coverage(&self, examples: &[Example]) -> Value {
        let groups: BTreeMap<_, _> = self.groups.iter().map(|g| (g.query, g)).collect();
        let (mut total, mut covered, mut missing_query) = (0, 0, 0);
        for e in examples {
            let positions: Vec<_> = if e.prompt.is_empty() {
                e.targets
                    .iter()
                    .enumerate()
                    .filter(|&(_, &y)| y >= 0)
                    .map(|(p, _)| p)
                    .collect()
            } else {
                vec![e.prompt.len() - 1]
            };
            for pos in positions {
                total += 1;
                let (q, items) = self.adapter.encode(&e.tokens[..=pos]);
                let Some(group) = groups.get(&q) else {
                    missing_query += 1;
                    continue;
                };
                let mut values = vec![self.adapter.pad(); self.adapter.keys()];
                for (k, v) in items {
                    values[k] = v;
                }
                if values
                    .iter()
                    .enumerate()
                    .all(|(k, &v)| group.seen.contains(&(k * self.adapter.values() + v)))
                {
                    covered += 1;
                }
            }
        }
        json!({"evaluated_inputs":total,"all_coordinates_observed":covered,"missing_query":missing_query,
            "no_span_certificate_due_to_missing_coordinate":total-covered,
            "scope":"first answer prefix for generation; all supervised prefixes otherwise",
            "complete_coordinate_coverage_is_not_a_span_certificate":true})
    }
    pub fn report(&mut self) -> Value {
        let mut loss = 0.;
        let mut gap = 0.;
        let mut total = 0;
        let mut groups = vec![];
        for group in &mut self.groups {
            let n = group.samples.len();
            let (mean_loss, mean_gap) = if group.realizable {
                (
                    ((self.adapter.vocab - 1) as f64 + std::f64::consts::E).ln() - 1.,
                    0.,
                )
            } else {
                let (gradient, loss) = group.gradient(self.adapter.values());
                let (_, _, cost) = group.oracle(&gradient);
                (
                    loss,
                    (gradient
                        .iter()
                        .zip(&group.model)
                        .map(|(&g, &w)| g * w)
                        .sum::<f64>()
                        - cost)
                        / n as f64,
                )
            };
            loss += mean_loss * n as f64;
            gap += mean_gap * n as f64;
            total += n;
            groups.push(
                json!({"query":group.query,"observations":n,"realizable":group.realizable,
                "mean_loss":mean_loss,"mean_fw_gap":mean_gap,"fixed":group.fixed}),
            );
        }
        json!({"observations":total,"mean_loss":loss/total as f64,"mean_fw_gap":gap/total as f64,
            "all_queries_realizable":self.groups.iter().all(|g|g.realizable),"groups":groups})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::cross_entropy;

    fn toy() -> (Adapter, Group) {
        let adapter = Adapter {
            vocab: 3,
            context: 2,
        };
        let samples = (0..2)
            .flat_map(|a| {
                (0..2).map(move |b| Sample {
                    label: a ^ b,
                    features: vec![a, adapter.values() + b],
                })
            })
            .collect();
        (adapter, Group::new(2, samples, adapter))
    }
    fn dense_loss(group: &Group, adapter: Adapter) -> f64 {
        let mut loss = 0.;
        for s in &group.samples {
            let mut selected = group.pads.clone();
            for &r in &s.features {
                selected[group.features[r] / adapter.values()] = r;
            }
            let mut logits = vec![0.; adapter.vocab];
            for r in selected {
                for (c, &label) in group.classes.iter().enumerate() {
                    logits[label] += group.model[r * group.classes.len() + c];
                }
            }
            loss += cross_entropy(&logits, group.classes[s.label]);
        }
        loss
    }
    #[test]
    fn sparse_gradient_matches_independent_dense_forward_and_finite_differences() {
        let (adapter, mut group) = toy();
        assert!(!group.realizable);
        let (gradient, loss) = group.gradient(adapter.values());
        assert!((loss * 4. - dense_loss(&group, adapter)).abs() < 1e-12);
        for (i, &expected) in gradient.iter().enumerate() {
            let original = group.model[i];
            group.model[i] = original + 1e-6;
            let plus = dense_loss(&group, adapter);
            group.model[i] = original - 1e-6;
            let minus = dense_loss(&group, adapter);
            group.model[i] = original;
            assert!(
                ((plus - minus) / 2e-6 - expected).abs() < 2e-8,
                "coordinate {i}"
            );
        }
    }
    #[test]
    fn restricted_oracle_equals_exhaustive_full_vocabulary_vertex_search() {
        let (adapter, group) = toy();
        let (gradient, _) = group.gradient(adapter.values());
        let (_, _, cost) = group.oracle(&gradient);
        let mut best = f64::INFINITY;
        for rows in &group.by_key {
            for assignment in 0..adapter.vocab.pow(adapter.values() as u32) {
                let mut score = 0.;
                for &r in rows {
                    let value = group.features[r] % adapter.values();
                    let label = (assignment / adapter.vocab.pow(value as u32)) % adapter.vocab;
                    score += match group.classes.binary_search(&label) {
                        Ok(c) => gradient[r * group.classes.len() + c],
                        // There is exactly one excluded class in this witness.
                        Err(_) => -gradient[r * group.classes.len()..(r + 1) * group.classes.len()]
                            .iter()
                            .sum::<f64>(),
                    };
                }
                best = best.min(score);
            }
        }
        assert!((cost - best).abs() < 1e-12);
    }
    #[test]
    fn convex_updates_preserve_every_shared_mass_and_saved_forward() -> Result<()> {
        let (adapter, mut group) = toy();
        for iteration in 2..=20 {
            group.step(iteration, adapter.values())?;
        }
        let model = Channel {
            adapter,
            queries: [(2, group.export(adapter.values()))].into(),
        };
        assert!(model.feasibility_error()? < 1e-12);
        let bytes = serde_json::to_vec(&model)?;
        let restored: Channel = serde_json::from_slice(&bytes)?;
        for prefix in [vec![0, 2], vec![1, 2], vec![2], vec![0, 1]] {
            let logits = model.logits(&prefix);
            assert_eq!(logits, restored.logits(&prefix));
            assert!((logits.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(logits.iter().all(|&x| x >= 0.));
        }
        Ok(())
    }
    #[test]
    fn first_vertex_fit_has_exact_ce_lower_bound_and_labels_do_not_enter_features() -> Result<()> {
        let adapter = Adapter {
            vocab: 6,
            context: 3,
        };
        let examples = vec![
            Example::generation("any", vec![1, 2], vec![4]),
            Example::generation("any", vec![3, 2], vec![5]),
        ];
        let mut trainer = Trainer::new(adapter, &examples, 0);
        assert!(trainer.fixed());
        let report = trainer.report();
        assert_eq!(report["mean_fw_gap"], 0.);
        let model = trainer.export();
        for e in &examples {
            assert_eq!(model.logits(&e.prompt)[e.answer[0] as usize], 1.);
        }
        let mut changed = examples[0].clone();
        changed.targets.fill(0);
        changed.answer.fill(0);
        assert_eq!(
            adapter.encode(&changed.tokens),
            adapter.encode(&examples[0].tokens)
        );
        assert_eq!(
            model.logits(&changed.prompt),
            model.logits(&examples[0].prompt)
        );
        Ok(())
    }
}

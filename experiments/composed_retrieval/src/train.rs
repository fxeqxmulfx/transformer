//! Optimizers take aligned original supervised prefixes. The fit loop has no
//! case identifier, evaluation records, role labels or selected checkpoint.
use crate::{
    Result,
    data::Protocol,
    gradient,
    model::{Layout, Model},
    operator::Tables,
    query,
};
use baseline::{
    channel::{argmax, cross_entropy},
    data::Example,
    hash,
    random::Random,
};
use bilinear::experiment::records;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Sample {
    pub prefix: Vec<usize>,
    pub target: usize,
}
pub fn samples(model: &Model, examples: &[Example]) -> Result<Vec<Vec<Sample>>> {
    (0..examples.len())
        .map(|i| {
            records(examples, &[i])?
                .into_iter()
                .map(|r| {
                    if r.target >= model.vocabulary {
                        return Err("target outside original vocabulary".into());
                    }
                    Ok(Sample {
                        prefix: model.objects(r.prefix)?,
                        target: r.target,
                    })
                })
                .collect()
        })
        .collect()
}
fn lengths(xs: &[&Sample]) -> Vec<usize> {
    let mut out: Vec<_> = xs
        .iter()
        .map(|s| s.prefix.len().saturating_sub(1))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}
pub fn analytic(m: &Model, t: &Tables, batch: &[&Sample]) -> Vec<f64> {
    let mut g = vec![0.; m.weights.len()];
    for s in batch {
        gradient::accumulate(m, t, &s.prefix, s.target, &mut g);
    }
    for x in &mut g {
        *x /= batch.len() as f64;
    }
    g
}
struct Adam {
    first: Vec<f64>,
    second: Vec<f64>,
    clock: Vec<u32>,
}
impl Adam {
    fn new(n: usize) -> Self {
        Self {
            first: vec![0.; n],
            second: vec![0.; n],
            clock: vec![0; n],
        }
    }
    fn update(
        &mut self,
        m: &mut Model,
        l: &Layout,
        coords: &[usize],
        grad: &[f64],
        p: &Protocol,
    ) -> Result<()> {
        if coords.len() != grad.len() || grad.iter().any(|g| !g.is_finite()) {
            return Err("invalid minibatch estimates".into());
        }
        for (&i, &g) in coords.iter().zip(grad) {
            self.clock[i] += 1;
            let time = self.clock[i] as i32;
            self.first[i] = p.beta1 * self.first[i] + (1. - p.beta1) * g;
            self.second[i] = p.beta2 * self.second[i] + (1. - p.beta2) * g * g;
            let lr = if i < l.bias.start {
                p.angle_lr
            } else if i == l.recency {
                p.recency_lr
            } else if i >= l.depth {
                p.gate_lr
            } else {
                p.routing_lr
            };
            let delta = lr * (self.first[i] / (1. - p.beta1.powi(time)))
                / ((self.second[i] / (1. - p.beta2.powi(time))).sqrt() + p.epsilon);
            m.weights[i] = l.project(i, m.weights[i] - delta);
        }
        m.validate()?;
        Ok(())
    }
    fn digest(&self) -> String {
        let mut bytes = vec![];
        for v in self.first.iter().chain(&self.second) {
            bytes.extend(v.to_le_bytes());
        }
        for v in &self.clock {
            bytes.extend(v.to_le_bytes());
        }
        hash(&bytes)
    }
}
struct Choices {
    tables: Vec<(Vec<usize>, usize)>,
    always: Vec<usize>,
}
impl Choices {
    fn new(l: &Layout, seed: u32) -> Self {
        let mut rng = Random::new(seed + 20000);
        let tables = [&l.key, &l.value, &l.phase, &l.bias, &l.relation]
            .into_iter()
            .map(|r| {
                let mut order: Vec<_> = r.clone().collect();
                rng.shuffle(&mut order);
                (order, 0)
            })
            .collect();
        Self {
            tables,
            always: l.start.clone().chain(l.scale..=l.product).collect(),
        }
    }
    fn next(&mut self, count: usize) -> Vec<usize> {
        let mut ids = self.always.clone();
        for (order, cursor) in &mut self.tables {
            for _ in 0..count.min(order.len()) {
                ids.push(order[*cursor]);
                *cursor = (*cursor + 1) % order.len();
            }
        }
        ids
    }
}
fn metrics(model: &Model, input: &[Vec<Sample>]) -> Result<Value> {
    let all: Vec<_> = input.iter().flatten().collect();
    let tab = Tables::new(model, &lengths(&all))?;
    let (mut correct, mut loss) = (0, 0.);
    for s in &all {
        let z = tab.logits(model, &tab.pass(model, &s.prefix).hidden);
        correct += usize::from(argmax(&z) == s.target);
        loss += cross_entropy(&z, s.target);
    }
    Ok(json!({"targets":all.len(),"correct":correct,"ce":loss/all.len() as f64}))
}
/// Read-only observations of the full minibatch immediately before Adam.
pub struct Update<'a> {
    pub number: usize,
    pub model: &'a Model,
    pub tables: &'a Tables,
    pub batch: &'a [&'a Sample],
    pub coordinates: &'a [usize],
    pub gradient: &'a [f64],
}
pub trait Observer {
    fn admit(&mut self) -> Result<()>;
    fn before_update(&mut self, _update: Update<'_>) -> Result<()> {
        Ok(())
    }
}
impl<F: FnMut() -> Result<()>> Observer for F {
    fn admit(&mut self) -> Result<()> {
        self()
    }
}
pub fn fit(
    m: &mut Model,
    rows: &[Example],
    batch_size: usize,
    mode: &str,
    seed: u32,
    p: &Protocol,
    mut observer: impl Observer,
) -> Result<Value> {
    let input = samples(m, rows)?;
    if input.is_empty() || batch_size == 0 || !p.modes.iter().any(|s| s == mode) {
        return Err("invalid training cell".into());
    }
    let l = m.validate()?;
    let before = metrics(m, &input)?;
    let mut adam = Adam::new(m.weights.len());
    let mut cycle = p
        .coordinates_per_table
        .map(|count| (Choices::new(&l, seed), count));
    let all: Vec<_> = (0..m.weights.len()).collect();
    let mut rng = Random::new(seed + 10000);
    let mut order: Vec<_> = (0..input.len()).collect();
    let mut cursor = order.len();
    let (mut examples, mut targets, mut queries) = (0, 0, 0);
    let mut trace = vec![];
    let mut batches = vec![];
    for step in 1..=p.updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        if step == 1 || step % 64 == 0 {
            observer.admit()?;
        }
        let end = (cursor + batch_size).min(order.len());
        let indices = &order[cursor..end];
        let batch: Vec<_> = indices.iter().flat_map(|&j| &input[j]).collect();
        for &idx in indices {
            batches.extend((idx as u64).to_le_bytes());
        }
        batches.extend(u64::MAX.to_le_bytes());
        let mut tab = Tables::new(m, &lengths(&batch))?;
        let selected;
        let (coords, g) = match mode {
            "mini" => (&all, analytic(m, &tab, &batch)),
            "forward" => {
                let coords = if let Some((choices, count)) = &mut cycle {
                    selected = choices.next(*count);
                    &selected
                } else {
                    &all
                };
                let g = query::differences(m, &mut tab, &batch, coords, p.query_step)?;
                queries += 2 * coords.len();
                (coords, g)
            }
            _ => return Err("unrecognized training mode".into()),
        };
        observer.before_update(Update {
            number: step,
            model: m,
            tables: &tab,
            batch: &batch,
            coordinates: coords,
            gradient: &g,
        })?;
        adam.update(m, &l, coords, &g, p)?;
        cursor = end;
        examples += indices.len();
        targets += batch.len();
        if step == 1 || step % 128 == 0 || step == p.updates {
            trace.push(json!({"update":step,"examples":examples,"targets":targets,"queries":queries,
                "optimizer_sha256":adam.digest(),"weights_sha256":hash(&serde_json::to_vec(&m.weights)?)}));
        }
    }
    Ok(
        json!({"before":before,"after":metrics(m,&input)?,"updates":p.updates,"examples":examples,
        "targets":targets,"forward_loss_queries":queries,"min_visits":adam.clock.iter().min(),"max_visits":adam.clock.iter().max(),
        "optimizer_sha256":adam.digest(),"batch_sha256":hash(&batches),"trace":trace}),
    )
}

use crate::{
    Result,
    cohort::Protocol,
    gradient::{Batch, Sample},
    model::Model,
};
use baseline::{
    channel::{argmax, cross_entropy},
    data::Example,
    random::Random,
};
use bilinear::experiment::records;
use serde_json::{Value, json};

pub fn training_samples(model: &Model, examples: &[Example]) -> Result<Vec<Vec<Sample>>> {
    (0..examples.len())
        .map(|i| {
            records(examples, &[i])?
                .into_iter()
                .map(|r| {
                    if r.target >= model.vocabulary {
                        return Err("training label outside vocabulary".into());
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

pub struct Adam {
    first: Vec<f64>,
    second: Vec<f64>,
    pub clocks: Vec<u32>,
}
impl Adam {
    pub fn new(parameters: usize) -> Self {
        Self {
            first: vec![0.; parameters],
            second: vec![0.; parameters],
            clocks: vec![0; parameters],
        }
    }
    pub fn step(
        &mut self,
        model: &mut Model,
        coords: &[usize],
        gradients: &[f64],
        p: &Protocol,
    ) -> Result<()> {
        if coords.len() != gradients.len() || gradients.iter().any(|g| !g.is_finite()) {
            return Err("invalid minibatch estimate".into());
        }
        for (&i, &g) in coords.iter().zip(gradients) {
            self.clocks[i] += 1;
            let t = self.clocks[i] as i32;
            self.first[i] = p.beta1 * self.first[i] + (1. - p.beta1) * g;
            self.second[i] = p.beta2 * self.second[i] + (1. - p.beta2) * g * g;
            let m = self.first[i] / (1. - p.beta1.powi(t));
            let v = self.second[i] / (1. - p.beta2.powi(t));
            let lr = if i < model.relative_start() {
                p.embedding_lr
            } else {
                p.routing_lr
            };
            model.weights[i] -= lr * m / (v.sqrt() + p.epsilon);
        }
        model.project_all();
        model.validate()
    }
}

struct Cycle {
    keys: Vec<usize>,
    values: Vec<usize>,
    cursor: usize,
}
impl Cycle {
    fn new(m: &Model, seed: u32) -> Self {
        let mut rng = Random::new(seed + 20000);
        let mut keys: Vec<_> = (0..m.table_len()).collect();
        let mut values: Vec<_> = (m.table_len()..m.relative_start()).collect();
        rng.shuffle(&mut keys);
        rng.shuffle(&mut values);
        Self {
            keys,
            values,
            cursor: 0,
        }
    }
    fn next(&mut self, m: &Model, count: usize) -> Vec<usize> {
        let mut coords: Vec<_> = (m.relative_start()..m.weights.len()).collect();
        for _ in 0..count.min(self.keys.len()) {
            coords.extend([self.keys[self.cursor], self.values[self.cursor]]);
            self.cursor = (self.cursor + 1) % self.keys.len();
        }
        coords
    }
}

fn train_metrics(model: &Model, examples: &[Vec<Sample>]) -> Value {
    let mut total = 0;
    let mut correct = 0;
    let mut ce = 0.;
    for s in examples.iter().flatten() {
        let link = model.links(s.prefix.len().saturating_sub(1));
        let z = model.pass(&s.prefix, &link).logits;
        total += 1;
        correct += usize::from(argmax(&z) == s.target);
        ce += cross_entropy(&z, s.target);
    }
    json!({"targets":total, "correct":correct, "ce":ce/total as f64})
}
pub fn fit(
    model: &mut Model,
    examples: &[Example],
    mode: &str,
    seed: u32,
    p: &Protocol,
    mut admit: impl FnMut() -> Result<()>,
) -> Result<Value> {
    let input = training_samples(model, examples)?;
    if input.is_empty() || !(mode == "mini" || mode == "forward") {
        return Err("invalid training cell".into());
    }
    let initial = model.clone();
    let before = train_metrics(&initial, &input);
    let mut rng = Random::new(seed + 10000);
    let mut order: Vec<_> = (0..input.len()).collect();
    let mut cursor = order.len();
    let mut cycle = Cycle::new(model, seed);
    let mut optimizer = Adam::new(model.weights.len());
    let all: Vec<_> = (0..model.weights.len()).collect();
    let mut example_count = 0;
    let mut targets = 0;
    let mut queries = 0usize;
    let mut trace = vec![];
    for update in 0..p.updates {
        if cursor == order.len() {
            admit()?;
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + p.batch).min(order.len());
        let batch = Batch::new(
            order[cursor..end]
                .iter()
                .flat_map(|&i| input[i].iter())
                .collect(),
        )?;
        let ce = batch.loss(model);
        if !ce.is_finite() {
            return Err("nonfinite batch loss".into());
        }
        if mode == "mini" {
            let grad = batch.analytic(model);
            optimizer.step(model, &all, &grad, p)?;
        } else {
            let coords = cycle.next(model, p.coordinates_per_table);
            let estimate = batch.finite(model, &coords, p.query_step);
            queries += 2 * coords.len();
            optimizer.step(model, &coords, &estimate, p)?;
        }
        example_count += end - cursor;
        targets += batch.records.len();
        cursor = end;
        if update == 0 || cursor == order.len() || update + 1 == p.updates {
            trace.push(json!({"update":update+1,"examples_seen":example_count,"targets_seen":targets,"batch_ce_before":ce}));
        }
    }
    let change = |lo: usize, hi: usize| {
        (lo..hi)
            .map(|i| (initial.weights[i] - model.weights[i]).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let k = model.table_len();
    let v = model.relative_start();
    let s = model.scale_index();
    Ok(
        json!({"updates":p.updates,"examples_seen":example_count,"targets_seen":targets,
        "initial_train":before,"final_train":train_metrics(model,&input),"minibatch_loss_queries":queries,
        "minimum_coordinate_visits":optimizer.clocks.iter().min(),"key_change_l2":change(0,k),
        "value_readout_change_l2":change(k,v),"relative_change_l2":change(v,s),"key_scale_change":model.scale()-initial.scale(),
        "trace":trace}),
    )
}

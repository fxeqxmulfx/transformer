//! Exact forward loss queries. A perturbation reuses only unaffected forward
//! values. Updating a single readout logit retains every other class in the
//! ordinary CE partition through independently accumulated prefix/suffix sums.
use crate::{
    Result,
    model::{Layout, Model, WIDTH},
    operator::{Pass, Tables},
    train::Sample,
};
use baseline::channel::cross_entropy;

#[derive(Clone, Copy)]
enum Change {
    Key(usize),
    Value(usize),
    Phase(usize),
    Start,
    Bias(usize),
    Link(usize),
    Scale,
    Recency,
    Depth,
    Product,
    OutputOffset(usize),
    Radius(usize),
}
impl Change {
    fn at(l: &Layout, i: usize) -> Self {
        if l.key.contains(&i) {
            Self::Key((i - l.key.start) / WIDTH)
        } else if l.value.contains(&i) {
            Self::Value((i - l.value.start) / WIDTH)
        } else if l.phase.contains(&i) {
            Self::Phase((i - l.phase.start) / WIDTH)
        } else if l.start.contains(&i) {
            Self::Start
        } else if l.bias.contains(&i) {
            Self::Bias(i - l.bias.start)
        } else if l.relation.contains(&i) {
            Self::Link((i - l.relation.start).abs_diff(l.relation.len() / 2))
        } else if i == l.scale {
            Self::Scale
        } else if i == l.recency {
            Self::Recency
        } else if i == l.depth {
            Self::Depth
        } else if l.output_offsets.contains(&i) {
            Self::OutputOffset(i - l.output_offsets.start)
        } else if l.radii.contains(&i) {
            Self::Radius(i - l.radii.start)
        } else {
            Self::Product
        }
    }
}

struct Original<'a> {
    sample: &'a Sample,
    pass: Pass,
    logits: Vec<f64>,
    other_mass: Vec<f64>,
    loss: f64,
    shift: f64,
}
impl<'a> Original<'a> {
    fn new(m: &Model, t: &Tables, sample: &'a Sample) -> Self {
        let pass = t.pass(m, &sample.prefix);
        let logits = t.logits(m, &pass.hidden);
        let loss = cross_entropy(&logits, sample.target);
        let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let terms: Vec<_> = logits.iter().map(|&z| (z - shift).exp()).collect();
        let mut left = 0.;
        let mut other_mass: Vec<_> = terms
            .iter()
            .map(|v| {
                let previous = left;
                left += v;
                previous
            })
            .collect();
        let mut right = 0.;
        for (entry, v) in other_mass.iter_mut().zip(terms).rev() {
            *entry += right;
            right += v;
        }
        Self {
            sample,
            pass,
            logits,
            other_mass,
            loss,
            shift,
        }
    }
    fn changed(&self, m: &Model, t: &Tables, change: Change) -> f64 {
        let prefix = &self.sample.prefix;
        let (q, memory) = prefix
            .split_last()
            .map_or((None, &[][..]), |(q, x)| (Some(*q), x));
        let n = memory.len();
        let one_logit_loss = |c: usize| {
            let z = t.class_logit(m, &self.pass.hidden, c);
            let label = if c == self.sample.target {
                z
            } else {
                self.logits[self.sample.target]
            };
            self.shift + (self.other_mass[c] + (z - self.shift).exp()).ln() - label
        };
        let hidden = match change {
            Change::OutputOffset(c) => return one_logit_loss(c),
            Change::Value(c) if !memory.contains(&c) => return one_logit_loss(c),
            Change::Value(_) => {
                let (s1, s2) = t.reads(memory, &self.pass.one, &self.pass.two);
                let product = t.product_read(m, memory, &self.pass.one, &s1);
                t.hidden(m, q.unwrap(), &s1, &s2, &product)
            }
            Change::Radius(o) if n == 0 || !prefix.contains(&o) => return self.loss,
            Change::Radius(o) if memory.contains(&o) => {
                let product = t.product_read(m, memory, &self.pass.one, &self.pass.s1);
                t.hidden(m, q.unwrap(), &self.pass.s1, &self.pass.s2, &product)
            }
            Change::Start if n > 0 => return self.loss,
            Change::Phase(o) if q != Some(o) => return self.loss,
            Change::Phase(_) if n > 0 => t.hidden(
                m,
                q.unwrap(),
                &self.pass.s1,
                &self.pass.s2,
                &self.pass.product_read,
            ),
            Change::Product if n == 0 => return self.loss,
            Change::Depth if n <= 1 => return self.loss,
            Change::Depth | Change::Product | Change::Radius(_) => t.hidden(
                m,
                q.unwrap(),
                &self.pass.s1,
                &self.pass.s2,
                &self.pass.product_read,
            ),
            Change::Key(o) if n <= 1 || !prefix.contains(&o) => return self.loss,
            Change::Bias(o) if n <= 1 || (o != m.vocabulary && !memory[..n - 1].contains(&o)) => {
                return self.loss;
            }
            Change::Link(offset) if n <= 1 || offset >= n => return self.loss,
            Change::Scale | Change::Recency if n <= 1 => return self.loss,
            _ => t.pass(m, prefix).hidden,
        };
        cross_entropy(&t.logits(m, &hidden), self.sample.target)
    }
}

pub fn differences(
    m: &mut Model,
    t: &mut Tables,
    batch: &[&Sample],
    coords: &[usize],
    step: f64,
    depth_log_odds: Option<f64>,
) -> Result<Vec<f64>> {
    if !(step > 0. && step.is_finite()) || batch.is_empty() {
        return Err("invalid forward query".into());
    }
    let cache: Vec<_> = batch.iter().map(|s| Original::new(m, t, s)).collect();
    let mut g = Vec::with_capacity(coords.len());
    for &i in coords {
        let original = m.weights[i];
        let (upper, lower, span) = if i == t.layout.depth
            && let Some(z) = depth_log_odds
        {
            let (hi, lo) = (z + step, z - step);
            (
                crate::model::sigmoid(hi),
                crate::model::sigmoid(lo),
                hi - lo,
            )
        } else {
            let (hi, lo) = (
                t.layout.project(i, original + step),
                t.layout.project(i, original - step),
            );
            (hi, lo, hi - lo)
        };
        if span <= 0. || !span.is_finite() {
            return Err("forward query displacement vanished".into());
        }
        let change = Change::at(&t.layout, i);
        let mut losses = [0.; 2];
        for (loss, weight) in losses.iter_mut().zip([upper, lower]) {
            m.weights[i] = weight;
            t.refresh(m, i);
            *loss = cache.iter().map(|s| s.changed(m, t, change)).sum::<f64>() / cache.len() as f64;
        }
        m.weights[i] = original;
        t.refresh(m, i);
        g.push((losses[0] - losses[1]) / span);
    }
    Ok(g)
}

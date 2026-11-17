//! Diagnostic task roles never enter the learned predictor or alter original labels.
use crate::Result;
use baseline::channel::{argmax, cross_entropy};
use bilinear::train::Record;
use serde_json::{Value, json};
#[derive(Debug)]
pub struct Context {
    pub values: [usize; 2],
    pub selected: usize,
    pub query_position: usize,
    pub target: usize,
}
impl Context {
    pub fn new(r: &Record<'_>) -> Result<Self> {
        let x = r.prefix;
        if !(6..=12).contains(&x.len())
            || x[0] != 1
            || !(36..68).contains(&x[1])
            || !(36..68).contains(&x[3])
            || x[1] == x[3]
            || !(68..100).contains(&x[2])
            || !(68..100).contains(&x[4])
            || x[5..x.len() - 1].iter().any(|v| !(68..100).contains(v))
        {
            return Err("original MQAR prefix shape changed".into());
        }
        let query = x[x.len() - 1];
        let selected = if query == x[1] {
            0
        } else if query == x[3] {
            1
        } else {
            return Err("query is not one of the stored keys".into());
        };
        let values = [x[2] as usize, x[4] as usize];
        if r.target != values[selected] {
            return Err("original target disagrees with its stored pair".into());
        }
        Ok(Self {
            values,
            selected,
            query_position: x.len() - 1,
            target: r.target,
        })
    }
}
pub struct Observation {
    pub category: usize,
    pub distinct: bool,
    pub choice_correct: bool,
    pub margin: f64,
    pub full_ce: f64,
    pub mass_ce: f64,
    pub pair_ce: f64,
    pub mass: f64,
    pub ce_error: f64,
}
impl Observation {
    pub fn new(c: &Context, logits: &[f64]) -> Result<Self> {
        if logits.len() != 100 || logits.iter().any(|x| !x.is_finite()) {
            return Err("diagnosis requires all100 original finite logits".into());
        }
        let predicted = argmax(logits);
        let other = c.values[1 - c.selected];
        let distinct = c.values[0] != c.values[1];
        let category = if predicted == c.target {
            0
        } else if predicted == other {
            1
        } else {
            2
        };
        let mut ids = c.values.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let pair: Vec<_> = ids.iter().map(|&id| logits[id]).collect();
        let target = ids
            .iter()
            .position(|&id| id == c.target)
            .ok_or("target missing from unique stored values")?;
        let choice = ids[argmax(&pair)];
        let pair_max = pair.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_pair = pair_max + pair.iter().map(|x| (x - pair_max).exp()).sum::<f64>().ln();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_all = max + logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        let full_ce = cross_entropy(logits, c.target);
        let mass_ce = log_all - log_pair;
        let pair_ce = cross_entropy(&pair, target);
        let ce_error = (full_ce - (mass_ce + pair_ce)).abs();
        if ce_error > 2e-12 || mass_ce < -2e-12 || pair_ce < -2e-12 || !full_ce.is_finite() {
            return Err("complete and conditional CE disagree".into());
        }
        Ok(Self {
            category,
            distinct,
            choice_correct: choice == c.target,
            margin: logits[c.target] - logits[other],
            full_ce,
            mass_ce,
            pair_ce,
            mass: (log_pair - log_all).exp(),
            ce_error,
        })
    }
}
#[derive(Default)]
pub struct Counts {
    pub total: usize,
    pub categories: [usize; 3],
    pub distinct: usize,
    pub choice_correct: usize,
    pub ties: usize,
    pub margin: f64,
    pub full_ce: f64,
    pub mass_ce: f64,
    pub pair_ce: f64,
    pub mass: f64,
    pub maximum_ce_error: f64,
}
fn ratio(sum: f64, n: usize) -> Value {
    if n == 0 {
        Value::Null
    } else {
        json!(sum / n as f64)
    }
}
impl Counts {
    pub fn observe(&mut self, o: &Observation) {
        self.total += 1;
        self.categories[o.category] += 1;
        if o.distinct {
            self.distinct += 1;
            self.choice_correct += usize::from(o.choice_correct);
            self.ties += usize::from(o.margin == 0.);
            self.margin += o.margin;
        }
        self.full_ce += o.full_ce;
        self.mass_ce += o.mass_ce;
        self.pair_ce += o.pair_ce;
        self.mass += o.mass;
        self.maximum_ce_error = self.maximum_ce_error.max(o.ce_error);
    }
    pub fn report(&self) -> Value {
        json!({"records":self.total,"correct":self.categories[0],"other_distinct_stored_value":self.categories[1],"outside_stored_values":self.categories[2],"original_accuracy":ratio(self.categories[0] as f64,self.total),"stored_value_prediction_rate":ratio((self.categories[0]+self.categories[1]) as f64,self.total),"distinct_value_records":self.distinct,"duplicate_value_records":self.total-self.distinct,"two_value_choice_correct":self.choice_correct,"two_value_choice_accuracy":ratio(self.choice_correct as f64,self.distinct),"two_value_ties":self.ties,"mean_correct_minus_other_logit":ratio(self.margin,self.distinct),"mean_full_ce":ratio(self.full_ce,self.total),"mean_mass_ce":ratio(self.mass_ce,self.total),"mean_conditional_pair_ce":ratio(self.pair_ce,self.total),"mean_stored_probability_mass":ratio(self.mass,self.total),"maximum_ce_decomposition_error":self.maximum_ce_error})
    }
}

//! JC2-JC4: complete splits, fixed old-margin deciles and retained old errors.
use crate::{
    domain::{Result, data::Data, model::W},
    forcing::Direction,
    inputs::require,
    measure::Snapshot,
    query_step::{self, Problem, Row, tolerance},
};
use serde_json::{Value, json};

#[derive(Default)]
struct Method {
    correct: usize,
    margin_up: usize,
    margin_down: usize,
    loss_down: usize,
    loss_up: usize,
    margins: [f64; 2],
    losses: [f64; 2],
}
impl Method {
    fn add(&mut self, r: &Row, method: usize) -> Result<()> {
        let b = r.intervals[method];
        let mt = tolerance(&[r.margin[0], r.margin[1], b.margin[0], b.margin[1]]);
        let lt = tolerance(&[r.loss[0], r.loss[1], b.loss[0], b.loss[1]]);
        let correct = b.margin[0] > mt;
        let margin_up = b.margin[0] > r.margin[0] + mt;
        let margin_down = b.margin[1] < r.margin[0] - mt;
        let loss_down = b.loss[1] < r.loss[0] - lt;
        let loss_up = b.loss[0] > r.loss[0] + lt;
        require(
            !correct || r.correct[1],
            "incorrect positive classification certificate",
        )?;
        require(
            !margin_up || r.margin[1] > r.margin[0],
            "incorrect margin improvement certificate",
        )?;
        require(
            !margin_down || r.margin[1] < r.margin[0],
            "incorrect margin regression certificate",
        )?;
        require(
            !loss_down || r.loss[1] < r.loss[0],
            "incorrect loss decrease certificate",
        )?;
        require(
            !loss_up || r.loss[1] > r.loss[0],
            "incorrect loss increase certificate",
        )?;
        self.correct += usize::from(correct);
        self.margin_up += usize::from(margin_up);
        self.margin_down += usize::from(margin_down);
        self.loss_down += usize::from(loss_down);
        self.loss_up += usize::from(loss_up);
        for j in 0..2 {
            self.margins[j] += b.margin[j];
            self.losses[j] += b.loss[j];
        }
        Ok(())
    }
    fn json(&self, n: usize) -> Value {
        let mean = |v: [f64; 2]| {
            if n > 0 {
                Some(v.map(|x| x / n as f64))
            } else {
                None
            }
        };
        json!({"certified_correct":self.correct,"certified_margin_improved":self.margin_up,
            "certified_margin_regressed":self.margin_down,"margin_inconclusive":n-self.margin_up-self.margin_down,
            "certified_loss_decreased":self.loss_down,"certified_loss_increased":self.loss_up,
            "loss_inconclusive":n-self.loss_down-self.loss_up,
            "mean_new_margin_bounds":mean(self.margins),"mean_new_loss_bounds":mean(self.losses),
            "mean_margin_interval_width":mean(self.margins).map(|x| x[1]-x[0]),
            "mean_loss_interval_width":mean(self.losses).map(|x| x[1]-x[0])})
    }
}

pub fn hardest(indices: &[usize], rows: &[Row]) -> Vec<usize> {
    let mut order = indices.to_vec();
    order.sort_by(|&a, &b| {
        rows[a].margin[0]
            .total_cmp(&rows[b].margin[0])
            .then(a.cmp(&b))
    });
    order.truncate(indices.len().div_ceil(10));
    order
}
pub fn freeze_difficult(old: &Snapshot, data: &Data) -> [Vec<usize>; 2] {
    [&data.train, &data.validation].map(|input| {
        let mut order: Vec<_> = input
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| {
                let q = (r[0] * 97 + r[1]) as usize;
                let target = r[2] as usize;
                let z = &old.logits[q * 97..(q + 1) * 97];
                let wrong = (0..97)
                    .filter(|&c| c != target)
                    .map(|c| z[c])
                    .fold(f64::NEG_INFINITY, f64::max);
                (q, z[target] - wrong)
            })
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        order.truncate(order.len().div_ceil(10));
        order.into_iter().map(|(q, _)| q).collect()
    })
}
pub fn subset(indices: &[usize], rows: &[Row]) -> Result<Value> {
    let mut methods: [Method; 2] = Default::default();
    let (mut margin, mut loss, mut correct) = ([0.; 2], [0.; 2], [0; 2]);
    let mut minimum = [f64::INFINITY; 2];
    let (mut up, mut down, mut loss_down, mut loss_up, mut gain, mut lost, mut switched) =
        (0, 0, 0, 0, 0, 0, 0);
    for &q in indices {
        let r = &rows[q];
        for j in 0..2 {
            margin[j] += r.margin[j];
            loss[j] += r.loss[j];
            correct[j] += usize::from(r.correct[j]);
            minimum[j] = minimum[j].min(r.margin[j]);
            methods[j].add(r, j)?;
        }
        up += usize::from(r.margin[1] > r.margin[0]);
        down += usize::from(r.margin[1] < r.margin[0]);
        loss_down += usize::from(r.loss[1] < r.loss[0]);
        loss_up += usize::from(r.loss[1] > r.loss[0]);
        gain += usize::from(!r.correct[0] && r.correct[1]);
        lost += usize::from(r.correct[0] && !r.correct[1]);
        switched += usize::from(r.worst[0] != r.worst[1]);
    }
    let n = indices.len();
    let mean = |a: [f64; 2]| {
        if n > 0 {
            Some(a.map(|x| x / n as f64))
        } else {
            None
        }
    };
    Ok(
        json!({"queries":n,"old_new_mean_margin":mean(margin),"old_new_mean_loss":mean(loss),
        "old_new_minimum_margin":if n>0 {Some(minimum)}else{None},"old_new_correct":correct,
        "actual_margin_improved":up,"actual_margin_regressed":down,
        "actual_loss_decreased":loss_down,"actual_loss_increased":loss_up,
        "correct_gained":gain,"correct_lost":lost,"worst_competitor_switched":switched,
        "joint":methods[0].json(n),"separate":methods[1].json(n),"false_certificates":0}),
    )
}

pub fn audit(
    old: &Snapshot,
    new: &Snapshot,
    old_e: &[f64],
    new_e: &[f64],
    d: &Direction,
    data: &Data,
    difficult: &[Vec<usize>; 2],
) -> Result<(Value, Vec<Row>)> {
    let mut labels = vec![usize::MAX; 9409];
    for r in data
        .train
        .as_chunks::<3>()
        .0
        .iter()
        .chain(data.validation.as_chunks::<3>().0)
    {
        labels[(r[0] * 97 + r[1]) as usize] = r[2] as usize;
    }
    let increment: Vec<_> = old_e
        .iter()
        .zip(&d.forcing)
        .map(|(e, f)| (d.decay - 1.) * e - d.eta * f)
        .collect();
    let rows = query_step::analyze(Problem {
        old_features: &old.features,
        new_features: &new.features,
        old_embedding: old_e,
        new_embedding: new_e,
        increment: &increment,
        rounding: &d.rounding,
        old_logits: &old.logits,
        new_logits: &new.logits,
        labels: &labels,
        width: W,
        classes: 97,
    })?;
    let mut result = json!({});
    for (split, (name, input)) in [("train", &data.train), ("validation", &data.validation)]
        .into_iter()
        .enumerate()
    {
        let indices: Vec<_> = input
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| (r[0] * 97 + r[1]) as usize)
            .collect();
        let all = subset(&indices, &rows)?;
        let difficult = &difficult[split];
        require(
            *difficult == hardest(&indices, &rows),
            "frozen difficult queries differ from old-margin selector",
        )?;
        let errors: Vec<_> = indices
            .iter()
            .copied()
            .filter(|&q| !rows[q].correct[0])
            .collect();
        for (j, s) in [old, new].iter().enumerate() {
            let measured = all["old_new_mean_loss"][j].as_f64().unwrap();
            let native = s.metrics[name]["loss"].as_f64().unwrap();
            require(
                (measured - native).abs() <= 2e-6 * (1. + native),
                "executed-logit CE versus GPU loss mismatch",
            )?;
            require(
                all["old_new_correct"][j] == s.metrics[name]["correct"],
                "query versus GPU accuracy mismatch",
            )?;
        }
        result[name] = json!({"all":all,"old_bottom_decile":subset(difficult,&rows)?,
            "old_error_rows":subset(&errors,&rows)?,"old_bottom_decile_indices":difficult,
            "old_error_indices":errors});
    }
    let max = |f: fn(&Row) -> f64| rows.iter().map(f).fold(0., f64::max);
    let terms: [f64; 6] = std::array::from_fn(|i| {
        rows.iter().map(|r| r.mean_abs_terms[i]).sum::<f64>() / rows.len() as f64
    });
    result["checks"] = json!({"queries":rows.len(),"competitor_pairs":rows.len()*96,
        "max_scaled_identity_error":max(|r|r.identity_error),"max_scaled_gap_bound_excess":max(|r|r.bound_error),
        "max_scaled_weighted_loss_error":max(|r|r.weighted_error),"passed":true});
    result["mean_absolute_contributions"] = json!({"names":["joint_first_order","separate_first_order_absolute_sum",
        "actual_mixed_term","mixed_rounding_execution_radius","actual_parameter_rounding",
        "actual_execution_change"],"values":terms});
    result["row_fields"] = json!(query_step::ROW_FIELDS);
    result["rows_by_canonical_pair_index"] = json!(rows.iter().map(Row::json).collect::<Vec<_>>());
    Ok((result, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn difficult_subset_is_fixed_before_the_new_outcome() {
        let rows = query_step::fixture(
            &[0., 1.],
            &[0., 0.],
            &[0., 0.],
            &[0.1, 0.5, -1.],
            &[-3., -2., 5.],
            &[1; 3],
            1,
        );
        assert_eq!(hardest(&[0, 1, 2], &rows), [2]);
        let s = subset(&[2], &rows).unwrap();
        assert_eq!(s["old_new_correct"], json!([0, 1]));
        assert_eq!(s["queries"], 1);
        assert!(subset(&[], &rows).unwrap()["old_new_mean_loss"].is_null());
    }
}

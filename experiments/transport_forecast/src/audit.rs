//! Post-freeze audit of every bridge endpoint, including uncertified queries.
use crate::{
    bridge::{Bridge, METHODS},
    domain::{Result, data::Data},
    inputs::require,
    measure::Snapshot,
    query_step::tolerance,
    reconstruction::RANKS,
};
use serde_json::{Value, json};
pub fn loss(gaps: &[f64]) -> f64 {
    let m = gaps.iter().copied().fold(0., f64::max);
    m + gaps.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}
fn winner(scores: &[f64]) -> usize {
    (1..scores.len()).fold(0, |best, c| if scores[c] > scores[best] { c } else { best })
}
pub fn run(b: &Bridge, old: &Snapshot, new: &Snapshot, data: &Data) -> Result<Value> {
    let mut result = json!({});
    let mut violations = 0;
    let mut false_certificates = 0;
    let mut query_records = vec![];
    for (name, rows) in [("train", &data.train), ("validation", &data.validation)] {
        let mut stats = vec![];
        for (method, upper) in b.upper.iter().enumerate() {
            let scores = if method % 3 == 0 {
                &old.logits
            } else {
                &new.logits
            };
            let mut count = 0;
            let mut certified = 0;
            let mut correct = 0;
            let mut abstained = 0;
            let mut ceiling = 0.;
            let mut floor = 0.;
            let mut min_margin = f64::INFINITY;
            let mut mean_margin = 0.;
            for row in rows.as_chunks::<3>().0 {
                let q = (row[0] * 97 + row[1]) as usize;
                let truth = row[2] as usize;
                count += 1;
                let z = &scores[q * 97..(q + 1) * 97];
                let right = winner(z) == truth;
                correct += usize::from(right);
                let Some(t) = b.labels[q] else {
                    abstained += 1;
                    continue;
                };
                require(
                    t == truth,
                    "transferred label disagrees with separate target audit",
                )?;
                let u = &upper[q * 97..(q + 1) * 97];
                let max_gap = (0..97)
                    .filter(|&c| c != t)
                    .map(|c| u[c])
                    .fold(f64::NEG_INFINITY, f64::max);
                let pass = max_gap < -tolerance(&[max_gap]);
                certified += usize::from(pass);
                false_certificates += usize::from(pass && !right);
                for c in 0..97 {
                    let actual = z[c] - z[t];
                    violations += usize::from(actual > u[c] + tolerance(&[actual, u[c]]));
                }
                let ce = loss(u);
                let actual_ce = loss(&z.iter().map(|v| v - z[t]).collect::<Vec<_>>());
                violations += usize::from(actual_ce > ce + tolerance(&[actual_ce, ce]));
                ceiling += ce;
                floor += (-ce).exp();
                min_margin = min_margin.min(-max_gap);
                mean_margin -= max_gap;
                query_records.push(json!({"split":name,"query":q,"rank":RANKS[method/3],"method":METHODS[method%3],"label":t,"correct":right,"certified":pass,"margin_lower":-max_gap,"ce_upper":ce,"probability_lower":(-ce).exp()}));
            }
            stats.push(json!({"rank":RANKS[method/3],"method":METHODS[method%3],"count":count,"correct":correct,"certified":certified,"coverage":certified as f64/count as f64,"abstained":abstained,
                "mean_ce_upper":if abstained==0 {Some(ceiling/count as f64)} else {None},"mean_probability_lower":floor/count as f64,"minimum_margin_lower":min_margin,"mean_margin_lower":mean_margin/(count-abstained).max(1) as f64}));
        }
        result[name] = json!(stats);
    }
    result["violations"] = json!(violations);
    result["false_certificates"] = json!(false_certificates);
    result["passed"] = json!(violations == 0 && false_certificates == 0);
    result["queries"] = json!(query_records);
    Ok(result)
}

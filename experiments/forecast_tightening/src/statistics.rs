//! All-query retained counts; frozen old-margin subsets are selected separately.
use crate::{
    intervals::{Frozen, NAMES, TRACKS},
    query_step::{Interval, tolerance},
};
use serde_json::{Value, json};
pub fn distribution(x: impl IntoIterator<Item = f64>) -> Value {
    let mut v: Vec<_> = x.into_iter().collect();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return Value::Null;
    }
    json!({"count":v.len(),"mean":v.iter().sum::<f64>()/v.len()as f64,"min":v[0],
        "median":v[v.len()/2],"p90":v[(9*v.len()).div_ceil(10)-1],"max":v[v.len()-1]})
}
pub struct Actual {
    pub old: Vec<Interval>,
    pub new: Vec<Interval>,
    pub correct: Vec<bool>,
}
pub fn subset(indices: &[usize], actual: &Actual, frozen: &Frozen) -> Value {
    let n = indices.len();
    let methods:Vec<_>=(0..TRACKS).map(|m|{
        let (mut correct,mut loss_down,mut loss_up,mut margin_up,mut margin_down,mut invalid)=(0,0,0,0,0,0);
        for &q in indices{
            let b=frozen.outcomes[m][q];let o=actual.old[q];let a=actual.new[q];
            // A valid interval already dominates the enclosed future value's
            // absolute scale. Strictness uses only information available at freeze.
            let mt=tolerance(&[o.margin[0],b.margin[0],b.margin[1]]);
            let lt=tolerance(&[o.loss[0],b.loss[0],b.loss[1]]);
            let c=b.margin[0]>mt;let md=b.margin[1]<o.margin[0]-mt;let mu=b.margin[0]>o.margin[0]+mt;
            let ld=b.loss[1]<o.loss[0]-lt;let lu=b.loss[0]>o.loss[0]+lt;
            correct+=usize::from(c);margin_up+=usize::from(mu);margin_down+=usize::from(md);
            loss_down+=usize::from(ld);loss_up+=usize::from(lu);
            invalid+=usize::from((c&&!actual.correct[q])||(md&&a.margin[0]>=o.margin[0])||
                (mu&&a.margin[0]<=o.margin[0])||(ld&&a.loss[0]>=o.loss[0])||(lu&&a.loss[0]<=o.loss[0]));
        }
        json!({"method":NAMES[m],"certified_correct":correct,"coverage":correct as f64/n as f64,
            "certified_loss_decreased":loss_down,"certified_loss_increased":loss_up,"loss_inconclusive":n-loss_down-loss_up,
            "certified_margin_improved":margin_up,"certified_margin_regressed":margin_down,"false_certificates":invalid,
            "margin_width":distribution(indices.iter().map(|&q|{let b=frozen.outcomes[m][q];b.margin[1]-b.margin[0]})),
            "loss_width":distribution(indices.iter().map(|&q|{let b=frozen.outcomes[m][q];b.loss[1]-b.loss[0]}))})
    }).collect();
    json!({"count":n,"actually_correct":indices.iter().filter(|&&q|actual.correct[q]).count(),"methods":methods})
}
